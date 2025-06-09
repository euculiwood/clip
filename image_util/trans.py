import torch
import torch.nn.functional as F
import random
import math
from torchvision import transforms



class CustomRandomHorizontalFlip(object):
    """随机水平翻转5通道图像的自定义转换类."""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if not torch.is_tensor(img):
            raise TypeError(f'输入图像应该是一个 torch.Tensor，但得到的是 {type(img)}')

        if torch.rand(1).item() < self.p:
            img = torch.flip(img, dims=[2])  # 假设形状为 (C, H, W)
        return img


class CustomRandomVerticalFlip(object):
    """随机垂直翻转多通道图像的自定义转换类."""

    def __init__(self, p=0.5):
        """
        Args:
            p (float): 图像垂直翻转的概率，默认为0.5。
        """
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (Tensor): 要翻转的图像，形状为 (C, H, W)。

        Returns:
            Tensor: 随机翻转后的图像。
        """
        if not torch.is_tensor(img):
            raise TypeError(f'输入图像应该是一个 torch.Tensor，但得到的是 {type(img)}')

        if torch.rand(1).item() < self.p:
            img = torch.flip(img, dims=[1])  # 假设形状为 (C, H, W)
            # print("Vertical flip performed.")
        # else:
        #     print("Vertical flip not performed.")
        return img


class CustomRandomRotation(object):
    """随机旋转多通道图像的自定义转换类，带有概率控制."""

    def __init__(self, degrees, p=0.5, resample='bilinear'):
        """
        Args:
            degrees (sequence or float or int): 旋转角度范围。如果是序列，表示范围 (min, max)。
                                               如果是一个数，则范围为 (-degrees, +degrees)。
            p (float): 执行旋转的概率。默认为0.5。
            resample (str): 插值方法，如 'nearest', 'bilinear'。默认为 'bilinear'。
        """
        if isinstance(degrees, (int, float)):
            self.degrees = (-degrees, degrees)
        else:
            self.degrees = degrees
        if resample not in ['nearest', 'bilinear']:
            raise ValueError(f"Unsupported resample mode: {resample}. Use 'nearest' or 'bilinear'.")
        self.resample = resample
        self.p = p  # 添加概率参数

    def __call__(self, img):
        """
        Args:
            img (Tensor): 要旋转的图像，形状为 (C, H, W)。

        Returns:
            Tensor: 随机旋转后的图像（根据概率决定是否旋转）。
        """
        if not torch.is_tensor(img):
            raise TypeError(f'输入图像应该是一个 torch.Tensor，但得到的是 {type(img)}')

        # 决定是否执行旋转
        if torch.rand(1).item() < self.p:
            angle = random.uniform(self.degrees[0], self.degrees[1])
            angle_rad = math.radians(angle)

            # 创建旋转矩阵
            theta = torch.tensor([
                [math.cos(angle_rad), -math.sin(angle_rad), 0],
                [math.sin(angle_rad), math.cos(angle_rad), 0]
            ], dtype=torch.float)
            theta = theta.unsqueeze(0)  # 扩展为 (1, 2, 3)

            # 生成仿射网格
            grid = F.affine_grid(theta, img.unsqueeze(0).size(), align_corners=False)

            # 选择插值模式
            if self.resample == 'bilinear':
                mode = 'bilinear'
            elif self.resample == 'nearest':
                mode = 'nearest'

            # 应用网格采样进行旋转
            rotated = F.grid_sample(img.unsqueeze(0), grid, mode=mode, padding_mode='zeros', align_corners=False)
            # print("Rotation performed.")

            return rotated.squeeze(0)
        else:
            # print("Rotation not performed.")
            return img


import torch


class CustomCenterCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        # 假设 sample 是一个 5 通道的图像，尺寸为 (C, H, W)
        c, h, w = sample.shape
        new_h, new_w = self.size, self.size

        # 计算裁剪区域的起始位置
        top = (h - new_h) // 2
        left = (w - new_w) // 2

        # 执行裁剪
        cropped_sample = sample[:, top:top + new_h, left:left + new_w]
        # print(f"Cropped sample shape: {cropped_sample.shape}")
        return cropped_sample


class CustomReshapePermute:
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        # print(f"Input tensor shape before reshape: {img.shape}")
        reshaped_tensor = img.reshape(self.size, self.size, 5)
        return reshaped_tensor.permute(2, 0, 1)  # 假设需要转置维度


class CustomExpStretchWithOffset:
    def __init__(self, a, b):
        self.a = a  # 控制指数拉伸的强度（α）
        self.b = b

    def __call__(self, tensor):
        # 执行指数拉伸并减去 1
        stretched_tensor = self.b * (torch.exp(self.a * tensor) - 1)

        # # 可以选择将图像数据归一化到 [0, 1] 或其他范围
        # stretched_tensor = stretched_tensor / stretched_tensor.max()
        # tensor_min = stretched_tensor.min()
        # tensor_max = stretched_tensor.max()
        # normalized_tensor = (stretched_tensor - tensor_min) / (tensor_max - tensor_min)

        return stretched_tensor  # 返回拉伸后的图像 Tensor


class CustomRandom:
    def __init__(self):
        pass

    def __call__(self, tensor):
        # 将原始 tensor 替换为 0 到 255 之间的随机整数
        random_tensor = torch.randint(0, 256, tensor.shape, dtype=torch.float32)  # 生成 [0, 255] 之间的随机整数
        # 执行指数拉伸并减去 1
        # stretched_tensor = self.b * (torch.exp(self.a * random_tensor) - 1)
        return random_tensor  # 返回拉伸后的图像 Tensor


class CustomExtinction:
    def __init__(self, u, g, r, i, z):
        self.u = u
        self.g = g
        self.r = r
        self.i = i
        self.z = z

    def __call__(self, tensor):
        corrected_u = tensor[0, :, :] * torch.pow(10.0, 0.4 * self.u)
        corrected_g = tensor[1, :, :] * torch.pow(10.0, 0.4 * self.g)
        corrected_r = tensor[2, :, :] * torch.pow(10.0, 0.4 * self.r)
        corrected_i = tensor[3, :, :] * torch.pow(10.0, 0.4 * self.i)
        corrected_z = tensor[4, :, :] * torch.pow(10.0, 0.4 * self.z)
        corrected_tensor = torch.stack([corrected_u, corrected_g, corrected_r, corrected_i, corrected_z], dim=0)
        return corrected_tensor

