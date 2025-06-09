# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging
from typing import List

import numpy as np
import skimage.filters
import skimage.transform
import torch
from torchvision import transforms

logger = logging.getLogger("dinov2")


class DataAugmentationAstroDINO(object):
    def __init__(
        self,
        global_crops_scale,  # 全局裁剪的缩放比例范围
        local_crops_scale,   # 局部裁剪的缩放比例范围
        local_crops_number,  # 局部裁剪的数量
        global_crops_size=144,  # 全局裁剪的目标尺寸
        local_crops_size=60,    # 局部裁剪的目标尺寸
    ):
        # 初始化参数赋值
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number
        self.global_crops_size = global_crops_size
        self.local_crops_size = local_crops_size

        # 记录数据增强参数到日志
        logger.info("###################################")
        logger.info("Using data augmentation parameters:")
        logger.info(f"global_crops_scale: {global_crops_scale}")
        logger.info(f"local_crops_scale: {local_crops_scale}")
        logger.info(f"local_crops_number: {local_crops_number}")
        logger.info(f"global_crops_size: {global_crops_size}")
        logger.info(f"local_crops_size: {local_crops_size}")
        logger.info("###################################")

        # 定义全局裁剪的几何变换组合
        self.geometric_augmentation_global = transforms.Compose(
            [
                transforms.RandomCrop(global_crops_size),  # 随机裁剪到指定尺寸
                transforms.RandomHorizontalFlip(p=0.5),      # 50%概率水平翻转
                transforms.RandomVerticalFlip(p=0.5),        # 50%概率垂直翻转
            ]
        )

        # 定义局部裁剪的几何变换组合
        self.geometric_augmentation_local = transforms.Compose(
            [
                transforms.RandomCrop(local_crops_size),  # 随机裁剪到局部尺寸
                transforms.RandomHorizontalFlip(p=0.5),     # 水平翻转
                transforms.RandomVerticalFlip(p=0.5),       # 垂直翻转
            ]
        )

        # 定义全局变换的额外增强操作（增强1）
        global_transfo1_extra = transforms.Compose(
            [
                RandomGaussianBlur(p=1.0),              # 100%概率高斯模糊
                RandomGaussianNoise(p=1.0, im_dim=global_crops_size),  # 100%概率高斯噪声
            ]
        )

        # 定义全局变换的额外增强操作（弱增强2）
        global_transfo2_extra = transforms.Compose(
            [
                RandomGaussianBlur(p=0.1),              # 10%概率高斯模糊
                RandomGaussianNoise(p=0.1, im_dim=global_crops_size),  # 10%概率高斯噪声
            ]
        )

        # 定义局部变换的额外增强操作
        local_transfo_extra = transforms.Compose(
            [
                RandomGaussianBlur(p=0.5),             # 50%概率高斯模糊
                RandomGaussianNoise(p=0.5, im_dim=local_crops_size),  # 50%概率高斯噪声
            ]
        )

        # 转换为RGB颜色空间
        # to_rgb = ToRGB()

        # 组合全局变换流程（强增强）
        # self.global_transfo1 = transforms.Compose([global_transfo1_extra, to_rgb])
        self.global_transfo1 = transforms.Compose([global_transfo1_extra])
        # 组合全局变换流程（弱增强）
        # self.global_transfo2 = transforms.Compose([global_transfo2_extra, to_rgb])
        self.global_transfo2 = transforms.Compose([global_transfo2_extra])
        # 组合局部变换流程
        # self.local_transfo = transforms.Compose([local_transfo_extra, to_rgb])
        self.local_transfo = transforms.Compose([local_transfo_extra])

    def __call__(self, image):
        output = {}

        # 生成全局裁剪1
        im1_base = np.array(self.geometric_augmentation_global(image))
        global_crop_1 = torch.tensor(self.global_transfo1(im1_base))

        # 生成全局裁剪2
        im2_base = np.array(self.geometric_augmentation_global(image))
        global_crop_2 = torch.tensor(self.global_transfo2(im2_base))

        # 存储全局裁剪结果
        output["global_crops"] = [global_crop_1, global_crop_2]
        # 为教师网络存储相同裁剪
        output["global_crops_teacher"] = [global_crop_1, global_crop_2]

        # 生成局部裁剪
        local_crops = [
            torch.tensor(
                self.local_transfo(np.array(self.geometric_augmentation_local(image)))
            )
            for _ in range(self.local_crops_number)  # 根据数量生成多个局部裁剪
        ]
        output["local_crops"] = local_crops
        output["offsets"] = ()  # 空偏移量占位符

        return output


class RandomGaussianBlur(transforms.RandomApply):
    """Randomly apply Gaussian blur to the image."""

    def __init__(self, *, p: float = 0.5):
        keep_p = 1 - p
        transform = GaussianBlur()
        super().__init__([transform], p=keep_p)


class RandomGaussianNoise(transforms.RandomApply):
    """Randomly apply Gaussian noise to the image."""

    def __init__(self, *, im_dim=144, p: float = 0.5):
        keep_p = 1 - p
        transform = GaussianNoise(im_dim=im_dim)
        super().__init__([transform], p=keep_p)


class ToRGB:
    """
    Transformation from raw image data (nanomaggies) to the rgb values displayed
    at the legacy viewer https://www.legacysurvey.org/viewer

    Code copied from
    https://github.com/legacysurvey/imagine/blob/master/map/views.py
    """

    def __init__(self, scales=None, m=0.03, Q=20, bands=["g", "r", "z"]):
        # 默认波段缩放参数：(plane, scale)
        rgb_scales = {
            "u": (2, 1.5),
            "g": (2, 6.0),
            "r": (1, 3.4),
            "i": (0, 1.0),
            "z": (0, 2.2),
        }
        if scales is not None:
            rgb_scales.update(scales)  # 允许自定义缩放参数

        self.rgb_scales = rgb_scales  # 保存波段缩放配置
        self.m = m                    # 线性变换截距项
        self.Q = Q                    # 亮度调整参数
        self.bands = bands            # 使用的波段顺序
        # 获取各波段的plane和scale参数
        self.axes, self.scales = zip(*[rgb_scales[bands[i]] for i in range(len(bands))])
        # 按照axes重新排列scale顺序以匹配图像通道
        self.scales = [self.scales[i] for i in self.axes]

    def __call__(self, imgs):
        # 处理输入图像，转换为C x H x W格式
        if imgs.shape[0] != len(self.bands):
            imgs = np.transpose(imgs, (2, 0, 1))

        I = 0
        # 对每个波段进行处理
        for img, band in zip(imgs, self.bands):
            plane, scale = self.rgb_scales[band]  # 获取波段参数
            img = np.maximum(0, img * scale + self.m)  # 应用线性变换并截断负值
            I = I + img  # 累加所有波段的强度
        I /= len(self.bands)  # 计算平均强度

        Q = 20
        # 计算增强后的强度值（使用双曲正弦函数）
        fI = np.arcsinh(Q * I) / np.sqrt(Q)
        I += (I == 0.0) * 1e-6  # 避免除以零，添加极小值
        H, W = I.shape  # 获取图像尺寸
        # 创建RGB输出数组
        rgb = np.zeros((H, W, 3), np.float32)
        # 将处理后的波段值分配到对应RGB通道
        for img, band in zip(imgs, self.bands):
            plane, scale = self.rgb_scales[band]
            # 应用变换并归一化
            rgb[:, :, plane] = (img * scale + self.m) * fI / I

        rgb = np.clip(rgb, 0, 1)  # 裁剪到[0,1]范围
        return rgb  # 返回处理后的RGB图像


class GaussianNoise:
    """
    Augmentations tuned to the Legacy Survey Data (with minor modifications).

    Code copied from
    https://github.com/georgestein/ssl-legacysurvey/blob/main/ssl_legacysurvey/data_loaders/decals_augmentations.py#L296
    """

    def __init__(
        self,
        scaling: List = [1.0],
        mean: float = 0,
        im_dim: int = 144,
        im_ch: int = 3,
        decals: bool = True,
        uniform: bool = False,
    ):
        self.mean = mean
        self.decals = decals
        self.im_ch = im_ch
        self.im_dim = im_dim
        self.uniform = uniform

        # Log normal fit paramaters
        self.shape_dist = np.array([0.2264926, 0.2431146, 0.1334844])
        self.loc_dist = np.array([-0.0006735, -0.0023663, -0.0143416])
        self.scale_dist = np.array([0.0037602, 0.0067417, 0.0260779])

        self.sigma_dist = np.log(self.scale_dist)

        # noise in channels is uncorrelated, as images taken at dirrerent times/telescopes
        self.noise_ch_min = np.array([0.001094, 0.001094, 0.001094])
        self.noise_ch_max = np.array([0.013, 0.018, 0.061])

    def __call__(self, image: np.ndarray):
        # draw 'true' noise level of each channel from lognormal fits
        self.sigma_true = (
            np.random.lognormal(self.sigma_dist, self.shape_dist) + self.loc_dist
        )

        if self.uniform:
            # draw desired augmented noise level from uniform, to target tails more
            self.sigma_final = np.random.uniform(self.noise_ch_min, self.noise_ch_max)
        else:
            self.sigma_final = (
                np.random.lognormal(self.sigma_dist, self.shape_dist) + self.loc_dist
            )

        # Gaussian noise adds as c^2 = a^2 + b^2
        self.sigma_augment = self.sigma_final**2 - self.sigma_true**2
        self.sigma_augment[self.sigma_augment < 0.0] = 0.0
        self.sigma_augment = np.sqrt(self.sigma_augment)

        for i in range(self.im_ch):
            if self.sigma_augment[i] > 0.0:
                image[i, :, :] += np.random.normal(
                    self.mean, self.sigma_augment[i], size=(self.im_dim, self.im_dim)
                )

        return image


class GaussianBlur:
    """
    Augmentations tuned to the Legacy Survey Data (with minor modifications).

    Code copied from
    https://github.com/georgestein/ssl-legacysurvey/blob/main/ssl_legacysurvey/data_loaders/decals_augmentations.py#L296
    """

    def __init__(
        self,
        scaling: List = [1.0],
        im_dim: int = 144,
        im_ch: int = 3,
        decals: bool = True,
        uniform: bool = False,
    ):
        self.decals = decals
        self.im_ch = im_ch
        self.im_dim = im_dim
        self.uniform = uniform

        # Log normal fit paramaters
        self.shape_dist = np.array([0.2109966, 0.3008485, 0.3471172])
        self.loc_dist = np.array([1.0807153, 1.2394326, 1.1928363])
        self.scale_dist = np.array([1.3153171, 0.9164757, 0.8233702])

        self.sigma_dist = np.log(self.scale_dist)

        self.psf_ch_min = np.array([1.3233109, 1.2667341, 1.2126263])
        self.psf_ch_max = np.array([5.0, 4.5, 4.25])

    def __call__(self, image: np.ndarray):
        # noise in channels is uncorrelated, as images taken at different times/telescopes
        # draw 'true' noise level of each channel from lognormal fits
        self.sigma_true = (
            np.random.lognormal(self.sigma_dist, self.shape_dist) + self.loc_dist
        )

        if self.uniform:
            # draw desired augmented noise level from uniform, to target tails more
            self.sigma_final = np.random.uniform(self.psf_ch_min, self.psf_ch_max)
        else:
            self.sigma_final = (
                np.random.lognormal(self.sigma_dist, self.shape_dist) + self.loc_dist
            )

        # Gaussian noise adds as c^2 = a^2 + b^2
        self.sigma_augment = self.sigma_final**2 - self.sigma_true**2
        self.sigma_augment[self.sigma_augment < 0.0] = 0.0
        self.sigma_augment = np.sqrt(self.sigma_augment)

        for i in range(self.im_ch):
            if self.sigma_augment[i] > 0.0:
                image[i, :, :] = skimage.filters.gaussian(
                    image[i, :, :], sigma=self.sigma_augment[i], mode="reflect"
                )

        return image
