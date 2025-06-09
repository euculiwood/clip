import torch
from torch.utils.data import Dataset
from image_util import CustomExtinction

def normalize(tensor):
    """进行 Min-Max 归一化"""
    min_val = tensor.min()
    max_val = tensor.max()
    return (tensor - min_val) / (max_val - min_val)


class PairDataset(Dataset):
    def __init__(self, data, transform=None, extinction=False, probs = False):
        self.data = data
        self.transform = transform
        self.extinction = extinction
        self.probs = probs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]['image']
        spec = self.data[idx]['spectrum']
        z = self.data[idx]['z']
        spec = spec.clone().detach().unsqueeze(-1)
        if self.probs:
            probs = self.data[idx]['params']
            diff = [abs(probs[i] - probs[j]) for i in range(len(probs)) for j in range(i + 1, len(probs))]
            diff = torch.tensor(diff)  # 将 diff 转换为 Tensor
            probs = torch.cat((probs, diff))  # 使用 torch.cat 进行拼接
            probs = normalize(probs)

        if self.extinction:
            params = self.data[idx]['params']
            # 确保 params 长度足够
            if len(params) >= 10:
                ext_u, ext_g, ext_r, ext_i, ext_z = params[5:10]  # 提取消光系数
                extinction = CustomExtinction(ext_u, ext_g, ext_r, ext_i, ext_z)
                img = extinction(img)
            else:
                raise ValueError(f"参数长度不足，无法提取消光系数，当前长度为 {len(params)}")

        if self.transform:
            img = self.transform(img)


        if self.probs:
            return {
                "image": img,
                "spectrum": spec,
                "probs": probs,
                "z": z
            }

        return {
            "image": img,
            "spectrum": spec,
            "z" :z
        }
