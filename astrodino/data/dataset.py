# Dataset file for DESI Legacy Survey data
import logging
import os
from enum import Enum
from typing import Any, Callable, Optional, Tuple, Union

import h5py
import numpy as np
import torch
from torchvision.datasets import VisionDataset
from datasets import load_from_disk

from dataset_util.PairDataset import normalize
from image_util import CustomExtinction

logger = logging.getLogger("astrodino")
_Target = float


class _SplitFull(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"  # NOTE: torchvision does not support the test split

    @property
    def length(self) -> int:
        split_lengths = {
            _SplitFull.TRAIN: 74_500_000,
            _SplitFull.VAL: 100_000,
            _SplitFull.TEST: 400_000,
        }
        return split_lengths[self]


# 定义LegacySurvey类，继承自VisionDataset
class LegacySurvey(VisionDataset):
    # 定义类型别名Target和Split，用于类型提示
    Target = Union[_Target]
    Split = Union[_SplitFull]

    def __init__(
        self,
        *,
        split: "LegacySurvey.Split",  # 数据集分割类型（训练/验证/测试）
        root: str,  # 数据集根目录路径
        extra: str = None,  # 额外数据路径（可选）
        transforms: Optional[Callable] = None,  # 图像变换函数（可选）
        transform: Optional[Callable] = None,  # 单例图像变换（可选）
        target_transform: Optional[Callable] = None,  # 目标变换函数（可选）
    ) -> None:
        # 调用父类构造函数初始化
        super().__init__(root, transforms, transform, target_transform)
        self._extra_root = extra  # 保存额外数据路径
        self._split = split  # 保存数据集分割类型

        # 打开HDF5文件（北方区域）
        self._files = [
            h5py.File(
                os.path.join(
                    root, "north/images_npix152_0%02d000000_0%02d000000.h5" % (i, i + 1)
                )
            )
            for i in range(14)  # 加载14个北方区域的HDF5文件
        ]
        # 添加南方区域的HDF5文件
        self._files += [
            h5py.File(
                os.path.join(
                    root, "south/images_npix152_0%02d000000_0%02d000000.h5" % (i, i + 1)
                )
            )
            for i in range(61)  # 加载61个南方区域的HDF5文件
        ]

        # 创建随机排列的索引数组（总大小7.5e7）
        rng = np.random.default_rng(seed=42)  # 设置随机种子保证可重复性
        self._indices = rng.permutation(int(7.5e7))
        # 根据分割类型划分数据集
        if split == LegacySurvey.Split.TRAIN.value:
            self._indices = self._indices[:74_500_000]  # 训练集
        elif split == LegacySurvey.Split.VAL.value:
            self._indices = self._indices[74_500_000:-400_000]  # 验证集
        else:
            self._indices = self._indices[-400_000:]  # 测试集

    @property
    def split(self) -> "LegacySurvey.Split":
        return self._split  # 返回当前数据集分割类型

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        # 将索引映射到实际文件位置
        true_index = self._indices[index]
        # 从对应文件中读取图像数据并转换为float32类型
        image = self._files[true_index // int(1e6)]["images"][
            true_index % int(1e6)
        ].astype("float32")
        image = torch.tensor(image)  # 转换为PyTorch张量
        target = None  # 当前无目标标签

        # 应用数据增强变换
        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target  # 返回图像和对应标签

    def __len__(self) -> int:
        return len(self._indices)  # 返回数据集长度

class QuasarDataset(VisionDataset):
    def __init__(
            self,
            *,
            split: "LegacySurvey.Split",  # 数据集分割类型（训练/验证/测试）
            root: str,  # 数据集根目录路径
            extra: str = None,  # 额外数据路径（可选）
            transforms: Optional[Callable] = None,  # 图像变换函数（可选）
            transform: Optional[Callable] = None,  # 单例图像变换（可选）
            target_transform: Optional[Callable] = None,  # 目标变换函数（可选）
            extinction=False,
            probs=False
    ) -> None:
        # 调用父类构造函数初始化
        super().__init__(root, transforms, transform, target_transform)
        self._split = split  # 保存数据集分割类型
        self.root  = root
        self.data= load_from_disk(self.root)
        self.extinction = extinction
        self.probs = probs

    @property
    def split(self):
        return self._split  # 返回当前数据集分割类型

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx) -> Tuple[Any, Any]:
        img = self.data[idx]['image']
        target = None  # 当前无目标标签
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

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img,target


class _SplitNorth(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"  # NOTE: torchvision does not support the test split

    @property
    def length(self) -> int:
        split_lengths = {
            _SplitNorth.TRAIN: 13_500_000,
            _SplitNorth.VAL: 100_000,
            _SplitNorth.TEST: 400_000,
        }
        return split_lengths[self]


class LegacySurveyNorth(VisionDataset):
    Target = Union[_Target]
    Split = Union[_SplitNorth]

    def __init__(
        self,
        *,
        split: "LegacySurvey.Split",
        root: str,
        extra: str = None,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self._extra_root = extra
        self._split = split

        # We start by opening the hdf5 files located at the root directory
        self._files = [
            h5py.File(
                os.path.join(
                    root, "north/images_npix152_0%02d000000_0%02d000000.h5" % (i, i + 1)
                )
            )
            for i in range(14)
        ]

        # Create randomized array of indices
        rng = np.random.default_rng(seed=42)
        self._indices = rng.permutation(int(1.4e7))
        if split == LegacySurvey.Split.TRAIN.value:
            self._indices = self._indices[:13_500_000]
        elif split == LegacySurvey.Split.VAL.value:
            self._indices = self._indices[13_500_000:-400_000]
        else:
            self._indices = self._indices[-400_000:]

    @property
    def split(self) -> "LegacySurvey.Split":
        return self._split

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        true_index = self._indices[index]
        image = self._files[true_index // int(1e6)]["images"][
            true_index % int(1e6)
        ].astype("float32")
        image = torch.tensor(image)
        target = None

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self) -> int:
        return len(self._indices)
