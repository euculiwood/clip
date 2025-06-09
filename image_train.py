import sys, os

from root_path import ROOT_PATH

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, random_split
# 添加缺失的导入
from datasets import load_from_disk  # 假设使用的是HuggingFace datasets库
from dataset_util.PairDataset import PairDataset
from models.resnet18 import resnet18
import torch
import torchvision.transforms as transforms
import os
from run_name import RUN_NAME, RATIO


# def get_mean_std(train_data):
#     """Calculate mean and std of training dataset."""
#     # 获取所有图像张量并堆叠
#     all_images = [sample['image'] for sample in train_data]
#     # 堆叠为 (N, 5, 64, 64) 的张量
#     images_tensor = torch.stack(all_images)
#
#     # 计算通道维度(0)的均值和标准差
#     mean = images_tensor.mean(dim=[0, 2, 3]).tolist()
#     std = images_tensor.std(dim=[0, 2, 3]).tolist()
#
#     return mean, std


def train_model(dataset, val_ratio=0.2, batch_size=32, max_epochs=100, ckpt_path=None, resume_ckpt=None):
    """
    训练模型
    
    Args:
        dataset: 完整的训练数据集
        val_ratio: 验证集比例
        batch_size: 批次大小
        max_epochs: 最大训练轮数
        ckpt_path: 模型检查点保存路径
        resume_ckpt: 恢复训练的检查点路径
    """
    # 确保检查点目录存在
    os.makedirs(ckpt_path, exist_ok=True)

    # 计算训练集和验证集的大小
    total_size = len(dataset)
    val_size = int(total_size * val_ratio)
    train_size = total_size - val_size

    # 随机拆分训练集和验证集
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # 设置随机种子以确保可重复性
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    model = resnet18()

    # 设置回调
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_path,
        filename='quasar-redshift-{epoch:02d}-{val_loss:.4f}',
        save_top_k=3,
        mode='min',
        monitor='val_loss',
        save_last=True,  # 保存最后一个检查点，用于恢复训练
    )

    # 初始化早停回调，当验证损失在20个epoch内无改善时停止训练
    early_stop_callback = EarlyStopping(
        monitor='val_loss',  # 监控验证集损失
        patience=25,  # 等待20个epoch没有改善才触发早停
        mode='min'  # 寻找监控指标的最小值（损失越小越好）
    )
    # 注：该回调将被传入Trainer以实现自动训练终止，防止过拟合

    # 设置logger
    logger = TensorBoardLogger(
        save_dir="/tf_logs",
        name=RUN_NAME,
        default_hp_metric=False
    )

    # 创建训练器
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator='auto',
        devices=2,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=logger,
        log_every_n_steps=10
    )

    # 开始训练
    trainer.fit(model, train_loader, val_loader, ckpt_path=resume_ckpt)

    # 返回检查点路径，方便后续测试
    return checkpoint_callback.best_model_path


if __name__ == "__main__":
    # 路径设置
    train_data_dir = f'{ROOT_PATH}/data/sample_{RATIO}/train_dataset'
    ckpt_dir = f"{ROOT_PATH}/model_checkpoints/{RUN_NAME}"

    # 检查是否存在最后的检查点
    last_ckpt = os.path.join(ckpt_dir, "last.ckpt") if os.path.exists(os.path.join(ckpt_dir, "last.ckpt")) else None

    # 加载数据
    train_data = load_from_disk(train_data_dir)

    # 原始的
    mean=[0.004, 0.006, 0.009, 0.012, 0.02]
    std=[0.419, 0.27, 0.376, 0.479, 1.325]
    # min_val = -8.609375
    # max_val = 2032.0
    # mean = [0.0042, 0.0042, 0.0042, 0.0042, 0.0042]
    # std = [0.0002, 0.0001, 0.0002, 0.0002, 0.0006]
    train_transform = transforms.Compose([
        # transforms.Lambda(lambda x: np.log1p(x)),
        # transforms.Lambda(lambda x: (x - min_val) / (max_val - min_val)),  # 新增min-max归一化
        transforms.Normalize(mean=mean, std=std)
    ])
    # 准备训练数据
    train_dataset = PairDataset(train_data, transform=train_transform)

    # print(train_dataset[0])

    # 开始训练
    best_model_path = train_model(
        train_dataset,
        batch_size=256,
        max_epochs=100,
        ckpt_path=ckpt_dir,
        resume_ckpt=last_ckpt
    )
    print(f"\n最佳模型保存在: {best_model_path}")

    # os.system('/root/a.sh')
