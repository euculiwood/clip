import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import ModelCheckpoint,EarlyStopping
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from pytorch_lightning import LightningModule, Trainer
from datasets import load_from_disk
from dataset_util.PairDataset import PairDataset
from pytorch_lightning.loggers import TensorBoardLogger

from root_path import ROOT_PATH


class ResNet18RedshiftPredictor(LightningModule):
    def __init__(self, pretrained=False, learning_rate=1e-3):
        super(ResNet18RedshiftPredictor, self).__init__()
        # 加载预训练的 ResNet18 模型结构
        resnet = resnet18(pretrained=pretrained)
        # 修改第一层卷积以适应 5 通道输入
        resnet.conv1 = nn.Conv2d(5, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # 替换最后的全连接层，改为回归任务（输出一个标量值）
        self.features = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            resnet.avgpool
        )
        self.fc = nn.Linear(512, 1)  # 输出一个标量值（红移）

        # 存储超参数
        self.learning_rate = learning_rate
        self.save_hyperparameters()

    def forward(self, x):
        # 提取特征
        x = self.features(x)
        x = torch.flatten(x, 1)  # 展平
        # 回归预测红移
        redshift = self.fc(x)
        return redshift

    def training_step(self, batch, batch_idx):
        images, redshifts = batch['image'], batch['z']
        outputs = self(images)  # 前向传播
        loss = nn.functional.mse_loss(outputs.squeeze(), redshifts)  # 计算 MSE 损失
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, redshifts = batch['image'], batch['z']
        outputs = self(images)  # 前向传播
        loss = nn.functional.mse_loss(outputs.squeeze(), redshifts)  # 计算 MSE 损失
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer



# 使用 PyTorch Lightning 的 Trainer 进行训练
def main(resume_ckpt: str):
    # 加载数据集
    train_dataset = load_from_disk(f"{ROOT_PATH}/data/sample_0.2/train_dataset")  # 替换为你的数据集路径
    val_dataset = load_from_disk(f"{ROOT_PATH}/data/sample_0.2/test_dataset")

    # 划分训练集和验证集
    train_dataset = PairDataset(train_dataset)
    val_dataset = PairDataset(val_dataset)

    print(train_dataset[0])
    # print(val_dataset[0])

    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=4)

    # 初始化模型
    model = ResNet18RedshiftPredictor(pretrained=False, learning_rate=1e-3)

    checkpoint_callback = ModelCheckpoint(
        dirpath=f'{ROOT_PATH}/model_checkpoints/dbx',
        monitor="val_loss",  # 监控验证损失（注意键名需与log一致）
        save_top_k=1,  # 保存最佳的前2个模型
        save_last=True,  # 同时保存最后一个模型
        mode="min"  # 以最小化损失为目标
    )

    logger = TensorBoardLogger(save_dir=f'/tf_logs/', name="dbx")
    #添加早停机制
    # 初始化早停回调，当验证损失在30个epoch内无改善时停止训练
    early_stop_callback = EarlyStopping(
        monitor='val_loss',  # 监控验证集损失
        patience=30,         # 等待30个epoch没有改善才触发早停
        mode='min'           # 寻找监控指标的最小值（损失越小越好）
    )

    # 初始化 Trainer
    trainer = Trainer(max_epochs=100,
                      accelerator="gpu",
                      devices=2,
                      callbacks=[checkpoint_callback, early_stop_callback],
                      logger = logger)
    # 如果没有 GPU，可以将 accelerator 改为 "cpu"

    # 开始训练
    trainer.fit(model, train_loader, val_loader,ckpt_path=resume_ckpt)


if __name__ == "__main__":
    resume_ckpt='/hy-tmp/model_checkpoints/dbx/last.ckpt'
    main(resume_ckpt=resume_ckpt)
    # model = ResNet18RedshiftPredictor(pretrained=False, learning_rate=1e-3)
    # print(model)
