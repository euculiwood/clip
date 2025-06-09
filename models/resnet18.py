import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchvision.models as models
from torchmetrics import MeanSquaredError, R2Score
import torchvision.transforms as transforms
import numpy as np
import math

from run_name import RUN_NAME


class resnet18(pl.LightningModule):
    def __init__(self, learning_rate=1e-3):
        super().__init__()
        # 修改ResNet18的第一层以适应5通道输入
        resnet = models.resnet18(pretrained=True)
        resnet.conv1 = nn.Conv2d(5, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_features= resnet.fc.in_features

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
            resnet.avgpool,
            nn.Flatten(),
        )
        # 特征扩展模块（独立分支，不影响原始特征）
        self.feature_extender = nn.Sequential(
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        # 最终回归头
        self.regressor = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 1)
        )

        # 存储超参数
        self.learning_rate = learning_rate
        self.save_hyperparameters()

        self.mse = MeanSquaredError()
        self.r2 = R2Score()

        # 用于存储测试结果
        self.test_outputs = []
        self.test_targets = []

    def forward(self, x):
        x = self.features(x)  # 获取512维基础特征
        x = self.feature_extender(x)  # 扩展到1024维

        return self.regressor(x)  # 正常预测红移值

    def feature_extractor(self, x):
        x=self.features(x)
        x=self.feature_extender(x)
        return x

    def training_step(self, batch, batch_idx):
        x = batch['image']
        y = batch['z']
        y_hat = self(x).squeeze()
        loss = F.mse_loss(y_hat, y)
        
        # 记录训练损失，用于监控训练过程
        # 'train_loss'：日志中的指标名称
        # loss：要记录的损失值
        # on_step=True：在每个训练步骤记录
        # on_epoch=True：在每个训练周期结束汇总
        # prog_bar=True：显示在进度条中
        # logger=True：记录到日志系统（如TensorBoard）
        # sync_dist=True：在分布式训练中同步所有设备的日志数据
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train_r2', self.r2(y_hat, y) ,on_epoch=True,sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch['image']
        y = batch['z']
        y_hat = self(x).squeeze()
        val_loss = F.mse_loss(y_hat, y)
        self.log('val_loss', val_loss,on_step=True, on_epoch=True, prog_bar=True,sync_dist=True)
        self.log('val_r2', self.r2(y_hat, y), on_epoch=True,sync_dist=True)
        return val_loss

    def test_step(self, batch, batch_idx):
        x = batch['image']
        y = batch['z']
        y_hat = self(x).squeeze()
        
        # 存储预测结果和真实值用于后续计算（确保分离计算图并移至CPU）
        self.test_outputs.append(y_hat.detach().cpu())
        self.test_targets.append(y.detach().cpu())
        
        # 计算和记录MSE损失
        test_loss = F.mse_loss(y_hat, y)
        self.log('test_loss', test_loss, on_step=True,prog_bar=True, sync_dist=True)
        return test_loss

    def on_test_epoch_end(self):
        # 合并所有批次的预测结果
        all_outputs = torch.cat(self.test_outputs)
        all_targets = torch.cat(self.test_targets)
        
        # 确保在CPU上计算最终指标
        all_outputs = all_outputs.cpu()
        all_targets = all_targets.cpu()
        # print(all_outputs.shape)
        # print(f"all_targets.shape:{all_targets.shape}")
        # 计算相对误差
        delta = torch.abs((all_outputs - all_targets) / (1 + all_targets))
        
        # 计算RMS
        rms = torch.sqrt(torch.mean(delta ** 2))
        
        # 计算准确率
        acc_0_1 = (delta < 0.1).float().mean()
        acc_0_2 = (delta < 0.2).float().mean()
        acc_0_3 = (delta < 0.3).float().mean()
        #如果文件不存在，就创建
        if not os.path.exists('/hy-tmp/result'):
            os.makedirs('/hy-tmp/result')
        # 控制台打印结果，并将其内容保存到/hy-tmp/clip/test_log.txt中
        with open(f'/hy-tmp/result/{RUN_NAME}.txt', 'a') as log_file:
            log_file.write(f'test_rms: {rms.item()}\n')
            log_file.write(f'test_acc_0.1: {acc_0_1.item()}\n')
            log_file.write(f'test_acc_0.2: {acc_0_2.item()}\n')
            log_file.write(f'test_acc_0.3: {acc_0_3.item()}\n')
            print(f'test_rms: {rms.item()}')
            print(f'test_acc_0.1: {acc_0_1.item()}')
            print(f'test_acc_0.2: {acc_0_2.item()}')
            print(f'test_acc_0.3: {acc_0_3.item()}')


        # 清空存储的结果
        self.test_outputs.clear()
        self.test_targets.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }

if __name__ == '__main__':
    model = resnet18()
    x=torch.randn(2,5,64,64)
    # y=models(x)
    # print(y.shape)
    y=model.feature_extractor(x)
    print(y.shape)
    z=model(x)
    print(z.shape)
    # print(model)