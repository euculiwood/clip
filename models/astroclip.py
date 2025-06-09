# 导入必要的库和模块
import os  # 操作系统接口模块
import sys  # 系统相关参数和函数
from typing import Tuple  # 类型提示支持

import lightning as L  # PyTorch Lightning深度学习框架
import numpy as np  # 数值计算库
import torch  # PyTorch深度学习框架
import torch.nn as nn  # 神经网络模块
import torch.nn.functional as F  # 神经网络函数
# from dinov2.eval.setup import setup_and_build_model  # DINOv2模型构建工具

from models.resnet18 import resnet18
# 导入自定义模块
from models.modules import MLP, CrossAttentionHead  # 多层感知机和交叉注意力头
from models.specformer import SpecFormer  # 光谱处理模型


# 定义AstroCLIP主模型类（继承自LightningModule）
class AstroClipModel(L.LightningModule):
    def __init__(
        self,
        image_encoder: nn.Module,  # 图像编码器
        spectrum_encoder: nn.Module,  # 光谱编码器
        temperature: float = 15.5,  # CLIP损失温度参数
        lr: float = 1e-4,  # 学习率
        weight_decay: float = 0.05,  # 权重衰减
        epochs: int = 100,  # 训练轮数
        eta_min: float = 5e-7,  # 最小学习率
        logit_scale: float = 15.5,  # logit缩放系数
        learnable_logit_scale: bool = False,  # 是否可学习logit缩放
    ):
        """
        The AstroCLIP model that takes an image and a spectrum and embeds them into a common space using CLIP loss.
        Note that you must provide the image and spectrum encoders to be used for the embedding.

        Args:
            image_encoder (nn.Module): The image encoder to be used for embedding.
            spectrum_encoder (nn.Module): The spectrum encoder to be used for embedding.
            temperature (float): The temperature parameter for the CLIP loss.
            lr (float): The learning rate for the optimizer.
            weight_decay (float): The weight decay for the optimizer.
            epochs (int): The number of epochs for training.
            eta_min (float): The minimum learning rate for the scheduler.
            logit_scale (float): The logit scale for the CLIP loss.
            learnable_logit_scale (bool): Whether the logit scale should be learnable.
        """
        super().__init__()  # 调用父类初始化
        self.save_hyperparameters()  # 保存超参数

        # 初始化编码器模块
        self.image_encoder = image_encoder  # 图像编码器实例
        self.spectrum_encoder = spectrum_encoder  # 光谱编码器实例

        # 初始化logit缩放参数
        if not learnable_logit_scale:  # 固定logit缩放
            self.logit_scale = np.log(logit_scale)  # 转换为对数形式
        else:  # 可学习logit缩放
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(logit_scale))

        # Use CLIP loss
        self.criterion = CLIPLoss()

    # 前向传播方法
    def forward(
        self,
        input: torch.Tensor,  # 输入张量
        input_type: str,  # 输入类型（image/spectrum）
    ):
        if input_type == "image":  # 图像处理分支
            return self.image_encoder(input)
        elif input_type == "spectrum":  # 光谱处理分支
            return self.spectrum_encoder(input)
        else:  # 异常处理
            raise ValueError("Input type must be either 'image' or 'spectrum'")

    # 训练步骤
    def training_step(self, batch, batch_idx):
        im, sp = batch["image"], batch["spectrum"]  # 解包批次数据

        # 特征提取
        image_features = self.image_encoder(im)  # 图像特征
        spectrum_features = self.spectrum_encoder(sp)  # 光谱特征

        # 计算两种损失（带logit和不带logit）
        loss_withlogit = self.criterion(
            image_features, spectrum_features, self.hparams.temperature
        )
        loss_nologit = self.criterion(
            image_features, spectrum_features, self.hparams.logit_scale
        )

        # 日志记录
        self.log("train_loss_withlogit", loss_withlogit)  # 带logit训练损失
        self.log("train_loss_nologit", loss_nologit)  # 不带logit训练损失
        self.log("scale", self.logit_scale)  # logit缩放值

        return loss_withlogit  # 返回主损失

    # 验证步骤
    def validation_step(self, batch, batch_idx):
        im, sp = batch["image"], batch["spectrum"]  # 解包验证数据

        # 特征提取
        image_features = self.image_encoder(im)  # 图像特征
        spectrum_features = self.spectrum_encoder(sp)  # 光谱特征

        # 计算两种验证损失
        val_loss_nologit = self.criterion(
            image_features, spectrum_features, self.hparams.logit_scale
        )
        val_loss_withlogit = self.criterion(
            image_features, spectrum_features, self.hparams.temperature
        )

        # 日志记录验证指标
        self.log("val_loss_nologit", val_loss_nologit)  # 不带logit验证损失
        self.log("val_loss_withlogit", val_loss_withlogit)  # 带logit验证损失

# CLIP对比损失类
class CLIPLoss(nn.Module):
    # 计算logits方法
    def get_logits(
        self,
        image_features: torch.FloatTensor,  # 图像特征
        spectrum_features: torch.FloatTensor,  # 光谱特征
        logit_scale: float,  # logit缩放系数
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        # 特征归一化处理
        image_features = F.normalize(image_features, dim=-1, eps=1e-3)  # L2归一化图像特征
        spectrum_features = F.normalize(spectrum_features, dim=-1, eps=1e-3)  # L2归一化光谱特征

        # 计算相似度矩阵
        logits_per_image = logit_scale * image_features @ spectrum_features.T  # 图像-光谱相似度
        return logits_per_image, logits_per_image.T  # 返回双向相似度矩阵

    # 损失计算主方法
    def forward(
        self,
        image_features: torch.FloatTensor,  # 图像特征
        spectrum_features: torch.FloatTensor,  # 光谱特征
        logit_scale: float,  # logit缩放系数
        output_dict: bool = False,  # 是否返回字典格式
    ) -> torch.FloatTensor:
        # 获取相似度矩阵
        logits_per_image, logits_per_spectrum = self.get_logits(
            image_features, spectrum_features, logit_scale
        )

        # 创建对比学习标签
        labels = torch.arange(
            logits_per_image.shape[0], device=image_features.device, dtype=torch.long
        )  # 顺序标签（0到batch_size-1）

        # 计算交叉熵损失
        total_loss = (
            F.cross_entropy(logits_per_image, labels)  # 图像到光谱方向
            + F.cross_entropy(logits_per_spectrum, labels)  # 光谱到图像方向
        ) / 2  # 平均损失

        return {"contrastive_loss": total_loss} if output_dict else total_loss  # 返回格式选择


# 图像处理头部网络
class ImageHead(nn.Module):
    def __init__(
        self,
        # config: str,  # 配置文件路径
        model_path: str,  # 模型权重路径
        # save_directory: str,  # 保存目录
        embed_dim: int = 1024,  # 嵌入维度
        n_head: int = 4,  # 注意力头数
        model_embed_dim: int = 1024,  # 模型嵌入维度
        dropout: float = 0.1,  # dropout概率
        freeze_backbone: bool = True,  # 是否冻结主干
    ):
        """
        Cross-attention image module that takes token outputs from the AstroDINO model and passes them through a
        cross-attention mechanism and MLP to get the final embedding.

        Args:
            save_directory (str): Path to the directory containing the AstroDINO model.
            config (str): Path to the configuration file of the AstroDINO model.
            model_weights (str): Path to the weights of the AstroDINO model.
            embed_dim (int): Dimension of the AstroCLIP embedding.
            n_head (int): Number of heads in the multihead attention.
            model_embed_dim (int): Dimension of the AstroDINO embedding.
            dropout (float): Dropout rate for MLP layers.
            freeze_backbone (bool): Whether to freeze the backbone of the AstroDINO model.
        """
        super().__init__()  # 父类初始化

        # 初始化resnet18主干网络
        self.backbone = resnet18(transfom_flag=False,pretrained=False)
        if model_path:
            checkpoint = torch.load(model_path)
            # checkpoint = torch.load(model_path,map_location='cpu')
            self.backbone.load_state_dict(checkpoint['state_dict'])


        # 设置主干网络冻结
        self.freeze_backbone = freeze_backbone
        if self.freeze_backbone:  # 冻结参数
            for param in self.backbone.parameters():
                param.requires_grad = False

        # # 打印参数的requires_grad状态
        # print("Model parameters requires_grad status:")
        # for name, param in self.backbone.named_parameters():
        #     print(f"{name}: {param.requires_grad}")

        # 初始化交叉注意力模块
        self.cross_attention = CrossAttentionHead(
            embed_dim=embed_dim,
            n_head=n_head,
            model_embed_dim=model_embed_dim,
            dropout=dropout,
        )

        # 初始化MLP模块
        self.mlp = MLP(
            in_features=embed_dim,
            hidden_features=4 * embed_dim,
            dropout=dropout,
        )

    # 前向传播
    def forward(self, x: torch.tensor, return_weights: bool = False):
        # 主干网络前向传播
        with torch.set_grad_enabled(not self.freeze_backbone):  # 梯度控制
            x = self.backbone.feature_extractor(x)  # 使用feature_extractor获取特征

        #(bn,patch_num,feature=1024)
        #(bn,1,feature=1024)
        # 交叉注意力处理
        x=x.unsqueeze(dim=1)
        x, attentions = self.cross_attention(x)

        # MLP处理
        x = self.mlp(x)

        # # 返回选择
        if return_weights:  # 需要返回注意力权重
            return x.squeeze(), attentions[1]
        return x.squeeze()


# 光谱处理头部网络
class SpectrumHead(nn.Module):
    def __init__(
        self,
        model_path: str,  # 模型路径
        embed_dim: int = 1024,  # 嵌入维度
        n_head: int = 4,  # 注意力头数
        model_embed_dim: int = 768,  # 模型嵌入维度
        dropout: float = 0.1,  # dropout概率
        freeze_backbone: bool = True,  # 是否冻结主干
        load_pretrained_weights=True,  # 是否加载预训练权重
    ):
        """
        Cross-attention spectrum module that takes a spectrum and passes it through a pretrained SpecFormer model and
        then through a cross-attention mechanism and MLP to get the final embedding.

        Args:
            save_path (str): Path to the checkpoint of the SpecFormer model.
            embed_dim (int): Dimension of the AstroCLIP embedding.
            n_head (int): Number of heads in the multihead attention.
            model_embed_dim (int): Dimension of the SpecFormer embedding.
            dropout (float): Dropout rate for MLP layers.
            freeze_backbone (bool): Whether to freeze the backbone of the SpecFormer model.
        """
        super().__init__()  # 父类初始化

        # 加载预训练模型
        checkpoint = torch.load(model_path)  # 加载检查点
        # checkpoint = torch.load(model_path,map_location='cpu')  # 加载检查点
        for i in ['lr', 'T_max', 'T_warmup', 'dropout','weight_decay']:
            checkpoint["hyper_parameters"].pop(i, None)
        self.backbone = SpecFormer(**checkpoint["hyper_parameters"])  # 初始化模型
        if load_pretrained_weights:  # 加载权重
            checkpoint["state_dict"] = self._remove_prefix(checkpoint["state_dict"], 'model.')
            self.backbone.load_state_dict(checkpoint["state_dict"])

        # 设置主干网络冻结
        self.freeze_backbone = freeze_backbone
        if self.freeze_backbone:  # 冻结参数
            for param in self.backbone.parameters():
                param.requires_grad = False

        # 初始化交叉注意力模块
        self.cross_attention = CrossAttentionHead(
            embed_dim=embed_dim,
            n_head=n_head,
            model_embed_dim=model_embed_dim,
            dropout=dropout,
        )

        # 初始化MLP模块
        self.mlp = MLP(
            in_features=embed_dim,
            hidden_features=4 * embed_dim,
            dropout=dropout,
        )

    @staticmethod
    def _remove_prefix(state_dict, prefix):
        """
        移除权重前缀，方便加载模型。
        """
        return {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in state_dict.items()}

    # 前向传播
    def forward(
        self, x: torch.tensor, y: torch.tensor = None, return_weights: bool = False
    ):
        # 主干网络处理
        with torch.set_grad_enabled(not self.freeze_backbone):  # 梯度控制
            embedding = self.backbone(x)["embedding"]  # 获取光谱嵌入

        # 交叉注意力处理
        x, attentions = self.cross_attention(embedding)

        # 残差连接+MLP
        x = x + self.mlp(x)  # 残差连接

        # 返回选择
        if return_weights:  # 需要返回注意力权重
            return x.squeeze(), attentions[1]
        return x.squeeze()

if __name__ == '__main__':
    img_weight_path="/hy-tmp/model_checkpoints/mine_4_data_20/last.ckpt"
    spec_weight_path = "/hy-tmp/model_checkpoints/spectrum_model/epoch=227-step=128820.ckpt"
    model=AstroClipModel(spec_weight_path=spec_weight_path,img_weight_path=img_weight_path)
    x={
        #转为float32类型
        "image": torch.randn(2,5,64,64),
        "spectrum": torch.randn(2,4501,1)
    }
    image=model(x["image"],input_type = "image")
    spectrum=model(x["spectrum"],input_type = "spectrum")
    print(image.shape,spectrum.shape)

