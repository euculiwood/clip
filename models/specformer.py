import math  # 导入数学库，用于基本数学运算

import lightning as L  # 导入PyTorch Lightning库，用于简化训练流程
# import pytorch_lightning as L
import numpy as np  # 导入NumPy库，用于数值计算
import torch  # 导入PyTorch库，用于构建神经网络
import torch.nn as nn  # 导入PyTorch神经网络模块
import torch.nn.functional as F  # 导入PyTorch神经网络函数模块
from torch import Tensor  # 从PyTorch导入Tensor类型
from models.modules import LayerNorm, TransformerBlock, _init_by_depth  # 导入自定义模块中的组件


class SpecFormer(L.LightningModule):  # 定义SpecFormer类，继承自LightningModule
    def __init__(
        self,
        input_dim: int=22,  # 输入特征维度
        embed_dim: int=768,  # 嵌入维度
        num_layers: int=6,  # Transformer块的数量
        num_heads: int=6,  # 多头注意力机制的头数
        max_len: int=800,  # 序列的最大长度
        mask_num_chunks: int = 6,  # 掩码分块数量
        mask_chunk_width: int = 50,  # 每个掩码块的宽度
        slice_section_length: int = 20,  # 切片段长度
        slice_overlap: int = 10,  # 切片重叠部分
        dropout: float = 0.1,  # Dropout概率
        norm_first: bool = False,  # 是否在Transformer中先进行归一化
    ):
        super().__init__()  # 调用父类初始化方法
        self.save_hyperparameters()  # 保存超参数

        self.data_embed = nn.Linear(input_dim, embed_dim)  # 数据嵌入层
        self.position_embed = nn.Embedding(max_len, embed_dim)  # 位置嵌入层
        self.dropout = nn.Dropout(dropout)  # Dropout层
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embedding_dim=embed_dim,  # 嵌入维度
                    num_heads=num_heads,  # 多头数量
                    causal=False,  # 非因果注意力
                    dropout=dropout,  # Dropout概率
                    bias=True,  # 使用偏置
                )
                for _ in range(num_layers)  # 创建多个Transformer块
            ]
        )
        self.final_layernorm = LayerNorm(embed_dim, bias=True)  # 最终层归一化
        self.head = nn.Linear(embed_dim, input_dim, bias=True)  # 输出头

        self._reset_parameters_datapt()  # 参数初始化

    def forward(self, x: Tensor) -> torch.Tensor:  # 前向传播方法
        """Forward pass through the model."""
        x = self.preprocess(x)  # 预处理输入
        # print(x.shape)
        return self.forward_without_preprocessing(x)  # 调用无预处理的前向方法

    def forward_without_preprocessing(self, x: Tensor):  # 无预处理的前向传播
        """Forward pass through the model.
        The training step performs masking before preprocessing,
        thus samples should not be preprocessed again as in forward()"""
        # 输入x形状：[256, N+1, 3]，N为切片后序列长度
        t = x.shape[1]  # 获取序列长度
        if t > self.hparams.max_len:  # 检查序列长度是否超过限制
            raise ValueError(
                f"Cannot forward sequence of length {t}, "
                f"block size is only {self.hparams.max_len}"
            )
        pos = torch.arange(0, t, dtype=torch.long, device=x.device)  # 生成位置索引[0,...,t-1]

        # forward the GPT model itself
        data_emb = self.data_embed(x)  # 数据嵌入：[256, t, 3] -> [256, t, embed_dim]
        pos_emb = self.position_embed(pos)  # 位置嵌入：[t, embed_dim]

        x = self.dropout(data_emb + pos_emb)  # 合并后形状：[256, t, embed_dim]
        for block in self.blocks:  # 通过每个Transformer块
            x = block(x)  # 保持形状[256, t, embed_dim]
        x = self.final_layernorm(x)  # 归一化后形状不变 [256,450,768]->[256,450,768]

        reconstructions = self.head(x)  # 输出头变换：[256, 450, 768] -> [256, 450,  22]

        return {"reconstructions": reconstructions, "embedding": x}  # 返回结果

    def training_step(self, batch):  # 训练步骤
        # slice the input and copy
        input = self.preprocess(batch["spectrum"])  # 预处理输入
        target = torch.clone(input)  # 克隆目标

        # mask parts of the input
        input = self.mask_sequence(input)  # 应用掩码
        # forward pass
        output = self.forward_without_preprocessing(input)["reconstructions"]  # 前向传播

        # find the mask locations
        locs = (input != target).type_as(output)  # 找到掩码位置
        loss = F.mse_loss(output * locs, target * locs, reduction="mean") / locs.mean()  # 计算损失
        # self.log("training_loss", loss, prog_bar=True)
        return loss  # 返回损失

    def validation_step(self, batch):  # 验证步骤
        # slice the input and copy
        input = self.preprocess(batch["spectrum"])  # 预处理输入
        target = torch.clone(input)  # 克隆目标

        # mask parts of the input
        input = self.mask_sequence(input)  # 应用掩码

        # forward pass
        output = self.forward_without_preprocessing(input)["reconstructions"]  # 前向传播

        # find the mask locations
        locs = (input != target).type_as(output)  # 找到掩码位置
        loss = F.mse_loss(output * locs, target * locs, reduction="mean") / locs.mean()  # 计算损失
        # self.log("val_training_loss", loss, prog_bar=True)
        return loss  # 返回损失

    def mask_sequence(self, x: Tensor):  # 掩码序列方法
        """Mask batched sequence"""
        return torch.stack([self._mask_seq(el) for el in x])  # 对每个样本应用掩码

    def preprocess(self, x):  # 预处理方法
        # print(x)
        std, mean = x.std(1, keepdim=True).clip_(0.2), x.mean(1, keepdim=True)  # [256,4501,1] -> std/mean: [256,1,1]
        x = (x - mean) / std  # 标准化：保持[256,4501,1]形状
        x = self._slice(x)  # 切片处理：[256,4501,1] -> [256, N, 20]，N取决于切片参数,=449
        x = F.pad(x, pad=(2, 0, 1, 0), mode="constant", value=0)  # 填充后形状：[256,450,22]
        x[:, 0, 0] = ((mean.squeeze() - 2) / 2)
        x[:, 0, 1] = ((std.squeeze() - 2) / 8)
        # print("preprocess shape: " + str(x.shape))
        return x  # [256,450,22]

    def _reset_parameters_datapt(self):  # 参数初始化方法
        # not scaling the initial embeddngs.
        for emb in [self.data_embed, self.position_embed]:  # 遍历嵌入层
            std = 1 / math.sqrt(self.hparams.embed_dim)  # 计算标准差
            nn.init.trunc_normal_(emb.weight, std=std, a=-3 * std, b=3 * std)  # 初始化权重

        # transformer block weights
        self.blocks.apply(lambda m: _init_by_depth(m, self.hparams.num_layers))  # 初始化Transformer块
        self.head.apply(lambda m: _init_by_depth(m, 1 / 2))  # 初始化输出头

    def _slice(self, x):  # 切片方法
        start_indices = np.arange(
            0,
            x.shape[1] - self.hparams.slice_overlap,
            self.hparams.slice_section_length - self.hparams.slice_overlap,
        )
        # print(x.shape)
        sections = [
            x[:, start : start + self.hparams.slice_section_length].transpose(1, 2)
            for start in start_indices
        ]

        # If the last section is not of length 'section_length', you can decide whether to keep or discard it
        if sections[-1].shape[-1] < self.hparams.slice_section_length:  # 检查最后一段长度
            sections.pop(-1)  # Discard the last section

        return torch.cat(sections, 1)  # 连接所有切片

    def _mask_seq(self, seq: torch.Tensor) -> torch.Tensor:  # 掩码序列方法
        """Randomly masks contiguous sections of an unbatched sequence,
        ensuring separation between chunks is at least chunk_width."""
        len_ = seq.shape[0]  # 获取序列长度
        num_chunks = self.hparams.mask_num_chunks  # 获取掩码块数
        chunk_width = self.hparams.mask_chunk_width  # 获取块宽度

        # Ensure there's enough space for the chunks and separations
        total_width_needed = num_chunks * chunk_width + (num_chunks - 1) * chunk_width  # 计算所需总宽度
        if total_width_needed > len_:  # 检查是否足够
            raise ValueError("Sequence is too short to mask")  # 抛出异常

        masked_seq = seq.clone()  # 克隆序列

        for i in range(num_chunks):  # 遍历每个块
            start = (i * len_) // num_chunks  # 计算起始位置
            loc = torch.randint(0, len_ // num_chunks - chunk_width, (1,)).item()  # 随机位置
            masked_seq[loc + start : loc + start + chunk_width] = 0  # 应用掩码

        return masked_seq  # 返回掩码后的序列

if __name__ == '__main__':
    x=torch.randn(256,4501,1)
    model=SpecFormer(
        input_dim=22,
        embed_dim=768,
        num_layers=1,
        num_heads=6,
        max_len=5000,
        mask_num_chunks=3,
        mask_chunk_width=50,
        slice_section_length=20,
        slice_overlap=10,
        dropout=0.,
        norm_first=False,
    )
    x={
        "spectrum":x,
    }
    out=model.training_step(x)