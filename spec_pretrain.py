import argparse
import os.path

import torch
import pytorch_lightning as pl
from datasets import load_from_disk
from torch.utils.data import DataLoader

from dataset_util.PairDataset import PairDataset
from models.specformer import SpecFormer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers
from root_path import ROOT_PATH
from spec_util.scheduler import CosineAnnealingWithWarmupLR

# 设置随机种子
pl.seed_everything(42)


# 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description="SpecFormer Training Script")

    # 数据路径
    parser.add_argument('--train_data_path', type=str, default=f'{ROOT_PATH}/data/sample_0.2/train_dataset',
                        help="Path to the training dataset")
    parser.add_argument('--test_data_path', type=str, default=f'{ROOT_PATH}/data/sample_0.2/test_dataset',
                        help="Path to the test dataset")

    # 模型超参数
    parser.add_argument('--input_dim', type=int, default=22, help="Input dimension")
    parser.add_argument('--embed_dim', type=int, default=768, help="Embedding dimension")
    parser.add_argument('--num_layers', type=int, default=6, help="Number of layers in the model")
    parser.add_argument('--num_heads', type=int, default=6, help="Number of attention heads")
    parser.add_argument('--max_len', type=int, default=800, help="Maximum sequence length")
    parser.add_argument('--mask_num_chunks', type=int, default=3, help="Number of chunks for masking")
    parser.add_argument('--mask_chunk_width', type=int, default=50, help="Width of each mask chunk")

    # 优化器超参数
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")  # 1e-5 -> 1e-4
    parser.add_argument('--weight_decay', type=float, default=1e-3, help="Weight decay")  # 1e-1 -> 1e-3
    parser.add_argument('--T_max', type=int, default=500_000, help="T_max for cosine annealing scheduler")
    parser.add_argument('--T_warmup', type=int, default=2000, help="T_warmup for cosine annealing scheduler")
    parser.add_argument('--dropout', type=float, default=0., help="dropout")

    # 训练参数
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size")
    parser.add_argument('--max_epochs', type=int, default=600, help="Number of epochs")
    parser.add_argument('--limit_val_batches', type=int, default=100, help="Limit validation batches")
    parser.add_argument('--gradient_clip_val', type=float, default=1.0, help="Gradient clipping value")

    # 输出路径
    parser.add_argument('--log_dir', type=str, default=f'{ROOT_PATH}/logs',
                        help="TensorBoard log directory")

    parser.add_argument('--output_dir', type=str, default=f'{ROOT_PATH}/model_checkpoints/specformer',
                        help="Output directory for checkpoints")

    return parser.parse_args()


# 训练模型
class SpecFormerLightning(pl.LightningModule):
    def __init__(self, input_dim, embed_dim, num_layers, num_heads, max_len, mask_num_chunks, mask_chunk_width, lr,
                 weight_decay, T_max, T_warmup, dropout):
        super(SpecFormerLightning, self).__init__()
        self.save_hyperparameters()  # 自动保存超参数到 self.hparams
        self.model = SpecFormer(input_dim=input_dim, embed_dim=embed_dim, num_layers=num_layers,
                                num_heads=num_heads, max_len=max_len, mask_num_chunks=mask_num_chunks,
                                mask_chunk_width=mask_chunk_width, dropout=dropout)

    def training_step(self, batch, batch_idx):
        loss = self.model.training_step(batch)
        self.log('train_loss', loss, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.model.validation_step(batch)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True,sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.hparams.lr,  # 从 hparams 获取 lr
            weight_decay=self.hparams.weight_decay,  # 从 hparams 获取 weight_decay
            betas=(0.9, 0.95)
        )
        eta_min = self.hparams.lr / 100  # 1/100 的学习率作为 eta_min

        scheduler = CosineAnnealingWithWarmupLR(
            optimizer,
            T_max=self.hparams.T_max,
            T_warmup=self.hparams.T_warmup,
            eta_min=eta_min
        )

        return [optimizer], [scheduler]


def main():
    # 解析命令行参数
    args = parse_args()

    # 加载数据集
    train_dataset = load_from_disk(args.train_data_path)
    test_dataset = load_from_disk(args.test_data_path)

    # 创建自定义数据集实例
    train_dataset = PairDataset(train_dataset)
    test_dataset = PairDataset(test_dataset)

    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1)

    # 初始化模型
    model = SpecFormerLightning(input_dim=args.input_dim, embed_dim=args.embed_dim, num_layers=args.num_layers,
                                num_heads=args.num_heads, max_len=args.max_len, mask_num_chunks=args.mask_num_chunks,
                                mask_chunk_width=args.mask_chunk_width, lr=args.lr, weight_decay=args.weight_decay,
                                T_max=args.T_max, T_warmup=args.T_warmup, dropout=args.dropout)

    # 设置训练器参数
    trainer = pl.Trainer(
        default_root_dir=args.output_dir,
        enable_checkpointing=True,
        gradient_clip_val=args.gradient_clip_val,
        max_epochs=args.max_epochs,
        limit_val_batches=args.limit_val_batches,
        logger=pl_loggers.TensorBoardLogger(args.log_dir,name="specformer"),
        callbacks=[
            LearningRateMonitor(logging_interval='step'),
            ModelCheckpoint(
                monitor='val_loss',  # 监控的指标
                save_top_k=2,  # 只保存最佳模型
                save_last=True,  # 也保存最后一个模型
                mode='min'  # 'min' 表示监控 val_loss 的最小值
            ),
        ],
        accelerator='gpu',
        devices=[0,1],
        strategy='ddp',
        # # enable_progress_bar=False,  # 禁用进度条刷新
    )
    ckpt_path=os.path.join(args.output_dir,'last.ckpt')
    # 使用 Trainer 进行训练
    trainer.fit(model, train_loader, test_loader, ckpt_path=ckpt_path)


if __name__ == '__main__':
    main()
