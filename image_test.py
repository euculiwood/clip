import pytorch_lightning as pl
from torchvision.transforms import transforms

from dataset_util.PairDataset import PairDataset
from models.resnet18 import resnet18  # 修改为正确的导入路径
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
from datasets import load_from_disk

from run_name import RATIO, RUN_NAME


def test_model(test_dataset, checkpoint_path, batch_size=256):
    """
    使用PyTorch Lightning进行模型测试
    
    Args:
        test_dataset: 测试数据集
        checkpoint_path: 模型检查点路径
        batch_size: 批次大小
    """
    # 创建测试数据加载器
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # 从检查点加载模型
    model = resnet18.load_from_checkpoint(checkpoint_path)
    model.eval()

    # 设置TensorBoard logger
    logger = TensorBoardLogger(
        save_dir="/tf_logs",
        name=f"test_{RUN_NAME}",
        default_hp_metric=False
    )
    
    # 创建训练器，使用所有可用GPU
    trainer = pl.Trainer(
        accelerator='auto',  # 自动选择加速器（CPU/GPU/TPU）
        devices=1,           # 使用1个设备进行测试
        logger=logger,       # 启用 TensorBoard 日志记录
    )


    # 运行测试
    trainer.test(model, test_loader)


if __name__ == "__main__":
    # 路径设置
    test_data_dir = f'/hy-tmp/data/sample_{RATIO}/test_dataset'
    ckpt_dir = f"/hy-tmp/model_checkpoints/{RUN_NAME}/last.ckpt"
    
    print(f"使用检查点: {ckpt_dir}")
    
    train_transform = transforms.Compose([
        transforms.Normalize(mean=[0.004, 0.006, 0.009, 0.012, 0.02], std=[0.419, 0.27, 0.376, 0.479, 1.325])
    ])
    test_data = load_from_disk(test_data_dir)
    test_dataset = PairDataset(test_data,transform=train_transform)

    print(len(test_dataset))

    # 运行测试
    test_model(test_dataset, ckpt_dir, batch_size=256)

    # os.system('/root/tainer.sh')
