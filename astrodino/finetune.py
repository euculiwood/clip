import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset # 导入 Dataset
import torch.distributed as dist
import torch.nn.parallel as DDP
from torch.utils.data.distributed import DistributedSampler

import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from argparse import ArgumentParser
import logging
from typing import List # 确保导入 List

# 导入 Hugging Face datasets 库
from datasets import load_from_disk

# --- 关键修改：将项目根目录添加到 sys.path ---
# 获取当前脚本的绝对路径
current_script_path = os.path.abspath(__file__)
# 假设 finetune.py 在 /hy-tmp/clip/astrodino/finetune.py
# 那么项目根目录 /hy-tmp/clip 就是当前脚本目录的上一级的上一级
project_root = os.path.dirname(os.path.dirname(current_script_path))

# 将项目根目录添加到 sys.path
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- 关键修改结束 ---

# 设置日志
logger = logging.getLogger("finetune_dino")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# 导入 AstroDINO 和 DINOv2 相关的模块 (现在应该能正确导入了)
from astrodino.utils import setup_astrodino
from astrodino.env import format_with_env
from astrodino.data.augmentations import DataAugmentationAstroDINO
from dinov2.data.loaders import make_data_loader, make_dataset # 保留 make_data_loader/make_dataset 只是为了避免其他地方可能存在的依赖，但不再主动调用它们
from downstream_tasks.property_utils.models import MLP

# --- 2. 定义 DINOv2 回归模型类 ---
class DinoV2RegressionModel(nn.Module):
    def __init__(self, dino_config_path, pretrained_weights_path, output_dim=1, hidden_dims: List[int] = [64, 64]):
        super().__init__()
        logger.info(f"Loading DINOv2 backbone from config: {dino_config_path} and weights: {pretrained_weights_path}")
        self.dino_backbone = setup_astrodino(
            astrodino_output_dir="./temp_finetune_output", # 临时输出目录
            astrodino_pretrained_weights=pretrained_weights_path,
            astrodino_config_file=dino_config_path
        )
        self.dino_backbone.train() # 确保 DINOv2 主干网在训练模式，允许微调
        dino_embed_dim = self.dino_backbone.embed_dim
        logger.info(f"DINOv2 backbone embedding dimension: {dino_embed_dim}")

        # 这里的 MLP 会使用你之前提供的 models.py 中的 MLP 定义
        # 确保 models.py 中的 MLP.__init__ 参数和这里匹配
        # (即 models.py 中 MLP 的 n_hidden 应该是 (16,16,16) 这样的元组，不是 List[int] 类型提示)
        self.regression_head = MLP(
            n_in=dino_embed_dim,
            n_out=output_dim,
            n_hidden=tuple(hidden_dims), # 将 List[int] 转换为 Tuple 传递给 MLP 的 n_hidden
            dropout=0.1
        )
        logger.info(f"Regression Head defined: {self.regression_head}")

    def forward(self, x):
        features = self.dino_backbone(x)
        output = self.regression_head(features)
        return output

# --- 定义 Hugging Face Dataset 的适配器类 ---
class HFDatasetWrapper(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.hf_dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        item = self.hf_dataset[idx]
        image = item["image"] # image 是 PIL Image 对象

        processed_image = None
        if self.transform:
            transformed_output = self.transform(image)
            # ... (这部分图像处理代码保持不变，因为它现在已经工作了)
            if isinstance(transformed_output, dict):
                if 'global_crops' in transformed_output:
                    global_crops_list = transformed_output['global_crops']
                    if isinstance(global_crops_list, list) and len(global_crops_list) > 0:
                        if torch.is_tensor(global_crops_list[0]):
                            processed_image = global_crops_list[0]
                        else:
                            raise TypeError(f"Element in 'global_crops' list is not a torch.Tensor. Type: {type(global_crops_list[0])}")
                    else:
                        raise ValueError("'global_crops' value is not a non-empty list or not a list.")
                else:
                    raise KeyError(f"Expected 'global_crops' key in DataAugmentationAstroDINO output, but found: {transformed_output.keys()}")
            elif isinstance(transformed_output, list):
                if len(transformed_output) > 0 and torch.is_tensor(transformed_output[0]):
                    processed_image = transformed_output[0]
                else:
                    raise TypeError(f"DataAugmentationAstroDINO returned a list but first element is not a torch.Tensor or list is empty.")
            elif torch.is_tensor(transformed_output):
                processed_image = transformed_output
            else:
                raise TypeError(f"DataAugmentationAstroDINO returned unexpected type: {type(transformed_output)}. Expected dict with 'global_crops', list of Tensors, or torch.Tensor.")
        else:
            from torchvision import transforms
            processed_image = transforms.ToTensor()(image)
            if processed_image.shape[0] == 1:
                processed_image = processed_image.repeat(3, 1, 1)

        if not torch.is_tensor(processed_image):
             raise ValueError("Final processed_image is not a torch.Tensor after transform/processing.")

        z_value = item["z"]
        # --- 关键修改：改回原始的 z 值处理方式 ---
        # 确保 z_value 是浮点张量，并保持维度一致 (1, )
        # 你的数据集里 z_value 可能是原始的 Python float 或 int
        if torch.is_tensor(z_value):
            return {"image": processed_image, "z": z_value.float().view(1)}
        else:
            # 如果是 Python number (int/float)，转换为 Tensor 并确保格式
            return {"image": processed_image, "z": torch.tensor(float(z_value), dtype=torch.float32).view(1)}
        # --- 修改结束 ---


# --- 3. 训练函数 (finetune_dino_regression) ---
def finetune_dino_regression(args):
    # --- 初始化分布式环境 ---
    is_distributed = False
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ and "LOCAL_RANK" in os.environ:
        try:
            rank = int(os.environ["RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
            local_rank = int(os.environ["LOCAL_RANK"])

            if world_size > 1:
                dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
                torch.cuda.set_device(local_rank)
                is_distributed = True
            else:
                rank = 0
                world_size = 1
                local_rank = 0

        except (KeyError, ValueError):
            rank = 0
            world_size = 1
            local_rank = 0
    else:
        rank = 0
        world_size = 1
        local_rank = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        if is_distributed:
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cuda:0")
    else:
        if rank == 0:
            logger.warning("CUDA is not available. Running on CPU. This will be very slow.")

    if rank == 0:
        logger.info("Starting finetune_dino_regression function.")
        logger.info(f"Using device: {device}, Global Rank: {rank}/{world_size}, Local Rank: {local_rank}, Is Distributed: {is_distributed}")
        logger.info(f"Arguments: {args}")
        os.makedirs(args.output_model_dir, exist_ok=True)

    if is_distributed:
        dist.barrier()


    # --- 数据增强定义 (DataAugmentationAstroDINO) ---
    data_transform = DataAugmentationAstroDINO(
        global_crops_scale=args.global_crops_scale,
        local_crops_scale=args.local_crops_scale,
        local_crops_number=0, # 微调时不生成局部图
        global_crops_size=args.image_size,
        local_crops_size=args.image_size
    )

    # --- 自定义 collate_fn ---
    def finetune_collate_fn(batch):
        processed_images = []
        zs = []
        for item in batch:
            processed_images.append(item["image"]) # 这现在已经是 Tensor 了
            # --- 关键修改：改回原始的 zs.append 方式，带上 torch.tensor() ---
            zs.append(torch.tensor(item["z"], dtype=torch.float32)) # 之前这里的 UserWarning 不影响运行，我们先让它能跑起来
            # --- 修改结束 ---

        images_tensor = torch.stack(processed_images) # (B, C=5, H, W)
        zs_tensor = torch.stack(zs).unsqueeze(1) # (B, 1) # 恢复这个 unsqueeze(1) 因为上面 zs.append 是 (1,) 的张量，stack后是(B,1)，如果再用 unsqueeze(1) 就变成 (B,1,1) 了，我们正是需要这个来重现你之前的错误，从而确认问题源头。

        return {"image": images_tensor, "z": zs_tensor}

    # --- 直接加载 Hugging Face 数据集并使用 HFDatasetWrapper ---
    # 训练数据集
    if rank == 0:
        logger.info(f"Loading train dataset from: {args.train_dataset_path}")
    train_hf_dataset = load_from_disk(args.train_dataset_path)
    train_dataset_obj = HFDatasetWrapper(train_hf_dataset, transform=data_transform)

    if is_distributed:
        train_sampler = DistributedSampler(
            train_dataset_obj,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=args.seed
        )
        shuffle_train_loader = False
    else:
        train_sampler = None
        shuffle_train_loader = True

    train_loader = DataLoader(
        dataset=train_dataset_obj,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sampler=train_sampler,
        shuffle=shuffle_train_loader,
        collate_fn=finetune_collate_fn,
        drop_last=True
    )

    # 测试数据集
    if rank == 0:
        logger.info(f"Loading test dataset from: {args.test_dataset_path}")
    test_hf_dataset = load_from_disk(args.test_dataset_path)
    test_dataset_obj = HFDatasetWrapper(test_hf_dataset, transform=data_transform)

    if is_distributed:
        test_sampler = DistributedSampler(
            test_dataset_obj,
            num_replicas=world_size,
            rank=rank,
            shuffle=False
        )
        shuffle_test_loader = False
    else:
        test_sampler = None
        shuffle_test_loader = False

    test_loader = DataLoader(
        dataset=test_dataset_obj,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sampler=test_sampler,
        shuffle=shuffle_test_loader,
        collate_fn=finetune_collate_fn,
        drop_last=False
    )

    if rank == 0:
        logger.info(f"Training dataset size: {len(train_dataset_obj)}, Test dataset size: {len(test_dataset_obj)}")

    # --- Z 值标准化器 ---
    scaler_z = StandardScaler()
    local_train_z = []
    for data in tqdm(train_loader, desc=f"Collecting Local Z for Scaler on Rank {rank}", disable=(rank != 0)):
        local_train_z.append(data["z"].cpu().numpy())

    local_train_z = np.concatenate(local_train_z, axis=0)
    local_train_z = local_train_z.reshape(-1, 1)

    scaler_z.fit(local_train_z)
    if is_distributed:
        gathered_z_list = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_z_list, local_train_z)

        if rank == 0:
            all_train_z_for_scaler = np.concatenate(gathered_z_list, axis=0)
            scaler_z.fit(all_train_z_for_scaler)
            logger.info("Z value scaler fitted by rank 0 using all training data.")

            scaler_mean = torch.from_numpy(scaler_z.mean_).to(device)
            scaler_scale = torch.from_numpy(scaler_z.scale_).to(device)
        else:
            scaler_mean = torch.empty(1, dtype=torch.float32, device=device)
            scaler_scale = torch.empty(1, dtype=torch.float32, device=device)
    else:
        scaler_z.fit(local_train_z)
        logger.info("Z value scaler fitted on single process using all training data.")
        scaler_mean = None
        scaler_scale = None


    if is_distributed:
        dist.broadcast(scaler_mean, src=0)
        dist.broadcast(scaler_scale, src=0)
        scaler_z.mean_ = scaler_mean.cpu().numpy()
        scaler_z.scale_ = scaler_scale.cpu().numpy()

    if is_distributed: # <--- 添加这个条件判断
        dist.barrier()
    if rank == 0:
        logger.info("Z value scaler broadcasted and set on all processes." if is_distributed else "Z value scaler ready on single process.")


    # --- 模型初始化 ---
    model = DinoV2RegressionModel(
        dino_config_path=args.dino_config_file,
        pretrained_weights_path=args.pretrained_weights_path,
        output_dim=1,
        hidden_dims=args.hidden_dims
    )

    model.to(device)
    if is_distributed:
        model = DDP.DistributedDataParallel(model, device_ids=[local_rank])

    if is_distributed:
        model.module.dino_backbone.train()
    else:
        model.dino_backbone.train()


    # --- 优化器设置---
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # --- 训练循环 ---
    if rank == 0:
        logger.info("Starting training loop...")
    best_val_rmse = float('inf')

    for epoch in range(args.epochs):
        if is_distributed:
            train_sampler.set_epoch(epoch)

        model.train()
        running_loss = 0.0

        train_loop_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} Training", disable=(rank != 0))
        for i, data in enumerate(train_loop_iterator):
            imgs = data["image"].to(device)
            labels_raw = data["z"].to(device)

            labels_scaled = torch.tensor(scaler_z.transform(labels_raw.cpu().numpy()), dtype=torch.float32).to(device)

            optimizer.zero_grad()
            outputs = model(imgs)

            loss = criterion(outputs, labels_scaled)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        total_running_loss = torch.tensor(running_loss, device=device)
        if is_distributed:
            dist.all_reduce(total_running_loss, op=dist.ReduceOp.SUM)

        avg_train_loss = total_running_loss.item() / len(train_loader) / world_size

        scheduler.step()

        # --- 验证阶段 ---
        model.eval()
        val_preds_scaled = []
        val_labels_scaled = []
        val_preds_raw = []
        val_labels_raw = []

        with torch.no_grad():
            val_loop_iterator = tqdm(test_loader, desc=f"Epoch {epoch+1}/{args.epochs} Validation", disable=(rank != 0))
            for i, data in enumerate(val_loop_iterator):
                imgs = data["image"].to(device)
                labels_raw_batch = data["z"].to(device)

                labels_scaled_batch = torch.tensor(scaler_z.transform(labels_raw_batch.cpu().numpy()), dtype=torch.float32).to(device)

                outputs_scaled = model(imgs)

                val_preds_scaled.append(outputs_scaled.cpu().numpy())
                val_labels_scaled.append(labels_scaled_batch.cpu().numpy())

                preds_raw_batch = scaler_z.inverse_transform(outputs_scaled.cpu().numpy())
                val_preds_raw.append(preds_raw_batch)
                val_labels_raw.append(labels_raw_batch.cpu().numpy())

        local_val_preds_scaled = np.concatenate(val_preds_scaled)
        local_val_labels_scaled = np.concatenate(val_labels_scaled)
        local_val_preds_raw = np.concatenate(val_preds_raw).flatten()
        local_val_labels_raw = np.concatenate(val_labels_raw).flatten()

        gathered_all_preds_raw = [None for _ in range(world_size)]
        gathered_all_labels_raw = [None for _ in range(world_size)]
        gathered_all_preds_scaled = [None for _ in range(world_size)]
        gathered_all_labels_scaled = [None for _ in range(world_size)]

        if is_distributed:
            dist.all_gather_object(gathered_all_preds_raw, local_val_preds_raw)
            dist.all_gather_object(gathered_all_labels_raw, local_val_labels_raw)
            dist.all_gather_object(gathered_all_preds_scaled, local_val_preds_scaled)
            dist.all_gather_object(gathered_all_labels_scaled, local_val_labels_scaled)
        else:
            gathered_all_preds_raw = [local_val_preds_raw]
            gathered_all_labels_raw = [local_val_labels_raw]
            gathered_all_preds_scaled = [local_val_preds_scaled]
            gathered_all_labels_scaled = [local_val_labels_scaled]


        if rank == 0:
            final_preds_raw = np.concatenate(gathered_all_preds_raw)
            final_labels_raw = np.concatenate(gathered_all_labels_raw)
            final_preds_scaled = np.concatenate(gathered_all_preds_scaled)
            final_labels_scaled = np.concatenate(gathered_all_labels_scaled)

            val_loss = criterion(torch.tensor(final_preds_scaled), torch.tensor(final_labels_scaled)).item()
            val_rmse = np.sqrt(mean_squared_error(final_labels_raw, final_preds_raw))
            val_r2 = r2_score(final_labels_raw, final_preds_raw)

            diff = np.abs(final_preds_raw - final_labels_raw)
            accuracy_0_1 = np.sum(diff < 0.1) / len(diff) * 100

            logger.info(f"Epoch {epoch+1}:")
            logger.info(f"  Training Loss (Avg across processes): {avg_train_loss:.4f}")
            logger.info(f"  Validation Loss (Scaled, Aggregated): {val_loss:.4f}")
            logger.info(f"  Validation RMSE (Raw, Aggregated): {val_rmse:.4f}")
            logger.info(f"  Validation R2 Score (Aggregated): {val_r2:.4f}")
            logger.info(f"  Accuracy (Error < 0.1, Aggregated): {accuracy_0_1:.2f}%")

            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                model_path = os.path.join(args.output_model_dir, "best_model.pth")
                save_model_state_dict = model.module.state_dict() if is_distributed else model.state_dict()
                torch.save(save_model_state_dict, model_path)
                logger.info(f"  Saved best model to {model_path} (RMSE: {best_val_rmse:.4f})")

            if (epoch + 1) % args.save_every_epochs == 0:
                model_path_epoch = os.path.join(args.output_model_dir, f"model_epoch_{epoch+1}.pth")
                save_model_state_dict = model.module.state_dict() if is_distributed else model.state_dict()
                torch.save(save_model_state_dict, model_path_epoch)
                logger.info(f"  Saved epoch {epoch+1} model to {model_path_epoch}")

    if is_distributed:
        dist.barrier()

    if is_distributed:
        dist.destroy_process_group()
    if rank == 0:
        logger.info("Finetuning finished.")


# --- 4. Main 函数和参数解析 ---
if __name__ == '__main__':
    parser = ArgumentParser(description="Finetune DINOv2 for Redshift Regression.")

    # resolved_astroclip_root 应该指向你的 CLIP 项目根目录 /hy-tmp/clip
    # 确保 format_with_env 能够正确解析环境变量或返回正确的路径
    # 如果环境没有设置 ASTROCLIP_ROOT，你也可以直接写死路径来测试
    ASTROCLIP_ROOT_PLACEHOLDER = "{ASTROCLIP_ROOT}"
    resolved_astroclip_root = format_with_env(ASTROCLIP_ROOT_PLACEHOLDER)
    # 如果上述无法解析，请暂时用以下代码，替换为你的实际项目根目录
    # resolved_astroclip_root = "/hy-tmp/clip"


    parser.add_argument("--train_dataset_path", type=str,
                        # 直接给出本地数据集路径，不再带 hf_datasets: 前缀
                        default=os.path.join(resolved_astroclip_root, "astrodino", "data", "train_dataset"),
                        help="Path to your training dataset (Hugging Face dataset format local path).")
    parser.add_argument("--test_dataset_path", type=str,
                        # 直接给出本地数据集路径
                        default=os.path.join(resolved_astroclip_root, "astrodino", "data", "test_dataset"),
                        help="Path to your test dataset (Hugging Face dataset format local path).")
    parser.add_argument("--pretrained_weights_path", type=str,
                        default="/astrodino/preweight/teacher_checkpoint.pth", # 请确认这个路径是否相对于文件系统根目录
                        help="Full path to the DINOv2 teacher_checkpoint.pth file.")
    parser.add_argument("--dino_config_file", type=str,
                        default=os.path.join(resolved_astroclip_root, "astrodino", "config.yaml"), # 确保这里路径正确
                        help="Path to DINOv2 config.yaml file. Default is astrodino/config.yaml under project root.")
    parser.add_argument("--output_model_dir", type=str,
                        default=os.path.join(resolved_astroclip_root, "finetuned_models", "dinov2_z_regression"),
                        help="Directory to save finetuned models.")

    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training and evaluation (per GPU).")
    parser.add_argument("--lr", type=float, default=1e-4, help="Unified learning rate for the entire model.")
    parser.add_argument("--hidden_dims", type=int, nargs='+', default=[64, 64],
                        help="Hidden dimensions for the regression head MLP. E.g., --hidden_dims 512 256")
    parser.add_argument("--save_every_epochs", type=int, default=10, help="Save model every N epochs.")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of data loader workers.")

    parser.add_argument("--image_size", type=int, default=64, help="Image size for global crops. Default is 64x64, matching the dataset.")
    parser.add_argument("--global_crops_scale", type=float, nargs=2, default=(0.4, 1.0), help="Global crops scale range. Default is (0.4, 1.0).")
    parser.add_argument("--local_crops_scale", type=float, nargs=2, default=(0.05, 0.4), help="Local crops scale range. Default is (0.05, 0.4).")

    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility across distributed processes.")

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    finetune_dino_regression(args)