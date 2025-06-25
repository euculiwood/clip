import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.distributed as dist
import torch.nn.parallel as DDP
from torch.utils.data.distributed import DistributedSampler

import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from argparse import ArgumentParser
import logging
from typing import List

# 导入 Hugging Face datasets 库
from datasets import load_from_disk

# 导入 timm 库 (现在在 vit.py 中使用，但 finetune.py 如果直接调用 DinoVisionTransformer 则不需要额外导入)
# import timm # 如果只修改 vit.py 内部，这里不需要 timm

# --- 关键修改：将项目根目录添加到 sys.path ---
current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_script_path))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 设置日志
logger = logging.getLogger("finetune_dino")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# 导入 AstroDINO 相关的模块 (DataAugmentationAstroDINO 保持不变)
from astrodino.env import format_with_env
from astrodino.data.augmentations import DataAugmentationAstroDINO

# 从 dinov2.models.vision_transformer 导入修改后的 DinoVisionTransformer
from dinov2.models.vision_transformer import DinoVisionTransformer

from dinov2.layers.patch_embed import PatchEmbed
from dinov2.layers.block import Block

# 确保 MLP 类可用，如果它在你的 downstream_tasks.property_utils.models 中
try:
    from downstream_tasks.property_utils.models import MLP
except ImportError:
    logger.warning("Could not import MLP from downstream_tasks.property_utils.models. Defining a simple MLP here.")
    # 如果 MLP 无法导入，则在此处定义一个简单的 MLP 类
    class MLP(nn.Module):
        def __init__(self, n_in: int, n_out: int, n_hidden: tuple = (), dropout: float = 0.0):
            super().__init__()
            layers = []
            current_dim = n_in
            for h_dim in n_hidden:
                layers.append(nn.Linear(current_dim, h_dim))
                layers.append(nn.ReLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
                current_dim = h_dim
            layers.append(nn.Linear(current_dim, n_out))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)

# --- 2. 定义新的 DinoV2RegressionModel 类，直接使用 DinoVisionTransformer ---
class DinoV2RegressionModel(nn.Module):
    def __init__(self, dino_model_config: dict, output_dim=1, hidden_dims: List[int] = [64, 64]):
        super().__init__()
        # 直接实例化 DinoVisionTransformer，并传入所有配置参数
        self.dino_backbone = DinoVisionTransformer(**dino_model_config)
        
        # 回归头将使用 DINOv2 输出的 CLS token 特征
        self.regression_head = MLP(
            n_in=self.dino_backbone.embed_dim, # DINOv2 模型的嵌入维度
            n_out=output_dim,
            n_hidden=tuple(hidden_dims),
            dropout=0.1
        )
        logger.info(f"Regression Head defined: {self.regression_head}")

    def forward(self, x):
        # DinoVisionTransformer 的 forward_features 方法返回一个字典
        dino_output = self.dino_backbone.forward_features(x)
        features = dino_output["x_norm_clstoken"] # 获取 CLS token 的归一化特征
        output = self.regression_head(features)
        return output

# --- 定义 Hugging Face Dataset 的适配器类 (保持不变) ---
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
            # 确保图像是 3 通道 (如果你的数据集图像是单通道，需要转换为 3 通道)
            if processed_image.shape[0] == 1:
                processed_image = processed_image.repeat(3, 1, 1)
            # 如果你的图像已经是 5 通道，这里不需要转换
            # 如果是其他通道数，请确保与 args.input_channels 匹配

        if not torch.is_tensor(processed_image):
             raise ValueError("Final processed_image is not a torch.Tensor after transform/processing.")

        z_value = item["z"]
        if torch.is_tensor(z_value):
            return {"image": processed_image, "z": z_value.float().view(1)}
        else:
            return {"image": processed_image, "z": torch.tensor(float(z_value), dtype=torch.float32).view(1)}

# --- 自定义 collate_fn (保持不变) ---
def finetune_collate_fn(batch):
    processed_images = []
    zs = []
    for item in batch:
        processed_images.append(item["image"])
        zs.append(torch.tensor(item["z"], dtype=torch.float32))

    images_tensor = torch.stack(processed_images)
    zs_tensor = torch.stack(zs).unsqueeze(1)

    return {"image": images_tensor, "z": zs_tensor}

# --- 训练函数 (finetune_dino_regression) ---
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

    # --- 直接加载 Hugging Face 数据集并使用 HFDatasetWrapper ---
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

    if is_distributed:
        dist.barrier()
    if rank == 0:
        logger.info("Z value scaler broadcasted and set on all processes." if is_distributed else "Z value scaler ready on single process.")


    # --- 模型初始化 ---
    # 根据 args.vit_model_type 确定 DINOv2 模型的配置
    if args.vit_model_type == "small":
        dino_embed_dim = 384
        dino_depth = 12
        dino_num_heads = 6
    elif args.vit_model_type == "base":
        dino_embed_dim = 768
        dino_depth = 12
        dino_num_heads = 12
    elif args.vit_model_type == "large":
        dino_embed_dim = 1024
        dino_depth = 24
        dino_num_heads = 16
    else:
        raise ValueError(f"Unknown ViT model type: {args.vit_model_type}")

    dino_model_config = {
        "img_size": args.image_size,
        "patch_size": args.vit_patch_size, # 原始 DINOv2 patch size，例如 16
        "in_chans": args.input_channels, # 输入通道数，例如 5 (用于 ResNet 的输入)
        "embed_dim": dino_embed_dim,
        "depth": dino_depth,
        "num_heads": dino_num_heads,
        "mlp_ratio": 4.0,
        "qkv_bias": True,
        "ffn_bias": True,
        "proj_bias": True,
        "drop_path_rate": 0.1, # 可调整
        "drop_path_uniform": False,
        "init_values": None,
        "embed_layer": PatchEmbed, # 从 dinov2.layers 导入
        "act_layer": nn.GELU,
        "block_fn": Block, # 从 dinov2.layers 导入
        "ffn_layer": "mlp",
        "block_chunks": 1,
        "num_register_tokens": 0,
        "interpolate_antialias": False,
        "interpolate_offset": 0.1,
        # 新增 ResNet 相关参数
        "resnet_model_name": args.resnet_model_name,
        "pretrained_resnet": args.pretrained_timm_models,
        "resnet_patch_size": args.resnet_patch_size
    }

    model = DinoV2RegressionModel(
        dino_model_config=dino_model_config,
        output_dim=1,
        hidden_dims=args.hidden_dims
    )

    # 如果有预训练的 DINOv2 权重，可以在这里尝试加载
    if args.pretrained_weights_path:
        if rank == 0:
            logger.info(f"Loading pretrained DINOv2 weights from: {args.pretrained_weights_path}")
        try:
            checkpoint = torch.load(args.pretrained_weights_path, map_location="cpu")
            # 过滤掉不匹配的权重 (例如 patch_embed.proj 和 pos_embed)
            model_state_dict = model.state_dict()
            pretrained_state_dict = checkpoint["teacher"] # 假设权重在 "model" 键下

            # 过滤掉不匹配的 keys
            new_pretrained_state_dict = {}
            for k, v in pretrained_state_dict.items():
                if k in model_state_dict and model_state_dict[k].shape == v.shape:
                    new_pretrained_state_dict[k] = v
                else:
                    if rank == 0:
                        logger.warning(f"Skipping loading of {k} due to shape mismatch or not found. Original shape: {v.shape}, Model shape: {model_state_dict.get(k, 'Not found in model').shape}")
            
            # 尝试加载过滤后的权重
            model.load_state_dict(new_pretrained_state_dict, strict=False)
            if rank == 0:
                logger.info("Successfully loaded compatible DINOv2 pretrained weights (non-matching layers skipped).")
        except Exception as e:
            if rank == 0:
                logger.error(f"Error loading pretrained DINOv2 weights: {e}")
                logger.warning("Model will be trained without DINOv2 pretrained weights.")


    model.to(device)
    if is_distributed:
        model = DDP.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    # 如果需要冻结 ResNet 或 ViT 的部分层，可以在这里添加逻辑
    # 例如：冻结 ResNet 的所有参数
    # if model.module.dino_backbone.resnet_backbone is not None:
    #     for param in model.module.dino_backbone.resnet_backbone.parameters():
    #         param.requires_grad = False
    #     if rank == 0:
    #         logger.info("Froze ResNet backbone parameters.")


    # --- 优化器设置---
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # --- 训练循环 (保持不变) ---
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
            labels_for_scaler = labels_raw.cpu().numpy().reshape(-1, 1)
            labels_scaled = torch.tensor(scaler_z.transform(labels_for_scaler), dtype=torch.float32).to(device)

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
                labels_for_scaler_val = labels_raw_batch.cpu().numpy().reshape(-1, 1)
                labels_scaled_batch = torch.tensor(scaler_z.transform(labels_for_scaler_val), dtype=torch.float32).to(device)

                outputs_scaled = model(imgs)

                val_preds_scaled.append(outputs_scaled.cpu().numpy())
                val_labels_scaled.append(labels_scaled_batch.cpu().numpy())

                preds_raw_batch = scaler_z.inverse_transform(outputs_scaled.cpu().numpy().reshape(-1, 1))
                val_preds_raw.append(preds_raw_batch)
                val_labels_raw.append(labels_raw_batch.cpu().numpy().reshape(-1, 1))

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


# --- 4. Main 函数和参数解析 (修改参数) ---
if __name__ == '__main__':
    parser = ArgumentParser(description="Finetune ResNet-ViT Hybrid Model for Redshift Regression.")

    ASTROCLIP_ROOT_PLACEHOLDER = "{ASTROCLIP_ROOT}"
    resolved_astroclip_root = format_with_env(ASTROCLIP_ROOT_PLACEHOLDER)

    parser.add_argument("--train_dataset_path", type=str,
                        default=os.path.join(resolved_astroclip_root, "astrodino", "data", "train_dataset"),
                        help="Path to your training dataset (Hugging Face dataset format local path).")
    parser.add_argument("--test_dataset_path", type=str,
                        default=os.path.join(resolved_astroclip_root, "astrodino", "data", "test_dataset"),
                        help="Path to your test dataset (Hugging Face dataset format local path).")
    # 增加 DINOv2 预训练权重路径参数
    parser.add_argument("--pretrained_weights_path", type=str, default=os.path.join(resolved_astroclip_root, "astrodino", "preweight", "teacher_checkpoint.pth"),
                        help="Path to DINOv2 pretrained weights (e.g., vit_large_dinov2.pth). Leave empty to train from scratch.")

    parser.add_argument("--output_model_dir", type=str,
                        default=os.path.join(resolved_astroclip_root, "finetuned_models", "resnet_dino_vit_hybrid_z_regression"), # 修改默认输出目录
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

    # *********** 新增模型相关的参数 ***********
    parser.add_argument("--vit_model_type", type=str, default="large", choices=["small", "base", "large"],
                        help="Type of DINOv2 ViT model to use (determines embed_dim, depth, num_heads).")
    parser.add_argument("--vit_patch_size", type=int, default=16,
                        help="Original patch size for DINOv2 ViT (e.g., 16).")
    parser.add_argument("--resnet_model_name", type=str, default="resnet50",
                        help="Name of the ResNet model from timm (e.g., 'resnet18', 'resnet50'). Set to None to use pure ViT (without ResNet frontend).")
    parser.add_argument("--pretrained_timm_models", action="store_true", default=False,
                        help="Whether to use pre-trained weights for ResNet from timm (ImageNet pre-trained).")
    parser.add_argument("--resnet_patch_size", type=int, default=1,
                        help="Effective patch size to apply on ResNet features (e.g., 1 means each feature map spatial location is a token).")
    parser.add_argument("--input_channels", type=int, default=5, # 根据你的天文图像通道数调整
                        help="Number of input channels for the images (e.g., 5 for astronomical images).")
    # **********************************************

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    finetune_dino_regression(args)