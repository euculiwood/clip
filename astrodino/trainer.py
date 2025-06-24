# AstroDino 训练脚本
# 基于 DINOv2 训练框架实现
# 动态获取当前脚本所在目录的父目录，即项目根目录 /hy-tmp/clip
# 这样可以确保 dinov2 和 provabgs 文件夹能够被找到

import os
import sys
current_file_dir = os.path.dirname(os.path.abspath(__file__)) # /hy-tmp/clip/astrodino
project_root = os.path.abspath(os.path.join(current_file_dir, '..')) # /hy-tmp/clip
sys.path.insert(0, project_root)
import argparse
import logging
import math
from functools import partial  # 用于固定函数参数
import torch
import wandb  # 权重与偏差可视化工具

from dinov2 import distributed as distributed  # 分布式训练工具
from dinov2.data import (
    # 原始DINO数据增强
    MaskingGenerator,      # 掩码生成器
    SamplerType,           # 采样器类型
    collate_data_and_cast, # 数据整理函数
)
from dinov2.fsdp import FSDPCheckpointer  # 分布式检查点保存
from dinov2.train.ssl_meta_arch import SSLMetaArch  # 自监督元架构
from dinov2.utils.config import setup  # 配置初始化
from dinov2.utils.utils import CosineScheduler  # 余弦调度器
from fvcore.common.checkpoint import PeriodicCheckpointer  # 周期检查点
from omegaconf import OmegaConf  # 配置文件处理

# 导入自定义模块
from astrodino.data.augmentations import DataAugmentationAstroDINO  # 天文专用数据增强
from astrodino.data.loaders import make_data_loader, make_dataset  # 数据加载器
from astrodino.utils import MetricLogger  # 指标记录器
from env import format_with_env  # 环境变量格式化

# PyTorch 1.12 默认关闭 TF32
torch.backends.cuda.matmul.allow_tf32 = True

logger = logging.getLogger("dinov2")  # 获取日志记录器
ASTROCLIP_ROOT = format_with_env("{ASTROCLIP_ROOT}")  # 项目根目录


def get_args_parser(add_help: bool = True):
    """创建命令行参数解析器"""
    parser = argparse.ArgumentParser("DINOv2 training", add_help=add_help)
    # 添加配置文件参数
    parser.add_argument(
        "--config-file",
        "-c",
        "--config",
        default=f"{ASTROCLIP_ROOT}/astrodino/config.yaml",
        metavar="FILE",
        help="path to config file",
    )
    # 添加不恢复训练参数
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Whether to not attempt to resume from the checkpoint directory. ",
    )
    parser.add_argument(
        "--eval-only",action="store_true", help="perform evaluation only"
    )
    parser.add_argument("--eval", type=str, default="", help="Eval type to perform")
    # 添加配置覆盖参数
    parser.add_argument(
        "opts",
        help="""
Modify config options at the end of the command. For Yacs configs, use
space-separated \"PATH.KEY VALUE\" pairs.
For python-based LazyConfig, use \"path.key=value\".
        """.strip(),
        default=None,
        nargs=argparse.REMAINDER,
    )  
    parser.add_argument("--run-name", default="01", help="run name for wandb")
    parser.add_argument("--group-name", default="test", help="group name for wandb")

    return parser


def build_optimizer(cfg, params_groups):
    """构建优化器"""
    return torch.optim.AdamW(
        params_groups,  # 参数组
        betas=(cfg.optim.adamw_beta1, cfg.optim.adamw_beta2)  # 动量参数
    )


def build_schedulers(cfg):
    """构建各种学习率调度器"""
    OFFICIAL_EPOCH_LENGTH = cfg.train.OFFICIAL_EPOCH_LENGTH
    # 学习率调度参数
    lr = dict(
        base_value=cfg.optim["lr"],
        final_value=cfg.optim["min_lr"],
        total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
        warmup_iters=cfg.optim["warmup_epochs"] * OFFICIAL_EPOCH_LENGTH,
        start_warmup_value=0,
    )
    # 权重衰减调度参数
    wd = dict(
        base_value=cfg.optim["weight_decay"],
        final_value=cfg.optim["weight_decay_end"],
        total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
    )
    # 动量调度参数
    momentum = dict(
        base_value=cfg.teacher["momentum_teacher"],
        final_value=cfg.teacher["final_momentum_teacher"],
        total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
    )
    # 教师网络温度调度
    teacher_temp = dict(
        base_value=cfg.teacher["teacher_temp"],
        final_value=cfg.teacher["teacher_temp"],
        total_iters=cfg.teacher["warmup_teacher_temp_epochs"] * OFFICIAL_EPOCH_LENGTH,
        warmup_iters=cfg.teacher["warmup_teacher_temp_epochs"] * OFFICIAL_EPOCH_LENGTH,
        start_warmup_value=cfg.teacher["warmup_teacher_temp"],
    )

    # 创建各种调度器
    lr_schedule = CosineScheduler(**lr)
    wd_schedule = CosineScheduler(**wd)
    momentum_schedule = CosineScheduler(**momentum)
    teacher_temp_schedule = CosineScheduler(**teacher_temp)
    last_layer_lr_schedule = CosineScheduler(**lr)

    # 冻结最后一层的初始阶段学习率为0
    last_layer_lr_schedule.schedule[
        : cfg.optim["freeze_last_layer_epochs"] * OFFICIAL_EPOCH_LENGTH
    ] = 0

    logger.info("Schedulers ready.")

    return (
        lr_schedule,
        wd_schedule,
        momentum_schedule,
        teacher_temp_schedule,
        last_layer_lr_schedule,
    )


def apply_optim_scheduler(optimizer, lr, wd, last_layer_lr):
    """应用优化器调度参数"""
    for param_group in optimizer.param_groups:
        is_last_layer = param_group["is_last_layer"]
        lr_multiplier = param_group["lr_multiplier"]
        wd_multiplier = param_group["wd_multiplier"]
        param_group["weight_decay"] = wd * wd_multiplier
        param_group["lr"] = (last_layer_lr if is_last_layer else lr) * lr_multiplier


def do_test(cfg, model, iteration):
    """执行测试阶段"""
    new_state_dict = model.teacher.state_dict()
    # 只在主进程中保存
    if distributed.is_main_process():
        iterstring = str(iteration)
        eval_dir = os.path.join(
            ASTROCLIP_ROOT,
            "outputs",
            "astroclip_image",
            wandb.run.id,
            "eval",
            iterstring,
        )
        os.makedirs(eval_dir, exist_ok=True)
        # 保存教师网络检查点
        teacher_ckp_path = os.path.join(eval_dir, "teacher_checkpoint.pth")
        torch.save({"teacher": new_state_dict}, teacher_ckp_path)


def do_train(cfg, model, run_name, group_name, resume=False):
    """执行训练阶段
    参数:
        cfg: 配置对象，包含训练参数
        model: 要训练的模型实例
        run_name: 当前运行的名称标识
        group_name: 实验组名称
        resume: 是否从检查点恢复训练
    """
    model.train()  # 设置模型为训练模式
    inputs_dtype = torch.half  # 输入数据类型设置为半精度浮点数
    fp16_scaler = model.fp16_scaler  # 获取混合精度训练使用的缩放器

    # 设置优化器，使用配置和模型参数组创建优化器实例
    optimizer = build_optimizer(cfg, model.get_params_groups())
    (
        lr_schedule,  # 学习率调度器
        wd_schedule,  # 权重衰减调度器
        momentum_schedule,  # 动量调度器
        teacher_temp_schedule,  # 教师网络温度调度器
        last_layer_lr_schedule,  # 最后一层学习率调度器
    ) = build_schedulers(cfg)

    # 检查点保存器
    checkpointer = FSDPCheckpointer(
        model,
        f"{ASTROCLIP_ROOT}/outputs/astroclip_image/{run_name}",
        optimizer=optimizer,
        save_to_disk=True,
    )

    # 加载检查点
    start_iter = (
        checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get(
            "iteration", -1
        )
        + 1
    )

    OFFICIAL_EPOCH_LENGTH = cfg.train.OFFICIAL_EPOCH_LENGTH
    max_iter = cfg.optim.epochs * OFFICIAL_EPOCH_LENGTH

    # 周期检查点
    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer,
        period=3 * OFFICIAL_EPOCH_LENGTH,
        max_iter=max_iter,
        max_to_keep=3,
    )

    # 设置数据预处理
    img_size = cfg.crops.global_crops_size #144
    patch_size = cfg.student.patch_size #12
    n_tokens = (img_size // patch_size) ** 2 #144
    mask_generator = MaskingGenerator(
        input_size=(img_size // patch_size, img_size // patch_size),
        max_num_patches=0.5 * img_size // patch_size * img_size // patch_size,
    )

    # 应用自定义天文数据增强
    data_transform = DataAugmentationAstroDINO(
        cfg.crops.global_crops_scale,  # 关键参数：全局裁剪比例
        cfg.crops.local_crops_scale,
        cfg.crops.local_crops_number,
        global_crops_size=cfg.crops.global_crops_size,
        local_crops_size=cfg.crops.local_crops_size,
    )

    # 数据整理函数
    collate_fn = partial(
        collate_data_and_cast,
        mask_ratio_tuple=cfg.ibot.mask_ratio_min_max,
        mask_probability=cfg.ibot.mask_sample_probability,
        n_tokens=n_tokens,
        mask_generator=mask_generator,
        dtype=inputs_dtype,
    )

    # 设置数据加载器
    dataset = make_dataset(
        dataset_str=format_with_env(cfg.train.dataset_path),
        transform=data_transform,
        target_transform=lambda _: (),
    )

    # sampler_type = SamplerType.INFINITE
    sampler_type = SamplerType.SHARDED_INFINITE
    data_loader = make_data_loader(
        dataset=dataset,
        batch_size=cfg.train.batch_size_per_gpu,
        num_workers=cfg.train.num_workers,
        shuffle=True,
        seed=start_iter,  # TODO: Fix this -- cfg.train.seed
        sampler_type=sampler_type,
        sampler_advance=0,  # TODO(qas): fix this -- start_iter * cfg.train.batch_size_per_gpu,
        drop_last=True,
        collate_fn=collate_fn,
    )
    # 设置 wandb
    global_rank = int(os.environ.get("RANK", 0))
    if global_rank == 0:
        wandb.init(
            project="astrodino",
            entity=format_with_env("{WANDB_ENTITY_NAME}"),
            name=run_name,
            group=group_name,
            resume="allow",
            dir=f"{ASTROCLIP_ROOT}/outputs/astroclip_image",
            allow_val_change=True,
            settings=wandb.Settings(init_timeout=300),
            mode="offline"
        )
        wandb.run.config.update(OmegaConf.to_object(cfg))

    # 训练循环
    iteration = start_iter

    logger.info("Starting training from iteration {}".format(start_iter))
    metrics_file = os.path.join(
        ASTROCLIP_ROOT,
        "outputs",
        "astroclip_image",
        run_name,
        "training_metrics.json",
    )
    metric_logger = MetricLogger(
        delimiter="  ", wandb=wandb.run, output_file=metrics_file
    )
    header = "Training"

    for data in metric_logger.log_every(data_loader, 25, header, max_iter, start_iter):
        current_batch_size = data["collated_global_crops"].shape[0] / 2
        if iteration > max_iter:
            return

        # 应用调度器
        lr = lr_schedule[iteration]
        wd = wd_schedule[iteration]
        mom = momentum_schedule[iteration]
        teacher_temp = teacher_temp_schedule[iteration]
        last_layer_lr = last_layer_lr_schedule[iteration]
        apply_optim_scheduler(optimizer, lr, wd, last_layer_lr)

        # 计算损失
        optimizer.zero_grad(set_to_none=True)
        loss_dict = model.forward_backward(data, teacher_temp=teacher_temp)

        # 梯度裁剪
        if fp16_scaler is not None:
            if cfg.optim.clip_grad:
                fp16_scaler.unscale_(optimizer)
                for v in model.student.values():
                    v.clip_grad_norm_(cfg.optim.clip_grad)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()
        else:
            if cfg.optim.clip_grad:
                for v in model.student.values():
                    v.clip_grad_norm_(cfg.optim.clip_grad)
            optimizer.step()

        # 教师网络 EMA 更新
        model.update_teacher(mom)

        # 日志记录
        if distributed.get_global_size() > 1:
            for v in loss_dict.values():
                torch.distributed.all_reduce(v)
        loss_dict_reduced = {
            k: v.item() / distributed.get_global_size() for k, v in loss_dict.items()
        }

        if math.isnan(sum(loss_dict_reduced.values())):
            logger.info("NaN detected")
            raise AssertionError
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        metric_logger.update(lr=lr)
        metric_logger.update(wd=wd)
        metric_logger.update(mom=mom)
        metric_logger.update(last_layer_lr=last_layer_lr)
        metric_logger.update(current_batch_size=current_batch_size)
        metric_logger.update(total_loss=losses_reduced, **loss_dict_reduced)
        # 检查点和测试

        if (
            cfg.evaluation.eval_period_iterations > 0
            and (iteration + 1) % cfg.evaluation.eval_period_iterations == 0
        ):
            do_test(cfg, model, f"training_{iteration}")
            torch.cuda.synchronize()
        periodic_checkpointer.step(iteration)

        iteration = iteration + 1
    metric_logger.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def main_cli(cli_args=None):
    print("启动训练程序...")

    """主命令行入口"""
    args = get_args_parser(add_help=True).parse_args(cli_args)

    run_name = str(args.run_name)
    args.output_dir = f"{ASTROCLIP_ROOT}/outputs/astroclip_image/{run_name}"

    cfg = setup(args)  # 初始化配置

    # 构建模型并准备分布式训练
    model = SSLMetaArch(cfg).to(torch.device("cuda"))
    model.prepare_for_distributed_training()

    logger.info("Model:\n{}".format(model))
    # 如果只执行评估
    if args.eval_only:
        iteration = (
            FSDPCheckpointer(model, save_dir=cfg.train.output_dir)
            .resume_or_load(cfg.MODEL.WEIGHTS, resume=not args.no_resume)
            .get("iteration", -1)
            + 1
        )
        return do_test(cfg, model, f"manual_{iteration}")

    # 执行训练
    do_train(cfg, model, run_name, args.group_name, resume=not args.no_resume)


if __name__ == "__main__":
    main_cli()  # 程序入口