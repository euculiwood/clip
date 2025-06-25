# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/main/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

from functools import partial
import math
import logging
from typing import Sequence, Tuple, Union, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F  # Added for F.interpolate
import torch.utils.checkpoint
from torch.nn.init import trunc_normal_
import os
import timm  # Added timm import

from dinov2.layers import Mlp, PatchEmbed, SwiGLUFFNFused, MemEffAttention, NestedTensorBlock as Block


logger = logging.getLogger("dinov2")


def named_apply(fn: Callable, module: nn.Module, name="", depth_first=True, include_root=False) -> nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = ".".join((name, child_name)) if name else child_name
        named_apply(fn=fn, module=child_module, name=child_name, depth_first=depth_first, include_root=True)
    if depth_first and include_root:
        fn(module=module, name=name)
    return module


class BlockChunk(nn.ModuleList):
    def forward(self, x):
        for b in self:
            x = b(x)
        return x


class DinoVisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=5,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        ffn_bias=True,
        proj_bias=True,
        drop_path_rate=0.0,
        drop_path_uniform=False,
        init_values=None,  # for layerscale: None or 0 => no layerscale
        embed_layer=PatchEmbed,  # Original PatchEmbed class
        act_layer=nn.GELU,
        block_fn=Block,
        ffn_layer="mlp",
        block_chunks=1,
        num_register_tokens=0,
        interpolate_antialias=False,
        interpolate_offset=0.1,
        # 新增 ResNet 集成参数
        resnet_model_name: str = None,
        pretrained_resnet: bool = False,
        # 新增参数: ResNet 特征图的有效“patch”大小（通常设为 1，即每个像素都是一个 token）
        resnet_patch_size: int = 1
    ):
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 1
        self.n_blocks = depth
        self.num_heads = num_heads
        self.original_patch_size = patch_size # 记录原始 ViT 的 patch_size
        self.num_register_tokens = num_register_tokens
        self.interpolate_antialias = interpolate_antialias
        self.interpolate_offset = interpolate_offset

        self.resnet_backbone = None
        self.use_resnet_frontend = False  # 标志，指示是否启用了 ResNet 前端

        current_in_chans = in_chans
        current_img_height = img_size # 假设输入图像是正方形
        current_img_width = img_size
        current_patch_size = patch_size # 初始化为原始 ViT 的 patch_size

        if resnet_model_name:
            self.use_resnet_frontend = True
            logger.info(f"DINOv2 backbone: Using ResNet frontend: {resnet_model_name}, pretrained: {pretrained_resnet}")
            self.resnet_backbone = timm.create_model(
                resnet_model_name,
                pretrained=pretrained_resnet,
                features_only=True,
                out_indices=[3]  # 获取最后一个阶段的特征 (通常 ResNet50 为 2048 通道)
            )
            manual_model_path = "/root/.cache/huggingface/hub/models--timm--resnet50.a1_in1k/snapshots/5d9e13b8fdab4d9718bcf2c8f5c3af01878367e8/pytorch_model.bin"

            if not os.path.exists(manual_model_path):
                raise FileNotFoundError(f"错误：手动指定的 ResNet 预训练模型文件未找到: {manual_model_path}")

            print(f"正在从本地路径加载 ResNet 权重: {manual_model_path}")
            # map_location='cpu' 可以确保权重加载到 CPU，避免显存不足
            state_dict = torch.load(manual_model_path, map_location='cpu')

            # 加载状态字典到模型中
            self.resnet_backbone.load_state_dict(state_dict, strict=False)
            print("ResNet 权重已成功从本地加载。")
            if in_chans != 3:
                # 获取原始的第一个卷积层
                original_conv1 = self.resnet_backbone.conv1

                # 创建一个新的 Conv2d 层，使其接受 in_chans 数量的输入通道
                # 复制原始 conv1 层的其他参数，如输出通道、卷积核大小、步长、填充等
                new_conv1 = nn.Conv2d(
                    in_channels=in_chans, # <-- 这里设置为你的 5 通道
                    out_channels=original_conv1.out_channels,
                    kernel_size=original_conv1.kernel_size,
                    stride=original_conv1.stride,
                    padding=original_conv1.padding,
                    bias=(original_conv1.bias is not None), # 保持是否有 bias
                    groups=original_conv1.groups # 保持 groups 参数
                )

                # 将原始 conv1 层（3 通道）的权重复制到新 conv1 层的前 3 个通道
                # 并将新增加的通道的权重初始化为零。
                with torch.no_grad():
                    # 复制原始的权重
                    new_conv1.weight[:, :original_conv1.in_channels, :, :].copy_(original_conv1.weight)
                    # 如果新通道数大于原始通道数，则将新增通道的权重置零
                    if in_chans > original_conv1.in_channels:
                         new_conv1.weight[:, original_conv1.in_channels:, :, :].zero_()

                    # 如果有 bias，也复制 bias
                    if original_conv1.bias is not None:
                        new_conv1.bias.copy_(original_conv1.bias)

                # 用新的 conv1 层替换模型中的原始 conv1 层
                self.resnet_backbone.conv1 = new_conv1
                print(f"ResNet conv1 层已成功从 3 通道适应为 {in_chans} 通道输入。")
            # 进行一次模拟前向传播以推断 ResNet 输出的形状和通道数
            dummy_input = torch.randn(1, in_chans, img_size, img_size)  # 使用原始输入通道和图像大小
            with torch.no_grad():
                resnet_output_features = self.resnet_backbone(dummy_input)[0]

            current_in_chans = resnet_output_features.shape[1]  # PatchEmbed 的新输入通道数
            current_img_height = resnet_output_features.shape[2] # PatchEmbed 的新有效图像高度
            current_img_width = resnet_output_features.shape[3]  # PatchEmbed 的新有效图像宽度
            current_patch_size = resnet_patch_size  # 使用 ResNet 特征图的“patch”大小 (例如 1)

            logger.info(f"ResNet output feature map shape for PatchEmbed: (C={current_in_chans}, H={current_img_height}, W={current_img_width})")
            logger.info(f"PatchEmbed will now use effective patch_size={current_patch_size}")

        # 根据是否使用 ResNet 前端，实例化 PatchEmbed
        # 如果 resnet_model_name 为 None，则使用原始参数。
        # 否则，它将使用 ResNet 的输出通道和 resnet_patch_size。
        self.patch_embed = embed_layer(
            img_size=(current_img_height, current_img_width), # PatchEmbed 的输入图像尺寸现在是特征图的尺寸
            patch_size=current_patch_size,  # PatchEmbed 的 patch_size 现在是 resnet_patch_size (例如 1)
            in_chans=current_in_chans,  # PatchEmbed 的输入通道现在是 ResNet 输出通道
            embed_dim=embed_dim,
        )
        # 更新主 patch_size 属性，供 interpolate_pos_encoding 使用
        self.patch_size = current_patch_size 

        num_patches = self.patch_embed.num_patches  # PatchEmbed 会根据当前参数正确计算 token 数量

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        assert num_register_tokens >= 0
        self.register_tokens = (
            nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim)) if num_register_tokens else None
        )

        if drop_path_uniform is True:
            dpr = [drop_path_rate] * depth
        else:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        if ffn_layer == "mlp":
            logger.info("using MLP layer as FFN")
            ffn_layer = Mlp
        elif ffn_layer == "swiglufused" or ffn_layer == "swiglu":
            logger.info("using SwiGLU layer as FFN")
            ffn_layer = SwiGLUFFNFused
        elif ffn_layer == "identity":
            logger.info("using Identity layer as FFN")

            def f(*args, **kwargs):
                return nn.Identity()

            ffn_layer = f
        else:
            raise NotImplementedError

        blocks_list = [
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                ffn_layer=ffn_layer,
                init_values=init_values,
            )
            for i in range(depth)
        ]
        if block_chunks > 0:
            self.chunked_blocks = True
            chunked_blocks = []
            chunksize = depth // block_chunks
            for i in range(0, depth, chunksize):
                chunked_blocks.append([nn.Identity()] * i + blocks_list[i : i + chunksize])
            self.blocks = nn.ModuleList([BlockChunk(p) for p in chunked_blocks])
        else:
            self.chunked_blocks = False
            self.blocks = nn.ModuleList(blocks_list)

        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()

        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))

        self.init_weights()

    def init_weights(self):
        trunc_normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=1e-6)
        if self.register_tokens is not None:
            nn.init.normal_(self.register_tokens, std=1e-6)
        named_apply(init_weights_vit_timm, self)

    def interpolate_pos_encoding(self, x, current_w, current_h):
        """
        根据当前输入（原始图像或 ResNet 特征图）的尺寸 (current_w, current_h)，
        对位置编码进行插值，以适应新的 token 数量。
        """
        previous_dtype = x.dtype
        npatch_current = (current_w // self.patch_size) * (current_h // self.patch_size) # 当前实际的 patch 数量
        
        # N 是 self.pos_embed 中原始 patch 的数量
        N_original_pos_embed = self.pos_embed.shape[1] - self.num_tokens - self.num_register_tokens 
        
        # 如果当前的 patch 数量与原始位置编码中的 patch 数量相同，则无需插值
        if npatch_current == N_original_pos_embed:
            return self.pos_embed.to(previous_dtype)

        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        
        # 分离 register token 的位置编码（如果存在）
        if self.num_register_tokens > 0:
            register_pos_embed = pos_embed[:, 1 : 1 + self.num_register_tokens]
            patch_pos_embed_orig = pos_embed[:, 1 + self.num_register_tokens :]
        else:
            register_pos_embed = None
            patch_pos_embed_orig = pos_embed[:, 1:]

        dim = x.shape[-1]
        
        # M 是原始 patch 网格的边长（例如，对于 224x224 图像和 16x16 patch，M = 14）
        M_original_grid_size = int(math.sqrt(N_original_pos_embed))
        assert M_original_grid_size * M_original_grid_size == N_original_pos_embed

        # 目标网格尺寸 (w0, h0)
        w0 = current_w // self.patch_size
        h0 = current_h // self.patch_size
        
        kwargs = {}
        if self.interpolate_offset:
            sx = float(w0 + self.interpolate_offset) / M_original_grid_size
            sy = float(h0 + self.interpolate_offset) / M_original_grid_size
            kwargs["scale_factor"] = (sx, sy)
        else:
            kwargs["size"] = (w0, h0)

        patch_pos_embed_interp = F.interpolate(
            patch_pos_embed_orig.reshape(1, M_original_grid_size, M_original_grid_size, dim).permute(0, 3, 1, 2),
            mode="bicubic",
            antialias=self.interpolate_antialias,
            **kwargs,
        )
        assert (w0, h0) == patch_pos_embed_interp.shape[-2:]
        patch_pos_embed_interp = patch_pos_embed_interp.permute(0, 2, 3, 1).view(1, -1, dim)

        # 拼接 CLS token、注册 token (如果存在) 和插值后的 patch 位置编码
        if register_pos_embed is not None:
            return torch.cat((class_pos_embed.unsqueeze(0), register_pos_embed, patch_pos_embed_interp), dim=1).to(previous_dtype)
        else:
            return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed_interp), dim=1).to(previous_dtype)


    def prepare_tokens_with_masks(self, x, masks=None):
        # 获取原始图像尺寸
        _B, _nc, original_img_h, original_img_w = x.shape

        current_w, current_h = original_img_w, original_img_h # 初始为原始图像尺寸

        if self.use_resnet_frontend:
            # 如果启用了 ResNet 前端，则首先通过 ResNet 进行特征提取
            x = self.resnet_backbone(x)[0]  # 输出是 (B, C_out, H_feat, W_feat)
            current_w, current_h = x.shape[3], x.shape[2]  # 更新为 ResNet 输出的特征图尺寸

        # 通过 PatchEmbed 将图像或特征图转换为 token
        x = self.patch_embed(x)

        if masks is not None:
            # 如果使用了 ResNet，需要将 mask 调整到特征图的尺寸
            if self.use_resnet_frontend:
                masks = F.interpolate(masks.unsqueeze(1).float(), size=(current_h, current_w), mode='nearest').squeeze(1).bool()
            
            x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)

        # 拼接 CLS token
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        
        # 将当前有效空间尺寸传递给位置编码插值函数
        x = x + self.interpolate_pos_encoding(x, current_w, current_h)

        # 拼接注册 token (如果存在)
        if self.num_register_tokens is not None and self.num_register_tokens > 0:
            x = torch.cat(
                (
                    x[:, :1],
                    self.register_tokens.expand(x.shape[0], -1, -1),
                    x[:, 1:],
                ),
                dim=1,
            )

        return x

    def forward_features_list(self, x_list, masks_list):
        x = [self.prepare_tokens_with_masks(x, masks) for x, masks in zip(x_list, masks_list)]
        for blk in self.blocks:
            x = blk(x)

        all_x = x
        output = []
        for x, masks in zip(all_x, masks_list):
            x_norm = self.norm(x)
            output.append(
                {
                    "x_norm_clstoken": x_norm[:, 0],
                    "x_norm_regtokens": x_norm[:, 1 : self.num_register_tokens + 1],
                    "x_norm_patchtokens": x_norm[:, self.num_register_tokens + 1 :],
                    "x_prenorm": x,
                    "masks": masks,
                }
            )
        return output

    def forward_features(self, x, masks=None):
        if isinstance(x, list):
            return self.forward_features_list(x, masks)

        x = self.prepare_tokens_with_masks(x, masks)

        for blk in self.blocks:
            x = blk(x)

        x_norm = self.norm(x)
        return {
            "x_norm_clstoken": x_norm[:, 0],
            "x_norm_regtokens": x_norm[:, 1 : self.num_register_tokens + 1],
            "x_norm_patchtokens": x_norm[:, self.num_register_tokens + 1 :],
            "x_prenorm": x,
            "masks": masks,
        }

    def _get_intermediate_layers_not_chunked(self, x, n=1):
        x = self.prepare_tokens_with_masks(x)
        # If n is an int, take the n last blocks. If it's a list, take them
        output, total_block_len = [], len(self.blocks)
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in blocks_to_take:
                output.append(x)
        assert len(output) == len(blocks_to_take), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    def _get_intermediate_layers_chunked(self, x, n=1):
        x = self.prepare_tokens_with_masks(x)
        output, i, total_block_len = [], 0, len(self.blocks[-1])
        # If n is an int, take the n last blocks. If it's a list, take them
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for block_chunk in self.blocks:
            for blk in block_chunk[i:]:  # Passing the nn.Identity()
                x = blk(x)
                if i in blocks_to_take:
                    output.append(x)
                i += 1
        assert len(output) == len(blocks_to_take), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n: Union[int, Sequence] = 1,  # Layers or n last layers to take
        reshape: bool = False,
        return_class_token: bool = False,
        norm=True,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]]]:
        if self.chunked_blocks:
            outputs = self._get_intermediate_layers_chunked(x, n)
        else:
            outputs = self._get_intermediate_layers_not_chunked(x, n)
        if norm:
            outputs = [self.norm(out) for out in outputs]
        class_tokens = [out[:, 0] for out in outputs]
        outputs = [out[:, 1 + self.num_register_tokens :] for out in outputs]
        if reshape:
            # Need original image size to reshape correctly, not current feature map size
            _B, _nc, original_img_h, original_img_w = x.shape
            
            # If using ResNet, calculate feature map dimensions
            if self.use_resnet_frontend:
                # Dummy forward pass to determine ResNet output dimensions for reshaping
                dummy_input = torch.randn(1, _nc, original_img_h, original_img_w)
                with torch.no_grad():
                    resnet_output = self.resnet_backbone(dummy_input)[0]
                H_feat, W_feat = resnet_output.shape[2], resnet_output.shape[3]
                reshape_h = H_feat // self.patch_size
                reshape_w = W_feat // self.patch_size
            else:
                reshape_h = original_img_h // self.patch_size
                reshape_w = original_img_w // self.patch_size

            outputs = [
                out.reshape(_B, reshape_h, reshape_w, -1).permute(0, 3, 1, 2).contiguous()
                for out in outputs
            ]
        if return_class_token:
            return tuple(zip(outputs, class_tokens))
        return tuple(outputs)

    def forward(self, *args, is_training=False, **kwargs):
        ret = self.forward_features(*args, **kwargs)
        if is_training:
            return ret
        else:
            return self.head(ret["x_norm_clstoken"])


def init_weights_vit_timm(module: nn.Module, name: str = ""):
    """ViT weight initialization, original timm impl (for reproducibility)"""
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def vit_small(
    patch_size=16,
    num_register_tokens=0,
    resnet_model_name: str = None,
    pretrained_resnet: bool = True,
    resnet_patch_size: int = 1,
    **kwargs
):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        resnet_model_name=resnet_model_name,
        pretrained_resnet=pretrained_resnet,
        resnet_patch_size=resnet_patch_size,
        **kwargs,
    )
    return model


def vit_base(
    patch_size=16,
    num_register_tokens=0,
    resnet_model_name: str = None,
    pretrained_resnet: bool = True,
    resnet_patch_size: int = 1,
    **kwargs
):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        resnet_model_name=resnet_model_name,
        pretrained_resnet=pretrained_resnet,
        resnet_patch_size=resnet_patch_size,
        **kwargs,
    )
    return model


def vit_large(
    patch_size=16,
    num_register_tokens=0,
    resnet_model_name: str = None,
    pretrained_resnet: bool = True,
    resnet_patch_size: int = 1,
    **kwargs
):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        resnet_model_name=resnet_model_name,
        pretrained_resnet=pretrained_resnet,
        resnet_patch_size=resnet_patch_size,
        **kwargs,
    )
    return model


def vit_giant2(
    patch_size=16,
    num_register_tokens=0,
    resnet_model_name: str = None,
    pretrained_resnet: bool = True,
    resnet_patch_size: int = 1,
    **kwargs
):
    """
    Close to ViT-giant, with embed-dim 1536 and 24 heads => embed-dim per head 64
    """
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1536,
        depth=40,
        num_heads=24,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        resnet_model_name=resnet_model_name,
        pretrained_resnet=pretrained_resnet,
        resnet_patch_size=resnet_patch_size,
        **kwargs,
    )
    return model