# -*- coding: utf-8 -*-
# =============================================================================
# Project     : Qwen3.5Seg
# File        : model/vision_neck.py
# Author      : Yuki
# Created     : 2026/3/12 11:14
# Description : 
# =============================================================================
# model/vision_neck.py
from typing import List, Dict

import torch


class Qwen35VisionFeatureExtractor:
    """
    在 Qwen3.5 ViT 的 merger 之前 hook 出多层特征。
    对 448x448 图像：每层输出 (784, 1152)，对应 28x28 空间网格。
    """

    def __init__(self, visual_model, hook_layers: List[int] = [6, 13, 20, 26]):
        self.visual_model = visual_model
        self.hook_layers = hook_layers
        self.features: Dict[int, torch.Tensor] = {}
        self._hooks = []
        self._register_hooks()

    def _register_hooks(self):
        for layer_idx in self.hook_layers:
            hook = self.visual_model.blocks[layer_idx].register_forward_hook(
                self._make_hook(layer_idx)
            )
            self._hooks.append(hook)

    def _make_hook(self, layer_idx: int):
        def hook_fn(module, input, output):
            # output shape: (seq_len, 1152)
            self.features[layer_idx] = output

        return hook_fn

    def get_features(self) -> Dict[int, torch.Tensor]:
        return self.features

    def clear(self):
        self.features = {}

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    @staticmethod
    def seq_to_spatial(
            feat: torch.Tensor,
            grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        """
        将序列特征 (seq_len, C) 还原为空间特征图 (B, C, H, W)。

        grid_thw: (num_images, 3)，每行是 [T, H_patches, W_patches]
        对图像输入通常 T=1（但 Conv3d temporal_patch_size=2，需注意）
        """
        spatial_features = []
        offset = 0

        for i in range(grid_thw.shape[0]):
            t, h, w = grid_thw[i]
            # 每个 image 的 token 数 = t * h * w
            # 由于 temporal_patch_size=2，单张图像 t=1
            num_tokens = (t * h * w).item()
            img_feat = feat[offset: offset + num_tokens]  # (H*W, C)

            # 若 t > 1（视频），先取平均；图像 t=1 直接跳过
            if t > 1:
                img_feat = img_feat.reshape(t, h * w, -1).mean(0)  # (H*W, C)

            img_feat = img_feat.reshape(h, w, -1)  # (H, W, C)
            img_feat = img_feat.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
            spatial_features.append(img_feat)
            offset += num_tokens

        # stack → (B, C, H, W)
        return torch.cat(spatial_features, dim=0)
