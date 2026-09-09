# =============================================================================
# model/adaptive_neck.py
# FPN Neck：Qwen3.5 ViT 多层特征 → SAM2 image_embedding (B, 256, 64, 64)。
#
# 变更记录：
#   v3: 移除浅层高分辨率 skip 分支（HighResHead）。
#       高分辨率 skip features 现由 CNNBypass 从原图直接提取，语义更准、边缘更锐利。
#       本模块专注输出 image_embedding，所有 hook_layers 均参与 FPN（不再区分深/浅）。
# =============================================================================
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBnGelu(nn.Sequential):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, k, s, p, bias=False),
            nn.GroupNorm(16, out_ch),
            nn.GELU(),
        )


class AdaptiveUpsample(nn.Module):
    """
    可学习的自适应上采样，将 ViT 的 28×28 特征图上采样至 64×64。
    两级 bilinear + Conv（替代 ConvTranspose2d，消除棋盘格伪影）+ AdaptiveAvgPool 精确对齐。
    """

    def __init__(self, channels: int = 256, target_size: int = 64):
        super().__init__()
        self.target_size = target_size

        def _up_block(c):
            # bilinear 上采样 2× 后接 3×3 Conv 精炼，无棋盘格伪影
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(c, c, 3, padding=1, bias=False),
                nn.GroupNorm(16, c), nn.GELU(),
            )

        self.up1 = _up_block(channels)
        self.up2 = _up_block(channels)
        self.final_align = nn.AdaptiveAvgPool2d((target_size, target_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[-2:]
        if H < self.target_size or W < self.target_size:
            x = self.up1(x)
        if x.shape[-2] < self.target_size or x.shape[-1] < self.target_size:
            x = self.up2(x)
        if x.shape[-2] != self.target_size or x.shape[-1] != self.target_size:
            x = self.final_align(x)
        return x


class FPNNeck(nn.Module):
    """
    FPN Neck：将 Qwen3.5 ViT 多层 hook 特征融合为 SAM2 image_embedding。

    所有 hook_layers 均参与 top-down FPN 融合，输出最浅层的 refined 特征
    （保留最多空间细节 + top-down 语义注入），经 AdaptiveUpsample 对齐至 64×64。

    高分辨率 skip features 已迁移至 CNNBypass，本模块不再涉及浅层分支。

    输出：image_embedding (B, 256, 64, 64)
    """

    TARGET_SIZE = 64

    def __init__(
            self,
            in_channels: int = 1152,
            out_channels: int = 256,
            hook_layers: Optional[List[int]] = None,
    ):
        super().__init__()

        if hook_layers is None:
            hook_layers = [3, 6, 13, 20, 26]
        self.hook_layers = hook_layers

        # 1×1 侧向投影：ViT 隐层 → FPN 统一通道
        self.laterals = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.GroupNorm(16, out_channels), nn.GELU(),
            )
            for _ in hook_layers
        ])

        # 3×3 精炼卷积
        self.refines = nn.ModuleList([
            ConvBnGelu(out_channels, out_channels) for _ in hook_layers
        ])

        self.adaptive_upsample = AdaptiveUpsample(out_channels, self.TARGET_SIZE)
        self.output_proj = nn.Conv2d(out_channels, out_channels, 1)

    # ── 工具方法 ─────────────────────────────────────────────────────────

    @staticmethod
    def _extract_single(feat_seq: torch.Tensor, offset: int, t: int, h: int, w: int) -> torch.Tensor:
        """(total_seq, C)[offset:] → (1, C, H, W)"""
        feat = feat_seq[offset: offset + t * h * w]
        if t > 1:
            feat = feat.reshape(t, h * w, -1).mean(0)
        return feat.reshape(h, w, -1).permute(2, 0, 1).unsqueeze(0)

    @staticmethod
    def _compute_offset(grid_thw: torch.Tensor, b: int) -> int:
        return sum(
            int(grid_thw[i, 0]) * int(grid_thw[i, 1]) * int(grid_thw[i, 2])
            for i in range(b)
        )

    def seq_to_spatial(self, feat: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        """供外部测试脚本调用：(total_seq, C) → (B, C, H, W)"""
        parts, offset = [], 0
        for i in range(grid_thw.shape[0]):
            t, h, w = [int(x) for x in grid_thw[i].tolist()]
            parts.append(self._extract_single(feat, offset, t, h, w))
            offset += t * h * w
        return torch.cat(parts, dim=0)

    def _process_single(
            self,
            multi_scale_feats: Dict[int, torch.Tensor],
            grid_thw: torch.Tensor,
            b: int,
    ) -> torch.Tensor:
        t, h, w = [int(x) for x in grid_thw[b].tolist()]
        offset = self._compute_offset(grid_thw, b)

        # 侧向投影
        projs = [
            self.laterals[i](self._extract_single(multi_scale_feats[li], offset, t, h, w))
            for i, li in enumerate(self.hook_layers)
        ]

        # top-down FPN 融合
        fpn = [projs[-1]]
        for i in range(len(projs) - 2, -1, -1):
            up = F.interpolate(fpn[-1], size=projs[i].shape[-2:],
                               mode='bilinear', align_corners=False)
            fpn.append(projs[i] + up)
        fpn.reverse()  # fpn[0] = 最浅层（最高分辨率 + top-down 语义）

        refined = [self.refines[i](f) for i, f in enumerate(fpn)]

        # 输出最浅层 refined：保留最多空间细节
        return self.output_proj(
            self.adaptive_upsample(refined[0])
        )  # (1, 256, 64, 64)

    def forward(
            self,
            multi_scale_feats: Dict[int, torch.Tensor],
            grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns:
            image_embedding : (B, 256, 64, 64)
        """
        embs = [self._process_single(multi_scale_feats, grid_thw, b)
                for b in range(grid_thw.shape[0])]
        return torch.cat(embs, dim=0)
