# =============================================================================
# model/cnn_bypass.py
# 轻量 CNN bypass，提供真实高分辨率图像特征，绕过 ViT 的 1/16 下采样限制。
#
# 设计动机：
#   Qwen3.5 ViT patch_size=16，最高分辨率特征图仅 H/16×W/16（896px→56×56）。
#   经 FPN 上采样到 64×64 后语义足够，但高频边缘信息已丢失。
#   CNN 直接从原图（1024×1024）提取像素级特征，stride=2/4/8 保留真实空间细节。
#
# 输入  : (B, 3, H, W)，bfloat16，ImageNet 归一化
#         推荐 H=W=1024；其他尺寸通过 AdaptiveAvgPool 自适应对齐。
# 输出  :
#   feat_s0 : (B, out_channels, 256, 256)  → mask_decoder.conv_s0
#   feat_s1 : (B, out_channels, 128, 128)  → mask_decoder.conv_s1
#
# 以 1024×1024 输入为例的分辨率链：
#   stem(s=2) → 512×512
#   layer1(s=2) → 256×256  ← feat_s0
#   layer2(s=2) → 128×128  ← feat_s1
#
# 参数量约 3.2M（out_channels=256），全 GroupNorm，bfloat16 兼容。
# =============================================================================
import torch
import torch.nn as nn


def _cb(in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int = 1) -> nn.Sequential:
    """Conv + GroupNorm(out_ch//8) + GELU，bfloat16 兼容基础块。"""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, k, stride=s, padding=p, bias=False),
        nn.GroupNorm(out_ch // 8, out_ch),
        nn.GELU(),
    )


class ResBlock(nn.Module):
    """残差块，两层 Conv-GN-GELU + 跳接。"""

    def __init__(self, ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.GroupNorm(ch // 8, ch),
            nn.GELU(),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.GroupNorm(ch // 8, ch),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.net(x) + x)


class CNNBypass(nn.Module):
    """
    轻量 CNN bypass：从原图提取真实高分辨率特征，直接对接 SAM2 MaskDecoder 高分辨率分支。

    输入  : (B, 3, H, W)，bfloat16，ImageNet 均值/标准差归一化
    输出  :
        feat_s0 : (B, out_channels, 256, 256)  → mask_decoder.conv_s0 输入
        feat_s1 : (B, out_channels, 128, 128)  → mask_decoder.conv_s1 输入
    """

    # ImageNet 归一化参数（在 dataset.py / infer.py 中使用）
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    INPUT_SIZE = 1024  # 推荐输入分辨率

    def __init__(self, out_channels: int = 256):
        super().__init__()
        assert out_channels % 16 == 0, "out_channels 需能被 16 整除"

        # Stem: stride=2, 32ch
        self.stem = nn.Sequential(
            _cb(3, 32, k=7, s=2, p=3),
            ResBlock(32),
        )
        # Layer1: stride=2 → 1/4 (256×256 at 1024px)，64ch → feat_s0 来源
        self.layer1 = nn.Sequential(
            _cb(32, 64, s=2),
            ResBlock(64),
        )
        # Layer2: stride=2 → 1/8 (128×128 at 1024px)，128ch → feat_s1 来源
        self.layer2 = nn.Sequential(
            _cb(64, 128, s=2),
            ResBlock(128),
        )

        # 投影到 out_channels，对接 SAM2 的 conv_s0 / conv_s1
        self.proj_s0 = nn.Sequential(
            nn.Conv2d(64, out_channels, 1, bias=False),
            nn.GroupNorm(16, out_channels),
            nn.GELU(),
        )
        self.proj_s1 = nn.Sequential(
            nn.Conv2d(128, out_channels, 1, bias=False),
            nn.GroupNorm(16, out_channels),
            nn.GELU(),
        )

        # 自适应对齐到 SAM2 期望的尺寸（以应对非 1024px 输入）
        self.pool_s0 = nn.AdaptiveAvgPool2d((256, 256))
        self.pool_s1 = nn.AdaptiveAvgPool2d((128, 128))

    def forward(self, x: torch.Tensor):
        """
        x: (B, 3, H, W)，bfloat16，ImageNet 归一化
        """
        x = self.stem(x)  # (B, 32,  H/2, W/2)
        s0 = self.layer1(x)  # (B, 64,  H/4, W/4)
        s1 = self.layer2(s0)  # (B, 128, H/8, W/8)
        feat_s0 = self.pool_s0(self.proj_s0(s0))  # (B, out_ch, 256, 256)
        feat_s1 = self.pool_s1(self.proj_s1(s1))  # (B, out_ch, 128, 128)
        return feat_s0, feat_s1
