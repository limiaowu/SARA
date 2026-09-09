# =============================================================================
# model/loss.py
# 分割损失函数
# =============================================================================
import torch
import torch.nn.functional as F


def seg_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """
    复合分割损失：BCE（主）+ Dice + 0.5×Focal + 0.5×Boundary-weighted BCE

    BCE 作为主要梯度来源（平滑、稳定）；Dice 处理类别不平衡；
    Focal 补充难样本挖掘；Boundary-weighted BCE 给边界区域额外监督。
    去掉 Boundary IoU（初始值接近 1.0，早期收敛慢）。

    pred : (B, H, W) raw logits
    gt   : (B, H, W) float binary mask ∈ [0, 1]
    """
    # ── 逐像素 BCE（复用于 Focal 和 Boundary-weighted BCE）───────────────────
    bce_pp = F.binary_cross_entropy_with_logits(pred, gt, reduction="none")

    # ── BCE（主项）───────────────────────────────────────────────────────────
    bce = bce_pp.mean()

    # ── Dice ──────────────────────────────────────────────────────────────────
    p = torch.sigmoid(pred)
    num = 2.0 * (p * gt).sum(dim=(-2, -1))
    den = p.sum(dim=(-2, -1)) + gt.sum(dim=(-2, -1)) + 1e-6
    dice = (1.0 - num / den).mean()

    # ── Focal (γ=2, α=0.25) ──────────────────────────────────────────────────
    p_t = torch.exp(-bce_pp)
    alpha_t = 0.25 * gt + 0.75 * (1.0 - gt)
    focal = (alpha_t * (1.0 - p_t) ** 2.0 * bce_pp).mean()

    # ── Boundary-weighted BCE（边界区域权重 5×）──────────────────────────────
    kernel = torch.ones(1, 1, 5, 5, device=gt.device, dtype=gt.dtype) / 25.0
    blurred = F.conv2d(gt.unsqueeze(1), kernel, padding=2).squeeze(1)
    is_boundary = ((blurred > 1e-3) & (blurred < 1.0 - 1e-3)).float()
    boundary_bce = (bce_pp * (1.0 + 4.0 * is_boundary)).mean()

    return bce + dice + 0.5 * focal + 0.5 * boundary_bce
