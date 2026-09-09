# =============================================================================
# model/sara.py
# SARA: unified referring and reasoning segmentation.
# =============================================================================
import os
import re
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict, TaskType
from transformers import Qwen3_5ForConditionalGeneration

from sam_decoder.mask_decoder import MaskDecoder
from sam_decoder.prompt_encoder import PromptEncoder
from sam_decoder.transformer import TwoWayTransformer
from .adaptive_neck import FPNNeck
from .cnn_bypass import CNNBypass
from .response_aggregation import ResponseAggregation
from .loss import seg_loss
from .vision_neck import Qwen35VisionFeatureExtractor

SEG_TOKEN = "[SEG]"

_DEFAULT_LORA_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",  # full_attention 层（8/32）
    "in_proj_qkv", "out_proj",  # GatedDeltaNet 层（24/32）
]


def _is_main() -> bool:
    import torch.distributed as _dist
    if _dist.is_available() and _dist.is_initialized():
        return _dist.get_rank() == 0
    return int(os.environ.get("LOCAL_RANK", 0)) == 0


# ── SAM2 配置 ─────────────────────────────────────────────────────────────────

# num_multimask_outputs=3 保留 → mask_tokens / hypernetworks 仍是 4 个，
# 与 SAM2 预训练权重 shape 对齐（可正常加载）。我们只在 forward 里传
# multimask_output=False，始终取单 mask token（index 0）。
# dynamic_multimask_via_stability 必须关闭：开启时 eval（model.eval()）会改走
# “从 3 个多义 token 里按稳定性挑一个”，导致训练（取 token 0）/推理不一致。
SAM2_TINY_CONFIG = dict(
    num_multimask_outputs=3,
    transformer_dim=256,
    iou_head_depth=3,
    iou_head_hidden_dim=256,
    use_high_res_features=False,
    iou_prediction_use_sigmoid=True,
    dynamic_multimask_via_stability=False,
    pred_obj_scores=True,
    pred_obj_scores_mlp=True,
    use_multimask_token_for_obj_ptr=False,
)

SAM2_LARGE_CONFIG = dict(
    num_multimask_outputs=3,
    transformer_dim=256,
    iou_head_depth=3,
    iou_head_hidden_dim=256,
    use_high_res_features=False,  # 会被 checkpoint 自动覆盖为 True
    iou_prediction_use_sigmoid=True,
    dynamic_multimask_via_stability=False,
    pred_obj_scores=True,  # Large checkpoint 包含 obj_score 层
    pred_obj_scores_mlp=True,
    use_multimask_token_for_obj_ptr=False,
)


def _load_sam_state_dict(sam2_ckpt_path: str) -> Dict[str, torch.Tensor]:
    ckpt = torch.load(sam2_ckpt_path, map_location="cpu", weights_only=True)
    return ckpt.get("model", ckpt)


def build_mask_decoder(sam2_ckpt_path: str, model_size: str = "tiny") -> MaskDecoder:
    config = dict(SAM2_TINY_CONFIG if model_size == "tiny" else SAM2_LARGE_CONFIG)
    state_dict = _load_sam_state_dict(sam2_ckpt_path)
    config["use_high_res_features"] = any(
        k.startswith("sam_mask_decoder.conv_s0") for k in state_dict
    )
    transformer = TwoWayTransformer(depth=2, embedding_dim=256, mlp_dim=2048, num_heads=8)
    decoder = MaskDecoder(transformer=transformer, **config)
    decoder_weights = {
        k.replace("sam_mask_decoder.", ""): v
        for k, v in state_dict.items()
        if k.startswith("sam_mask_decoder.")
    }
    missing, unexpected = decoder.load_state_dict(decoder_weights, strict=False)
    if _is_main():
        print(f"[MaskDecoder] use_high_res_features={decoder.use_high_res_features}")
        if missing:    print(f"[MaskDecoder] Missing:    {missing}")
        if unexpected: print(f"[MaskDecoder] Unexpected: {unexpected}")
    return decoder


def build_prompt_encoder(sam2_ckpt_path: str) -> PromptEncoder:
    encoder = PromptEncoder(
        embed_dim=256,
        image_embedding_size=(64, 64),
        input_image_size=(1024, 1024),
        mask_in_chans=16,
    )
    state_dict = _load_sam_state_dict(sam2_ckpt_path)
    prompt_weights = {
        k.replace("sam_prompt_encoder.", ""): v
        for k, v in state_dict.items()
        if k.startswith("sam_prompt_encoder.")
    }
    missing, unexpected = encoder.load_state_dict(prompt_weights, strict=False)
    if _is_main():
        if missing:    print(f"[PromptEncoder] Missing:    {missing}")
        if unexpected: print(f"[PromptEncoder] Unexpected: {unexpected}")
    return encoder


# ── 主模型 ────────────────────────────────────────────────────────────────────

class SARAModel(nn.Module):
    """
    单塔指代图像分割模型。

    数据流：
        原图 → CNNBypass → feat_s0 (256×256), feat_s1 (128×128)
        图像 → Qwen3.5 ViT（hook 多层特征）→ FPNNeck → image_embedding (64×64)
        文本 → Qwen3.5 LLM → 生成序列 assistant token hidden states
                            → ResponseAggregation (N query × 2层 TransformerDecoder)
                            → (B, N, 256) sparse embeddings
                            → PromptEncoder(text_embeddings, boxes)
        image_embedding + sparse/dense prompts + high_res_features → MaskDecoder → mask
    """

    def __init__(
            self,
            qwen_model_path: str,
            sam2_ckpt_path: str,
            sam2_model_size: str = "tiny",
            hook_layers: Optional[List[int]] = None,
            lora_r: int = 16,
            lora_alpha: int = 32,
            lora_dropout: float = 0.05,
            lora_target_modules: Optional[List[str]] = None,
            seg_token_idx: Optional[int] = None,
            num_queries: int = 32,
            mask_loss_weight: float = 2.0,
            attn_implementation: str = "flash_attention_2",
            use_cnn_bypass: bool = True,
    ):
        super().__init__()
        self._keys_to_ignore_on_save = None
        self.mask_loss_weight = mask_loss_weight
        self.use_cnn_bypass = use_cnn_bypass

        if hook_layers is None:
            hook_layers = [3, 6, 13, 20, 26]

        # ── 1. Qwen3.5 backbone ──────────────────────────────────────────
        self.qwen = Qwen3_5ForConditionalGeneration.from_pretrained(
            qwen_model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation=attn_implementation,
        )

        # ── 2. ViT 全量冻结 ───────────────────────────────────────────────
        for param in self.qwen.model.visual.parameters():
            param.requires_grad = False
        if _is_main():
            print("Vision encoder fully frozen.")

        # ── 3. LoRA（覆盖全 32 层 LLM）+ TrainableTokens（[SEG] 行）────
        # 检测 embed_tokens 与 lm_head 是否权重共享（tie）：Qwen3.5-4B 及以下
        # tie_word_embeddings=True（共享同一份权重），9B 为 False（两份独立）。
        # 直接比对权重存储指针，比读 config.tie_word_embeddings 更可靠（多模态 config
        # 里该字段可能挂在 text_config 上）。顺便取 LLM 隐藏维度供下游 CQE 用，避免写死。
        _emb_w = self.qwen.get_input_embeddings().weight
        llm_hidden = _emb_w.shape[1]
        try:
            _out_w = self.qwen.get_output_embeddings().weight
            tie_emb = (_out_w.data_ptr() == _emb_w.data_ptr())
        except Exception:
            tie_emb = bool(getattr(self.qwen.config, "tie_word_embeddings", False))
        # ViT 隐藏维度（hook 取的是 merger 之前的 block 输出）：9B=1152，4B=1024，
        # 各尺寸不同，从 vision_config 自动读取，避免写死导致 FPNNeck 通道不匹配。
        _vis_cfg = getattr(self.qwen.config, "vision_config", None)
        vit_hidden = 1152
        if _vis_cfg is not None:
            vit_hidden = (getattr(_vis_cfg, "hidden_size", None)
                          or getattr(_vis_cfg, "embed_dim", None) or 1152)
        if _is_main():
            print(f"LLM hidden_size={llm_hidden}, tie_word_embeddings={tie_emb}, "
                  f"ViT hidden_size={vit_hidden}")

        # tie 时 PEFT 只能 target embed_tokens（改动会自动同步到共享的 lm_head）；
        # 若再显式 target lm_head，会因 lm_head 是 tied 的非常规层而触发
        # "TrainableTokensLayer wraps an unknown layer type"。非 tie（9B）时两处都 target。
        if seg_token_idx is None:
            _tt_indices = None
        elif tie_emb:
            _tt_indices = {"embed_tokens": [seg_token_idx]}
        else:
            _tt_indices = {"embed_tokens": [seg_token_idx], "lm_head": [seg_token_idx]}
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=lora_target_modules or _DEFAULT_LORA_MODULES,
            bias="none",
            trainable_token_indices=_tt_indices,
        )
        self.qwen = get_peft_model(self.qwen, lora_config)

        # ── 4. 记录 seg_token_idx ─────────────────────────────────────────
        self.seg_token_idx = seg_token_idx

        if _is_main():
            self.qwen.print_trainable_parameters()

        # ── 5. ViT 特征提取 hook ──────────────────────────────────────────
        self.feature_extractor = Qwen35VisionFeatureExtractor(
            visual_model=self.qwen.base_model.model.model.visual,
            hook_layers=hook_layers,
        )
        self.hook_layers = hook_layers

        # ── 6. FPN Neck ───────────────────────────────────────────────────
        self.neck = FPNNeck(
            in_channels=vit_hidden,  # 9B=1152，4B=1024，从 vision_config 自动取
            out_channels=256,
            hook_layers=hook_layers,
        )

        # ── 7. CNN Bypass（原图 → 真实高分辨率 skip features）────────────
        # 消融开关：use_cnn_bypass=False 时不建 CNN，decoder 高分辨率分支改吃
        # FPN 输出（image_embedding 64²）上采样得到的 feat_s0/feat_s1（见 _high_res_features）。
        if use_cnn_bypass:
            self.cnn_bypass = CNNBypass(out_channels=256)
        else:
            self.cnn_bypass = None
            if _is_main():
                print("CNN bypass DISABLED（消融）：高分辨率特征 = FPN 输出上采样到 256²/128²。")

        # ── 8. Response Aggregation ──────────────────────────────────────
        # 2 层 TransformerDecoder，从 LLM 生成序列的全部 assistant hidden states
        # 中提取语义，聚合到 num_queries 个可学习 query 上。
        self.response_aggregation = ResponseAggregation(
            llm_dim=llm_hidden,  # 9B=4096，4B 等小型号不同，从 embedding 自动取
            embed_dim=256,
            num_queries=num_queries,
            num_heads=4,
            num_layers=2,
        )

        # ── 9. SAM2 MaskDecoder + PromptEncoder ───────────────────────────
        if _is_main():
            print(f"Loading SAM2 MaskDecoder ({sam2_model_size}) from {sam2_ckpt_path}...")
        self.mask_decoder = build_mask_decoder(sam2_ckpt_path, sam2_model_size)
        self.prompt_encoder = build_prompt_encoder(sam2_ckpt_path)

        for param in self.prompt_encoder.parameters():
            param.requires_grad = True

        # ── 10. 统一 dtype ────────────────────────────────────────────────
        self.neck = self.neck.bfloat16()
        if self.cnn_bypass is not None:
            self.cnn_bypass = self.cnn_bypass.bfloat16()
        self.response_aggregation = self.response_aggregation.bfloat16()
        self.mask_decoder = self.mask_decoder.bfloat16()
        self.prompt_encoder = self.prompt_encoder.bfloat16()

    # ── Trainer 接口 ─────────────────────────────────────────────────────

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.qwen.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
        )

    def gradient_checkpointing_disable(self):
        self.qwen.gradient_checkpointing_disable()

    def get_main_device(self) -> torch.device:
        return next(self.neck.parameters()).device

    # ── 可训练参数统计 ───────────────────────────────────────────────────

    def print_trainable_parameters(self):
        """按模块分组统计可训练参数量，并打印汇总表（仅主进程）。"""
        if not _is_main():
            return

        def _count(module) -> tuple:
            """返回 (trainable, total)。"""
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            total = sum(p.numel() for p in module.parameters())
            return trainable, total

        # Qwen 内部：LoRA adapter 与 [SEG] token embed/lm_head 行都是可训练的，
        # 用名字区分以便单独列出 LoRA 与 trainable-token 两部分。
        lora_trainable, token_trainable = 0, 0
        for name, p in self.qwen.named_parameters():
            if not p.requires_grad:
                continue
            if "lora_" in name:
                lora_trainable += p.numel()
            else:
                token_trainable += p.numel()
        qwen_total = sum(p.numel() for p in self.qwen.parameters())

        groups = [
            ("Qwen LoRA",              lora_trainable,            qwen_total),
            ("Qwen trainable tokens",  token_trainable,           None),
            ("FPNNeck",                *_count(self.neck)),
            ("CNNBypass",              *(_count(self.cnn_bypass) if self.cnn_bypass is not None
                                         else (0, 0))),
            ("ResponseAggregation",    *_count(self.response_aggregation)),
            ("MaskDecoder",            *_count(self.mask_decoder)),
            ("PromptEncoder",          *_count(self.prompt_encoder)),
        ]

        total_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_all = sum(p.numel() for p in self.parameters())

        def _fmt(n):
            return f"{n / 1e6:>10.3f} M" if n is not None else f"{'-':>12}"

        print("\n" + "=" * 60)
        print("Trainable parameters by module")
        print("-" * 60)
        print(f"{'Module':<26}{'Trainable':>14}{'Total':>14}")
        print("-" * 60)
        for name, tr, tot in groups:
            print(f"{name:<26}{_fmt(tr):>14}{_fmt(tot):>14}")
        print("-" * 60)
        print(f"{'TOTAL':<26}{_fmt(total_trainable):>14}{_fmt(total_all):>14}")
        print(f"Trainable ratio: {100 * total_trainable / total_all:.4f}%")
        print("=" * 60 + "\n")

    # ── 高分辨率 skip features 来源 ───────────────────────────────────────

    def _high_res_features(
            self,
            image_embedding: torch.Tensor,
            cnn_images: Optional[torch.Tensor],
            device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """生成 SAM decoder 高分辨率分支的 feat_s0(256²)/feat_s1(128²)。

        - use_cnn_bypass=True ：CNNBypass 从原图提取真实像素级高分辨率特征。
        - use_cnn_bypass=False（消融）：不引入独立高分辨率来源，直接把 FPN 输出
          image_embedding(B,256,64²) 双线性上采样到 256²/128²。两条路径输出通道
          都是 256，下游 conv_s0/conv_s1 完全一致，可公平对照 CNN bypass 的增益。
        """
        if self.use_cnn_bypass:
            return self.cnn_bypass(cnn_images.to(device=device, dtype=torch.bfloat16))
        feat_s0 = F.interpolate(
            image_embedding, size=(256, 256), mode="bilinear", align_corners=False,
        )
        feat_s1 = F.interpolate(
            image_embedding, size=(128, 128), mode="bilinear", align_corners=False,
        )
        return feat_s0, feat_s1

    # ── Forward ──────────────────────────────────────────────────────────

    def forward(
            self,
            input_ids: torch.LongTensor,
            attention_mask: torch.Tensor,
            pixel_values: torch.Tensor,
            image_grid_thw: torch.Tensor,
            cnn_images: torch.Tensor,
            mm_token_type_ids: Optional[torch.Tensor] = None,
            labels: Optional[torch.LongTensor] = None,
            gt_masks: Optional[torch.Tensor] = None,
            gt_boxes: Optional[torch.Tensor] = None,
            segmentation_mask: Optional[torch.Tensor] = None,
            original_size: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, torch.Tensor]:
        device = self.get_main_device()
        self.feature_extractor.clear()

        # ── Step 1: Qwen3.5 前向 ─────────────────────────────────────────
        qwen_outputs = self.qwen(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
            labels=labels,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )

        # ── Step 2: 提取 assistant token hidden states 作为 context ─────
        last_hidden = qwen_outputs.hidden_states[-1]  # (B, seq_len, 4096)
        B_size = last_hidden.shape[0]

        if segmentation_mask is None:
            segmentation_mask = torch.ones(B_size, dtype=torch.bool, device=device)
        else:
            segmentation_mask = segmentation_mask.to(device=device, dtype=torch.bool)

        # Homogeneous language-only batches do not need any segmentation module.
        if not segmentation_mask.any():
            result = {}
            if labels is not None:
                result["lm_loss"] = qwen_outputs.loss
                result["loss"] = qwen_outputs.loss
            return result

        context_list, max_ctx_len = [], 0
        for b in range(B_size):
            if labels is not None:
                ctx = last_hidden[b][labels[b] != -100]  # (n_assistant, 4096)
            else:
                seg_pos = (input_ids[b] == self.seg_token_idx).nonzero(as_tuple=True)[0]
                idx = seg_pos[0].item() if len(seg_pos) > 0 else -1
                ctx = last_hidden[b, idx:idx + 1, :]
            if ctx.shape[0] == 0:
                ctx = last_hidden[b, -1:, :]
            context_list.append(ctx)
            max_ctx_len = max(max_ctx_len, ctx.shape[0])

        padded_ctx = torch.zeros(
            B_size, max_ctx_len, last_hidden.shape[-1],
            device=device, dtype=torch.bfloat16,
        )
        ctx_pad_mask = torch.ones(B_size, max_ctx_len, dtype=torch.bool, device=device)
        for b, ctx in enumerate(context_list):
            L = ctx.shape[0]
            padded_ctx[b, :L] = ctx.bfloat16()
            ctx_pad_mask[b, :L] = False

        text_emb = self.response_aggregation(padded_ctx, ctx_pad_mask)  # (B, M, 256)

        # ── Step 3: ViT 特征 → FPN Neck → image_embedding ────────────────
        vit_feats = {k: v.to(device) for k, v in self.feature_extractor.get_features().items()}
        image_embedding = self.neck(vit_feats, image_grid_thw.to(device)).bfloat16()

        # ── Step 4: 高分辨率 skip features（CNN bypass 或 FPN 上采样）──────
        # 实际特征在 Step 7 按需生成（仅 use_high_res_features 时）。

        # ── Step 5: Box prompt ───────────────────────────────────────────
        boxes_input = None
        if gt_boxes is not None:
            boxes_input = gt_boxes.to(device=device, dtype=torch.bfloat16)

        # ── Step 6: PromptEncoder ─────────────────────────────────────────
        image_pe = self.prompt_encoder.get_dense_pe().bfloat16()
        sparse_prompts, dense_prompts = self.prompt_encoder(
            points=None, boxes=boxes_input, masks=None,
            text_embeddings=text_emb,
        )
        sparse_prompts = sparse_prompts.to(device=device, dtype=torch.bfloat16)
        dense_prompts = dense_prompts.to(device=device, dtype=torch.bfloat16)

        # ── Step 7: 高分辨率 skip features ───────────────────────────────
        high_res_features = None
        if self.mask_decoder.use_high_res_features:
            feat_s0, feat_s1 = self._high_res_features(image_embedding, cnn_images, device)
            high_res_features = [
                self.mask_decoder.conv_s0(feat_s0),
                self.mask_decoder.conv_s1(feat_s1),
            ]

        # ── Step 8: MaskDecoder（multimask_output=False，单 mask token）───
        # 只取单 mask 输出：iou_pred 最高的多义 mask 实测未必分割最好，
        # 单 mask token（SAM 为无歧义场景训练的那一路）更稳定、也更省。
        low_res_masks, iou_preds, _, _ = self.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompts,
            dense_prompt_embeddings=dense_prompts,
            multimask_output=False,  # (B, 1, H_low, W_low), iou_preds (B, 1)
            repeat_image=False,
            high_res_features=high_res_features,
        )

        # ── Step 9: 取单 mask ────────────────────────────────────────────
        # 训练 loss 全部在 SAM decoder 原生低分辨率（low_res_masks，≈256²）上算，
        # Compute the loss at the decoder resolution to avoid unnecessary upsampling.
        Hlow, Wlow = low_res_masks.shape[-2:]
        mask_low = low_res_masks[:, 0]                          # (B, Hlow, Wlow)
        iou_pred = iou_preds[:, 0]                              # (B,)

        # 返回给上层的 pred_masks 上采到 original_size（供可视化/接口一致；
        # 训练 backward 不依赖它）。
        target_h, target_w = original_size if original_size is not None else (Hlow, Wlow)
        pred_masks = F.interpolate(
            mask_low.unsqueeze(1), size=(target_h, target_w),
            mode="bilinear", align_corners=False,
        ).squeeze(1)                                            # (B, H, W)

        result = {"pred_masks": pred_masks, "iou_predictions": iou_pred}

        # ── Step 10: Loss（全部在低分辨率上计算）──────────────────────────
        if labels is not None:
            result["lm_loss"] = qwen_outputs.loss

        if gt_masks is not None and segmentation_mask.any():
            # 统一用 bfloat16，与模型输出一致，避免 backward dtype 冲突
            gt = gt_masks.to(device=device, dtype=torch.bfloat16)[segmentation_mask]
            gt_low = F.interpolate(
                gt.unsqueeze(1), size=(Hlow, Wlow), mode="nearest",
            ).squeeze(1)                                        # (B, Hlow, Wlow)

            mask_low_for_loss = mask_low[segmentation_mask]
            iou_pred_for_loss = iou_pred[segmentation_mask]

            # 单 mask seg_loss
            mask_loss = seg_loss(mask_low_for_loss, gt_low)

            # IoU head 监督：MSE(predicted_iou, true_iou)，true IoU 在低分辨率上算
            with torch.no_grad():
                p_bin = (torch.sigmoid(mask_low_for_loss) > 0.5).to(dtype=torch.bfloat16)
                inter = (p_bin * gt_low).sum(dim=(-2, -1))
                union = (p_bin + gt_low - p_bin * gt_low).sum(dim=(-2, -1))
                true_ious = (inter / (union + 1e-6)).to(dtype=iou_pred_for_loss.dtype)
            iou_mse_loss = F.mse_loss(iou_pred_for_loss, true_ious)

            result["mask_loss"] = mask_loss + 0.2 * iou_mse_loss

        if "lm_loss" in result and "mask_loss" in result:
            result["loss"] = result["lm_loss"] + self.mask_loss_weight * result["mask_loss"]
        elif "lm_loss" in result:
            result["loss"] = result["lm_loss"]
        elif "mask_loss" in result:
            result["loss"] = result["mask_loss"]

        return result

    # ── 推理 ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def generate_with_mask(
            self,
            input_ids: torch.LongTensor,
            attention_mask: torch.Tensor,
            pixel_values: torch.Tensor,
            image_grid_thw: torch.Tensor,
            cnn_images: torch.Tensor,
            mm_token_type_ids: Optional[torch.Tensor] = None,
            original_size: tuple = (512, 512),
            max_new_tokens: int = 256,
            tokenizer=None,
            oracle_boxes: Optional[torch.Tensor] = None,
            decode_masks: bool = True,
    ) -> dict:
        """
        自回归推理，收集所有生成 token 的 hidden states 作为 context。
        input_ids 必须使用左填充（left padding）。
        """
        device = self.get_main_device()
        B = input_ids.shape[0]

        self.feature_extractor.clear()

        _eos_ids = []
        _pad_id = None
        if tokenizer is not None:
            _pad_id = tokenizer.pad_token_id
            if tokenizer.eos_token_id is not None:
                _eos_ids.append(tokenizer.eos_token_id)
            _im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
            if _im_end_id not in (_eos_ids + [tokenizer.unk_token_id]):
                _eos_ids.append(_im_end_id)

        generate_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            max_new_tokens=max_new_tokens,
            output_hidden_states=True,
            return_dict_in_generate=True,
            use_cache=True,
        )
        if _eos_ids:
            generate_kwargs["eos_token_id"] = _eos_ids
        if _pad_id is not None:
            generate_kwargs["pad_token_id"] = _pad_id
        if mm_token_type_ids is not None:
            generate_kwargs["mm_token_type_ids"] = mm_token_type_ids

        outputs = self.qwen.generate(**generate_kwargs)

        prompt_len = input_ids.shape[1]
        generated_ids = outputs.sequences[:, prompt_len:]
        num_steps = len(outputs.hidden_states)

        # 解析生成文本中的 bbox（0-1000 → SAM 1024 空间）
        _BOX_RE = re.compile(
            r'"bbox_2d"\s*:\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]'
        )
        pred_boxes = []
        generated_texts = None
        if tokenizer is not None:
            generated_texts = [
                tokenizer.decode(generated_ids[b], skip_special_tokens=True)
                for b in range(B)
            ]
            for text in generated_texts:
                m = _BOX_RE.search(text)
                if m:
                    x1, y1, x2, y2 = [int(v) / 1000 * 1024 for v in m.groups()]
                    pred_boxes.append(torch.tensor(
                        [x1, y1, x2, y2], device=device, dtype=torch.bfloat16
                    ))
                else:
                    pred_boxes.append(None)
        else:
            pred_boxes = [None] * B

        pred_boxes_tensor = torch.full(
            (B, 4), float("nan"), device=device, dtype=torch.bfloat16,
        )
        box_founds = torch.tensor(
            [box is not None for box in pred_boxes], device=device, dtype=torch.bool,
        )
        for index, box in enumerate(pred_boxes):
            if box is not None:
                pred_boxes_tensor[index] = box

        boxes_tensor = None
        if any(b is not None for b in pred_boxes):
            boxes_tensor = torch.stack([
                b if b is not None
                else torch.tensor([0., 0., 1024., 1024.], device=device, dtype=torch.bfloat16)
                for b in pred_boxes
            ])

        seg_founds = [
            len((generated_ids[b] == self.seg_token_idx).nonzero(as_tuple=True)[0]) > 0
            for b in range(B)
        ]

        # Language-only inference avoids all segmentation-side compute unless an
        # oracle-box diagnostic was explicitly requested.
        if not decode_masks or (not any(seg_founds) and oracle_boxes is None):
            target_h, target_w = original_size
            empty_masks = torch.zeros(
                B, target_h, target_w, device=device, dtype=torch.bfloat16,
            )
            empty_ious = torch.zeros(B, device=device, dtype=torch.bfloat16)
            return {
                "generated_ids": generated_ids,
                "generated_text": generated_texts,
                "pred_masks": empty_masks,
                "iou_predictions": empty_ious,
                "all_masks": empty_masks.unsqueeze(1),
                "all_iou_predictions": empty_ious.unsqueeze(1),
                "seg_found": seg_founds,
                "pred_boxes": pred_boxes_tensor,
                "box_found": box_founds,
            }

        # 收集生成 token 的 hidden states（跳过 step 0，那是 prompt prefill）
        # step 0: shape (B, prompt_len, D)，取的是最后一个 prompt token，不是生成 token
        # step 1+: shape (B, 1, D)，每个生成 token 的 hidden state
        gen_steps = list(range(1, num_steps)) or [0]  # 极端情况兜底
        all_step_hiddens = [
            outputs.hidden_states[s][-1][:, -1, :].detach().clone()  # (B, 4096)
            for s in gen_steps
        ]
        context_hidden = torch.stack(all_step_hiddens, dim=1)  # (B, gen_len, 4096)

        # 构造 padding mask：EOS 之后的步骤置 True（屏蔽），避免垃圾 hidden state 污染 context
        # step s 对应 generated_ids[:, s-1]
        ctx_pad_mask = torch.zeros(B, len(gen_steps), dtype=torch.bool, device=device)
        if B > 1 and len(_eos_ids) > 0:
            for b in range(B):
                is_eos = torch.zeros(generated_ids.shape[1], dtype=torch.bool, device=device)
                for eid in _eos_ids:
                    is_eos |= (generated_ids[b] == eid)
                eos_pos = is_eos.nonzero(as_tuple=True)[0]
                if len(eos_pos) > 0:
                    first_eos = eos_pos[0].item()
                    # first_eos 对应 step first_eos+1，EOS token 本身保留，之后全部 mask
                    for i, s in enumerate(gen_steps):
                        if s - 1 > first_eos:
                            ctx_pad_mask[b, i:] = True
                            break

        # FPN（高分辨率 skip features 在下方按需生成）
        vit_feats = {k: v.to(device) for k, v in self.feature_extractor.get_features().items()}
        image_embedding = self.neck(vit_feats, image_grid_thw.to(device)).bfloat16()

        # Response Aggregation
        text_emb = self.response_aggregation(
            context_hidden.to(device).bfloat16(), ctx_pad_mask,
        )

        high_res_features = None
        if self.mask_decoder.use_high_res_features:
            feat_s0, feat_s1 = self._high_res_features(image_embedding, cnn_images, device)
            high_res_features = [
                self.mask_decoder.conv_s0(feat_s0),
                self.mask_decoder.conv_s1(feat_s1),
            ]

        image_pe = self.prompt_encoder.get_dense_pe().bfloat16()

        def decode_with_boxes(prompt_boxes):
            sparse_prompts, dense_prompts = self.prompt_encoder(
                points=None,
                boxes=prompt_boxes,
                masks=None,
                text_embeddings=text_emb,
            )
            low_res_masks, iou_preds, _, _ = self.mask_decoder(
                image_embeddings=image_embedding,
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse_prompts.to(
                    device=device, dtype=torch.bfloat16
                ),
                dense_prompt_embeddings=dense_prompts.to(
                    device=device, dtype=torch.bfloat16
                ),
                multimask_output=False,
                repeat_image=False,
                high_res_features=high_res_features,
            )
            mask_up = F.interpolate(
                low_res_masks, size=original_size,
                mode="bilinear", align_corners=False,
            )
            return mask_up, iou_preds

        if any(seg_founds):
            mask_up, iou_preds = decode_with_boxes(boxes_tensor)
            pred_masks = mask_up[:, 0]
            iou_pred = iou_preds[:, 0]
        else:
            target_h, target_w = original_size
            pred_masks = torch.zeros(
                B, target_h, target_w, device=device, dtype=torch.bfloat16,
            )
            iou_pred = torch.zeros(B, device=device, dtype=torch.bfloat16)
            mask_up = pred_masks.unsqueeze(1)
            iou_preds = iou_pred.unsqueeze(1)

        oracle_mask_up = None
        oracle_iou_preds = None
        if oracle_boxes is not None:
            oracle_boxes = oracle_boxes.to(device=device, dtype=torch.bfloat16)
            oracle_mask_up, oracle_iou_preds = decode_with_boxes(oracle_boxes)

        return {
            "generated_ids": generated_ids,
            "generated_text": generated_texts,
            "pred_masks": pred_masks,           # (B, H, W)
            "iou_predictions": iou_pred,        # (B,)
            "all_masks": mask_up,               # (B, 1, H, W) 兼容旧接口（单张）
            "all_iou_predictions": iou_preds,   # (B, 1)
            "seg_found": seg_founds,
            "pred_boxes": pred_boxes_tensor,
            "box_found": box_founds,
            "oracle_pred_masks": (
                oracle_mask_up[:, 0] if oracle_mask_up is not None else None
            ),
            "oracle_iou_predictions": (
                oracle_iou_preds[:, 0] if oracle_iou_preds is not None else None
            ),
        }

    # ── Checkpoint ───────────────────────────────────────────────────────

    def load_trainable(self, checkpoint) -> None:
        """Load a SARA trainable checkpoint."""
        if isinstance(checkpoint, (str, os.PathLike)):
            checkpoint = torch.load(checkpoint, map_location="cpu", weights_only=False)

        checkpoint_seg_idx = checkpoint.get("seg_token_idx")
        if checkpoint_seg_idx is not None and checkpoint_seg_idx != self.seg_token_idx:
            raise ValueError(
                f"[SEG] token mismatch: model={self.seg_token_idx}, "
                f"checkpoint={checkpoint_seg_idx}"
            )

        set_peft_model_state_dict(self.qwen, checkpoint["lora"])
        self.neck.load_state_dict(checkpoint["neck"])
        if self.cnn_bypass is not None:
            self.cnn_bypass.load_state_dict(checkpoint["cnn_bypass"])
        self.response_aggregation.load_state_dict(checkpoint["response_aggregation"])
        self.mask_decoder.load_state_dict(checkpoint["mask_decoder"])
        self.prompt_encoder.load_state_dict(checkpoint["prompt_encoder"])

        if _is_main():
            print("Loaded SARA initialization checkpoint.")

    def save_trainable(self, path: str):
        """保存所有可训练参数到单个 .pt 文件。"""
        from peft import get_peft_model_state_dict
        os.makedirs(os.path.dirname(path), exist_ok=True)

        state = {
            "lora": get_peft_model_state_dict(self.qwen),
            "neck": self.neck.state_dict(),
            "response_aggregation": self.response_aggregation.state_dict(),
            "mask_decoder": self.mask_decoder.state_dict(),
            "prompt_encoder": self.prompt_encoder.state_dict(),
            "seg_token_idx": self.seg_token_idx,
            "hook_layers": self.hook_layers,
            "use_cnn_bypass": self.use_cnn_bypass,
        }
        if self.cnn_bypass is not None:
            state["cnn_bypass"] = self.cnn_bypass.state_dict()

        torch.save(state, path)
        if _is_main():
            print(f"Saved trainable weights → {path}")
