# =============================================================================
# model/qseg.py
# QSeg: Single-Tower Reasoning Segmentation via Qwen3.5 + SAM2 MaskDecoder
#
# 架构变更记录：
#   v4: 引入 QueryExtractor，将单 [SEG] sparse embedding 替换为 N 个可学习 query。
#       N 个 query 通过 cross-attention 从 [SEG] hidden state 提取多角度语义，
#       作为 N 个 sparse embeddings 进入 MaskDecoder 的 TwoWayTransformer，
#       让图像特征从多个语义角度被激活，改善定位精度。
#   v3: 移除 ViT 解冻逻辑（num_unfreeze_vit_layers），ViT 重新全量冻结。
#       移除 adaptive_neck 的浅层高分辨率分支，引入 CNNBypass 替代。
#       CNNBypass 从原图（1024×1024）以 stride=2/4/8 提取真实高频特征，
#       完全绕过 ViT 的 1/16 下采样瓶颈。
#       PromptEncoder 新增 text_embeddings 参数（LISA 风格文本分支），
#       [SEG] hidden state 经 QueryExtractor 投影后直接进入 PromptEncoder，
#       与稠密 prompt 一同送入 MaskDecoder，接口更统一。
#   v2: mm_token_type_ids 透传修复，左填充，Boundary-weighted 损失。
# =============================================================================
import os
import re
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model, TaskType
from transformers import Qwen3_5ForConditionalGeneration

from sam_decoder.mask_decoder import MaskDecoder
from sam_decoder.prompt_encoder import PromptEncoder
from sam_decoder.transformer import TwoWayTransformer
from .adaptive_neck import FPNNeck
from .cnn_bypass import CNNBypass
from .context_query import ContextQueryExtractor
from .vision_neck import Qwen35VisionFeatureExtractor

SEG_TOKEN = "[SEG]"


# QueryExtractor 已替换为 ContextQueryExtractor（见 model/context_query.py）


def _is_main() -> bool:
    import torch.distributed as _dist
    if _dist.is_available() and _dist.is_initialized():
        return _dist.get_rank() == 0
    return int(os.environ.get("LOCAL_RANK", 0)) == 0


# ── SAM2 配置 ─────────────────────────────────────────────────────────────────

SAM2_TINY_CONFIG = dict(
    num_multimask_outputs=3,
    transformer_dim=256,
    iou_head_depth=3,
    iou_head_hidden_dim=256,
    use_high_res_features=False,
    iou_prediction_use_sigmoid=True,
    dynamic_multimask_via_stability=True,
    dynamic_multimask_stability_delta=0.05,
    dynamic_multimask_stability_thresh=0.98,
    pred_obj_scores=True,
    pred_obj_scores_mlp=True,
    use_multimask_token_for_obj_ptr=False,
)

SAM2_LARGE_CONFIG = dict(
    num_multimask_outputs=3,
    transformer_dim=256,
    iou_head_depth=3,
    iou_head_hidden_dim=256,
    use_high_res_features=False,
    iou_prediction_use_sigmoid=True,
    dynamic_multimask_via_stability=True,
    dynamic_multimask_stability_delta=0.05,
    dynamic_multimask_stability_thresh=0.98,
    pred_obj_scores=False,
    pred_obj_scores_mlp=False,
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

class QSegModel(nn.Module):
    """
    单塔推理分割模型（v3）。

    数据流：
        原图 → CNNBypass → feat_s0 (256×256), feat_s1 (128×128)  [真实高分辨率]
        图像 → Qwen3.5 ViT（hook 多层特征）→ FPNNeck → image_embedding (64×64)
        文本 → Qwen3.5 LLM → [SEG] token hidden state (4096-d)
                            → QueryExtractor（N 可学习 query × cross-attention）
                            → (B, N, 256) sparse embeddings
                            → PromptEncoder.text_embeddings → sparse prompts
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
            seg_token_idx: Optional[int] = None,
            num_queries: int = 4,
    ):
        super().__init__()
        self._keys_to_ignore_on_save = None

        if hook_layers is None:
            hook_layers = [3, 6, 13, 20, 26]

        # ── 1. Qwen3.5 backbone ──────────────────────────────────────────
        self.qwen = Qwen3_5ForConditionalGeneration.from_pretrained(
            qwen_model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )

        # ── 2. ViT 全量冻结 ───────────────────────────────────────────────
        for param in self.qwen.model.visual.parameters():
            param.requires_grad = False
        if _is_main():
            print("Vision encoder fully frozen.")

        # ── 3. resize embedding（Qwen3.5 预留了词表空间，通常无需 resize）──
        # seg_token_idx 由 add_tokens() 分配，落在预留范围内，embed 不会越界。

        # ── 4. LoRA + TrainableTokens（只训练 [SEG] 所在行）────────────
        # 注：lm_head.weight 与 embed_tokens.weight 在 Qwen3.5 中是 tied 的（同一 tensor）。
        # 两者同时写入 trainable_token_indices 以明确语义；PEFT 内部会正确处理。
        # 之前 lr=0 的根因是 Sobel sqrt 反传 NaN（已修复），与本机制无关。
        _tt_indices = (
            {"embed_tokens": [seg_token_idx], "lm_head": [seg_token_idx]}
            if seg_token_idx is not None else None
        )
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            # full_attention 层：q/k/v/o_proj（8/32 层）
            # GatedDeltaNet 层：in_proj_qkv/out_proj（24/32 层）
            # 两类合计覆盖全部 32 层 LLM
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                             "in_proj_qkv", "out_proj"],
            bias="none",
            trainable_token_indices=_tt_indices,
        )
        self.qwen = get_peft_model(self.qwen, lora_config)

        # ── 5. 记录 seg_token_idx ────────────────────────────────────────
        self.seg_token_idx = seg_token_idx

        if _is_main():
            self.qwen.print_trainable_parameters()

        # ── 6. ViT 特征提取 hook ─────────────────────────────────────────
        self.feature_extractor = Qwen35VisionFeatureExtractor(
            visual_model=self.qwen.base_model.model.model.visual,
            hook_layers=hook_layers,
        )
        self.hook_layers = hook_layers

        # ── 7. FPN Neck（仅输出 image_embedding，无浅层分支）────────────
        self.neck = FPNNeck(
            in_channels=1152,
            out_channels=256,
            hook_layers=hook_layers,
        )

        # ── 8. CNN Bypass（原图 → 真实高分辨率 skip features）───────────
        self.cnn_bypass = CNNBypass(out_channels=256)

        # ── 9. Context Query Extractor（LENS 风格）───────────────────────
        # 从 LLM 生成序列的全部 hidden states 中提取语义，
        # 通过 2 层 TransformerDecoder 聚合到 num_queries 个可学习 query 上。
        # 相比原 QueryExtractor：KV 从单 [SEG] token 扩展为完整生成序列，
        # queries 4→16，单层 cross-attn → 2 层带 self-attn+FFN 的 decoder block。
        self.context_query_extractor = ContextQueryExtractor(
            llm_dim=4096,
            embed_dim=256,
            num_queries=num_queries,
            num_heads=4,
            num_layers=2,
        )

        # ── 10. SAM2 MaskDecoder + PromptEncoder ─────────────────────────
        if _is_main():
            print(f"Loading SAM2 MaskDecoder from {sam2_ckpt_path}...")
        self.mask_decoder = build_mask_decoder(sam2_ckpt_path, sam2_model_size)
        self.prompt_encoder = build_prompt_encoder(sam2_ckpt_path)

        for param in self.prompt_encoder.parameters():
            param.requires_grad = True

        # ── 11. 统一 dtype ───────────────────────────────────────────────
        self.neck = self.neck.bfloat16()
        self.cnn_bypass = self.cnn_bypass.bfloat16()
        self.context_query_extractor = self.context_query_extractor.bfloat16()
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
            original_size: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        参数：
            cnn_images : (B, 3, 1024, 1024)，ImageNet 归一化，bfloat16 或 float32
            gt_boxes   : (B, 4)，[x1,y1,x2,y2] in SAM 1024×1024 space，训练时使用
        """
        device = self.get_main_device()
        self.feature_extractor.clear()

        # ── Step 1: Qwen3.5 前向，触发 ViT hooks ─────────────────────────
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

        # ── Step 2: 提取 assistant token hidden states 作为 context ────────
        # labels != -100 标记 assistant 回复的所有 token（bbox + 语义句 + [SEG]）
        last_hidden = qwen_outputs.hidden_states[-1]  # (B, seq_len, 4096)
        B_size = last_hidden.shape[0]

        context_list, max_ctx_len = [], 0
        for b in range(B_size):
            if labels is not None:
                ctx = last_hidden[b][labels[b] != -100]     # (n_assistant_tokens, 4096)
            else:
                # fallback：无 labels 时退化为单 [SEG] hidden state
                seg_pos = (input_ids[b] == self.seg_token_idx).nonzero(as_tuple=True)[0]
                idx = seg_pos[0].item() if len(seg_pos) > 0 else -1
                ctx = last_hidden[b, idx:idx + 1, :]        # (1, 4096)
            if ctx.shape[0] == 0:
                ctx = last_hidden[b, -1:, :]                # 兜底：最后一个 token
            context_list.append(ctx)
            max_ctx_len = max(max_ctx_len, ctx.shape[0])

        # 填充到批次内最长 context，生成 key_padding_mask
        padded_ctx = torch.zeros(
            B_size, max_ctx_len, last_hidden.shape[-1],
            device=device, dtype=torch.bfloat16,
        )
        ctx_pad_mask = torch.ones(B_size, max_ctx_len, dtype=torch.bool, device=device)
        for b, ctx in enumerate(context_list):
            L = ctx.shape[0]
            padded_ctx[b, :L] = ctx.bfloat16()
            ctx_pad_mask[b, :L] = False   # False = 有效位置（参与 attention）

        # ContextQueryExtractor → (B, num_queries, 256) sparse embeddings
        text_emb = self.context_query_extractor(padded_ctx, ctx_pad_mask)  # (B, M, 256)

        # ── Step 3: ViT 特征 → FPN Neck → image_embedding ───────────────
        vit_feats = {k: v.to(device) for k, v in self.feature_extractor.get_features().items()}
        image_embedding = self.neck(vit_feats, image_grid_thw.to(device)).bfloat16()

        # ── Step 4: CNN bypass → 真实高分辨率 skip features ─────────────
        feat_s0, feat_s1 = self.cnn_bypass(
            cnn_images.to(device=device, dtype=torch.bfloat16)
        )

        # ── Step 5: PromptEncoder（文本分支 + box prompt）────────────────
        # gt_boxes: (B, 4) in SAM 1024 space；训练时用 GT box 作为空间先验
        boxes_input = None
        if gt_boxes is not None:
            boxes_input = gt_boxes.to(device=device, dtype=torch.bfloat16)
        image_pe = self.prompt_encoder.get_dense_pe().bfloat16()
        sparse_prompts, dense_prompts = self.prompt_encoder(
            points=None, boxes=boxes_input, masks=None,
            text_embeddings=text_emb,
        )
        sparse_prompts = sparse_prompts.to(device=device, dtype=torch.bfloat16)
        dense_prompts = dense_prompts.to(device=device, dtype=torch.bfloat16)

        # ── Step 6: 高分辨率 skip features ──────────────────────────────
        high_res_features = None
        if self.mask_decoder.use_high_res_features:
            high_res_features = [
                self.mask_decoder.conv_s0(feat_s0),
                self.mask_decoder.conv_s1(feat_s1),
            ]

        # ── Step 7: MaskDecoder ──────────────────────────────────────────
        low_res_masks, iou_preds, _, _ = self.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompts,
            dense_prompt_embeddings=dense_prompts,
            multimask_output=False,
            repeat_image=False,
            high_res_features=high_res_features,
        )

        # ── Step 8: 上采样到目标尺寸 ────────────────────────────────────
        target_h, target_w = original_size if original_size is not None else (256, 256)
        pred_masks = F.interpolate(
            low_res_masks,
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)  # (B, H, W)

        result = {"pred_masks": pred_masks, "iou_predictions": iou_preds}

        # ── Step 9: Loss ─────────────────────────────────────────────────
        if labels is not None:
            result["lm_loss"] = qwen_outputs.loss

        if gt_masks is not None:
            gt = gt_masks.to(device).float()
            if gt.shape[-2:] != pred_masks.shape[-2:]:
                gt = F.interpolate(gt.unsqueeze(1), size=pred_masks.shape[-2:],
                                   mode="nearest").squeeze(1)
            result["mask_loss"] = self._mask_loss(pred_masks, gt)

        if "lm_loss" in result and "mask_loss" in result:
            result["loss"] = result["lm_loss"] + 2.0 * result["mask_loss"]
        elif "lm_loss" in result:
            result["loss"] = result["lm_loss"]
        elif "mask_loss" in result:
            result["loss"] = result["mask_loss"]

        return result

    # ── 损失函数 ─────────────────────────────────────────────────────────

    @staticmethod
    def _sobel(x: torch.Tensor) -> torch.Tensor:
        """提取 sigmoid mask 的 Sobel 边缘幅值图，用于边缘形状监督。"""
        kx = torch.tensor(
            [[1., 0., -1.], [2., 0., -2.], [1., 0., -1.]],
            dtype=x.dtype, device=x.device,
        ).view(1, 1, 3, 3)
        ky = kx.transpose(-2, -1).contiguous()
        x_in = x.unsqueeze(1)  # (B, 1, H, W)
        gx = F.conv2d(x_in, kx, padding=1)
        gy = F.conv2d(x_in, ky, padding=1)
        return (gx ** 2 + gy ** 2 + 1e-6).sqrt().squeeze(1)  # (B, H, W)；+1e-6 防止反传时除以零

    @staticmethod
    def _mask_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """
        Focal + Dice + Boundary-weighted BCE + Sobel 边缘损失。
        权重：1 : 1 : 1 : 0.5
        """
        # Focal (γ=2, α=0.25)
        bce_per_pixel = F.binary_cross_entropy_with_logits(pred, gt, reduction="none")
        p_t = torch.exp(-bce_per_pixel)
        alpha_t = 0.25 * gt + 0.75 * (1.0 - gt)
        focal_loss = (alpha_t * (1.0 - p_t) ** 2.0 * bce_per_pixel).mean()

        # Dice
        p = torch.sigmoid(pred)
        num = 2.0 * (p * gt).sum(dim=(-2, -1))
        den = p.sum(dim=(-2, -1)) + gt.sum(dim=(-2, -1)) + 1e-6
        dice_loss = 1.0 - (num / den).mean()

        # Boundary-weighted BCE（5×5 均值模糊 GT，边界区域权重 5×）
        kernel = torch.ones(1, 1, 5, 5, device=gt.device, dtype=gt.dtype) / 25.0
        blurred = F.conv2d(gt.unsqueeze(1), kernel, padding=2).squeeze(1)
        is_boundary = ((blurred > 1e-3) & (blurred < 1.0 - 1e-3)).float()
        boundary_weight = 1.0 + 4.0 * is_boundary
        boundary_loss = (bce_per_pixel * boundary_weight).mean()

        # Sobel 边缘损失：监督预测边缘形状与 GT 边缘形状的几何吻合
        pred_edges = QSegModel._sobel(p)  # 保留梯度
        gt_edges = QSegModel._sobel(gt)  # gt 本无梯度
        edge_loss = F.l1_loss(pred_edges, gt_edges)

        return focal_loss + dice_loss + boundary_loss + 0.5 * edge_loss

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
            max_new_tokens: int = 128,
            tokenizer=None,
    ) -> dict:
        """
        自回归推理：让模型自由生成，找到生成序列中的 [SEG] token 取其 hidden state。
        若未生成 [SEG]，fallback 到最后一步 hidden state。

        参数：
            cnn_images : (B, 3, 1024, 1024)，ImageNet 归一化
            input_ids  : 必须使用左填充（left padding）
        """
        device = self.get_main_device()
        B = input_ids.shape[0]

        self.feature_extractor.clear()

        # 生成（第一步触发完整 ViT forward，hooks 在此填充）
        # 从 tokenizer 动态获取停止符，避免硬编码（Qwen3.5 词表已扩充至 ~24 万）
        _eos_ids = []
        _pad_id = None
        if tokenizer is not None:
            # pad token
            _pad_id = tokenizer.pad_token_id
            # eos token（通常是 <|endoftext|>）
            if tokenizer.eos_token_id is not None:
                _eos_ids.append(tokenizer.eos_token_id)
            # <|im_end|>：Qwen chat 模板的轮次终止符，必须加入以防止跨轮生成
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
        generated_ids = outputs.sequences[:, prompt_len:]  # (B, gen_len)
        num_steps = len(outputs.hidden_states)

        # ── 提前解码文本，解析 bbox（用于 box prompt）────────────────────────
        # 格式：{"bbox_2d": [x1, y1, x2, y2]}，0-1000 相对坐标 → SAM 1024 空间
        _BOX_RE = re.compile(
            r'"bbox_2d"\s*:\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]'
        )
        pred_boxes = []
        for b in range(B):
            box_sam = None
            if tokenizer is not None:
                gen_text = tokenizer.decode(generated_ids[b], skip_special_tokens=True)
                m = _BOX_RE.search(gen_text)
                if m:
                    x1, y1, x2, y2 = [int(v) / 1000 * 1024 for v in m.groups()]
                    box_sam = torch.tensor(
                        [x1, y1, x2, y2], device=device, dtype=torch.bfloat16
                    )
            pred_boxes.append(box_sam)

        # 若任意样本解析到 box，构建 (B, 4) boxes tensor；否则不传 box prompt
        if any(b is not None for b in pred_boxes):
            boxes_tensor = torch.stack([
                b if b is not None
                else torch.tensor([0., 0., 1024., 1024.], device=device, dtype=torch.bfloat16)
                for b in pred_boxes
            ])  # (B, 4)
        else:
            boxes_tensor = None

        # 逐样本找 [SEG]（用于 seg_found 标志，不再用于提取单点 hidden）
        seg_founds = []
        for b in range(B):
            pos = (generated_ids[b] == self.seg_token_idx).nonzero(as_tuple=True)[0]
            seg_founds.append(len(pos) > 0)

        # ── 收集所有生成步的 last-layer hidden state ─────────────────────────
        # outputs.hidden_states: tuple[num_steps] of tuple[num_layers] of (B, 1, hidden)
        # 所有 batch item 的步数相同（batched generation 时等长，pad 后统一）
        all_step_hiddens = [
            outputs.hidden_states[s][-1][:, -1, :]   # (B, 4096)
            for s in range(num_steps)
        ]
        context_hidden = torch.stack(all_step_hiddens, dim=1)  # (B, num_steps, 4096)
        # 推理时无填充，全部位置均有效
        ctx_pad_mask = torch.zeros(B, num_steps, dtype=torch.bool, device=device)

        # FPN Neck
        vit_feats = {k: v.to(device) for k, v in self.feature_extractor.get_features().items()}
        image_embedding = self.neck(vit_feats, image_grid_thw.to(device)).bfloat16()

        # CNN bypass
        feat_s0, feat_s1 = self.cnn_bypass(
            cnn_images.to(device=device, dtype=torch.bfloat16)
        )

        # ContextQueryExtractor → (B, num_queries, 256) sparse embeddings
        text_emb = self.context_query_extractor(
            context_hidden.to(device).bfloat16(), ctx_pad_mask,
        )  # (B, M, 256)

        image_pe = self.prompt_encoder.get_dense_pe().bfloat16()
        sparse_prompts, dense_prompts = self.prompt_encoder(
            points=None,
            boxes=boxes_tensor,  # (B, 4) 已解析的预测 box，或 None
            masks=None,
            text_embeddings=text_emb,
        )
        sparse_prompts = sparse_prompts.to(device=device, dtype=torch.bfloat16)
        dense_prompts = dense_prompts.to(device=device, dtype=torch.bfloat16)

        high_res_features = None
        if self.mask_decoder.use_high_res_features:
            high_res_features = [
                self.mask_decoder.conv_s0(feat_s0),
                self.mask_decoder.conv_s1(feat_s1),
            ]

        low_res_masks, iou_preds, _, _ = self.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompts,
            dense_prompt_embeddings=dense_prompts,
            multimask_output=False,
            repeat_image=False,
            high_res_features=high_res_features,
        )

        target_h, target_w = original_size
        pred_masks = F.interpolate(
            low_res_masks, size=(target_h, target_w),
            mode="bilinear", align_corners=False,
        ).squeeze(1)  # (B, H, W)

        # generated_texts 已在 box parsing 阶段解码；tokenizer 为 None 时保持 None
        generated_texts = None
        if tokenizer is not None:
            generated_texts = [
                tokenizer.decode(generated_ids[b], skip_special_tokens=True)
                for b in range(B)
            ]

        return {
            "generated_ids": generated_ids,
            "generated_text": generated_texts,
            "pred_masks": pred_masks,
            "iou_predictions": iou_preds,
            "seg_found": seg_founds,
        }

    # ── Checkpoint ───────────────────────────────────────────────────────

    def save_trainable(self, path: str):
        """
        保存所有可训练参数：
            lora          : LoRA 权重（q/k/v/o_proj）
                            + TrainableTokens delta（[SEG] 行的 embed/lm_head 增量）
                            两者统一由 get_peft_model_state_dict 打包
            neck                    : FPN Neck
            cnn_bypass              : CNN bypass（高分辨率 skip 分支）
            context_query_extractor : ContextQueryExtractor（LENS 风格，N 个 query + 2 层 decoder）
            mask_decoder            : SAM2 MaskDecoder
            prompt_encoder          : SAM2 PromptEncoder（含文本分支）
            seg_token_idx           : [SEG] token 的 ID（整数）
        """
        from peft import get_peft_model_state_dict
        os.makedirs(os.path.dirname(path), exist_ok=True)

        state = {
            # TrainableTokens delta 自动包含在 lora key 里，无需单独保存
            "lora": get_peft_model_state_dict(self.qwen),
            "neck": self.neck.state_dict(),
            "cnn_bypass": self.cnn_bypass.state_dict(),
            "context_query_extractor": self.context_query_extractor.state_dict(),
            "mask_decoder": self.mask_decoder.state_dict(),
            "prompt_encoder": self.prompt_encoder.state_dict(),
            "seg_token_idx": self.seg_token_idx,
        }

        torch.save(state, path)
        if _is_main():
            print(f"Saved trainable weights → {path}")
