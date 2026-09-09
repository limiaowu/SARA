# =============================================================================
# train.py
# =============================================================================
import argparse
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from transformers import AutoProcessor, TrainingArguments, Trainer, TrainerCallback
from transformers.trainer_utils import get_last_checkpoint

from data.dataset import collate_fn
from data.builder import build_from_specs
from model.sara import SARAModel, SEG_TOKEN
from utils.metrics_logger import MetricsLogger


# ─────────────────────────────────────────────────────────────────────────────

def _is_main() -> bool:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return int(os.environ.get("LOCAL_RANK", 0)) == 0


# ─────────────────────────────────────────────────────────────────────────────

# IoU 评估
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_val_iou(
        model,
        val_dataset,
        mask_size: int = 256,
        eval_batch_size: int = 4,
        pad_token_id: int = 0,
        tokenizer=None,
) -> tuple:
    """
    在【整个】验证集上同时计算两个指标：
      cIoU (cumulative IoU) = 累计交 / 累计并，受大物体主导
      gIoU (global/mean IoU)  = 每张图单独算 IoU 后取均值，与目标大小无关
    返回 (cIoU, gIoU)。各卡推理不重叠子集，通过 all_reduce 汇总。
    """
    try:
        from tqdm import tqdm
        use_tqdm = True
    except ImportError:
        use_tqdm = False

    model.eval()
    unwrapped = model.module if hasattr(model, "module") else model
    device = unwrapped.get_main_device()

    all_indices = list(range(len(val_dataset)))
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        indices = all_indices[rank::world_size]
    else:
        indices = all_indices

    subset = Subset(val_dataset, indices)
    loader = DataLoader(
        subset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=16,
        collate_fn=lambda b: collate_fn(b, pad_token_id=pad_token_id),
        drop_last=False,
    )

    total_intersection = torch.tensor(0.0, device=device)
    total_union = torch.tensor(0.0, device=device)
    total_per_sample_iou = torch.tensor(0.0, device=device)
    total_samples_t = torch.tensor(0, device=device)
    seg_found_count = 0
    total_samples = 0

    iter_loader = (
        tqdm(loader, desc="[Val]", dynamic_ncols=True, disable=not _is_main())
        if use_tqdm else loader
    )

    for batch in iter_loader:
        mm_ttids = batch.get("mm_token_type_ids")
        outputs = unwrapped.generate_with_mask(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            pixel_values=batch["pixel_values"].to(device),
            image_grid_thw=batch["image_grid_thw"].to(device),
            cnn_images=batch["cnn_images"].to(device),
            mm_token_type_ids=mm_ttids.to(device) if mm_ttids is not None else None,
            original_size=(mask_size, mask_size),
            tokenizer=tokenizer,
        )

        seg_found_count += sum(outputs["seg_found"])
        batch_n = len(outputs["seg_found"])
        total_samples += batch_n
        total_samples_t += batch_n

        pred = (torch.sigmoid(outputs["pred_masks"].float()) > 0.5).float()
        gt = batch["gt_masks"].to(device).float()
        if gt.shape[-2:] != pred.shape[-2:]:
            gt = F.interpolate(gt.unsqueeze(1), size=pred.shape[-2:],
                               mode="nearest").squeeze(1)

        # cIoU: 累计分子/分母
        total_intersection += (pred * gt).sum()
        total_union += ((pred + gt) > 0).float().sum()

        # gIoU: 每张图单独算 IoU 后累加
        for b in range(pred.shape[0]):
            inter_b = (pred[b] * gt[b]).sum()
            union_b = ((pred[b] + gt[b]) > 0).float().sum()
            total_per_sample_iou += inter_b / (union_b + 1e-6)

        if use_tqdm and _is_main():
            cur_ciou = (total_intersection / (total_union + 1e-6)).item()
            cur_giou = (total_per_sample_iou / max(total_samples, 1)).item()
            iter_loader.set_postfix(
                seg=f"{seg_found_count}/{total_samples}",
                cIoU=f"{cur_ciou:.4f}",
                gIoU=f"{cur_giou:.4f}",
            )

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(total_intersection, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_union, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_per_sample_iou, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_samples_t, op=dist.ReduceOp.SUM)

    n_total = total_samples_t.item()
    ciou = (total_intersection / (total_union + 1e-6)).item()
    giou = (total_per_sample_iou / max(n_total, 1)).item()
    if _is_main():
        print(f"  [Eval] seg_found: {seg_found_count}/{total_samples}, "
              f"cIoU={ciou:.4f}, gIoU={giou:.4f}")
    model.train()
    return ciou, giou


# ─────────────────────────────────────────────────────────────────────────────
# Epoch-end 评估回调
# ─────────────────────────────────────────────────────────────────────────────

class _EpochEvalCallback(TrainerCallback):
    """在每个 epoch 结束时触发一次 cIoU 评估并保存最优权重。"""

    def __init__(self, trainer: "SARATrainer"):
        self._trainer = trainer

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        if model is None or self._trainer._val_dataset is None:
            return
        self._trainer._run_eval_and_save(
            model=model,
            step=state.global_step,
            tag=f"epoch{int(state.epoch)}",
        )


# ─────────────────────────────────────────────────────────────────────────────
# 自定义 Trainer
# ─────────────────────────────────────────────────────────────────────────────

class SARATrainer(Trainer):

    def __init__(
            self,
            *args,
            best_ckpt_path: str = None,
            val_dataset=None,
            mask_size: int = 256,
            eval_steps: int = 500,
            eval_batch_size: int = 4,
            pad_token_id: int = 0,
            tokenizer=None,
            best_metric: str = "ciou",
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if best_metric not in {"ciou", "giou"}:
            raise ValueError("best_metric must be 'ciou' or 'giou'")
        self._best_metric = best_metric
        self._best_score = -1.0
        self._eval_steps = eval_steps
        self.best_ckpt_path = best_ckpt_path or os.path.join(
            os.path.dirname(self.args.output_dir), "best_model", "sara_weights.pt"
        )
        self._val_dataset = val_dataset
        self._mask_size = mask_size
        self._eval_batch_size = eval_batch_size
        self._pad_token_id = pad_token_id
        self._tokenizer = tokenizer

        self._ema_mask_loss: float = None
        self._ema_alpha = 0.05
        self._last_eval_step = -1

        # CSV 指标记录器（仅 rank-0 写入）
        self._metrics_logger = (
            MetricsLogger(self.args.output_dir) if _is_main() else None
        )

        # 每 epoch 结束时评估一次（保证步数极少时也能触发）
        self.add_callback(_EpochEvalCallback(self))

    def _run_eval_and_save(self, model, step: int, tag: str = "") -> None:
        """运行一次评估（cIoU + gIoU），若 cIoU 刷新最优则保存权重。"""
        if self._val_dataset is None:
            return
        if step == self._last_eval_step:
            return
        self._last_eval_step = step

        ciou, giou = evaluate_val_iou(
            model,
            self._val_dataset,
            mask_size=self._mask_size,
            eval_batch_size=self._eval_batch_size,
            pad_token_id=self._pad_token_id,
            tokenizer=self._tokenizer,
        )
        suffix = f"_{tag}" if tag else "_subset"
        self.log({
            f"val_ciou{suffix}": round(ciou, 4),
            f"val_giou{suffix}": round(giou, 4),
        })

        score = ciou if self._best_metric == "ciou" else giou
        if score > self._best_score and _is_main():
            self._best_score = score
            unwrapped = model.module if hasattr(model, "module") else model
            unwrapped.save_trainable(self.best_ckpt_path)
            label = f" [{tag}]" if tag else ""
            print(f"[Step {step}{label}] New best {self._best_metric}={score:.4f} "
                  f"(cIoU={ciou:.4f}, gIoU={giou:.4f}), "
                  f"saved → {self.best_ckpt_path}")

    def log(self, logs, start_time=None):
        # 先走 HF Trainer 原有逻辑（写 tensorboard / state.log_history）
        if start_time is not None:
            super().log(logs, start_time)
        else:
            super().log(logs)
        # 再写 CSV
        if self._metrics_logger is not None:
            self._metrics_logger.record(self.state.global_step, logs)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pixel_values=inputs["pixel_values"],
            image_grid_thw=inputs["image_grid_thw"],
            cnn_images=inputs["cnn_images"],
            mm_token_type_ids=inputs.get("mm_token_type_ids"),
            labels=inputs["labels"],
            gt_masks=inputs["gt_masks"],
            gt_boxes=inputs.get("gt_boxes"),
            original_size=(self._mask_size, self._mask_size),
        )
        loss = outputs["loss"]

        step = self.state.global_step
        if step % self.args.logging_steps == 0:
            mask_loss_val = outputs.get("mask_loss", torch.tensor(0.)).item()
            lm_loss_val = outputs.get("lm_loss", torch.tensor(0.)).item()

            if "mask_loss" in outputs:
                if self._ema_mask_loss is None:
                    self._ema_mask_loss = mask_loss_val
                else:
                    self._ema_mask_loss = (
                        self._ema_alpha * mask_loss_val
                        + (1 - self._ema_alpha) * self._ema_mask_loss
                    )

            self.log({
                "lm_loss": round(lm_loss_val, 4),
                "mask_loss": round(mask_loss_val, 4),
                "mask_loss_ema": round(self._ema_mask_loss or 0.0, 4),
            })

        # 每 eval_steps 步做一次步级验证（eval_steps=0 时仅依赖 epoch-end 回调）
        if (step > 0
                and self._eval_steps > 0
                and step % self._eval_steps == 0
                and self._val_dataset is not None):
            self._run_eval_and_save(model, step)

        return (loss, outputs) if return_outputs else loss


# ─────────────────────────────────────────────────────────────────────────────
# 参数
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--local_rank", type=int, default=-1)

    # 路径
    parser.add_argument("--qwen_model_path", type=str, required=True)
    parser.add_argument("--sam2_ckpt_path", type=str, required=True)
    parser.add_argument("--sam2_model_size", type=str, default="large",
                        choices=["tiny", "large"])
    parser.add_argument("--refcoco_root", type=str, default=None)
    parser.add_argument("--coco_image_root", type=str, default=None)
    parser.add_argument("--reasoning_seg_root", type=str, default=None,
                        help="ReasoningSeg 根目录（含 train/val 子目录）")
    parser.add_argument("--base_output_dir", type=str,
                        default="./outputs")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--init_weights", type=str, default=None,
                        help="Existing SARA checkpoint used for initialization")

    # 数据：以 "name:split[:repeat]" 形式的 spec 列表声明训练 / 验证集
    #   训练例：refcoco:train refcoco+:train refcocog:train reasoning_seg:train:100
    #   验证例：refcoco:val
    parser.add_argument("--train_specs", nargs="+",
                        default=["refcoco:train", "refcoco+:train", "refcocog:train"],
                        help="训练集 spec 列表，格式 name:split[:repeat]，repeat 为过采样倍数")
    parser.add_argument("--val_specs", nargs="+",
                        default=["refcoco:val"],
                        help="验证集 spec 列表，格式 name:split（repeat 忽略）")
    parser.add_argument("--image_size", type=int, default=896)
    parser.add_argument("--mask_size", type=int, default=512)

    # 模型结构
    parser.add_argument("--hook_layers", type=int, nargs="+", default=[3, 6, 13, 20, 26])
    parser.add_argument("--num_queries", type=int, default=16,
                        help="ContextQueryExtractor 的可学习 query 数量")
    parser.add_argument("--no_cnn_bypass", action="store_true",
                        help="消融：关闭 CNN bypass，decoder 高分辨率特征改用 FPN 输出上采样")

    # LoRA
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", nargs="+", default=None,
                        help="LoRA 作用的模块名列表，默认覆盖全 32 层 LLM")

    # 损失
    parser.add_argument("--mask_loss_weight", type=float, default=2.0,
                        help="total_loss = lm_loss + mask_loss_weight × mask_loss")

    # 训练
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=-1,
                        help="Positive values override num_epochs; useful for smoke tests")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_steps", type=float, default=200)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--best_metric", choices=("ciou", "giou"), default="ciou",
                        help="Validation metric used to select sara_weights.pt")
    parser.add_argument("--skip_mask_eval", action="store_true",
                        help="Disable generation-based mask validation")
    parser.add_argument("--no_gradient_checkpointing", action="store_true",
                        help="关闭 gradient checkpointing（GatedDeltaNet 与 grad_ckpt 有兼容性问题时使用）")

    # DeepSpeed
    parser.add_argument("--deepspeed", type=str, default=None)
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2",
                        choices=["flash_attention_2", "sdpa", "eager"])

    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    args.gradient_checkpointing = not args.no_gradient_checkpointing
    from datetime import datetime
    run_name = args.run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.base_output_dir, run_name)
    best_ckpt_path = os.path.join(output_dir, "best_model", "sara_weights.pt")
    os.makedirs(output_dir, exist_ok=True)

    if _is_main():
        print(f"Experiment : {run_name}")
        print(f"Output dir : {output_dir}")

    # ── Processor ────────────────────────────────────────────────────────
    processor = AutoProcessor.from_pretrained(args.qwen_model_path, max_pixels=1024 * 1024)
    num_added = processor.tokenizer.add_tokens([SEG_TOKEN])
    seg_token_idx = processor.tokenizer.convert_tokens_to_ids(SEG_TOKEN)
    pad_token_id = processor.tokenizer.pad_token_id or 0

    if _is_main():
        print(f"Added {num_added} token(s). [SEG] idx = {seg_token_idx}, "
              f"pad_token_id = {pad_token_id}")
        processor.save_pretrained(os.path.join(output_dir, "processor"))
        print(f"Processor saved → {output_dir}/processor")

    init_checkpoint = None
    if args.init_weights:
        init_checkpoint = torch.load(args.init_weights, map_location="cpu", weights_only=False)
        if "hook_layers" in init_checkpoint:
            args.hook_layers = list(init_checkpoint["hook_layers"])
        query_state = init_checkpoint.get("context_query_extractor", {})
        if "queries" in query_state:
            args.num_queries = int(query_state["queries"].shape[0])
        if "use_cnn_bypass" in init_checkpoint:
            args.no_cnn_bypass = not bool(init_checkpoint["use_cnn_bypass"])
        if _is_main():
            print(
                f"Initialization checkpoint: {args.init_weights} "
                f"(hooks={args.hook_layers}, queries={args.num_queries})"
            )

    # ── 模型 ─────────────────────────────────────────────────────────────
    model = SARAModel(
        qwen_model_path=args.qwen_model_path,
        sam2_ckpt_path=args.sam2_ckpt_path,
        sam2_model_size=args.sam2_model_size,
        hook_layers=args.hook_layers,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=args.lora_target_modules,
        seg_token_idx=seg_token_idx,
        num_queries=args.num_queries,
        mask_loss_weight=args.mask_loss_weight,
        attn_implementation=args.attn_implementation,
        use_cnn_bypass=not args.no_cnn_bypass,
    )
    if init_checkpoint is not None:
        model.load_trainable(init_checkpoint)

    # ── 可训练参数统计 ────────────────────────────────────────────────────
    model.print_trainable_parameters()

    # ── 数据集 ────────────────────────────────────────────────────────────
    # train_specs / val_specs 统一走 data.builder：
    #   训练集按各 spec 的 repeat 倍数过采样后混合（如 reasoning_seg:train:100）；
    #   验证集 for_generation=True（评测走 generate_with_mask）。
    _paths = dict(
        refcoco_root=args.refcoco_root,
        coco_image_root=args.coco_image_root,
        reasoning_seg_root=args.reasoning_seg_root,
    )
    train_dataset = build_from_specs(
        args.train_specs,
        processor=processor,
        for_generation=False,
        image_size=args.image_size,
        mask_size=args.mask_size,
        seg_token_idx=seg_token_idx,
        **_paths,
    )
    val_dataset = None
    if not args.skip_mask_eval:
        val_dataset = build_from_specs(
            args.val_specs,
            processor=processor,
            for_generation=True,
            image_size=args.image_size,
            mask_size=args.mask_size,
            seg_token_idx=seg_token_idx,
            **_paths,
        )

    # ── TrainingArguments ─────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.num_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        max_grad_norm=args.max_grad_norm,
        bf16=True,
        tf32=True,
        dataloader_num_workers=args.num_workers,
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        eval_strategy="no",
        load_best_model_at_end=False,
        report_to="tensorboard",
        remove_unused_columns=False,
        deepspeed=args.deepspeed,
        gradient_checkpointing=args.gradient_checkpointing,
        ddp_find_unused_parameters=False,
    )

    # ── Trainer ───────────────────────────────────────────────────────────
    trainer = SARATrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=lambda b: collate_fn(b, pad_token_id=pad_token_id),
        best_ckpt_path=best_ckpt_path,
        val_dataset=val_dataset,
        eval_steps=args.eval_steps,
        mask_size=args.mask_size,
        eval_batch_size=args.eval_batch_size,
        pad_token_id=pad_token_id,
        tokenizer=processor.tokenizer,
        best_metric=args.best_metric,
    )

    last_ckpt = get_last_checkpoint(output_dir) if os.path.isdir(output_dir) else None
    if last_ckpt and _is_main():
        print(f"Resuming from checkpoint: {last_ckpt}")

    trainer.train(resume_from_checkpoint=last_ckpt)
    trainer.save_model(output_dir)

    if _is_main():
        print("Training complete.")


if __name__ == "__main__":
    main()
