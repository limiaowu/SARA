# =============================================================================
# train.py
#
# 改动记录：
#   - [BUG FIX] mm_token_type_ids 透传到 model.forward() 和 generate_with_mask()
#   - 去除可视化代码（evaluate_subset_iou 不再生成图片）
#   - 所有 print 统一通过 _is_main() 守卫，避免 4 卡重复输出
#   - save_trainable 改为先 unwrap DDP/DeepSpeed 再调用
#   - eval DataLoader 使用左填充 collate_fn（与训练共用）
# =============================================================================
import argparse
import os
import random

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from transformers import AutoProcessor, TrainingArguments, Trainer
from transformers.trainer_utils import get_last_checkpoint

from data.dataset import RefCOCODataset, collate_fn
from model.qseg import QSegModel, SEG_TOKEN


# ─────────────────────────────────────────────────────────────────────────────
# 工具：是否是主进程
# ─────────────────────────────────────────────────────────────────────────────

def _is_main() -> bool:
    """返回 True 当且仅当当前进程是 rank-0（或非分布式模式）。"""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    # torchrun / deepspeed 启动时会设置 LOCAL_RANK；dist 未初始化时用它判断
    return int(os.environ.get("LOCAL_RANK", 0)) == 0


# ─────────────────────────────────────────────────────────────────────────────
# IoU 评估
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_subset_iou(
        model,
        val_dataset,
        n: int = 4000,
        mask_size: int = 256,
        eval_batch_size: int = 4,
        pad_token_id: int = 0,
        tokenizer=None,
) -> float:
    """
    在验证子集上计算 mIoU。各卡各自推理一个不重叠子集，通过 all_reduce 汇总。

    参数：
        eval_batch_size : 每卡 batch 大小，使用左填充，> 1 可显著提速。
        pad_token_id    : collate_fn 的 pad id，透传自 processor.tokenizer。
        tokenizer       : 用于动态获取 eos_token_id / pad_token_id，避免硬编码。
    """
    try:
        from tqdm import tqdm
        use_tqdm = True
    except ImportError:
        use_tqdm = False

    model.eval()

    # unwrap DDP / DeepSpeed
    unwrapped = model.module if hasattr(model, "module") else model
    device = unwrapped.get_main_device()

    # 分布式采样：n 为总样本数，各卡均分不重叠子集
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        rng = random.Random(42)
        all_indices = rng.sample(range(len(val_dataset)), min(n, len(val_dataset)))
        indices = all_indices[rank::world_size]
    else:
        indices = random.sample(range(len(val_dataset)), min(n, len(val_dataset)))

    subset = Subset(val_dataset, indices)
    loader = DataLoader(
        subset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=8,
        collate_fn=lambda b: collate_fn(b, pad_token_id=pad_token_id),
        drop_last=False,
    )

    total_intersection = torch.tensor(0.0, device=device)
    total_union = torch.tensor(0.0, device=device)
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
        total_samples += len(outputs["seg_found"])

        pred = (torch.sigmoid(outputs["pred_masks"].float()) > 0.5).float()
        gt = batch["gt_masks"].to(device).float()
        if gt.shape[-2:] != pred.shape[-2:]:
            gt = F.interpolate(gt.unsqueeze(1), size=pred.shape[-2:],
                               mode="nearest").squeeze(1)
        total_intersection += (pred * gt).sum()
        total_union += ((pred + gt) > 0).float().sum()

        if use_tqdm and _is_main():
            cur_miou = (total_intersection / (total_union + 1e-6)).item()
            iter_loader.set_postfix(
                seg=f"{seg_found_count}/{total_samples}",
                miou=f"{cur_miou:.4f}",
            )

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(total_intersection, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_union, op=dist.ReduceOp.SUM)

    miou = (total_intersection / (total_union + 1e-6)).item()
    if _is_main():
        print(f"  [Eval] seg_found: {seg_found_count}/{total_samples}, mIoU={miou:.4f}")
    model.train()
    return miou


# ─────────────────────────────────────────────────────────────────────────────
# 自定义 Trainer
# ─────────────────────────────────────────────────────────────────────────────

class QSegTrainer(Trainer):

    def __init__(
            self,
            *args,
            best_ckpt_path: str = None,
            val_dataset=None,
            eval_subset_n: int = 500,
            mask_size: int = 256,
            eval_steps: int = 500,
            eval_batch_size: int = 4,
            pad_token_id: int = 0,
            tokenizer=None,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.best_miou = -1.0
        self._eval_steps = eval_steps
        self.best_ckpt_path = best_ckpt_path or os.path.join(
            os.path.dirname(self.args.output_dir), "best_model", "qseg_weights.pt"
        )
        self._val_dataset = val_dataset
        self._eval_subset_n = eval_subset_n
        self._mask_size = mask_size
        self._eval_batch_size = eval_batch_size
        self._pad_token_id = pad_token_id
        self._tokenizer = tokenizer

        # EMA 平滑 mask_loss，避免早期随机低值影响日志
        self._ema_mask_loss: float = None
        self._ema_alpha = 0.05

        # 防止 grad_accum > 1 时同一 step 触发两次验证
        self._last_eval_step = -1

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
            original_size=(args.mask_size, args.mask_size),
        )
        loss = outputs["loss"]

        step = self.state.global_step
        if step % self.args.logging_steps == 0:
            mask_loss_val = outputs.get("mask_loss", torch.tensor(0.)).item()
            lm_loss_val = outputs.get("lm_loss", torch.tensor(0.)).item()

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
                "mask_loss_ema": round(self._ema_mask_loss, 4),
            })

        # 每 eval_steps 步做一次 val 子集 IoU 评估
        # _last_eval_step 防止 grad_accum > 1 时同一 step 被 compute_loss 调用两次
        if (step > 0
                and self._eval_steps > 0
                and step % self._eval_steps == 0
                and step != self._last_eval_step
                and self._val_dataset is not None):
            self._last_eval_step = step
            miou = evaluate_subset_iou(
                model,
                self._val_dataset,
                n=self._eval_subset_n,
                mask_size=self._mask_size,
                eval_batch_size=self._eval_batch_size,
                pad_token_id=self._pad_token_id,
                tokenizer=self._tokenizer,
            )
            self.log({"val_miou_subset": round(miou, 4)})

            if miou > self.best_miou and _is_main():
                self.best_miou = miou
                # unwrap 后再调用 save_trainable，兼容 DDP / DeepSpeed
                unwrapped = model.module if hasattr(model, "module") else model
                unwrapped.save_trainable(self.best_ckpt_path)
                print(f"[Step {step}] New best val_miou={miou:.4f}, "
                      f"saved → {self.best_ckpt_path}")

        return (loss, outputs) if return_outputs else loss


# ─────────────────────────────────────────────────────────────────────────────
# 参数
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--local_rank", type=int, default=-1)
    # 路径
    parser.add_argument("--qwen_model_path", type=str,
                        default="/chenmei/Models/Qwen/Qwen3.5-9B")
    parser.add_argument("--sam2_ckpt_path", type=str,
                        default="/chenmei/Models/SAM2/sam2.1_hiera_tiny.pt")
    parser.add_argument("--sam2_model_size", type=str, default="tiny",
                        choices=["tiny", "large"])
    parser.add_argument("--refcoco_root", type=str,
                        default="/chenmei/Datasets/refcoco")
    parser.add_argument("--coco_image_root", type=str,
                        default="/chenmei/Datasets/coco/train2014")
    parser.add_argument("--base_output_dir", type=str,
                        default="/chenmei/Projects/Qwen3Seg/outputs")
    parser.add_argument("--run_name", type=str, default=None,
                        help="实验名称，为空则用时间戳自动生成")

    # 数据
    parser.add_argument("--dataset_names", nargs="+",
                        default=["refcoco", "refcoco+", "refcocog"])
    parser.add_argument("--image_size", type=int, default=896)
    parser.add_argument("--mask_size", type=int, default=512)

    # 模型结构
    parser.add_argument("--hook_layers", type=int, nargs="+", default=[3, 6, 13, 20, 26],
                        help="从 Qwen ViT 哪些 block 抽取特征（供 FPN 使用）")
    parser.add_argument("--num_queries", type=int, default=4,
                        help="QueryExtractor 的 query 数量（替代单 [SEG] embedding）")

    # LoRA
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)


    # 训练
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_steps", type=float, default=100)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--eval_subset_n", type=int, default=500)
    parser.add_argument("--eval_batch_size", type=int, default=8,
                        help="验证时每卡 batch 大小（左填充）")

    # DeepSpeed
    parser.add_argument("--deepspeed", type=str, default=None)

    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    global args
    args = parse_args()

    from datetime import datetime
    run_name = args.run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.base_output_dir, run_name)
    best_ckpt_path = os.path.join(output_dir, "best_model", "qseg_weights.pt")
    os.makedirs(output_dir, exist_ok=True)

    if _is_main():
        print(f"Experiment : {run_name}")
        print(f"Output dir : {output_dir}")

    # ── Processor ────────────────────────────────────────────────────────
    processor = AutoProcessor.from_pretrained(args.qwen_model_path)
    num_added = processor.tokenizer.add_tokens([SEG_TOKEN])
    seg_token_idx = processor.tokenizer.convert_tokens_to_ids(SEG_TOKEN)
    pad_token_id = processor.tokenizer.pad_token_id or 0

    if _is_main():
        print(f"Added {num_added} token(s). [SEG] idx = {seg_token_idx}, "
              f"pad_token_id = {pad_token_id}")
        processor.save_pretrained(os.path.join(output_dir, "processor"))
        print(f"Processor saved → {output_dir}/processor")

    # ── 模型 ─────────────────────────────────────────────────────────────
    model = QSegModel(
        qwen_model_path=args.qwen_model_path,
        sam2_ckpt_path=args.sam2_ckpt_path,
        sam2_model_size=args.sam2_model_size,
        hook_layers=args.hook_layers,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        seg_token_idx=seg_token_idx,
        num_queries=args.num_queries,
    )

    # ── 数据集 ────────────────────────────────────────────────────────────
    train_dataset = RefCOCODataset(
        refcoco_root=args.refcoco_root,
        coco_image_root=args.coco_image_root,
        processor=processor,
        dataset_names=args.dataset_names,
        split="train",
        for_generation=False,
        image_size=args.image_size,
        mask_size=args.mask_size,
        seg_token_idx=seg_token_idx,
    )

    val_dataset = RefCOCODataset(
        refcoco_root=args.refcoco_root,
        coco_image_root=args.coco_image_root,
        processor=processor,
        dataset_names=["refcoco"],
        split="val",
        for_generation=True,  # 验证时让模型自由生成
        image_size=args.image_size,
        mask_size=args.mask_size,
        seg_token_idx=seg_token_idx,
    )

    # ── TrainingArguments ─────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        max_grad_norm=args.max_grad_norm,
        bf16=True,
        tf32=True,
        dataloader_num_workers=16,
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        eval_strategy="no",
        load_best_model_at_end=False,
        report_to="tensorboard",
        remove_unused_columns=False,
        deepspeed=args.deepspeed,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False,
    )

    # ── Trainer ───────────────────────────────────────────────────────────
    trainer = QSegTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=lambda b: collate_fn(b, pad_token_id=pad_token_id),
        best_ckpt_path=best_ckpt_path,
        val_dataset=val_dataset,
        eval_steps=args.eval_steps,
        eval_subset_n=args.eval_subset_n,
        mask_size=args.mask_size,
        eval_batch_size=args.eval_batch_size,
        pad_token_id=pad_token_id,
        tokenizer=processor.tokenizer,
    )

    # 断点续训
    last_ckpt = get_last_checkpoint(output_dir) if os.path.isdir(output_dir) else None
    if last_ckpt and _is_main():
        print(f"Resuming from checkpoint: {last_ckpt}")

    trainer.train(resume_from_checkpoint=last_ckpt)
    trainer.save_model(output_dir)

    if _is_main():
        print("Training complete.")


if __name__ == "__main__":
    main()
