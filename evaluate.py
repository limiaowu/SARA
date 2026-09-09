#!/usr/bin/env python3
# =============================================================================
# evaluate.py
# SARA offline evaluation on RefCOCO, RefCOCO+, RefCOCOg, and ReasoningSeg.
#
# 单卡:
#   python evaluate.py --ckpt ... --refcoco_root ... --coco_image_root ...
#
# 多卡 (torchrun):
#   torchrun --nproc_per_node 4 evaluate.py --ckpt ... --refcoco_root ... --coco_image_root ...
# =============================================================================
import argparse
import json
import os
import time
from datetime import datetime

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from peft import set_peft_model_state_dict
from transformers import AutoProcessor

from data.dataset import collate_fn
from data.builder import build_dataset, eval_splits, ALL_DATASETS
from model.sara import SARAModel, SEG_TOKEN
from model.cnn_bypass import CNNBypass

_CNN_SIZE = CNNBypass.INPUT_SIZE


# ── 分布式工具 ────────────────────────────────────────────────────────────────

def setup_dist():
    """初始化分布式环境（torchrun 自动设置 env 变量）。"""
    if "RANK" not in os.environ:
        return 0, 1  # 单卡模式
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return rank, dist.get_world_size()


def cleanup_dist():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main(rank):
    return rank == 0


# ── 参数 ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate SARA on segmentation benchmarks")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--qwen_model_path", type=str, required=True)
    p.add_argument("--sam2_ckpt", type=str, required=True)
    p.add_argument("--sam2_model_size", type=str, default="large",
                   choices=["tiny", "large"])
    p.add_argument("--output_dir", type=str, default="./outputs/evaluation")
    p.add_argument("--refcoco_root", type=str, default=None)
    p.add_argument("--coco_image_root", type=str, default=None)
    p.add_argument("--reasoning_seg_root", type=str, default=None,
                   help="ReasoningSeg 根目录（含 val/test 子目录）")
    p.add_argument("--datasets", nargs="+",
                   default=["refcoco", "refcoco+", "refcocog"],
                   choices=ALL_DATASETS)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=16)
    p.add_argument("--mask_size", type=int, default=1024)
    p.add_argument("--num_queries", type=int, default=32)
    p.add_argument("--lora_r", type=int, default=64)
    p.add_argument("--lora_alpha", type=int, default=128)
    p.add_argument("--hook_layers", type=int, nargs="+",
                   default=[3, 6, 13, 20, 26])
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--attn_implementation", type=str, default="flash_attention_2",
                   choices=["flash_attention_2", "sdpa", "eager"])
    return p.parse_args()


# ── 模型加载 ──────────────────────────────────────────────────────────────────

def load_model(args, rank):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")

    if is_main(rank):
        print(f"Loading processor from {args.qwen_model_path} ...")
    # max_pixels 必须在 from_pretrained 时传（与 train.py 一致）！
    # 这版 Qwen image processor 会【静默忽略】调用 processor(...) 时再传的 max_pixels，
    # 所以 dataset 里那个 per-call max_pixels 实际不生效。若此处不设上限，ReasoningSeg
    # 的大尺寸手机原图（~12MP）会以近原分辨率进 ViT，产生上万 patch、超长视觉序列，
    # 直接触发 CUDA illegal memory access（RefCOCO 图小 <1MP 故侥幸不崩）。
    processor = AutoProcessor.from_pretrained(args.qwen_model_path, max_pixels=1024 * 1024)
    processor.tokenizer.add_tokens([SEG_TOKEN])
    seg_token_idx = processor.tokenizer.convert_tokens_to_ids(SEG_TOKEN)
    if is_main(rank):
        print(f"  seg_token_idx = {seg_token_idx}")

    if is_main(rank):
        print(f"Loading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)

    # hook_layers 必须与训练时一致：不同尺寸 Qwen 的 ViT 层数不同（9B 27 层 / 4B 24 层），
    # hook 层索引也随之不同。训练 checkpoint 已保存 hook_layers，优先用它，避免 CLI
    # 默认值（可能含越界层索引）与实际权重不匹配。
    hook_layers = ckpt.get("hook_layers", args.hook_layers)
    if is_main(rank) and list(hook_layers) != list(args.hook_layers):
        print(f"  [hook_layers] 用 checkpoint 的 {hook_layers}（忽略 CLI 默认 {args.hook_layers}）")

    # 旧 checkpoint 无 use_cnn_bypass 字段，默认 True（与历史权重一致）
    use_cnn_bypass = ckpt.get("use_cnn_bypass", True)
    model = SARAModel(
        qwen_model_path=args.qwen_model_path,
        sam2_ckpt_path=args.sam2_ckpt,
        sam2_model_size=args.sam2_model_size,
        hook_layers=hook_layers,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        seg_token_idx=seg_token_idx,
        num_queries=args.num_queries,
        attn_implementation=args.attn_implementation,
        use_cnn_bypass=use_cnn_bypass,
    )

    set_peft_model_state_dict(model.qwen, ckpt["lora"])
    model.neck.load_state_dict(ckpt["neck"])
    if model.cnn_bypass is not None:
        model.cnn_bypass.load_state_dict(ckpt["cnn_bypass"])
    model.response_aggregation.load_state_dict(ckpt["response_aggregation"])
    model.mask_decoder.load_state_dict(ckpt["mask_decoder"])
    model.prompt_encoder.load_state_dict(ckpt["prompt_encoder"])

    model = model.eval().to(device)
    if is_main(rank):
        print(f"  Model on {device}  (world_size={dist.get_world_size() if dist.is_initialized() else 1})")
    return model, processor, seg_token_idx


# ── 评测核心循环（与数据集类型无关）─────────────────────────────────────────────

_EMPTY = {"ciou": float("nan"), "giou": float("nan"),
          "n_samples": 0, "seg_rate": 0.0, "elapsed": 0.0}


@torch.no_grad()
def _eval_dataset(
    model, dataset, pad_token_id: int, tokenizer, args,
    rank: int, world_size: int, desc: str,
) -> dict:
    """给定任意 Dataset，运行推理并返回 cIoU / gIoU / seg_rate。"""
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")

    if world_size > 1:
        sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=rank,
            shuffle=False, drop_last=False,
        )
        loader = DataLoader(
            dataset, batch_size=args.batch_size, sampler=sampler,
            num_workers=args.num_workers, pin_memory=True,
            collate_fn=lambda b: collate_fn(b, pad_token_id=pad_token_id),
        )
    else:
        loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True,
            collate_fn=lambda b: collate_fn(b, pad_token_id=pad_token_id),
            drop_last=False,
        )

    total_inter = torch.tensor(0.0, device=device)
    total_union = torch.tensor(0.0, device=device)
    total_per_sample_iou = torch.tensor(0.0, device=device)
    total_seg_found = 0
    total_samples = 0

    iter_loader = (
        tqdm(loader, desc=f"[rank{rank}] {desc}", dynamic_ncols=True,
             disable=not is_main(rank))
        if tqdm is not None else loader
    )

    t0 = time.time()
    for batch in iter_loader:
        mm_ttids = batch.get("mm_token_type_ids")
        outputs = model.generate_with_mask(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            pixel_values=batch["pixel_values"].to(device),
            image_grid_thw=batch["image_grid_thw"].to(device),
            cnn_images=batch["cnn_images"].to(device),
            mm_token_type_ids=mm_ttids.to(device) if mm_ttids is not None else None,
            original_size=(args.mask_size, args.mask_size),
            max_new_tokens=args.max_new_tokens,
            tokenizer=tokenizer,
        )

        pred = (torch.sigmoid(outputs["pred_masks"].float()) > 0.5).float()
        gt = batch["gt_masks"].to(device).float()
        if gt.shape[-2:] != pred.shape[-2:]:
            gt = F.interpolate(
                gt.unsqueeze(1), size=pred.shape[-2:], mode="nearest"
            ).squeeze(1)

        # ReasoningSeg 的 ignore 区域以哨兵值 255 标记，从 IoU 中整体排除（对齐 LISA
        # 口径）：ignore 处的预测既不算 TP 也不算 FP。RefCOCO 无 255，valid 恒为 1。
        valid = (gt != 255).float()
        gt_fg = (gt == 1).float()
        pred = pred * valid

        total_inter += (pred * gt_fg).sum()
        total_union += ((pred + gt_fg) > 0).float().sum()
        for i in range(pred.shape[0]):
            inter_i = (pred[i] * gt_fg[i]).sum()
            union_i = ((pred[i] + gt_fg[i]) > 0).float().sum()
            total_per_sample_iou += inter_i / (union_i + 1e-6)
        total_seg_found += sum(outputs["seg_found"])
        total_samples += pred.shape[0]

        if tqdm is not None and is_main(rank):
            sps = total_samples / (time.time() - t0)
            iter_loader.set_postfix(
                cIoU=f"{(total_inter / (total_union + 1e-6)).item():.4f}",
                gIoU=f"{(total_per_sample_iou / max(total_samples, 1)).item():.4f}",
                seg=f"{total_seg_found}/{total_samples}",
                sps=f"{sps:.2f}",
            )

    if world_size > 1:
        dist.all_reduce(total_inter, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_union, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_per_sample_iou, op=dist.ReduceOp.SUM)
        seg_tensor = torch.tensor(
            [total_seg_found, total_samples], dtype=torch.long, device=device
        )
        dist.all_reduce(seg_tensor, op=dist.ReduceOp.SUM)
        total_seg_found = seg_tensor[0].item()
        total_samples = seg_tensor[1].item()

    elapsed = time.time() - t0
    return {
        "ciou": (total_inter / (total_union + 1e-6)).item(),
        "giou": (total_per_sample_iou / max(total_samples, 1)).item(),
        "n_samples": total_samples,
        "seg_rate": total_seg_found / max(total_samples, 1),
        "elapsed": elapsed,
    }


# ── 单个 split 评测（任意数据集，统一走 data.builder）──────────────────────────

def evaluate_split(
    model, processor, args,
    dataset_name: str, split: str,
    rank: int, world_size: int,
) -> dict:
    pad_token_id = processor.tokenizer.pad_token_id or 0
    seg_token_idx = processor.tokenizer.convert_tokens_to_ids(SEG_TOKEN)
    try:
        dataset = build_dataset(
            dataset_name, split,
            processor=processor,
            for_generation=True,
            image_size=_CNN_SIZE,
            mask_size=args.mask_size,
            seg_token_idx=seg_token_idx,
            refcoco_root=args.refcoco_root,
            coco_image_root=args.coco_image_root,
            reasoning_seg_root=args.reasoning_seg_root,
        )
    except Exception as e:
        if is_main(rank):
            print(f"  [{dataset_name} {split}] build failed: {e}, skipping.")
        return _EMPTY
    if len(dataset) == 0:
        if is_main(rank):
            print(f"  [{dataset_name} {split}] No samples found, skipping.")
        return _EMPTY
    return _eval_dataset(model, dataset, pad_token_id, processor.tokenizer,
                         args, rank, world_size, desc=f"{dataset_name}/{split}")


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    rank, world_size = setup_dist()

    model, processor, _ = load_model(args, rank)

    results = {}

    # 评测覆盖每个数据集“除 train 外的全部 split”（eval_splits）
    for dataset_name in args.datasets:
        for split in eval_splits(dataset_name):
            if is_main(rank):
                print(f"\n{'='*60}")
                print(f"Evaluating {dataset_name} / {split} ...")

            if world_size > 1:
                dist.barrier()

            metrics = evaluate_split(
                model, processor, args, dataset_name, split, rank, world_size,
            )
            results[(dataset_name, split)] = metrics

            if is_main(rank):
                print(f"  cIoU={metrics['ciou']:.4f}  "
                      f"gIoU={metrics['giou']:.4f}  "
                      f"[SEG]={metrics['seg_rate']:.3f}  "
                      f"n={metrics['n_samples']}  "
                      f"time={metrics['elapsed']:.0f}s")

    if not is_main(rank):
        cleanup_dist()
        return

    # ── 汇总表（仅 rank 0）────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  SARA Evaluation Results")
    print(f"  Checkpoint : {args.ckpt}")
    print(f"  mask_size  : {args.mask_size}   batch_size : {args.batch_size}")
    print(f"  world_size : {world_size}")
    print(f"{'='*60}")

    col_map = {name: eval_splits(name) for name in ALL_DATASETS}
    row_fmt = "  {:<20s}"
    val_fmt = "  {:>8s}"

    # RefCOCO 汇总表
    refcoco_datasets = [d for d in ["refcoco", "refcoco+", "refcocog"] if d in args.datasets]
    if refcoco_datasets:
        header = row_fmt.format("Dataset/Metric")
        for d in refcoco_datasets:
            for s in col_map[d]:
                header += val_fmt.format(s)
        print(header)
        n_cols = sum(len(col_map[d]) for d in refcoco_datasets)
        print("  " + "-" * (22 + 10 * n_cols))
        for d in refcoco_datasets:
            for metric, label in [("ciou", f"{d}/cIoU"), ("giou", f"{d}/gIoU")]:
                row = row_fmt.format(label)
                for s in col_map[d]:
                    key = (d, s)
                    if key in results and results[key]["n_samples"] > 0:
                        row += val_fmt.format(f"{results[key][metric]:.4f}")
                    else:
                        row += val_fmt.format("-")
                print(row)

    # ReasoningSeg 汇总表
    if "reasoning_seg" in args.datasets:
        print()
        header = row_fmt.format("ReasoningSeg/Metric")
        for s in col_map["reasoning_seg"]:
            header += val_fmt.format(s)
        print(header)
        print("  " + "-" * (22 + 10 * len(col_map["reasoning_seg"])))
        for metric, label in [("ciou", "cIoU"), ("giou", "gIoU")]:
            row = row_fmt.format(label)
            for s in col_map["reasoning_seg"]:
                key = ("reasoning_seg", s)
                if key in results and results[key]["n_samples"] > 0:
                    row += val_fmt.format(f"{results[key][metric]:.4f}")
                else:
                    row += val_fmt.format("-")
            print(row)

    print(f"{'='*60}\n")

    # ── 保存 JSON ─────────────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"eval_{timestamp}.json")
    json_data = {
        "timestamp": timestamp,
        "checkpoint": args.ckpt,
        "mask_size": args.mask_size,
        "batch_size": args.batch_size,
        "world_size": world_size,
        "hook_layers": model.hook_layers,
        "num_queries": args.num_queries,
        "lora_r": args.lora_r,
        "results": {
            f"{d}/{s}": {
                "ciou": round(m["ciou"], 6),
                "giou": round(m["giou"], 6),
                "n_samples": m["n_samples"],
                "seg_rate": round(m["seg_rate"], 4),
                "elapsed_s": round(m["elapsed"], 1),
            }
            for (d, s), m in results.items()
        },
    }
    with open(out_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"Results saved to: {out_path}\n")

    cleanup_dist()


if __name__ == "__main__":
    main()
