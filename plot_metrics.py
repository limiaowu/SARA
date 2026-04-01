#!/usr/bin/env python3
# =============================================================================
# plot_metrics.py
# 解析训练日志，生成可视化图表 + CSV + 控制台摘要
#
# 用法：
#   python plot_metrics.py                               # 自动找最新 run
#   python plot_metrics.py --log outputs/run/train.log  # 指定日志
#   python plot_metrics.py --log outputs/run/train.log --output_dir metrics/
# =============================================================================
import argparse
import ast
import csv
import os
import re
import sys
from pathlib import Path


def _find_latest_log(base_dir: str = "outputs") -> str | None:
    """自动找 outputs/ 下最新修改的 train.log"""
    logs = sorted(
        Path(base_dir).rglob("train.log"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return str(logs[0]) if logs else None


def parse_log(log_path: str):
    """
    解析日志，返回 (train_steps, val_points)。

    每条日志行是一个 Python dict 字面量，大致有三类：
      - lm_loss / mask_loss 行：per micro-step 自定义日志
      - loss / learning_rate 行：HuggingFace Trainer 的 optimizer step 日志
      - val_miou_subset 行：验证集评估结果
    """
    train_steps = []   # list of dict
    val_points  = []   # list of dict
    pending     = {}   # 暂存当前 optimizer step 的自定义指标
    global_step = 0

    with open(log_path, "r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            m = re.search(r"\{.*\}", raw)
            if not m:
                continue
            try:
                d = ast.literal_eval(m.group())
            except Exception:
                continue

            if "val_miou_subset" in d:
                val_points.append({
                    "step":     global_step,
                    "epoch":    float(d.get("epoch", 0)),
                    "val_miou": float(d["val_miou_subset"]),
                })

            elif "learning_rate" in d and "loss" in d:
                # Trainer 主日志：一个 optimizer step 完成
                global_step += 1
                train_steps.append({
                    "step":          global_step,
                    "epoch":         float(d.get("epoch", 0)),
                    "loss":          float(d.get("loss", 0)),
                    "grad_norm":     float(d.get("grad_norm", 0)),
                    "lr":            float(d.get("learning_rate", 0)),
                    "lm_loss":       float(pending.get("lm_loss", 0)),
                    "mask_loss":     float(pending.get("mask_loss", 0)),
                    "mask_loss_ema": float(pending.get("mask_loss_ema", 0)),
                })
                pending = {}

            elif "lm_loss" in d or "mask_loss" in d:
                # 自定义 micro-step 日志，暂存（取最后一次）
                pending.update({k: v for k, v in d.items() if k != "epoch"})

    return train_steps, val_points


def save_csv(train_steps, val_points, output_dir: str):
    """保存训练指标和验证指标为 CSV。"""
    if train_steps:
        train_path = os.path.join(output_dir, "train_metrics.csv")
        with open(train_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=train_steps[0].keys())
            w.writeheader()
            w.writerows(train_steps)
        print(f"  Saved: {train_path}")

    if val_points:
        val_path = os.path.join(output_dir, "val_metrics.csv")
        with open(val_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=val_points[0].keys())
            w.writeheader()
            w.writerows(val_points)
        print(f"  Saved: {val_path}")


def print_summary(train_steps, val_points, run_name: str):
    """控制台打印最新指标摘要。"""
    sep = "─" * 60
    print(f"\n{sep}")
    print(f"  Run: {run_name}")
    print(sep)

    if train_steps:
        last = train_steps[-1]
        best_loss = min(s["loss"] for s in train_steps)
        print(f"  Total optimizer steps : {last['step']}")
        print(f"  Current epoch         : {last['epoch']:.3f}")
        print(f"  Latest loss           : {last['loss']:.4f}  "
              f"(lm={last['lm_loss']:.4f}  mask_ema={last['mask_loss_ema']:.4f})")
        print(f"  Best loss             : {best_loss:.4f}")
        print(f"  Latest lr             : {last['lr']:.2e}")
        print(f"  Latest grad_norm      : {last['grad_norm']:.3f}")

    if val_points:
        best_val = max(v["val_miou"] for v in val_points)
        last_val = val_points[-1]
        print(f"  Val mIoU (latest)     : {last_val['val_miou']:.4f}  @ step {last_val['step']}")
        print(f"  Val mIoU (best)       : {best_val:.4f}")
        print(f"  Total val checkpoints : {len(val_points)}")

    print(sep + "\n")


def plot(train_steps, val_points, output_path: str):
    """生成四格折线图：loss / lr / grad_norm / val mIoU。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        print("[WARN] matplotlib not available, skipping plot.")
        return

    if not train_steps:
        print("[WARN] No training steps to plot.")
        return

    steps      = [s["step"]          for s in train_steps]
    loss       = [s["loss"]          for s in train_steps]
    lm_loss    = [s["lm_loss"]       for s in train_steps]
    mask_ema   = [s["mask_loss_ema"] for s in train_steps]
    lr         = [s["lr"]            for s in train_steps]
    grad_norm  = [s["grad_norm"]     for s in train_steps]

    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.3)

    # ── Panel 1: Loss ──────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(steps, loss,     label="total loss",    linewidth=1.2, alpha=0.9)
    ax1.plot(steps, lm_loss,  label="lm loss",       linewidth=1.0, alpha=0.7, linestyle="--")
    ax1.plot(steps, mask_ema, label="mask loss (EMA)", linewidth=1.0, alpha=0.7, linestyle=":")
    ax1.set_title("Training Loss")
    ax1.set_xlabel("Optimizer Step")
    ax1.set_ylabel("Loss")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # ── Panel 2: Learning Rate ─────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(steps, lr, color="tab:orange", linewidth=1.2)
    ax2.set_title("Learning Rate")
    ax2.set_xlabel("Optimizer Step")
    ax2.set_ylabel("LR")
    ax2.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    ax2.grid(True, alpha=0.3)

    # ── Panel 3: Gradient Norm ─────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(steps, grad_norm, color="tab:green", linewidth=1.0, alpha=0.8)
    ax3.set_title("Gradient Norm")
    ax3.set_xlabel("Optimizer Step")
    ax3.set_ylabel("Grad Norm")
    ax3.set_yscale("log")
    ax3.grid(True, alpha=0.3)

    # ── Panel 4: Validation mIoU ───────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    if val_points:
        val_steps = [v["step"]     for v in val_points]
        val_miou  = [v["val_miou"] for v in val_points]
        ax4.plot(val_steps, val_miou, marker="o", markersize=4,
                 color="tab:red", linewidth=1.2, label="val mIoU")
        best_v = max(val_miou)
        best_s = val_steps[val_miou.index(best_v)]
        ax4.axhline(best_v, linestyle="--", color="gray", linewidth=0.8,
                    label=f"best={best_v:.4f} @ step {best_s}")
        ax4.legend(fontsize=8)
    else:
        ax4.text(0.5, 0.5, "No val data yet", ha="center", va="center",
                 transform=ax4.transAxes, fontsize=12, color="gray")
    ax4.set_title("Validation mIoU (subset)")
    ax4.set_xlabel("Optimizer Step")
    ax4.set_ylabel("mIoU")
    ax4.set_ylim(0, 1.0)
    ax4.grid(True, alpha=0.3)

    run_name = Path(output_path).stem
    fig.suptitle(f"Training Metrics — {run_name}", fontsize=13, y=0.98)

    plt.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Parse training log and generate metrics report")
    parser.add_argument("--log",        type=str, default=None,
                        help="Path to train.log (auto-detected if omitted)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Where to save outputs (defaults to same dir as log)")
    args = parser.parse_args()

    # ── Find log ──────────────────────────────────────────────────────────
    log_path = args.log
    if log_path is None:
        log_path = _find_latest_log()
    if log_path is None or not os.path.exists(log_path):
        print(f"[ERR] train.log not found. Specify with --log.")
        sys.exit(1)

    output_dir = args.output_dir or os.path.dirname(log_path)
    os.makedirs(output_dir, exist_ok=True)
    run_name = Path(log_path).parent.name

    print(f"Parsing: {log_path}")
    train_steps, val_points = parse_log(log_path)
    print(f"  {len(train_steps)} optimizer steps, {len(val_points)} val checkpoints")

    print_summary(train_steps, val_points, run_name)

    print("Saving outputs...")
    save_csv(train_steps, val_points, output_dir)
    plot_path = os.path.join(output_dir, "metrics.png")
    plot(train_steps, val_points, plot_path)


if __name__ == "__main__":
    main()