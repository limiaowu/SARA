# =============================================================================
# utils/metrics_logger.py
# 训练指标 CSV 记录器
#
# 设计原则：
#   - 追加写入（append mode），断点续训不会覆盖历史记录
#   - 每次写入后立即 flush，进程中断不丢数据
#   - 内存/显存零占用：只是文件 I/O
#   - 仅 rank-0 写入
#
# 输出文件（均在 output_dir 下）：
#   train_metrics.csv  —— step, lm_loss, mask_loss, mask_loss_ema, loss, learning_rate, grad_norm
#   val_metrics.csv    —— step, val_ciou
# =============================================================================
import csv
import os
from typing import Any, Dict


class MetricsLogger:

    TRAIN_FILE = "train_metrics.csv"
    VAL_FILE   = "val_metrics.csv"

    # 字段顺序决定 CSV 列顺序
    TRAIN_FIELDS = ["step", "lm_loss", "mask_loss", "mask_loss_ema",
                    "loss", "learning_rate", "grad_norm"]
    VAL_FIELDS   = ["step", "val_ciou"]

    # log() 里哪些 key 属于训练指标
    _TRAIN_KEYS = {"lm_loss", "mask_loss", "mask_loss_ema",
                   "loss", "learning_rate", "grad_norm"}

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self._train_path = os.path.join(output_dir, self.TRAIN_FILE)
        self._val_path   = os.path.join(output_dir, self.VAL_FILE)
        self._ensure_header(self._train_path, self.TRAIN_FIELDS)
        self._ensure_header(self._val_path,   self.VAL_FIELDS)

    # ── 公开接口 ─────────────────────────────────────────────────────────

    def record(self, step: int, logs: Dict[str, Any]) -> None:
        """由 SARATrainer.log() 调用，自动路由到对应 CSV。"""
        if "val_ciou_subset" in logs:
            self._append(self._val_path, self.VAL_FIELDS, {
                "step":     step,
                "val_ciou": logs["val_ciou_subset"],
            })

        if self._TRAIN_KEYS & set(logs.keys()):
            row = {"step": step}
            for k in self.TRAIN_FIELDS[1:]:
                v = logs.get(k)
                row[k] = "" if v is None else round(float(v), 6)
            self._append(self._train_path, self.TRAIN_FIELDS, row)

    # ── 内部方法 ─────────────────────────────────────────────────────────

    @staticmethod
    def _ensure_header(path: str, fields: list) -> None:
        """若文件不存在则写表头；已存在则直接追加（断点续训不重置）。"""
        if not os.path.exists(path):
            with open(path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=fields).writeheader()

    @staticmethod
    def _append(path: str, fields: list, row: dict) -> None:
        with open(path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writerow(row)
            f.flush()
