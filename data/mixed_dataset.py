# =============================================================================
# data/mixed_dataset.py
# 多数据集混合器，按指定重复倍数拼接多个 Dataset。
#
# 示例：
#   # RefCOCO 1 倍 + ReasoningSeg 3 倍（样本少，过采样到大致相近规模）
#   MixedDataset(
#       datasets=[refcoco_train, reasoning_train],
#       repeat_factors=[1.0, 3.0],
#   )
#
# 输出格式与各子数据集一致，可直接用于同一 collate_fn。
# =============================================================================
import math
import os
import random
from typing import List, Optional

from torch.utils.data import Dataset


def _is_main() -> bool:
    import torch.distributed as _dist
    if _dist.is_available() and _dist.is_initialized():
        return _dist.get_rank() == 0
    return int(os.environ.get("LOCAL_RANK", 0)) == 0


class MixedDataset(Dataset):
    """
    将多个 Dataset 按指定重复倍数拼接。

    参数：
        datasets       : Dataset 列表
        repeat_factors : 各数据集的重复倍数（float）。
                         n = ceil(len(ds) * factor)，超出部分循环取样。
                         默认各数据集等权重（1.0）。
        seed           : 用于初始打乱的随机种子（保证多卡一致）。
    """

    def __init__(
        self,
        datasets: List[Dataset],
        repeat_factors: Optional[List[float]] = None,
        seed: int = 42,
    ):
        assert len(datasets) > 0, "至少需要一个数据集"
        if repeat_factors is None:
            repeat_factors = [1.0] * len(datasets)
        assert len(datasets) == len(repeat_factors), \
            "datasets 和 repeat_factors 长度须一致"

        self._datasets = datasets

        # 构建全局索引表 [(ds_idx, item_idx), ...]
        self._index_map: List[tuple] = []
        for ds_idx, (ds, r) in enumerate(zip(datasets, repeat_factors)):
            n = math.ceil(len(ds) * float(r))
            ds_len = len(ds)
            for i in range(n):
                self._index_map.append((ds_idx, i % ds_len))

        # 打乱一次，使不同数据集的样本交错（Trainer 训练时还会再 shuffle）
        rng = random.Random(seed)
        rng.shuffle(self._index_map)

        if _is_main():
            sizes = [f"{ds.__class__.__name__}×{r}" for ds, r in zip(datasets, repeat_factors)]
            print(f"[MixedDataset] {' + '.join(sizes)} → total {len(self._index_map)} samples")

    def __len__(self) -> int:
        return len(self._index_map)

    def __getitem__(self, idx: int) -> dict:
        ds_idx, item_idx = self._index_map[idx]
        return self._datasets[ds_idx][item_idx]
