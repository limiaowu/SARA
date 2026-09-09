# =============================================================================
# data/builder.py
# 数据集注册表 + 统一构建入口。
#
# train.py / evaluate.py 共用本模块，避免在多处重复维护
# “数据集 → 类 / split / 路径” 的映射关系。
#
# ── 核心概念：dataset spec ─────────────────────────────────────────────────
#   一条 spec 是 "name:split[:repeat]" 形式的字符串：
#       refcoco:train            → RefCOCO train，过采样 1.0 倍
#       refcocog:val             → RefCOCOg val
#       reasoning_seg:train:100  → ReasoningSeg train，过采样 100 倍
#
#   训练用 spec 列表（含 repeat），验证 / 评测用 spec 列表（repeat 无意义，忽略）。
#
# ── 评测覆盖范围 ────────────────────────────────────────────────────────────
#   evaluate 默认跑“某数据集除 train 外的所有 split”，由 eval_splits() 给出。
# =============================================================================
from typing import List, Optional, Tuple

from torch.utils.data import Dataset

from data.dataset import RefCOCODataset
from data.reasoning_seg import ReasoningSegDataset
from data.mixed_dataset import MixedDataset

# ── 数据集 → 全部 split（按出现顺序，train 必在首位）────────────────────────
DATASET_SPLITS = {
    "refcoco":       ["train", "val", "testA", "testB"],
    "refcoco+":      ["train", "val", "testA", "testB"],
    "refcocog":      ["train", "val", "test"],
    "reasoning_seg": ["train", "val", "test"],
}

# RefCOCO 系列共用同一份 refcoco_root / coco_image_root
REFCOCO_NAMES = {"refcoco", "refcoco+", "refcocog"}

ALL_DATASETS = list(DATASET_SPLITS.keys())


def eval_splits(name: str) -> List[str]:
    """返回某数据集“除 train 外的全部 split”，供 evaluate 遍历。"""
    return [s for s in DATASET_SPLITS[name] if s != "train"]


# ── spec 解析 ────────────────────────────────────────────────────────────────

def parse_spec(spec: str) -> Tuple[str, str, float]:
    """
    解析 "name:split[:repeat]" → (name, split, repeat)。
    repeat 缺省为 1.0。非法 name / split 直接抛错（fail fast）。
    """
    parts = spec.split(":")
    if len(parts) < 2 or len(parts) > 3:
        raise ValueError(
            f"非法 dataset spec: {spec!r}，应为 'name:split' 或 'name:split:repeat'"
        )
    name, split = parts[0], parts[1]
    repeat = float(parts[2]) if len(parts) == 3 else 1.0

    if name not in DATASET_SPLITS:
        raise ValueError(f"未知数据集 {name!r}，可选：{ALL_DATASETS}")
    if split not in DATASET_SPLITS[name]:
        raise ValueError(
            f"数据集 {name!r} 无 split {split!r}，可选：{DATASET_SPLITS[name]}"
        )
    if repeat <= 0:
        raise ValueError(f"repeat 须为正数，spec={spec!r}")
    return name, split, repeat


# ── 单个数据集构建 ─────────────────────────────────────────────────────────

def build_dataset(
    name: str,
    split: str,
    *,
    processor,
    for_generation: bool,
    image_size: int,
    mask_size: int,
    seg_token_idx: Optional[int] = None,
    refcoco_root: Optional[str] = None,
    coco_image_root: Optional[str] = None,
    reasoning_seg_root: Optional[str] = None,
) -> Dataset:
    """按 name/split 构建对应 Dataset。缺失必要路径时抛错。"""
    common = dict(
        processor=processor,
        split=split,
        for_generation=for_generation,
        image_size=image_size,
        mask_size=mask_size,
        seg_token_idx=seg_token_idx,
    )

    if name in REFCOCO_NAMES:
        if not refcoco_root or not coco_image_root:
            raise ValueError(
                f"{name} 需要 refcoco_root 与 coco_image_root，但未提供"
            )
        return RefCOCODataset(
            refcoco_root=refcoco_root,
            coco_image_root=coco_image_root,
            dataset_names=[name],
            **common,
        )

    if name == "reasoning_seg":
        if not reasoning_seg_root:
            raise ValueError("reasoning_seg 需要 reasoning_seg_root，但未提供")
        return ReasoningSegDataset(root=reasoning_seg_root, **common)

    raise ValueError(f"未知数据集 {name!r}")


# ── 多 spec → 单个（可混合 / 过采样）Dataset ────────────────────────────────

def build_from_specs(
    specs: List[str],
    *,
    processor,
    for_generation: bool,
    image_size: int,
    mask_size: int,
    seg_token_idx: Optional[int] = None,
    refcoco_root: Optional[str] = None,
    coco_image_root: Optional[str] = None,
    reasoning_seg_root: Optional[str] = None,
    seed: int = 42,
) -> Optional[Dataset]:
    """
    将一组 spec 构建并混合为单个 Dataset：
      - 每条 spec 独立构建；
      - 按各自 repeat 倍数过采样后交错混合（MixedDataset）；
      - 单条 spec 且 repeat==1.0 时直接返回原 Dataset（省掉 MixedDataset 包装）。
    specs 为空时返回 None。
    """
    if not specs:
        return None

    paths = dict(
        refcoco_root=refcoco_root,
        coco_image_root=coco_image_root,
        reasoning_seg_root=reasoning_seg_root,
    )

    datasets, repeats = [], []
    for spec in specs:
        name, split, repeat = parse_spec(spec)
        datasets.append(build_dataset(
            name, split,
            processor=processor,
            for_generation=for_generation,
            image_size=image_size,
            mask_size=mask_size,
            seg_token_idx=seg_token_idx,
            **paths,
        ))
        repeats.append(repeat)

    if len(datasets) == 1 and repeats[0] == 1.0:
        return datasets[0]
    return MixedDataset(datasets, repeats, seed=seed)
