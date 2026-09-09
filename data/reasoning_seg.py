# =============================================================================
# data/reasoning_seg.py
# LISA ReasoningSeg 数据集
#
# 目录结构：
#   root/
#   ├── train/  image_1.jpg  image_1.json  ...
#   └── val/    image_1.jpg  image_1.json  ...
#
# JSON 格式（关键字段）：
#   "text"      : str 或 [str, ...]，训练时随机选一条
#   "is_sentence": bool
#   "shapes"    : [{"label": "target"/"ignore", "shape_type": "polygon",
#                   "points": [[x,y], ...], ...}]
#
# 输出格式与 RefCOCODataset 完全一致，可直接用于同一 collate_fn。
# =============================================================================
import json
import os
import random
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset

_CNN_SIZE = 1024
_CNN_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_CNN_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

SEG_TOKEN = "[SEG]"

ANSWER_TEMPLATES = [
    "Sure, [SEG].",
    "It is [SEG].",
    "Of course, [SEG].",
    "The segmentation result is [SEG].",
    "Sure, the segmentation result is [SEG].",
]


def _is_main() -> bool:
    import torch.distributed as _dist
    if _dist.is_available() and _dist.is_initialized():
        return _dist.get_rank() == 0
    return int(os.environ.get("LOCAL_RANK", 0)) == 0


class ReasoningSegDataset(Dataset):
    """
    LISA ReasoningSeg 数据集。

    参数：
        root          : 数据集根目录（含 train/val 子目录）
        split         : "train" 或 "val"
        for_generation: False（训练）→ 完整对话 + labels；
                        True（推理）→ 仅 user 消息 + labels 全 -100
    """

    def __init__(
        self,
        root: str,
        processor,
        split: str = "train",
        for_generation: bool = False,
        image_size: int = 1024,
        mask_size: int = 512,
        seg_token_idx: Optional[int] = None,
    ):
        self.processor = processor
        self.split = split
        self.for_generation = for_generation
        self.image_size = image_size
        self.mask_size = mask_size
        self.seg_token_idx = seg_token_idx

        split_dir = Path(root) / split
        self.samples: List[dict] = []

        for json_path in sorted(split_dir.glob("*.json")):
            img_path = json_path.with_suffix(".jpg")
            if not img_path.exists():
                continue
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    ann = json.load(f)
            except Exception:
                continue

            # 过滤掉无 target shape 的样本
            targets = [s for s in ann.get("shapes", []) if s.get("label") == "target"]
            if not targets:
                continue

            texts = ann.get("text", [])
            if isinstance(texts, str):
                texts = [texts]
            if not texts:
                continue

            self.samples.append({
                "image_path": str(img_path),
                "texts": texts,
                "shapes": ann["shapes"],
            })

        if _is_main():
            print(f"[ReasoningSegDataset] split={split}: {len(self.samples)} samples from {split_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        # ── 图像（EXIF orientation 校正，与多边形标注坐标系对齐）──────────
        # ReasoningSeg 多为手机原图，竖拍图常以横向像素 + orientation 标签存储；
        # labelme 标注的是校正后画面，故须 exif_transpose 后再用，否则 mask 错位。
        image = ImageOps.exif_transpose(Image.open(sample["image_path"])).convert("RGB")
        orig_w, orig_h = image.size

        # CNN bypass 输入：1024×1024, ImageNet 归一化
        cnn_img = image.resize((_CNN_SIZE, _CNN_SIZE), Image.BILINEAR)
        cnn_tensor = torch.from_numpy(np.array(cnn_img)).float() / 255.0
        cnn_tensor = (cnn_tensor.permute(2, 0, 1) - _CNN_MEAN) / _CNN_STD

        # ── GT mask（target=1）────────────────────────────────────────────
        gt_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
        for shape in sample["shapes"]:
            if shape.get("label") != "target":
                continue
            pts = np.array(shape["points"], dtype=np.float32).reshape(-1, 2)
            cv2.fillPoly(gt_mask, [pts.astype(np.int32)], 1)

        # 评测路径额外把 ignore 多边形标成哨兵值 255（覆盖与 target 的重叠区），
        # evaluate.py 算 IoU 时会把 ==255 的像素整体排除（对齐 LISA ReasonSeg 口径）。
        # 训练路径保持 {0,1}（ignore 当背景），不改变既有训练行为。RefCOCO 无 ignore，
        # 始终是 {0,1}，下游 `gt != 255` 对它恒为真，无影响。
        if self.for_generation:
            for shape in sample["shapes"]:
                if "ignore" not in str(shape.get("label", "")).lower():
                    continue
                pts = np.array(shape["points"], dtype=np.float32).reshape(-1, 2)
                cv2.fillPoly(gt_mask, [pts.astype(np.int32)], 255)

        gt_mask_rs = cv2.resize(
            gt_mask, (self.mask_size, self.mask_size), interpolation=cv2.INTER_NEAREST
        )
        gt_mask_tensor = torch.from_numpy(gt_mask_rs).float()

        # ── Bbox（target 多边形的 AABB）───────────────────────────────────
        all_pts = []
        for shape in sample["shapes"]:
            if shape.get("label") != "target":
                continue
            pts = np.array(shape["points"], dtype=np.float32)
            all_pts.append(pts)

        if all_pts:
            pts_cat = np.concatenate(all_pts, axis=0)
            x1 = float(np.clip(pts_cat[:, 0].min(), 0, orig_w))
            y1 = float(np.clip(pts_cat[:, 1].min(), 0, orig_h))
            x2 = float(np.clip(pts_cat[:, 0].max(), 0, orig_w))
            y2 = float(np.clip(pts_cat[:, 1].max(), 0, orig_h))
        else:
            x1, y1, x2, y2 = 0.0, 0.0, float(orig_w), float(orig_h)

        # Qwen3-VL 0-1000 坐标
        bx1 = round(x1 / orig_w * 1000)
        by1 = round(y1 / orig_h * 1000)
        bx2 = round(x2 / orig_w * 1000)
        by2 = round(y2 / orig_h * 1000)

        # SAM PromptEncoder 1024 坐标
        gt_box = torch.tensor([
            x1 / orig_w * 1024, y1 / orig_h * 1024,
            x2 / orig_w * 1024, y2 / orig_h * 1024,
        ], dtype=torch.float32)

        # ── 对话构建 ─────────────────────────────────────────────────────
        query = random.choice(sample["texts"])
        bbox_prefix = f'{{"bbox_2d": [{bx1}, {by1}, {bx2}, {by2}]}}\n'
        answer = bbox_prefix + random.choice(ANSWER_TEMPLATES)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": query},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": answer}],
            },
        ]

        if not self.for_generation:
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False, enable_thinking=False,
            )
        else:
            text = self.processor.apply_chat_template(
                [messages[0]], tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )

        # ── Processor ────────────────────────────────────────────────────
        # max_pixels 只在 AutoProcessor.from_pretrained 时生效（调用时再传会被
        # 这版 Qwen image processor 静默忽略），故此处不传，见 train/evaluate.py。
        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
        )

        input_ids       = inputs["input_ids"].squeeze(0)
        attention_mask  = inputs["attention_mask"].squeeze(0)
        pixel_values    = inputs["pixel_values"]
        image_grid_thw  = inputs["image_grid_thw"].squeeze(0)
        labels          = self._build_labels(input_ids)

        mm_token_type_ids = None
        if "mm_token_type_ids" in inputs:
            mm_token_type_ids = inputs["mm_token_type_ids"].squeeze(0)

        item = {
            "input_ids":       input_ids,
            "attention_mask":  attention_mask,
            "pixel_values":    pixel_values,
            "image_grid_thw":  image_grid_thw,
            "cnn_image":       cnn_tensor,
            "labels":          labels,
            "gt_masks":        gt_mask_tensor,
            "gt_boxes":        gt_box,
            "image_path":      sample["image_path"],
        }
        if mm_token_type_ids is not None:
            item["mm_token_type_ids"] = mm_token_type_ids

        return item

    def _build_labels(self, input_ids: torch.Tensor) -> torch.Tensor:
        labels = input_ids.clone()
        labels[:] = -100
        if self.for_generation:
            return labels

        im_start_id  = self.processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
        im_end_id    = self.processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
        assistant_id = self.processor.tokenizer.encode("assistant", add_special_tokens=False)[0]

        i = 0
        while i < len(input_ids):
            if (input_ids[i] == im_start_id
                    and i + 1 < len(input_ids)
                    and input_ids[i + 1] == assistant_id):
                j = i
                while j < len(input_ids) and input_ids[j] != im_end_id:
                    j += 1
                labels[i: j + 1] = input_ids[i: j + 1]
                i = j + 1
            else:
                i += 1
        return labels
