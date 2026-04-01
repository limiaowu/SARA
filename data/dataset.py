# =============================================================================
# data/dataset.py
# RefCOCO / RefCOCO+ / RefCOCOg 数据集
#
# 改动记录：
#   - [BUG FIX] pixel_values 不应 squeeze(0)：dim-0 是 total_patches，而非 batch
#   - [BUG FIX] 捕获并返回 mm_token_type_ids（M-RoPE 必需，processor 会输出此字段）
#   - split 职责拆分：for_generation 控制 chat template 格式；
#     split 仅用于数据集分割过滤
#   - collate_fn 统一左填充（训练 / 验证均可用，batch generate 要求左填充）
#   - mm_token_type_ids 在 collate_fn 中以 0（text 类型）左填充
#   - 所有 print 用 _is_main() 守卫，避免 4 卡重复输出
# =============================================================================
import json
import os
import random
from typing import List, Dict, Optional

import cv2
import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_utils
from torch.utils.data import Dataset

# CNN bypass 输入参数（与 model/cnn_bypass.py 保持一致）
_CNN_SIZE = 1024
_CNN_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_CNN_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

SEG_TOKEN = "[SEG]"


def _is_main() -> bool:
    """主进程检测（兼容 dist 未初始化的数据集加载阶段）。"""
    import torch.distributed as _dist
    if _dist.is_available() and _dist.is_initialized():
        return _dist.get_rank() == 0
    return int(os.environ.get("LOCAL_RANK", 0)) == 0


ANSWER_TEMPLATES = [
    "Sure, [SEG].",
    "It is [SEG].",
    "Of course, [SEG].",
    "The segmentation result is [SEG].",
    "Sure, the segmentation result is [SEG].",
]

QUESTION_TEMPLATES = [
    "Please segment {description} in the image.",
    "Can you segment {description}?",
    "Please output a segmentation mask for {description}.",
    "Segment {description} in this image.",
    "Where is {description}? Please segment it.",
]


class RefCOCODataset(Dataset):
    """
    支持 RefCOCO、RefCOCO+、RefCOCOg 三个数据集。

    参数：
        split         : 数据分割，"train"/"val"/"test"，决定加载哪些样本。
        for_generation: 控制 chat template 格式：
                        False（训练）→ 完整对话（user+assistant），labels 覆盖 assistant；
                        True （推理）→ 仅 user 消息 + add_generation_prompt，labels 全 -100。

    数据目录结构：
        refcoco_root/
        ├── refcoco/
        │   ├── instances.json
        │   └── refs(unc).p
        ├── refcoco+/
        └── refcocog/
        coco_image_root/   ← COCO 2014 train images
    """

    def __init__(
            self,
            refcoco_root: str,
            coco_image_root: str,
            processor,
            dataset_names: Optional[List[str]] = None,
            split: str = "train",
            for_generation: bool = False,
            image_size: int = 448,
            mask_size: int = 256,
            seg_token_idx: int = None,
    ):
        self.processor = processor
        self.image_size = image_size
        self.mask_size = mask_size
        self.seg_token_idx = seg_token_idx
        self.split = split
        self.for_generation = for_generation

        if dataset_names is None:
            dataset_names = ["refcoco", "refcoco+", "refcocog"]
        self.samples = []
        for name in dataset_names:
            self._load_dataset(refcoco_root, coco_image_root, name, split)

        if _is_main():
            mode_str = "generation" if for_generation else "training"
            print(f"[Dataset] split={split} mode={mode_str}: {len(self.samples)} samples "
                  f"from {dataset_names}")

    # ── 数据加载 ──────────────────────────────────────────────────────────────

    def _load_dataset(self, root, image_root, name, split):
        try:
            import pickle
            dataset_dir = os.path.join(root, name)

            refs_file = os.path.join(dataset_dir, "refs(unc).p")
            if not os.path.exists(refs_file):
                refs_file = os.path.join(dataset_dir, "refs(umd).p")
            if not os.path.exists(refs_file):
                refs_file = os.path.join(dataset_dir, "refs(google).p")

            with open(refs_file, "rb") as f:
                refs = pickle.load(f)
            with open(os.path.join(dataset_dir, "instances.json"), "r") as f:
                instances = json.load(f)

            ann_map = {ann["id"]: ann for ann in instances["annotations"]}
            img_map = {img["id"]: img for img in instances["images"]}

            for ref in refs:
                if ref["split"] not in self._get_splits(split):
                    continue
                ann = ann_map.get(ref["ann_id"])
                img_info = img_map.get(ref["image_id"])
                if ann is None or img_info is None:
                    continue
                for sent in ref["sentences"]:
                    self.samples.append({
                        "image_path": os.path.join(image_root, img_info["file_name"]),
                        "description": sent["sent"].strip(),
                        "segmentation": ann["segmentation"],
                        "bbox": ann.get("bbox"),  # [x, y, w, h] COCO 格式，绝对像素坐标
                        "image_h": img_info["height"],
                        "image_w": img_info["width"],
                    })
        except Exception as e:
            if _is_main():
                print(f"[Dataset] Failed to load {name}: {e}")

    def _get_splits(self, split):
        if split == "train":
            return ["train"]
        elif split == "val":
            return ["val"]
        else:
            return [split]

    # ── __getitem__ ───────────────────────────────────────────────────────────

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # ── 加载图像 ──────────────────────────────────────────────────────────
        image = Image.open(sample["image_path"]).convert("RGB")
        orig_w, orig_h = image.size

        # CNN bypass 输入：1024×1024, ImageNet 归一化
        cnn_img = image.resize((_CNN_SIZE, _CNN_SIZE), Image.BILINEAR)
        cnn_tensor = torch.from_numpy(np.array(cnn_img)).float() / 255.0
        cnn_tensor = cnn_tensor.permute(2, 0, 1)  # (3, H, W)
        cnn_tensor = (cnn_tensor - _CNN_MEAN) / _CNN_STD  # ImageNet 归一化

        # ── 生成 GT mask ──────────────────────────────────────────────────────
        gt_mask = self._decode_mask(sample["segmentation"], orig_h, orig_w)
        gt_mask = cv2.resize(
            gt_mask.astype(np.uint8),
            (self.mask_size, self.mask_size),
            interpolation=cv2.INTER_NEAREST,
        )
        gt_mask = torch.from_numpy(gt_mask).float()

        # ── Box prompt：从标注计算两种格式的 bbox ────────────────────────────
        bbox_raw = sample.get("bbox")  # [x, y, w, h] 绝对像素
        if bbox_raw is not None and len(bbox_raw) == 4:
            bx, by, bw, bh = bbox_raw
            x1 = max(0.0, float(bx))
            y1 = max(0.0, float(by))
            x2 = min(float(orig_w), float(bx + bw))
            y2 = min(float(orig_h), float(by + bh))
        else:
            # 从 mask 计算 AABB 作为后备
            ys, xs = np.where(gt_mask.numpy() > 0.5)
            if len(xs) > 0:
                x1, y1 = float(xs.min()), float(ys.min())
                x2, y2 = float(xs.max() + 1), float(ys.max() + 1)
                # gt_mask 已经是 mask_size 尺度，需换算回原图尺度
                x1 = x1 / self.mask_size * orig_w
                y1 = y1 / self.mask_size * orig_h
                x2 = x2 / self.mask_size * orig_w
                y2 = y2 / self.mask_size * orig_h
            else:
                x1, y1, x2, y2 = 0.0, 0.0, float(orig_w), float(orig_h)

        # Qwen3-VL 格式：0-1000 相对坐标（用于 LM 训练目标）
        bx1 = round(x1 / orig_w * 1000)
        by1 = round(y1 / orig_h * 1000)
        bx2 = round(x2 / orig_w * 1000)
        by2 = round(y2 / orig_h * 1000)

        # SAM PromptEncoder 格式：1024×1024 绝对坐标（供 mask decoder 使用）
        gt_box = torch.tensor([
            x1 / orig_w * 1024, y1 / orig_h * 1024,
            x2 / orig_w * 1024, y2 / orig_h * 1024,
        ], dtype=torch.float32)  # (4,)

        # ── 构建对话 ──────────────────────────────────────────────────────────
        description = sample["description"]
        question = random.choice(QUESTION_TEMPLATES).format(description=description)
        # bbox_2d 前缀让 LLM 学习输出定位坐标（Qwen3-VL 风格），[SEG] 用于 mask 解码
        bbox_prefix = f'{{"bbox_2d": [{bx1}, {by1}, {bx2}, {by2}]}}\n'
        answer = bbox_prefix + random.choice(ANSWER_TEMPLATES)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": answer}],
            },
        ]

        # ── Chat template ─────────────────────────────────────────────────────
        # for_generation=False（训练）：完整对话，labels 覆盖 assistant 回答
        # for_generation=True （推理）：仅 user，让模型自由生成
        if not self.for_generation:
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=False,
            )
        else:
            text = self.processor.apply_chat_template(
                [messages[0]],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )

        # ── Processor ─────────────────────────────────────────────────────────
        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            max_pixels=self.image_size * self.image_size,
        )

        input_ids = inputs["input_ids"].squeeze(0)  # (seq_len,)
        attention_mask = inputs["attention_mask"].squeeze(0)  # (seq_len,)
        labels = self._build_labels(input_ids)

        # pixel_values: (total_patches, C*temporal*pH*pW)
        # Qwen processor 输出的 pixel_values 没有 batch 维——dim-0 是 total_patches。
        # 直接赋值；collate_fn 用 cat(dim=0) 拼接各样本的 patches。
        pixel_values = inputs["pixel_values"]

        # image_grid_thw: 单图时 (1, 3) → squeeze(0) → (3,)
        image_grid_thw = inputs["image_grid_thw"].squeeze(0)

        # mm_token_type_ids: (seq_len,)，标记每个 token 的模态（0=text,1=image,2=video）
        # 用于 M-RoPE 的 3D 位置编码计算，processor 在有图像时必定返回此字段
        mm_token_type_ids = None
        if "mm_token_type_ids" in inputs:
            mm_token_type_ids = inputs["mm_token_type_ids"].squeeze(0)

        item = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "cnn_image": cnn_tensor,  # (3, 1024, 1024), float32
            "labels": labels,
            "gt_masks": gt_mask,
            "gt_boxes": gt_box,       # (4,) [x1,y1,x2,y2] in SAM 1024 space
            "image_path": sample["image_path"],
        }
        if mm_token_type_ids is not None:
            item["mm_token_type_ids"] = mm_token_type_ids

        return item

    # ── 工具方法 ──────────────────────────────────────────────────────────────

    def _decode_mask(self, segmentation, h, w) -> np.ndarray:
        mask = np.zeros((h, w), dtype=np.uint8)
        if isinstance(segmentation, list):
            for poly in segmentation:
                pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
                cv2.fillPoly(mask, [pts], 1)
        elif isinstance(segmentation, dict):
            rle = segmentation
            if isinstance(rle.get("counts"), list):
                rle = mask_utils.frPyObjects(rle, h, w)
            mask = mask_utils.decode(rle)
        return mask

    def _build_labels(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        只让 assistant 回答部分（<|im_start|>assistant ... <|im_end|>）参与 loss，
        其余填 -100。
        for_generation=True 时 input_ids 里没有 assistant 块，结果全是 -100，正确。
        """
        labels = input_ids.clone()
        labels[:] = -100

        im_start_id = self.processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
        im_end_id = self.processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
        assistant_id = self.processor.tokenizer.encode(
            "assistant", add_special_tokens=False
        )[0]

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


# =============================================================================
# collate_fn：左填充，训练和验证统一使用
#
# 左填充原因：
#   generate() 要求 batch 内所有序列右端对齐（新 token 紧接真实内容生成）；
#   forward()  左右填充均可（attention_mask 屏蔽 padding）。
#   统一左填充 → 训练/验证共用同一 collate，不需要维护两套逻辑。
#
# mm_token_type_ids 以 0（text 类型）填充左侧，语义上"padding 位置是文本"。
# pad_token_id 建议传入 processor.tokenizer.pad_token_id（Qwen 默认为 eos_token_id）。
# =============================================================================

def collate_fn(batch: List[Dict], pad_token_id: int = 0) -> Dict:
    """
    左填充 collate，训练和验证共用。

    参数：
        batch        : Dataset.__getitem__ 返回的样本列表
        pad_token_id : tokenizer 的 pad token ID，用于填充 input_ids
    """
    max_len = max(x["input_ids"].shape[0] for x in batch)
    has_mm_ttids = all("mm_token_type_ids" in x for x in batch)

    input_ids_list, attn_mask_list, labels_list, mm_ttids_list = [], [], [], []

    for x in batch:
        pad_len = max_len - x["input_ids"].shape[0]
        input_ids_list.append(torch.cat([
            torch.full((pad_len,), pad_token_id, dtype=torch.long),
            x["input_ids"],
        ]))
        attn_mask_list.append(torch.cat([
            torch.zeros(pad_len, dtype=torch.long),
            x["attention_mask"],
        ]))
        labels_list.append(torch.cat([
            torch.full((pad_len,), -100, dtype=torch.long),
            x["labels"],
        ]))
        if has_mm_ttids:
            mm_ttids_list.append(torch.cat([
                torch.zeros(pad_len, dtype=torch.long),  # 0 = text
                x["mm_token_type_ids"],
            ]))

    result = {
        "input_ids": torch.stack(input_ids_list),
        "attention_mask": torch.stack(attn_mask_list),
        # pixel_values：各样本 patch 数不同，cat 在 dim=0 拼接
        "pixel_values": torch.cat([x["pixel_values"] for x in batch], dim=0),
        # image_grid_thw：每样本 (3,) → unsqueeze → cat → (B, 3)
        "image_grid_thw": torch.cat(
            [x["image_grid_thw"].unsqueeze(0) for x in batch], dim=0
        ),
        # cnn_image：每样本 (3, 1024, 1024)，直接 stack → (B, 3, 1024, 1024)
        "cnn_images": torch.stack([x["cnn_image"] for x in batch]),
        "labels": torch.stack(labels_list),
        "gt_masks": torch.stack([x["gt_masks"] for x in batch]),
        "gt_boxes": torch.stack([x["gt_boxes"] for x in batch]),  # (B, 4)
        "image_path": [x["image_path"] for x in batch],
    }
    if has_mm_ttids:
        result["mm_token_type_ids"] = torch.stack(mm_ttids_list)

    return result
