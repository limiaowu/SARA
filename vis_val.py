# =============================================================================
# vis_val.py
# 随机可视化验证集样本，对比 GT 掩码与预测掩码
# 用法：
#   python vis_val.py [--ckpt path] [--output_dir path] [--n 20]
#                     [--datasets refcoco refcoco+ refcocog]
# 输出每张图为四格：原图 | GT掩码叠加 | 预测掩码叠加 | 置信度热图
# GT框绿色，预测框橙色
# =============================================================================
import argparse
import json
import os
import pickle
import random
import re

import cv2
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_utils

font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
fm.fontManager.addfont(font_path)
prop = fm.FontProperties(fname=font_path)
plt.rcParams['font.sans-serif'] = [prop.get_name()] + plt.rcParams['font.sans-serif']
plt.rcParams['axes.unicode_minus'] = False

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

from peft import set_peft_model_state_dict
from transformers import AutoProcessor
from model.qseg import QSegModel, SEG_TOKEN
from model.cnn_bypass import CNNBypass

# ── 路径配置（与 infer.py 保持一致）────────────────────────────────────────
QWEN_PATH     = "/chenmei/Models/Qwen/Qwen3.5-9B"
SAM2_CKPT     = "/chenmei/Models/SAM2/sam2.1_hiera_tiny.pt"
CKPT_PATH     = "/chenmei/Projects/Qwen3Seg/outputs/896px_cnn_query_box_single_row/best_model/qseg_weights.pt"
REFCOCO_ROOT  = "/chenmei/Datasets/refcoco"
COCO_IMAGES   = "/chenmei/Datasets/coco/train2014"
OUTPUT_DIR    = "/chenmei/Projects/Qwen3Seg/vis_val"

# CNN bypass 归一化参数
_CNN_SIZE = CNNBypass.INPUT_SIZE
_CNN_MEAN = torch.tensor(CNNBypass.MEAN).view(3, 1, 1)
_CNN_STD  = torch.tensor(CNNBypass.STD).view(3, 1, 1)

_THINK_RE = re.compile(r'<think>.*?</think>', re.DOTALL)
_BBOX_RE  = re.compile(r'"bbox_2d"\s*:\s*\[([^]]+)\]')


def _clean_response(text: str) -> str:
    text = _THINK_RE.sub('', text)
    return re.sub(r'\n{2,}', '\n', text).strip()


def _parse_bbox(response: str, img_w: int, img_h: int):
    """Qwen 0-1000 坐标 → 像素 (x1,y1,x2,y2)，解析失败返回 None。"""
    m = _BBOX_RE.search(response)
    if m is None:
        return None
    try:
        coords = [int(x.strip()) for x in m.group(1).split(',')]
        if len(coords) != 4:
            return None
        x1 = round(coords[0] / 1000 * img_w)
        y1 = round(coords[1] / 1000 * img_h)
        x2 = round(coords[2] / 1000 * img_w)
        y2 = round(coords[3] / 1000 * img_h)
        return x1, y1, x2, y2
    except Exception:
        return None


def _decode_mask(segmentation, h: int, w: int) -> np.ndarray:
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


# ── 数据加载 ──────────────────────────────────────────────────────────────

def load_val_samples(refcoco_root: str, coco_images: str,
                     dataset_names: list, n: int) -> list:
    """从各数据集的 val split 加载原始样本，随机采样 n 条。"""
    all_samples = []
    for name in dataset_names:
        dataset_dir = os.path.join(refcoco_root, name)
        refs_file = next(
            (os.path.join(dataset_dir, f)
             for f in ["refs(unc).p", "refs(umd).p", "refs(google).p"]
             if os.path.exists(os.path.join(dataset_dir, f))),
            None
        )
        if refs_file is None:
            print(f"[WARN] refs file not found for {name}, skipped")
            continue
        with open(refs_file, "rb") as f:
            refs = pickle.load(f)
        with open(os.path.join(dataset_dir, "instances.json"), "r") as f:
            instances = json.load(f)
        ann_map = {a["id"]: a for a in instances["annotations"]}
        img_map = {i["id"]: i for i in instances["images"]}
        for ref in refs:
            if ref["split"] != "val":
                continue
            ann      = ann_map.get(ref["ann_id"])
            img_info = img_map.get(ref["image_id"])
            if ann is None or img_info is None:
                continue
            for sent in ref["sentences"]:
                all_samples.append({
                    "image_path":  os.path.join(coco_images, img_info["file_name"]),
                    "description": sent["sent"].strip(),
                    "segmentation": ann["segmentation"],
                    "bbox":         ann.get("bbox"),   # [x,y,w,h] COCO 格式
                    "image_h":      img_info["height"],
                    "image_w":      img_info["width"],
                    "dataset":      name,
                })
    random.shuffle(all_samples)
    return all_samples[:n]


# ── 模型加载（与 infer.py 逻辑相同）────────────────────────────────────────

def load_model(ckpt_path: str):
    processor = AutoProcessor.from_pretrained(QWEN_PATH)
    processor.tokenizer.add_tokens([SEG_TOKEN])
    seg_token_idx = processor.tokenizer.convert_tokens_to_ids(SEG_TOKEN)

    model = QSegModel(
        qwen_model_path=QWEN_PATH,
        sam2_ckpt_path=SAM2_CKPT,
        seg_token_idx=seg_token_idx,
    ).eval().cuda()

    if ckpt_path and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        set_peft_model_state_dict(model.qwen, ckpt["lora"])
        model.neck.load_state_dict(ckpt["neck"])
        if "context_query_extractor" in ckpt:
            model.context_query_extractor.load_state_dict(ckpt["context_query_extractor"])
        elif "query_extractor" in ckpt:
            print("[WARN] Old query_extractor key; context_query_extractor uses random weights.")
        model.mask_decoder.load_state_dict(ckpt["mask_decoder"])
        if "prompt_encoder" in ckpt:
            model.prompt_encoder.load_state_dict(ckpt["prompt_encoder"])
        if "cnn_bypass" in ckpt:
            model.cnn_bypass.load_state_dict(ckpt["cnn_bypass"])

        lm      = model.qwen.base_model.model.model.language_model
        lm_head = model.qwen.base_model.model.lm_head
        seg_idx = ckpt.get("seg_token_idx", seg_token_idx)
        if "embed_tokens_seg_row" in ckpt:
            with torch.no_grad():
                lm.embed_tokens.weight[seg_idx] = ckpt["embed_tokens_seg_row"].to(
                    device=lm.embed_tokens.weight.device, dtype=torch.bfloat16)
        elif "embed_tokens" in ckpt:
            lm.embed_tokens.load_state_dict(
                {k: v.bfloat16() for k, v in ckpt["embed_tokens"].items()})
        if "lm_head_seg_row" in ckpt:
            with torch.no_grad():
                lm_head.weight[seg_idx] = ckpt["lm_head_seg_row"].to(
                    device=lm_head.weight.device, dtype=torch.bfloat16)
        elif "lm_head" in ckpt:
            lm_head.load_state_dict(
                {k: v.bfloat16() for k, v in ckpt["lm_head"].items()})
        print(f"Loaded checkpoint: {ckpt_path}")
    else:
        print(f"[WARN] No checkpoint at {ckpt_path}, using random weights.")

    return model, processor, seg_token_idx


# ── 推理 ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def infer_single(model, processor, image: Image.Image, query: str):
    device = model.get_main_device()

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text",  "text":  query},
        ],
    }]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    inputs = processor(text=[text], images=[image], return_tensors="pt")

    cnn_img    = image.resize((_CNN_SIZE, _CNN_SIZE), Image.BILINEAR)
    cnn_tensor = torch.from_numpy(np.array(cnn_img)).float() / 255.0
    cnn_tensor = (cnn_tensor.permute(2, 0, 1) - _CNN_MEAN) / _CNN_STD
    cnn_images = cnn_tensor.unsqueeze(0).to(device=device, dtype=torch.bfloat16)

    outputs = model.generate_with_mask(
        input_ids=inputs["input_ids"].to(device),
        attention_mask=inputs["attention_mask"].to(device),
        pixel_values=inputs["pixel_values"].to(device),
        image_grid_thw=inputs["image_grid_thw"].to(device),
        cnn_images=cnn_images,
        original_size=(512, 512),
        max_new_tokens=64,
        tokenizer=processor.tokenizer,
    )
    pred_mask = torch.sigmoid(outputs["pred_masks"][0].float()).cpu().numpy()
    iou_pred  = outputs["iou_predictions"][0].float().cpu().item()
    response  = outputs["generated_text"][0]
    return pred_mask, iou_pred, response


# ── 可视化 ────────────────────────────────────────────────────────────────

def visualize_sample(
        image: Image.Image,
        gt_mask: np.ndarray,
        pred_mask: np.ndarray,
        gt_box_px,            # (x1,y1,x2,y2) 像素坐标或 None
        pred_box_px,          # (x1,y1,x2,y2) 像素坐标或 None
        iou_pred: float,
        query: str,
        response: str,
        save_path: str,
) -> float:
    """四格可视化：原图 | GT叠加 | 预测叠加 | 热图。返回实际 IoU。"""
    img_np = np.array(image.convert("RGB"))
    h, w   = img_np.shape[:2]

    pred_resized = cv2.resize(pred_mask, (w, h), interpolation=cv2.INTER_LINEAR)
    pred_binary  = (pred_resized > 0.5).astype(np.uint8)
    gt_resized   = cv2.resize(gt_mask.astype(np.float32), (w, h),
                               interpolation=cv2.INTER_NEAREST)
    gt_binary    = (gt_resized > 0.5).astype(np.uint8)

    # GT 叠加（蓝色）+ GT 框（绿色）
    gt_overlay = img_np.copy()
    gt_overlay[gt_binary == 1] = (
        gt_overlay[gt_binary == 1] * 0.4 + np.array([60, 60, 255]) * 0.6
    ).astype(np.uint8)
    if gt_box_px is not None:
        x1, y1, x2, y2 = gt_box_px
        cv2.rectangle(gt_overlay, (x1, y1), (x2, y2), (0, 220, 0), 2)

    # 预测叠加（红色）+ 预测框（橙色）
    pred_overlay = img_np.copy()
    pred_overlay[pred_binary == 1] = (
        pred_overlay[pred_binary == 1] * 0.4 + np.array([255, 60, 60]) * 0.6
    ).astype(np.uint8)
    if pred_box_px is not None:
        x1, y1, x2, y2 = pred_box_px
        cv2.rectangle(pred_overlay, (x1, y1), (x2, y2), (255, 165, 0), 2)

    # 计算实际 IoU
    inter = int((pred_binary & gt_binary).sum())
    union = int((pred_binary | gt_binary).sum())
    iou   = inter / union if union > 0 else 0.0

    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    axes[0].imshow(img_np)
    axes[0].set_title("原图")
    axes[0].axis("off")

    axes[1].imshow(gt_overlay)
    axes[1].set_title("GT 掩码（蓝）+ GT 框（绿）")
    axes[1].axis("off")

    axes[2].imshow(pred_overlay)
    axes[2].set_title(f"预测掩码（红）+ 预测框（橙）\nIoU={iou:.3f}  iou_pred={iou_pred:.3f}")
    axes[2].axis("off")

    axes[3].imshow(pred_resized, cmap="hot", vmin=0, vmax=1)
    axes[3].set_title("置信度热图")
    axes[3].axis("off")

    clean = _clean_response(response)
    q_str = f"Q: {query[:120]}"
    a_str = f"A: {clean[:200]}"
    plt.suptitle(f"{q_str}\n{a_str}", fontsize=9, x=0.01, ha='left', y=1.03,
                 fontproperties=prop)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    return iou


# ── 主程序 ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Visualize val set predictions")
    parser.add_argument("--ckpt",       type=str,   default=CKPT_PATH)
    parser.add_argument("--output_dir", type=str,   default=OUTPUT_DIR)
    parser.add_argument("--n",          type=int,   default=20,
                        help="Number of samples to visualize")
    parser.add_argument("--datasets",   nargs="+",
                        default=["refcoco", "refcoco+", "refcocog"])
    parser.add_argument("--seed",       type=int,   default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading model...")
    model, processor, _ = load_model(args.ckpt)

    print(f"Sampling {args.n} val examples from {args.datasets}...")
    samples = load_val_samples(REFCOCO_ROOT, COCO_IMAGES, args.datasets, args.n)
    if not samples:
        print("[ERR] No samples loaded.")
        return
    print(f"Got {len(samples)} samples. Running inference...")

    ious = []
    for i, sample in enumerate(samples):
        image = Image.open(sample["image_path"]).convert("RGB")
        img_w, img_h = image.size
        query = sample["description"]

        # GT mask（原始尺寸）
        gt_mask = _decode_mask(sample["segmentation"], sample["image_h"], sample["image_w"])

        # GT bbox → 像素坐标
        gt_box_px = None
        raw_bbox  = sample.get("bbox")
        if raw_bbox and len(raw_bbox) == 4:
            bx, by, bw, bh = raw_bbox
            gt_box_px = (int(bx), int(by), int(bx + bw), int(by + bh))

        # 推理
        try:
            pred_mask, iou_pred, response = infer_single(model, processor, image, query)
        except Exception as e:
            print(f"  [{i+1}/{len(samples)}] ERROR: {e}")
            continue

        pred_box_px = _parse_bbox(response, img_w, img_h)

        save_path = os.path.join(args.output_dir, f"{i+1:04d}_vis.png")
        iou = visualize_sample(
            image, gt_mask, pred_mask,
            gt_box_px, pred_box_px,
            iou_pred, query, response, save_path,
        )
        ious.append(iou)
        clean = _clean_response(response)[:60]
        print(f"  [{i+1:>3}/{len(samples)}] IoU={iou:.3f}  A: {clean}")

    if ious:
        print(f"\nmIoU over {len(ious)} visualized samples: {np.mean(ious):.4f}")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()