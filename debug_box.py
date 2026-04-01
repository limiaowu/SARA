"""
debug_box.py  ——  可视化数据集中的坐标对齐情况
用法：python debug_box.py --n 16 --out vis/debug_box
输出：每张图叠加 GT mask + GT box（0-1000 归一化框，还原为像素坐标）
"""
import argparse
import os
import pickle
import json
import random
import numpy as np
import cv2
from PIL import Image

# ─── 路径配置（按需修改）────────────────────────────────────────────────────
REFCOCO_ROOT   = "/chenmei/Datasets/refcoco"   # 包含 refcoco/refcoco+/refcocog 子目录
COCO_IMAGE_ROOT = "/chenmei/Datasets/coco/train2014"
DATASET_NAME   = "refcoco"  # "refcoco" / "refcoco+" / "refcocog"
SPLIT          = "val"
# ────────────────────────────────────────────────────────────────────────────


def load_samples(root, image_root, name, split, n):
    """加载 n 个随机样本（不依赖 dataset.py，裸读 pickle）"""
    from pycocotools import mask as mask_utils

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

    ann_map = {a["id"]: a for a in instances["annotations"]}
    img_map = {i["id"]: i for i in instances["images"]}

    valid = [r for r in refs if r["split"] == split and
             ann_map.get(r["ann_id"]) is not None and
             img_map.get(r["image_id"]) is not None]

    random.shuffle(valid)
    samples = []
    for ref in valid[:n]:
        ann = ann_map[ref["ann_id"]]
        img_info = img_map[ref["image_id"]]
        samples.append({
            "image_path": os.path.join(image_root, img_info["file_name"]),
            "description": ref["sentences"][0]["sent"].strip(),
            "segmentation": ann["segmentation"],
            "bbox": ann.get("bbox"),          # [x, y, w, h] 像素
            "image_h": img_info["height"],
            "image_w": img_info["width"],
        })
    return samples


def decode_mask(segmentation, h, w):
    from pycocotools import mask as mask_utils
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


def visualize_sample(sample, out_path):
    img = cv2.imread(sample["image_path"])
    if img is None:
        print(f"  [SKIP] 无法读取图像: {sample['image_path']}")
        return
    orig_h, orig_w = img.shape[:2]

    # ── GT Mask（红色半透明叠加）──────────────────────────────────────────
    gt_mask = decode_mask(sample["segmentation"], orig_h, orig_w)
    overlay = img.copy()
    overlay[gt_mask == 1] = (
        overlay[gt_mask == 1] * 0.4 + np.array([0, 0, 200]) * 0.6
    ).astype(np.uint8)

    # ── GT Box（COCO [x,y,w,h] → 像素绝对值 → 绿色矩形）─────────────────
    bbox_raw = sample.get("bbox")
    if bbox_raw and len(bbox_raw) == 4:
        bx, by, bw, bh = bbox_raw
        x1, y1 = int(bx), int(by)
        x2, y2 = int(bx + bw), int(by + bh)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # ── 同步验证 0-1000 坐标还原是否一致 ─────────────────────────────
        bx1_norm = round(x1 / orig_w * 1000)
        by1_norm = round(y1 / orig_h * 1000)
        bx2_norm = round(x2 / orig_w * 1000)
        by2_norm = round(y2 / orig_h * 1000)

        # 0-1000 → 还原回像素
        rx1 = int(bx1_norm / 1000 * orig_w)
        ry1 = int(by1_norm / 1000 * orig_h)
        rx2 = int(bx2_norm / 1000 * orig_w)
        ry2 = int(by2_norm / 1000 * orig_h)
        # 黄色虚线框（还原后坐标，与绿框几乎重合才说明没有坐标系混乱）
        cv2.rectangle(overlay, (rx1, ry1), (rx2, ry2), (0, 255, 255), 1)

        label_txt = (f"COCO:[{x1},{y1},{x2},{y2}]  "
                     f"0-1000:[{bx1_norm},{by1_norm},{bx2_norm},{by2_norm}]")
        cv2.putText(overlay, label_txt, (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)

    # ── SAM 1024 空间坐标（蓝色小标注，右下角）──────────────────────────
    if bbox_raw and len(bbox_raw) == 4:
        bx, by, bw, bh = bbox_raw
        x1_sam = x1 / orig_w * 1024
        y1_sam = y1 / orig_h * 1024
        x2_sam = x2 / orig_w * 1024
        y2_sam = y2 / orig_h * 1024
        sam_txt = f"SAM1024:[{x1_sam:.0f},{y1_sam:.0f},{x2_sam:.0f},{y2_sam:.0f}]"
        cv2.putText(overlay, sam_txt, (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 180, 0), 1, cv2.LINE_AA)

    # ── 描述文字 ──────────────────────────────────────────────────────────
    desc = sample["description"][:60]
    cv2.putText(overlay, desc, (10, orig_h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imwrite(out_path, overlay)
    print(f"  保存: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=16, help="可视化样本数")
    parser.add_argument("--out", type=str, default="vis/debug_box")
    parser.add_argument("--dataset", type=str, default=DATASET_NAME)
    parser.add_argument("--split", type=str, default=SPLIT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    print(f"加载 {args.dataset}/{args.split} 中 {args.n} 个样本...")
    samples = load_samples(REFCOCO_ROOT, COCO_IMAGE_ROOT,
                           args.dataset, args.split, args.n)
    print(f"实际加载 {len(samples)} 个样本")

    for i, sample in enumerate(samples):
        out_path = os.path.join(args.out, f"{i:04d}.jpg")
        visualize_sample(sample, out_path)

    print(f"\n完成，结果保存在: {args.out}")
    print("图例：")
    print("  绿色实线框 = COCO 原始 GT box（像素坐标）")
    print("  黄色细线框 = 0-1000 归一化再还原的坐标（应与绿框几乎重合）")
    print("  红色半透明 = GT mask")


if __name__ == "__main__":
    main()