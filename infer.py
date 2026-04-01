# =============================================================================
# infer.py
# 交互式推理脚本 —— 模型只加载一次，循环处理多张图像
# 用法：
#   python infer.py [--ckpt path] [--vis_dir path]
#   启动后按提示输入图像路径和查询；输入 q / quit / exit 或按 Ctrl+C 退出
# =============================================================================
import argparse
import os
import re
try:
    import readline  # noqa: F401 — 导入即可，让 input() 支持退格/方向键等行编辑
except ImportError:
    pass  # Windows 下无此模块，忽略

import cv2
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
fm.fontManager.addfont(font_path)
prop = fm.FontProperties(fname=font_path)

# 用实际加载到的字体名（JP也可以正常显示中文，字形是一样的）
plt.rcParams['font.sans-serif'] = [prop.get_name()] + plt.rcParams['font.sans-serif']
plt.rcParams['axes.unicode_minus'] = False

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

from peft import set_peft_model_state_dict
from transformers import AutoProcessor
from model.qseg import QSegModel, SEG_TOKEN
from model.cnn_bypass import CNNBypass

# CNN bypass 归一化参数
_CNN_SIZE = CNNBypass.INPUT_SIZE
_CNN_MEAN = torch.tensor(CNNBypass.MEAN).view(3, 1, 1)
_CNN_STD = torch.tensor(CNNBypass.STD).view(3, 1, 1)

# ── 配置 ──────────────────────────────────────────────────────────────────────
QWEN_PATH = "/chenmei/Models/Qwen/Qwen3.5-9B"
SAM2_CKPT = "/chenmei/Models/SAM2/sam2.1_hiera_tiny.pt"
CKPT_PATH = "/chenmei/Projects/Qwen3Seg/outputs/896px_cnn_query_box_single_row/best_model/qseg_weights.pt"
VIS_DIR = "infer_vis"


def load_model(ckpt_path):
    processor = AutoProcessor.from_pretrained(QWEN_PATH)
    processor.tokenizer.add_tokens([SEG_TOKEN])
    seg_token_idx = processor.tokenizer.convert_tokens_to_ids(SEG_TOKEN)
    print(f"seg_token_idx={seg_token_idx}")

    model = QSegModel(
        qwen_model_path=QWEN_PATH,
        sam2_ckpt_path=SAM2_CKPT,
        seg_token_idx=seg_token_idx,
    ).eval().cuda()

    if ckpt_path and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        # lora key 里自动包含 TrainableTokens delta（embed/lm_head [SEG] 行增量）
        set_peft_model_state_dict(model.qwen, ckpt["lora"])
        model.neck.load_state_dict(ckpt["neck"])
        # 兼容三代 checkpoint 键名
        if "context_query_extractor" in ckpt:
            model.context_query_extractor.load_state_dict(ckpt["context_query_extractor"])
        elif "query_extractor" in ckpt:
            print("[WARN] Checkpoint uses old query_extractor key; context_query_extractor uses random weights.")
        elif "seg_projector" in ckpt:
            print("[WARN] Checkpoint uses old seg_projector; context_query_extractor uses random weights.")
        model.mask_decoder.load_state_dict(ckpt["mask_decoder"])
        if "prompt_encoder" in ckpt:
            model.prompt_encoder.load_state_dict(ckpt["prompt_encoder"])
        if "cnn_bypass" in ckpt:
            model.cnn_bypass.load_state_dict(ckpt["cnn_bypass"])

        # 向后兼容：旧格式 checkpoint 有单独的 embed/lm_head 键，手动写回
        lm = model.qwen.base_model.model.model.language_model
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


@torch.no_grad()
def infer_single(model, processor, image: Image.Image, query: str):
    """对单张图推理，返回 pred_mask (H,W) numpy array（0~1 float）"""
    device = model.get_main_device()

    # Qwen 多模态输入
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": query},
        ],
    }]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    inputs = processor(text=[text], images=[image], return_tensors="pt")

    # CNN bypass 输入：1024×1024, ImageNet 归一化
    cnn_img = image.resize((_CNN_SIZE, _CNN_SIZE), Image.BILINEAR)
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

    response = outputs["generated_text"][0]
    seg_found = outputs["seg_found"][0]
    pred_mask = torch.sigmoid(outputs["pred_masks"][0].float()).cpu().numpy()
    iou_pred = outputs["iou_predictions"][0].float().cpu().item()

    print(f"  模型回答: {_clean_response(response)}")
    if not seg_found:
        print("  [WARN] 模型未生成 [SEG]，已 fallback 到最后一步 hidden state")

    return pred_mask, iou_pred, response


def visualize(image: Image.Image, pred_mask: np.ndarray, iou_pred: float,
              query: str, response: str, save_path: str):
    img_np = np.array(image.convert("RGB"))
    h, w = img_np.shape[:2]

    pred_resized = cv2.resize(pred_mask, (w, h), interpolation=cv2.INTER_LINEAR)
    pred_binary = (pred_resized > 0.5).astype(np.uint8)

    overlay = img_np.copy()
    overlay[pred_binary == 1] = (
            overlay[pred_binary == 1] * 0.4 + np.array([255, 60, 60]) * 0.6
    ).astype(np.uint8)
    bbox = _parse_bbox(response, w, h)
    if bbox is not None:
        cv2.rectangle(overlay, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(img_np)
    axes[0].set_title("原图")
    axes[0].axis("off")
    axes[1].imshow(overlay)
    axes[1].set_title(f"分割叠加（iou_pred={iou_pred:.3f}）")
    axes[1].axis("off")
    axes[2].imshow(pred_resized, cmap="hot", vmin=0, vmax=1)
    axes[2].set_title("置信度热图")
    axes[2].axis("off")

    # 左对齐文字：Q 和 A 各一行，去掉 <think> 标签和多余空行
    clean_resp = _clean_response(response)
    q_str = f"Q: {query[:120]}"
    a_str = f"A: {clean_resp[:180]}"
    plt.suptitle(f"{q_str}\n{a_str}", fontsize=9, x=0.01, ha='left', y=1.03,
                 fontproperties=prop)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  保存到: {save_path}")


# Qwen3 关闭 thinking 模式时仍输出空 <think></think> 标签（正常现象）
# <think>/<think> 在部分版本不在 skip_special_tokens 列表里，需手动清除
_THINK_RE = re.compile(r'<think>.*?</think>', re.DOTALL)
_BBOX_RE = re.compile(r'"bbox_2d"\s*:\s*\[([^\]]+)\]')


def _clean_response(text: str) -> str:
    """去掉 <think>…</think> 块并规范空白，用于打印和可视化显示。"""
    text = _THINK_RE.sub('', text)
    text = re.sub(r'\n{2,}', '\n', text)
    return text.strip()


def _parse_bbox(response: str, img_w: int, img_h: int):
    """从回答中解析 bbox_2d（Qwen 0-1000 坐标），返回像素 (x1,y1,x2,y2) 或 None。"""
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


def _prompt(msg: str) -> str:
    """带颜色的输入提示。

    readline 已在文件顶部导入，在 Linux 下会自动处理退格/删除等行编辑；
    此处额外清理 \\x08（Backspace）和 \\x7f（DEL）防止非标准终端传入残留控制字符。
    """
    try:
        raw = input(f"\001\033[96m\002{msg}\001\033[0m\002")
        # 处理少数终端将退格/DEL 原样传入的情况
        while '\x08' in raw or '\x7f' in raw:
            raw = re.sub(r'[^\x08\x7f][\x08\x7f]', '', raw)  # 字符 + 退格 → 消除
            raw = re.sub(r'^[\x08\x7f]', '', raw)             # 行首孤立退格 → 消除
        return raw.strip()
    except EOFError:
        return "quit"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=CKPT_PATH)
    parser.add_argument("--vis_dir", type=str, default=VIS_DIR)
    args = parser.parse_args()

    os.makedirs(args.vis_dir, exist_ok=True)

    print("Loading model...")
    model, processor, _ = load_model(args.ckpt)
    print("\n\033[92m模型加载完成。输入 q / quit / exit 或按 Ctrl+C 退出。\033[0m")

    counter = 0
    try:
        while True:
            # ── 输入图像路径 ───────────────────────────────────────────────
            img_path = _prompt("\n图像路径 > ")
            if img_path.lower() in ("q", "quit", "exit", ""):
                break
            if not os.path.exists(img_path):
                print(f"  [ERR] 文件不存在: {img_path}")
                continue

            # ── 输入查询 ──────────────────────────────────────────────────
            query = _prompt("查    询 > ")
            if query.lower() in ("q", "quit", "exit"):
                break
            if not query:
                print("  [ERR] 查询不能为空")
                continue

            # ── 推理 ──────────────────────────────────────────────────────
            try:
                image = Image.open(img_path).convert("RGB")
                pred_mask, iou_pred, response = infer_single(
                    model, processor, image, query
                )
            except Exception as e:
                print(f"  [ERR] 推理失败: {e}")
                continue

            # ── 可视化 ────────────────────────────────────────────────────
            counter += 1
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            save_path = os.path.join(args.vis_dir, f"{counter:04d}_{img_name}_vis.png")
            visualize(image, pred_mask, iou_pred, query, response, save_path)

    except KeyboardInterrupt:
        pass

    print("\n\033[93m已退出。\033[0m")


if __name__ == "__main__":
    main()
