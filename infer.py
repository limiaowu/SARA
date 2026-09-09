"""Run SARA on one image or in an interactive loop."""

import argparse
import os
import re
from datetime import datetime

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps
from transformers import AutoProcessor

from model.cnn_bypass import CNNBypass
from model.sara import SARAModel, SEG_TOKEN


_CNN_SIZE = CNNBypass.INPUT_SIZE
_CNN_MEAN = torch.tensor(CNNBypass.MEAN).view(3, 1, 1)
_CNN_STD = torch.tensor(CNNBypass.STD).view(3, 1, 1)
_KNOWN_HOOK_CONFIGS = {
    5: [3, 6, 13, 20, 26],
    8: [1, 3, 6, 8, 13, 17, 21, 26],
}
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _clean_response(text: str) -> str:
    text = _THINK_RE.sub("", text)
    return re.sub(r"\n{2,}", "\n", text).strip()


def _infer_hook_layers(checkpoint: dict, fallback: list[int]) -> list[int]:
    if "hook_layers" in checkpoint:
        return list(checkpoint["hook_layers"])

    neck_state = checkpoint.get("neck", {})
    lateral_indices = {
        int(key.split(".")[1])
        for key in neck_state
        if key.startswith("laterals.")
    }
    return _KNOWN_HOOK_CONFIGS.get(len(lateral_indices), fallback)


def _infer_lora_rank(checkpoint: dict, fallback: int = 64) -> int:
    for key, value in checkpoint.get("lora", {}).items():
        if "lora_A" in key and getattr(value, "ndim", 0) == 2:
            return int(value.shape[0])
    return fallback


def load_model(args):
    if not os.path.isfile(args.ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")

    checkpoint = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    hook_layers = _infer_hook_layers(checkpoint, args.hook_layers)
    query_state = checkpoint.get("response_aggregation", {})
    num_queries = args.num_queries
    if num_queries is None and "queries" in query_state:
        num_queries = int(query_state["queries"].shape[0])
    num_queries = num_queries or 32

    lora_r = args.lora_r or _infer_lora_rank(checkpoint)
    lora_alpha = args.lora_alpha or 2 * lora_r

    processor = AutoProcessor.from_pretrained(
        args.qwen_model_path,
        max_pixels=args.image_size * args.image_size,
    )
    processor.tokenizer.add_tokens([SEG_TOKEN])
    seg_token_idx = processor.tokenizer.convert_tokens_to_ids(SEG_TOKEN)
    checkpoint_seg_idx = checkpoint.get("seg_token_idx")
    if checkpoint_seg_idx is not None and checkpoint_seg_idx != seg_token_idx:
        raise ValueError(
            "The [SEG] token ID differs from the checkpoint. "
            "Use the same base tokenizer that was used for training."
        )

    model = SARAModel(
        qwen_model_path=args.qwen_model_path,
        sam2_ckpt_path=args.sam2_ckpt,
        sam2_model_size=args.sam2_model_size,
        hook_layers=hook_layers,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        seg_token_idx=seg_token_idx,
        num_queries=num_queries,
        attn_implementation=args.attn_implementation,
        use_cnn_bypass=checkpoint.get("use_cnn_bypass", True),
    )
    model.load_trainable(checkpoint)
    return model.eval().to(args.device), processor


@torch.no_grad()
def infer_single(model, processor, image: Image.Image, query: str, args):
    device = model.get_main_device()
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": query},
        ],
    }]
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = processor(text=[prompt], images=[image], return_tensors="pt")

    cnn_image = image.resize((_CNN_SIZE, _CNN_SIZE), Image.Resampling.BILINEAR)
    cnn_tensor = torch.from_numpy(np.asarray(cnn_image).copy()).float() / 255.0
    cnn_tensor = (cnn_tensor.permute(2, 0, 1) - _CNN_MEAN) / _CNN_STD

    mm_token_type_ids = inputs.get("mm_token_type_ids")
    outputs = model.generate_with_mask(
        input_ids=inputs["input_ids"].to(device),
        attention_mask=inputs["attention_mask"].to(device),
        pixel_values=inputs["pixel_values"].to(device),
        image_grid_thw=inputs["image_grid_thw"].to(device),
        cnn_images=cnn_tensor.unsqueeze(0).to(device=device, dtype=torch.bfloat16),
        mm_token_type_ids=(
            mm_token_type_ids.to(device) if mm_token_type_ids is not None else None
        ),
        original_size=(args.mask_size, args.mask_size),
        max_new_tokens=args.max_new_tokens,
        tokenizer=processor.tokenizer,
    )

    mask = torch.sigmoid(outputs["pred_masks"][0].float()).cpu().numpy()
    return mask, _clean_response(outputs["generated_text"][0]), outputs["seg_found"][0]


def save_result(image, mask, query, response, seg_found, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    image_array = np.asarray(image.convert("RGB"))
    height, width = image_array.shape[:2]
    mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)
    binary = mask > 0.5

    Image.fromarray((binary.astype(np.uint8) * 255), mode="L").save(
        os.path.join(output_dir, "mask.png")
    )
    overlay = image_array.astype(np.float32)
    overlay[binary] = overlay[binary] * 0.45 + np.array([255, 64, 64]) * 0.55
    Image.fromarray(overlay.clip(0, 255).astype(np.uint8)).save(
        os.path.join(output_dir, "overlay.png")
    )
    with open(os.path.join(output_dir, "response.txt"), "w", encoding="utf-8") as handle:
        handle.write(f"Query: {query}\n\nResponse: {response}\n\n[SEG] found: {seg_found}\n")


def run_once(model, processor, image_path: str, query: str, args):
    image = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
    mask, response, seg_found = infer_single(model, processor, image, query, args)
    stem = os.path.splitext(os.path.basename(image_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"{stem}_{timestamp}")
    save_result(image, mask, query, response, seg_found, output_dir)
    print(f"Response: {response}")
    print(f"[SEG] found: {seg_found}")
    print(f"Saved to: {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="SARA image inference")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--qwen_model_path", required=True)
    parser.add_argument("--sam2_ckpt", required=True)
    parser.add_argument("--image", default=None)
    parser.add_argument("--query", default=None)
    parser.add_argument("--output_dir", default="./outputs/inference")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sam2_model_size", choices=["tiny", "large"], default="large")
    parser.add_argument("--hook_layers", type=int, nargs="+", default=[3, 6, 13, 20, 26])
    parser.add_argument("--num_queries", type=int, default=None)
    parser.add_argument("--lora_r", type=int, default=None)
    parser.add_argument("--lora_alpha", type=int, default=None)
    parser.add_argument("--image_size", type=int, default=1024)
    parser.add_argument("--mask_size", type=int, default=1024)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument(
        "--attn_implementation",
        choices=["flash_attention_2", "sdpa", "eager"],
        default="flash_attention_2",
    )
    args = parser.parse_args()
    if (args.image is None) != (args.query is None):
        parser.error("--image and --query must be provided together")
    return args


def main():
    args = parse_args()
    model, processor = load_model(args)

    if args.image is not None:
        run_once(model, processor, args.image, args.query, args)
        return

    print("Model loaded. Enter 'quit' to exit.")
    while True:
        image_path = input("Image path: ").strip()
        if image_path.lower() in {"q", "quit", "exit"}:
            break
        query = input("Query: ").strip()
        if query.lower() in {"q", "quit", "exit"}:
            break
        if not os.path.isfile(image_path) or not query:
            print("Provide an existing image path and a non-empty query.")
            continue
        run_once(model, processor, image_path, query, args)


if __name__ == "__main__":
    main()
