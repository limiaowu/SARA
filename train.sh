#!/bin/bash
# =============================================================================
# train.sh  —  多卡启动脚本
# 用法：CUDA_VISIBLE_DEVICES=4,5,6,7 bash train.sh
# =============================================================================

# ── 指定空闲的卡（根据 nvidia-smi 实际情况修改）──────────────────────────
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"1,2,3,5"}
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

echo "Using GPUs: $CUDA_VISIBLE_DEVICES  (${NUM_GPUS} cards)"

# ── 路径配置（按你的实际路径修改）───────────────────────────────────────
QWEN_PATH="/chenmei/Models/Qwen/Qwen3.5-9B"
SAM2_CKPT="/chenmei/Models/SAM2/sam2.1_hiera_tiny.pt"
REFCOCO_ROOT="/chenmei/Datasets/refcoco"
COCO_IMAGES="/chenmei/Datasets/coco/train2014"
BASE_OUTPUT_DIR="/chenmei/Projects/Qwen3Seg/outputs"
RUN_NAME="896px_context_query_full_lora"    # v3: ContextQueryExtractor + LoRA 覆盖全 32 层

OUTPUT_DIR="${BASE_OUTPUT_DIR}/${RUN_NAME}"
mkdir -p "$OUTPUT_DIR"

# ── 模型结构 ──────────────────────────────────────────────────────────────
SAM2_SIZE="tiny"              # tiny / large
HOOK_LAYERS=(3 6 13 20 26)  # Qwen ViT 特征抽取层，务必覆盖深层（语义关键）
NUM_QUERIES=16              # ContextQueryExtractor query 数（LENS 实验最优 64，我们用 16）

# ── LoRA ──────────────────────────────────────────────────────────────────
LORA_R=16
LORA_ALPHA=32
LORA_DROPOUT=0.05

# ── 数据 ──────────────────────────────────────────────────────────────────
IMAGE_SIZE=896   # processor max_pixels = IMAGE_SIZE^2
MASK_SIZE=512    # GT mask 及预测 mask 的边长

# ── 训练超参 ──────────────────────────────────────────────────────────────
NUM_EPOCHS=3
BATCH_SIZE=16        # 每卡 per-device batch size
GRAD_ACCUM=2        # 梯度累积步数
LR=1e-4
WARMUP_STEPS=200
WEIGHT_DECAY=0.0
MAX_GRAD_NORM=1.0
SAVE_STEPS=500
LOGGING_STEPS=10

# ── 验证 ──────────────────────────────────────────────────────────────────
EVAL_STEPS=500        # 每隔多少 optimizer step 做一次验证
EVAL_SUBSET_N=4000     # 验证总样本数（各卡均分，非每卡 500）
EVAL_BATCH_SIZE=8     # 验证时每卡 batch size

# ── 启动训练 ──────────────────────────────────────────────────────────────
deepspeed --master_port 29500 \
    --include "localhost:${CUDA_VISIBLE_DEVICES}" \
    train.py \
    --qwen_model_path $QWEN_PATH \
    --sam2_ckpt_path $SAM2_CKPT \
    --sam2_model_size $SAM2_SIZE \
    --refcoco_root $REFCOCO_ROOT \
    --coco_image_root $COCO_IMAGES \
    --base_output_dir $BASE_OUTPUT_DIR \
    --run_name $RUN_NAME \
    --dataset_names refcoco refcoco+ refcocog \
    --hook_layers "${HOOK_LAYERS[@]}" \
    --num_queries $NUM_QUERIES \
    --image_size $IMAGE_SIZE \
    --mask_size $MASK_SIZE \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT \
    --num_epochs $NUM_EPOCHS \
    --batch_size $BATCH_SIZE \
    --grad_accum $GRAD_ACCUM \
    --lr $LR \
    --warmup_steps $WARMUP_STEPS \
    --weight_decay $WEIGHT_DECAY \
    --max_grad_norm $MAX_GRAD_NORM \
    --save_steps $SAVE_STEPS \
    --logging_steps $LOGGING_STEPS \
    --eval_steps $EVAL_STEPS \
    --eval_subset_n $EVAL_SUBSET_N \
    --eval_batch_size $EVAL_BATCH_SIZE \
    --deepspeed ds_config.json \
    2>&1 | tee $OUTPUT_DIR/train.log
