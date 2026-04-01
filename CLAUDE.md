# CLAUDE.md

本文件是 Claude Code 的项目指导文档，新会话开始时请先读此文件。

## 项目概述

QSeg 是一个单塔式指代图像分割（Referring Image Segmentation）模型，使用 **Qwen3.5-9B** 作为视觉语言主干，结合 **SAM2 的 MaskDecoder**，通过自然语言驱动图像分割。

训练数据：RefCOCO / RefCOCO+ / RefCOCOg（约 32 万样本）。
目标环境：多卡 H20 + DeepSpeed（ZeRO stage 0），bfloat16。

---

## 常用命令

```bash
# 训练（服务器端执行）
CUDA_VISIBLE_DEVICES=4,5,6,7 bash train.sh

# 交互式推理（模型加载一次，循环输入；q / Ctrl+C 退出）
python infer.py [--ckpt /path/to/qseg_weights.pt] [--vis_dir /path/to/vis]

# 基础模型加载测试（裸 Qwen3.5，不含分割模块）
python test.py
```

`train.py` 主要参数：`--qwen_model_path`、`--sam2_ckpt_path`、`--refcoco_root`、`--coco_image_root`、`--base_output_dir`、`--run_name`、`--deepspeed ds_config.json`

---

## 整体架构

```
原图(1024×1024)
    └─ CNNBypass (stride=2/4/8, ~3.2M params)
         ├─ feat_s0: (B, 256, 256, 256)    → mask_decoder.conv_s0
         └─ feat_s1: (B, 256, 128, 128)    → mask_decoder.conv_s1

输入图像 → Qwen3.5 ViT（全量冻结，hook 第 [3,6,13,20,26] 层，共 27 层）
    └─ FPNNeck (top-down FPN) → image_embedding (B, 256, 64, 64)

文本查询 → Qwen3.5 LLM（LoRA 作用于 full_attention 层的 q/k/v/o_proj）
    └─ [SEG] token 的 last-layer hidden state (B, 4096)
         └─ QueryExtractor (4 可学习 query × cross-attention)
              └─ (B, 4, 256) sparse embeddings
                   └─ PromptEncoder(text_embeddings=…, boxes=…)
                        └─ sparse + dense prompts
                             └─ SAM2 MaskDecoder → 低分辨率 mask → 上采样至目标尺寸
```

---

## 关键设计决策

- **ViT 全量冻结**：`CNNBypass` 从原图（1024×1024）提供真实高频特征，替代解冻浅层的方案
- **CNNBypass**：7×7 stem + 两级 stride-2 + ResBlock，全 GroupNorm，bfloat16 兼容，约 3.2M 参数
- **FPNNeck**：hook 5 层 ViT 特征（pre-merger，hidden_size=1152），top-down FPN 融合后上采样到 64×64；用 bilinear+Conv2d 替代 ConvTranspose2d 消除棋盘格伪影
- **QueryExtractor**：4 个可学习 query 通过 cross-attention 从 [SEG] hidden state 提取多角度语义，输出 (B, 4, 256) 作为 sparse embeddings，替代旧版单个 seg_projector
- **PromptEncoder** 新增 `text_embeddings` 参数（LISA 风格），QueryExtractor 输出直接进入 sparse prompts；同时支持 `boxes` 参数（SAM 1024 空间）传入空间先验
- **Box prompt**：答案模板包含 `{"bbox_2d": [x1,y1,x2,y2]}`（Qwen3-VL 0-1000 相对坐标，无需 smart_resize）；训练时传 GT box，推理时从生成文本 regex 解析 box，均进入 PromptEncoder
- **`[SEG]` token**：通过 `processor.tokenizer.add_tokens()` 添加，embed_tokens 和 lm_head 全量可训练
- **mm_token_type_ids** 必须贯穿整个 pipeline，供 M-RoPE 3D 位置编码使用（Qwen3.5 强制要求，缺失会抛异常）
- **左填充（left-padding）**：collate_fn 统一左填充，批量 generate 时 M-RoPE 的必要条件
- **损失函数**：`LM_loss + 2.0 × mask_loss`，其中 `mask_loss = Focal + Dice + Boundary-weighted BCE + 0.5 × Sobel edge loss`
- **Qwen3.5 混合架构**：LLM 共 32 层，其中 8 层是 full_attention（有 q/k/v/o_proj），24 层是 linear_attention（GatedDeltaNet，有 in_proj_qkv/out_proj）。LoRA target_modules 只匹配 full_attention 的 8 层，PEFT 自动跳过其余层不报错
- **eos token 动态获取**：Qwen3.5 词表约 24 万，generate 时从 tokenizer 动态获取 eos_token_id 和 im_end_id，禁止硬编码

---

## 核心文件

| 文件 | 作用 |
|------|------|
| `model/qseg.py` | 主模型 `QSegModel`，包含 `QueryExtractor`、forward、generate_with_mask、save_trainable |
| `model/cnn_bypass.py` | `CNNBypass`，原图高分辨率特征提取（替代解冻 ViT 浅层的方案） |
| `model/adaptive_neck.py` | `FPNNeck`，ViT 多层特征 FPN 融合 → image_embedding (64×64) |
| `model/vision_neck.py` | `Qwen35VisionFeatureExtractor`，在 ViT block 上注册 forward hook |
| `sam_decoder/mask_decoder.py` | 修改版 SAM2 解码器（支持动态尺寸 + bfloat16 + high_res_features） |
| `sam_decoder/prompt_encoder.py` | SAM2 prompt 编码器，新增 `text_embeddings` 文本分支和 `boxes` 参数 |
| `data/dataset.py` | `RefCOCODataset`，每个样本返回 `cnn_image (3,1024,1024)` 和 `gt_boxes (4,)` |
| `train.py` | `QSegTrainer`（继承 HF Trainer）；`evaluate_subset_iou()` 用于子集推理验证 |
| `train.sh` | DeepSpeed 多卡启动脚本，所有超参集中定义于此 |
| `ds_config.json` | DeepSpeed ZeRO stage 0，bfloat16，梯度裁剪 1.0 |
| `infer.py` | 交互式推理脚本，模型加载一次后循环处理，Ctrl+C 或输入 q/quit/exit 退出 |
| `test.py` | 基础模型加载测试（裸 Qwen3.5，不含分割模块） |

**忽略目录**：`Qwen3.5/`（Qwen3.5 transformers 源码，仅供参考，不参与训练）、`infer_vis/`、`vis/`（输出图片）、`outputs/`（训练日志）、`test_images/`（测试图片）

---

## 数据集模式

`RefCOCODataset` 通过 `for_generation` 参数控制两种模式：
- `False`（训练）：完整对话模板（user + assistant）；labels 仅覆盖 assistant 回复部分；answer 格式为 `{"bbox_2d": [x1,y1,x2,y2]}\n` + 随机句式 + `[SEG]`
- `True`（推理/验证）：仅 user 消息 + `add_generation_prompt=True`；所有 labels 置为 -100

每个样本额外包含：
- `cnn_image (3,1024,1024)`：ImageNet 归一化，collate 后为 `cnn_images (B,3,1024,1024)`
- `gt_boxes (4,)`：`[x1,y1,x2,y2]`，SAM 1024×1024 空间，collate 后为 `gt_boxes (B,4)`

---

## 模型路径（PEFT 包装后）

```python
model.qwen                                                       # PEFT 模型
model.qwen.base_model.model                                      # Qwen3_5ForConditionalGeneration
model.qwen.base_model.model.model                                # Qwen3_5Model
model.qwen.base_model.model.model.visual                         # Qwen3_5VisionModel（ViT，depth=27）
model.qwen.base_model.model.model.visual.blocks[i]               # ViT block（hook 挂这里）
model.qwen.base_model.model.model.language_model                 # Qwen3_5TextModel
model.qwen.base_model.model.model.language_model.embed_tokens    # nn.Embedding（全量可训练）
model.qwen.base_model.model.lm_head                              # nn.Linear（全量可训练）
```

---

## 训练器（QSegTrainer）关键行为

- **双次触发防护**：`grad_accum > 1` 时 `compute_loss` 每 optimizer step 被调 2 次；`_last_eval_step` 守卫确保验证只触发一次
- **验证样本分配**：`eval_subset_n` 是**总样本数**，各卡均分不重叠子集（例如 n=4000，4卡 → 每卡 1000 样本）
- **Best checkpoint**：每次验证 mIoU 超过历史最优时，仅在 rank-0 调用 `save_trainable()`，保存到 `{output_dir}/best_model/qseg_weights.pt`
- **EMA mask loss 日志**：α=0.05 的 EMA 平滑 mask_loss，防止早期噪声影响日志可读性

---

## Checkpoint 格式

`save_trainable()` 保存的 `.pt` 文件包含以下键：

| 键 | 内容 |
|---|---|
| `lora` | LoRA 权重（仅 full_attention 层的 q/k/v/o_proj） |
| `neck` | FPN Neck 权重 |
| `cnn_bypass` | CNNBypass 权重 |
| `query_extractor` | QueryExtractor（4 query + cross-attention）权重 |
| `mask_decoder` | SAM2 MaskDecoder 权重 |
| `prompt_encoder` | SAM2 PromptEncoder 权重（含 text_embeddings 参数） |
| `embed_tokens` | 完整 embedding 表（含 [SEG] token 行） |
| `lm_head` | 语言模型头权重 |
| `seg_token_idx` | [SEG] token 的 ID（整数） |

加载时兼容旧版（`seg_projector` key）→ query_extractor 使用随机初始化权重并打印 WARN。

---

## 已知问题与局限

- **边缘模糊**：可查看vis文件夹下的可视化结果，边缘模糊是当前模型的主要问题之一，可能与训练数据、Qwen3.5ViT架构（如无窗口注意力等）相关
- **LoRA 范围有限**：Qwen3.5 混合架构中仅 8/32 层是 full_attention，LoRA 实际作用层数较少；如需增加可训参数，可在 target_modules 中加入 `in_proj_qkv`、`out_proj` 等 GatedDeltaNet 模块名
---

## 未来改进方向（参考 LENS 论文 arXiv:2508.14153）

1. **GRPO RL 微调阶段**（优先级最高）：SFT 收敛后，用真实 mask IoU 作为奖励信号做 GRPO，直接优化目标指标
2. **QueryExtractor 扩容**：queries 4→16，cross-attention 1 层→2~3 层，增强 context 表达
3. **CoT reasoning**：启用 `enable_thinking=True`，利用推理链辅助分割（收益待验证）
4. **单输入架构**：目前增加了CNN旁路，虽然占用很小，但是能否更换为单塔架构（优先级较低暂不考虑）
