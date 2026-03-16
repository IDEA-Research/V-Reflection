# Stage 2 自回归 Teacher-Student 蒸馏

## 概述

Stage 2 蒸馏在 Stage 1（BoxFeatureResampler 联合训练）的基础上，将 **BoxFeatureResampler 冻结为 Teacher**，新增 **DynamicAutoregressiveResampler 作为 Student**，用 LLM 在 8 个连续 `<lvr>` 位置的 **自回归 Hidden States** 作为动态 Queries，在全图特征上做 Cross-Attention，输出与 Teacher Target 对齐的 latent tokens，通过 MSE 蒸馏。

## 动机

Stage 1 中，BoxFeatureResampler 使用 **可学习固定 Queries** 对 bbox 区域特征做 Cross-Attention，与 LLM 通过双向对称损失联合训练。推理时，LLM 自回归生成 8 个 `<lvr>` token，其 hidden states 应能表达与 resampler 压缩结果一致的信息。

Stage 2 的目标是：**显式学习从 LLM 自回归 hidden states 到 latent tokens 的映射**，使推理阶段无需再调用 BoxFeatureResampler，仅依赖 LLM 输出 + 轻量 Cross-Attention 即可得到 latent 表示。

## 架构

```
Teacher (frozen): BoxFeatureResampler
    输入: bbox_region_features [B, max_N, D]
    输出: target_latent_tokens [B, 8, D]  (no_grad)

Student (trainable): DynamicAutoregressiveResampler
    输入:
      - lvr_hidden_states [B, 8, D_llm]  -- LLM 在 8 个 <lvr> 位置的自回归 hidden states
      - full_image_features [B, Seq_Len, D_vis]  -- 全图特征（非 bbox 裁剪）
    输出: predicted_latent_tokens [B, 8, D]

Loss: MSE(predicted_latent_tokens, target_latent_tokens.detach())
```

### DynamicAutoregressiveResampler 实现细节

**结构图**：

```
lvr_hidden_states [B, 8, D_llm]     full_image_features [B, Seq_Len, D_vis]
        |                                      |
        v (q_proj if D_llm != hidden_size)     v (kv_proj if D_vis != hidden_size)
        |                                      |
        Q [B, 8, hidden_size]         K/V [B, Seq_Len, hidden_size]
        |                                      |
        +-------- Cross-Attention --------------+
        |   nn.MultiheadAttention(Q, K, V)     |
        +--------------------------------------+
                        |
                        v LayerNorm
        predicted_latent_tokens [B, 8, hidden_size]
```

**构造函数参数**（`lvr_heads.py`）：

| 参数 | 说明 | 默认 |
|------|------|------|
| `hidden_size` | 输出维度，与 Teacher latent 对齐 | - |
| `llm_hidden_size` | LLM hidden 维度，用于 q_proj | hidden_size |
| `vision_dim` | 视觉特征维度，用于 kv_proj | hidden_size |
| `num_queries` | Query 数量 | 8 |
| `num_heads` | Cross-Attention 头数 | min(8, hidden_size//64) |

**前向接口**：

```python
def forward(lvr_hidden_states, full_image_features, key_padding_mask=None, return_attention=False):
    # lvr_hidden_states: [B, 8, D_llm]
    # full_image_features: [B, Seq_Len, D_vis]
    # key_padding_mask: [B, Seq_Len], True=padding(忽略), False=有效
    # return_attention: True 时返回 (out, attn_weights)
    # 返回: [B, 8, hidden_size] 或 ((out, attn_weights)) 其中 attn_weights [B, 8, Seq_Len]
```

**设计要点**：
- **无 MLP query 投影**：Queries 直接来自 LLM hidden states，保持自回归语义
- **全图特征作为 K/V**：Student 从整张图 attend，而非仅 bbox 区域，与 Teacher 的 bbox-only 输入形成互补
- **投影层**：`q_proj`、`kv_proj` 仅在 `llm_hidden_size != hidden_size` 或 `vision_dim != hidden_size` 时创建，通常 Qwen2.5-VL 下可省略
- **单层 Cross-Attention**：`nn.MultiheadAttention`，`batch_first=True`，`dropout=0.0`

## 训练流程

### 1. 前置条件

- 必须加载 **Stage 1 checkpoint**（`use_box_feature_resampler=True` 训练得到的模型）
- `use_box_feature_resampler=True` 且 `use_stage2_distillation=True`
- Stage 2 会 **禁用 `resume_from_checkpoint`**，仅用 `checkpoint_name` 加载模型权重，避免 DeepSpeed 恢复 Stage 1 的完整训练状态（Stage 1 无 `student_resampler`，会导致 state_dict 不匹配）

### 2. Forward 流程

```
Step A: Teacher 前向（no_grad）
  bbox_feats, key_pad_mask = _prepare_bbox_region_features(...)
  target_latent_tokens = box_feature_resampler(bbox_feats, key_padding_mask=key_pad_mask)

Step B: 收集 LLM 在 8 个 <lvr> 位置的 hidden states
  seq_positions = 每个 <lvr> 在序列中的位置（不做 -1，即自回归输出位置）
  lvr_hidden_states = hidden_states[batch_indices, seq_positions]  # [num_bboxes*8, D]
  lvr_hidden_states = lvr_hidden_states.view(num_bboxes, 8, -1)

Step C: 准备全图特征
  batch_indices_per_bbox = batch_indices[0::8]  # 每个 bbox 取一个 batch index
  full_image_features, image_attention_mask = _prepare_batched_image_embeds(
      image_embeds, total_tokens, batch_indices_per_bbox, target_dtype=compute_dtype
  )

Step D: Student 前向
  predicted_latent_tokens = student_resampler(
      lvr_hidden_states=lvr_hidden_states,
      full_image_features=full_image_features,
      key_padding_mask=~image_attention_mask
  )

Step E: MSE Loss + Pixel-level Attention Distillation
  loss_lvr_resampler = F.mse_loss(
      predicted_latent_tokens.to(float32),
      target_latent_tokens.detach().to(float32)
  )
  # 若 loss_attn_transfer_lambda > 0：额外加入 KL 蒸馏
  target_attn = get_aligned_teacher_attn_from_lvr_tokens(teacher_attn, lvr_tokens, key_pad_mask, seq_len_full, device)
  loss_attn_transfer = F.kl_div(log(student_attn), target_attn, reduction='batchmean')
  loss_lvr_resampler += loss_attn_transfer_lambda * loss_attn_transfer
```

### 3. Pixel-level Attention Distillation

Teacher 的 Cross-Attention 仅在 bbox 内 token 上计算（形状 `[B, 8, max_N]`），而 Student 的 attention 在全图 token 上（形状 `[B, 8, Seq_Len]`）。为让 Student 在全图上的 attention 分布与 Teacher 在 bbox 内的分布对齐，引入 **像素级 Attention 蒸馏**：

1. **Teacher attention 对齐到全图**：`get_aligned_teacher_attn_from_lvr_tokens` 将 Teacher 的 `[B, 8, max_N]` 映射到 `[B, 8, max_seq_len]`。`teacher_attn[i,:,j]` 对应 `lvr_tokens[i][j]` 位置，按 `lvr_tokens` 将 Teacher 的列索引映射到全图 token 索引，非 bbox 位置填 0。
2. **KL 散度蒸馏**：`loss_attn_transfer = F.kl_div(student_attn_log, target_attn, reduction='batchmean')`，使 Student 的全图 attention 分布拟合 Teacher 的 bbox 内分布（零填充后）。
3. **触发条件**：`loss_attn_transfer_lambda > 0` 时，Teacher 以 `return_attention=True` 前向，Student 始终返回 attention。

### 4. 与 Stage 1 的差异

| 项目 | Stage 1 | Stage 2 |
|------|---------|---------|
| BoxFeatureResampler | 可训练 | 冻结 |
| Queries 来源 | 可学习参数 | LLM 自回归 hidden states |
| K/V 来源 | bbox 区域特征 | 全图特征 |
| Loss | 双向对称（resampler ↔ LLM） | 单向 MSE + 可选 Pixel-level Attention KL 蒸馏 |
| 填充 inputs_embeds | resampler 输出 detach | 仍用 Teacher 输出 detach（DiT 50/50 条件） |

### 5. 总 Loss

**Forward 内 `loss_lvr_resampler` 计算**（`monkey_patch_forward_lvr.py`）：

```python
loss_lvr_resampler = F.mse_loss(predicted_latent_tokens, target_latent_tokens.detach())
if loss_attn_transfer_lambda > 0 and teacher_attn is not None and student_attn is not None:
    target_attn = get_aligned_teacher_attn_from_lvr_tokens(teacher_attn, lvr_tokens, key_pad_mask, seq_len_full, device)
    student_attn_log = torch.log(student_attn.float() + 1e-9)
    loss_attn_transfer = F.kl_div(student_attn_log, target_attn, reduction='batchmean')
    loss_lvr_resampler = loss_lvr_resampler + loss_attn_transfer_lambda * loss_attn_transfer
```

**Trainer 总 Loss**（`lvr_trainer.py`，Stage 2 且 `LVR_HEAD=False` 时 `loss_lvr` 为 None）：

```
loss = loss_ce + loss_lvr_resampler_lambda * loss_lvr_resampler
```

**展开形式**：

```
loss = loss_ce
     + loss_lvr_resampler_lambda * [ MSE(predicted, target)
                                   + loss_attn_transfer_lambda * KL(student_attn || target_attn) ]
```

**默认脚本权重**（`LOSS_LVR_RESAMPLER_LAMBDA=0.1`, `LOSS_ATTN_TRANSFER_LAMBDA=1.0`）：

```
loss = loss_ce + 0.1 * MSE + 0.1 * KL
```

**梯度流向**：`target_latent_tokens.detach()`、`teacher_attn` 在 `no_grad` 下，梯度仅回传到 **Student**（DynamicAutoregressiveResampler），Teacher（BoxFeatureResampler）冻结。

## 配置参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--use_stage2_distillation` | 启用 Stage 2 蒸馏 | False |
| `--use_box_feature_resampler` | 必须为 True（Teacher 依赖） | - |
| `--num_latent_tokens` | 每个 bbox 的 latent 数 | 8 |
| `--loss_lvr_resampler_lambda` | MSE 蒸馏 loss 权重 | 0.1 |
| `--loss_attn_transfer_lambda` | Pixel-level attention 蒸馏（KL）权重，0 表示关闭 | 1.0 |
| `--checkpoint_name` / `CHECKPOINT_PATH` | Stage 1 checkpoint 路径 | 必填 |

## 使用示例

### 训练脚本

```bash
# scripts/sbatch/sft_7b_stage2_distillation.sh
CHECKPOINT_PATH="result/box_resampler/SFT_box_resampler_steps2500_b4_LVR0.1_resampler0.1_acc8/checkpoint-2500"

deepspeed src/train/train_lvr.py \
    --use_box_feature_resampler True \
    --use_stage2_distillation True \
    --num_latent_tokens 8 \
    --loss_lvr_resampler_lambda 0.1 \
    --loss_attn_transfer_lambda 1.0 \
    --checkpoint_name "$CHECKPOINT_PATH" \
    ...
```

### 环境变量

```bash
export CHECKPOINT_PATH="path/to/stage1_checkpoint"
sbatch scripts/sbatch/sft_7b_stage2_distillation.sh
```

## 文件结构

```
src/
├── model/
│   ├── lvr_heads.py          # DynamicAutoregressiveResampler 类定义
│   └── qwen_lvr_model.py     # _init_dynamic_autoregressive_resampler, 冻结 Teacher
├── train/
│   ├── train_lvr.py          # Stage 2 配置、跳过 resume、configure_lvr_head 中 Student 可训练
│   └── monkey_patch_forward_lvr.py  # loss 分支：Teacher target、Student 前向、MSE、get_aligned_teacher_attn_from_lvr_tokens、KL 蒸馏
└── params.py                 # use_stage2_distillation、loss_attn_transfer_lambda 参数
```

## 实现细节

### dtype 与 ZeRO-3

- 使用 `compute_dtype = hidden_states.dtype` 而非 `next(student_resampler.parameters()).dtype`，因 DeepSpeed ZeRO-3 下参数可能分片，`parameters().dtype` 不可靠
- `lvr_hidden_states`、`full_image_features` 保持与模型一致（如 bf16），MSE 计算前转为 float32 以提升数值稳定性

### DiT 50/50 条件

- `_resampler_output_for_dit` 仍使用 Teacher 输出，与 Stage 1 一致，保证 DiT 重建头的条件输入稳定

### 数值稳定性

- 若 `predicted_latent_tokens` 或 `target_latent_tokens` 含 NaN/Inf，则 `loss_lvr_resampler = 0.0`（带 `requires_grad=True` 以保持计算图）
- `loss_attn_transfer` 计算前对 `student_attn` 加 `eps=1e-9` 避免 log(0)；若 KL 结果为 NaN/Inf 则跳过该 loss

## 与其他模块的关系

| 模块 | 关系 |
|------|------|
| BoxFeatureResampler | **Teacher**，Stage 2 中冻结 |
| LVR Head | 通常关闭（`LVR_HEAD=False`），仅做蒸馏 |
| Vision Encoder | 冻结，与 Stage 1 一致 |
