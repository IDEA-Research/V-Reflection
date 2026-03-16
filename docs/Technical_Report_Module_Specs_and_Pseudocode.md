# Active-Coconut 技术报告：模块参数、可学习 Query 与伪代码

本文档基于代码实现，详细说明 BCM/DAR 的 Cross-Attention 参数、教师模型可学习 Query 的序列长度，以及核心创新的 PyTorch 风格伪代码。

---

## 1. 模块参数：BCM 与 DAR 的 Cross-Attention

### 1.1 概述

- **BCM (BoxFeatureResampler)**：Teacher（Stage 1 可训练，Stage 2 冻结），将 bbox 区域视觉特征压缩为固定 latent tokens。

- **DAR (DynamicAutoregressiveResampler)**：Student（Stage 2 可训练），用 LLM 自回归 hidden states 作为 Query，在全图特征上做 Cross-Attention，输出与 Teacher Target 对齐的 latent tokens。

### 1.2 Cross-Attention 层参数

| 参数 | BCM | DAR |
|------|-----|-----|
| **层数 (Number of Layers)** | 1 | 1 |
| **注意力头数 (Number of Heads)** | 8 | 8 |
| **隐藏层维度 (Hidden Dimension $D$)** | `config.hidden_size` | `config.hidden_size` |


**实现细节**（`src/model/lvr_heads.py`）：

```python
# BCM: num_heads = num_heads or min(8, hidden_size // 64)
# DAR: num_heads = num_heads or min(8, hidden_size // 64)
# 对于 Qwen2.5-VL-7B（hidden_size=3584）: num_heads = min(8, 56) = 8

self.cross_attn = nn.MultiheadAttention(
    embed_dim=hidden_size,
    num_heads=num_heads,
    dropout=0.0,
    batch_first=True,
)
```

- **BCM**：`embed_dim = hidden_size`，`num_heads = min(8, hidden_size // 64)`，单层 Cross-Attention。
- **DAR**：`embed_dim = hidden_size`，`num_heads = min(8, hidden_size // 64)`，单层 Cross-Attention。
- **Qwen2.5-VL-7B**：`hidden_size = 3584`，因此 $D = 3584$，`num_heads = 8`。

---

## 2. Learnable Queries：教师模型 $Z_T$ 的序列长度

### 2.1 定义

教师模型（BoxFeatureResampler）使用**静态可学习 Query** $Z_T$ 作为 Cross-Attention 的 Query 输入，对 bbox 区域特征做 Cross-Attention，输出与 LLM 对齐的 latent tokens。

### 2.2 序列长度

```python
# src/model/lvr_heads.py
self.queries = nn.Parameter(torch.randn(1, num_queries, hidden_size) * 0.02)
# num_queries 默认 8
```

- **$Z_T$ 的序列长度**：**8**（`num_queries`）

- 形状：`[1, 8, D]`，`D = hidden_size`。

- 初始化：`randn(1, 8, D) * 0.02`。

---

## 3. 伪代码 (Pseudocode)

### 3.1 Stochastic Decoupled Alignment Strategy（随机解耦对齐策略）

训练阶段通过**双向对称损失**实现 Resampler 与 LLM 的解耦对齐：两边各自只接收自己的梯度，避免互相“作弊”。

```python
# Stage 1: Bidirectional Symmetric Loss (src/train/monkey_patch_forward_lvr.py)
# 对应 docs/BoxFeatureResampler.md 中的设计

def stochastic_decoupled_alignment_loss(
    llm_hidden_states: torch.Tensor,   # [B, 8, D]  LLM 在 8 个 <lvr> 位置的 hidden states
    bbox_region_features: torch.Tensor, # [B, max_N, D]  bbox 区域视觉特征
    box_feature_resampler: nn.Module,
    lvr_loss_fct: callable,
) -> torch.Tensor:
    """
    Stochastic Decoupled Alignment: 双向对称损失，梯度解耦。
    - Resampler 学习输出接近 LLM 自然会产生的 hidden state
    - LLM 学习输出接近 Resampler 压缩结果的 hidden state
    """
    # Resampler: Q = learnable queries [1,8,D], K/V = bbox_region_features
    resampler_output = box_feature_resampler(bbox_region_features)  # [B, 8, D]

    # Clamp for numerical stability
    llm_clamped = torch.clamp(llm_hidden_states, min=-1e4, max=1e4)
    resampler_clamped = torch.clamp(resampler_output, min=-1e4, max=1e4)

    # Decoupled: each side receives only its own gradient
    loss_resampler = lvr_loss_fct(llm_clamped.detach(), resampler_clamped)  # train resampler
    loss_llm = lvr_loss_fct(llm_clamped, resampler_clamped.detach())        # train LLM

    loss = (loss_resampler + loss_llm) / 2
    return loss
```

**Stage 2 蒸馏**：Teacher 冻结，Student 用 LLM hidden states 作为 Query，在全图特征上做 Cross-Attention，输出与 Teacher Target 对齐：

```python
# Stage 2: Teacher-Student MSE Distillation
with torch.no_grad():
    target_latent_tokens = box_feature_resampler(bbox_feats, key_padding_mask=key_pad_mask)

predicted_latent_tokens = student_resampler(
    lvr_hidden_states=lvr_hidden_states,      # [B, 8, D] from LLM
    full_image_features=full_image_features,   # [B, Seq_Len, D]
    key_padding_mask=key_padding_mask,
)
loss_lvr_resampler = F.mse_loss(predicted_latent_tokens, target_latent_tokens.detach())
```

### 3.2 Coconut-style 自回归推理过程

推理时使用 **Coconut 风格**：在 LVR 模式内，将 `last_position_hidden_state` 作为下一步的 input embedding，而非 token embedding，实现隐状态“回环”。

```python
# 基于 src/model/qwen_lvr_model.py (_lvr_deocding_by_steps)
# 及 src/train/monkey_patch_forward_lvr.py (qwen2_5_mixed_modality_forward_lvr_with_resampler_inference)

def coconut_style_lvr_inference(
    model,
    input_ids: torch.LongTensor,
    generation_config,
    lvr_steps: int = 8,
) -> torch.LongTensor:
    """
    Coconut-style autoregressive inference: 在 LVR 区间内用 hidden state 替代 token embedding。
    """
    batch_size = input_ids.shape[0]
    lvr_start_id = model.config.lvr_start_id
    lvr_end_id = model.config.lvr_end_id

    lvr_mode_switch = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)
    last_position_hidden_state = None
    lvr_remaining_steps = torch.tensor([lvr_steps] * batch_size, device=input_ids.device)

    while not all_finished:
        # 1. 准备输入
        model_inputs = model.prepare_inputs_for_generation(input_ids, **model_kwargs)
        model_inputs["lvr_mode_switch"] = lvr_mode_switch
        model_inputs["last_position_hidden_state"] = last_position_hidden_state

        # 2. Forward
        outputs = model(**model_inputs, return_dict=True)

        # 3. Coconut 核心：在 LVR 模式下，last_position_hidden_state 会在 forward 内被写入 inputs_embeds
        #    inputs_embeds[lvr_mode_switch, -1, :] = last_position_hidden_state[lvr_mode_switch]
        last_position_hidden_state = outputs.last_hidden_state[:, -1, :]

        # 4. 采样 next token
        next_token_logits = outputs.logits[:, -1, :]
        next_tokens = sample_or_argmax(next_token_logits)

        # 5. LVR 模式切换逻辑
        last_tokens = input_ids[:, -1]
        lvr_start_switch = (last_tokens == lvr_start_id)

        # 进入：刚生成 <lvr_start> 或已在 LVR 区间
        new_mode_switch = lvr_mode_switch | lvr_start_switch

        # 刚进入时重置 quota
        just_entered = (~lvr_mode_switch) & new_mode_switch
        lvr_remaining_steps = torch.where(just_entered, lvr_steps, lvr_remaining_steps)

        # 已在内则减少 quota
        lvr_remaining_steps = lvr_remaining_steps - lvr_mode_switch.long()

        # 退出：quota 用完
        lvr_mode_switch = new_mode_switch & (lvr_remaining_steps > 0)

        # 6. 更新序列
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        model_kwargs = update_model_kwargs(outputs, model_kwargs)

    return input_ids
```

**Forward 内部 Coconut 替换**（`monkey_patch_forward_lvr.py`）：

```python
# 当 lvr_mode_switch 为 True 且 last_position_hidden_state 非 None 时：
if last_position_hidden_state is not None and lvr_mode_switch is not None:
    # 用 last_position_hidden_state 作为当前步的 input embedding
    inputs_embeds[lvr_mode_switch, -1, :] = last_position_hidden_state[lvr_mode_switch]
```

---

## 4. 总结

| 项目 | 值 |
|------|-----|
| BCM Cross-Attention 层数 | 1 |
| DAR Cross-Attention 层数 | 1 |
| 注意力头数 | 8 |
| 隐藏层维度 $D$ | 3584（Qwen2.5-VL-7B） |
| 教师可学习 Query $Z_T$ 序列长度 | 8 |

核心创新：
1. **Stochastic Decoupled Alignment**：双向对称损失 + detach，实现 Resampler 与 LLM 的解耦对齐。
2. **Coconut-style 推理**：LVR 区间内用 `last_position_hidden_state` 替代 token embedding，实现 8 步隐状态回环的纯 LLM decode。

---

## 5. Visual CoT 数据集与模型输入输出格式

### 5.1 数据集格式

训练使用 **Visual CoT (viscot)** 风格数据集，遵循 LLaVA 规范。每条样本为 JSON 对象：

```json
{
  "dataset": "sroie",
  "split": "train",
  "question_id": 0,
  "image": ["viscot/sroie/X51006555072.jpg"],
  "conversations": [
    {
      "from": "human",
      "value": "<image>\nWhat is the company in the invoice shown in the picture?\nProvide a short and direct response."
    },
    {
      "from": "gpt",
      "value": "<lvr>\n<answer> DIGI TELECOMMUNICATIONS SDN BHD </answer>"
    }
  ],
  "bboxes": [[0.096, 0.182, 0.929, 0.540]]
}
```

- **`<image>`**：占位符，数据 collate 时替换为 Qwen2-VL 的视觉 token（`<|vision_start|><|image_pad|><|vision_end|>`）。
- **`<lvr>`**：占位符，替换为 `<|lvr_start|>` + 8×`<|lvr|>` + `<|lvr_end|>`（或含 `<|lvr_latent_end|>`）。
- **`bboxes`**：归一化坐标 `[x0, y0, x1, y1]`，范围 [0, 1]，与图像分辨率无关。训练时通过 `bbox_to_token_idxs` 映射到 image token 索引，用于提取 bbox 区域特征。

### 5.2 模型输入输出

| 阶段 | 输入 | 输出 |
|------|------|------|
| **训练** | System + User(图像+问题) + Assistant 前缀 | 生成 `<|lvr_start|><|lvr|>×8<|lvr_end|>\n<answer>...</answer>` |
| **推理** | 同上 | 同上；LVR 区间内用 hidden state 回环，不生成实际 `<|lvr|>` token |

---

## 6. 训练指令：System Prompt 与 User Instruction

### 6.1 Stage 1 与 Stage 2

| 项目 | 内容 |
|------|------|
| **System Prompt** | `"You are a helpful assistant."`（`src/constants.py` 中 `SYSTEM_MESSAGE`） |
| **User Instruction** | 由 `conversations` 中 human 的 `value` 经 `replace_image_tokens` 得到，格式为 `<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n` |

**说明**：`LVR_SYSTEM_MESSAGE = "Put your final answer in <answer> </answer>."` 在代码中定义但**未在数据集中使用**，实际训练仍用 `SYSTEM_MESSAGE`。答案格式由 GPT 的 `value` 模板（`<lvr>\n<answer>...</answer>`）隐式约束。

### 6.2 完整 Prompt 结构

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
<|vision_start|><|image_pad|><|vision_end|>
What is the company in the invoice shown in the picture?
Provide a short and direct response.<|im_end|>
<|im_start|>assistant
```

模型需生成：

```
<|lvr_start|><|lvr|><|lvr|><|lvr|><|lvr|><|lvr|><|lvr|><|lvr|><|lvr_end|>
<answer> DIGI TELECOMMUNICATIONS SDN BHD </answer><|im_end|>
```

---

## 7. 特殊 Token 与边界框表示

### 7.1 特殊 Token

| Token | 作用 |
|-------|------|
| `<|lvr_start|>` | 触发连续隐状态推理（LVR 模式入口） |
| `<|lvr|>` | 8 个 latent 槽位，训练时由 BoxFeatureResampler 输出填充 inputs_embeds |
| `<|lvr_latent_end|>` | 可选，用于 mode switch loss；启用 `latent_end_token` 时插入 |
| `<|lvr_end|>` | LVR 区间结束 |

**实现**（`src/train/train_lvr.py`）：

```python
processor.tokenizer.add_tokens("<|lvr_start|>", special_tokens=True)
processor.tokenizer.add_tokens("<|lvr|>", special_tokens=True)
processor.tokenizer.add_tokens("<|lvr_latent_end|>", special_tokens=True)
processor.tokenizer.add_tokens("<|lvr_end|>", special_tokens=True)
```

### 7.2 边界框 (Bounding Box) 表示

**边界框不在文本中表示**，而是单独存在 `bboxes` 字段：

- **格式**：`[[x0, y0, x1, y1], ...]`，归一化坐标 (0–1)
- **用途**：通过 `bbox_to_token_idxs(bboxes, image_grid_thw)` 转为 image token 索引，用于提取 bbox 区域特征并送入 BoxFeatureResampler

### 7.3 实际 Text Prompt 示例

**示例 1（SROIE 发票）**：

```
User: <image>
What is the company in the invoice shown in the picture?
Provide a short and direct response.

Assistant: <lvr>
<answer> DIGI TELECOMMUNICATIONS SDN BHD </answer>
```

**示例 2（Flickr30k）**：

```
User: <image>
Can you describe the lower apparel of the child on the swing?
Provide a short and direct response.

Assistant: <lvr>
<answer> The child on the swing is wearing dark blue denim shorts. </answer>
```

**示例 3（多 bbox，多 `<lvr>` 占位符）**：

若 `conversations` 中有多个 `<lvr>`，每个对应一个 bbox，替换后为多个 `<|lvr_start|>...<|lvr_end|>` 段。

**训练时实际 token 序列**（单 bbox，8 latent）：

```
<|lvr_start|><|lvr|><|lvr|><|lvr|><|lvr|><|lvr|><|lvr|><|lvr|><|lvr_end|>
<answer> ... </answer><|im_end|>
```
