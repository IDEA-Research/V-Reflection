# Stage 2 Distillation 推理效率分析

本文档总结 Stage 2 Distillation 模型（`evaluation_7b_stage2_distillation.sh`）相比正常 LLM 推理引入的额外参数与计算开销。

---

## 1. 额外参数量（Parameter Overhead）

Stage 2 在 Qwen2.5-VL-7B（`hidden_size=3584`）基础上引入两个模块：

| 模块 | 结构 | 参数量 | 推理时是否被调用 |
|------|------|--------|----------------|
| **Box-Guided Compression** | learnable queries [1,8,H] + MHA(H,H,8 heads) + LN | **51.43M** | **否**（Teacher，仅训练用） |
| **Dynamic Autoregressive Compression** | MHA(H,H,8 heads) + LN | **51.40M** | **否**（Student，加载但不调用） |
| **合计额外** | — | **102.83M** | **0 FLOPs** |

### 参数占比

```
102.83M / ~7,070M (7B base) ≈ +1.45% 参数量
显存开销 (bf16): 196 MB（相比基础模型 13.2 GB 仅 +1.5%）
```

**关键发现：** 两个 compression 模块在推理 forward 中**均未被调用**。代码注释明确说明 "Do NOT apply box_feature_resampler enhancement here"。它们仅占用显存，不产生推理计算开销。

---

## 2. 计算开销（Compute Overhead）

真正的推理开销来自 **Coconut-style 连续思考步骤**：

```
evaluation_7b_stage2_distillation.sh:
EVAL_STEP_LIST="${EVAL_STEP_LIST:-8}"   # 默认 8 步 LVR 思考
```

### `_lvr_deocding_by_steps` 工作机制

1. 模型正常生成，遇到 `<|lvr_start|>` 进入 LVR 模式
2. 接下来 **8 个额外 decode step**：每步将 `last_position_hidden_state`（LLM 最后位置的隐状态）作为下一步的 input embedding，而非 token embedding
3. 8 步耗尽后退出 LVR 模式，继续生成 answer

每一个 LVR 思考步等价于 **1 个完整的 KV-cached decoder forward pass**（28 层 Transformer），即生成 1 个普通 token 的计算量。

| 开销维度 | Baseline | Stage 2（step=8） | 倍数 |
|---------|----------|-------------------|------|
| Prefill pass | 1× | 1×（相同） | 1.0× |
| Decode steps | T 步（答案长度） | T + 8 步 | (T+8)/T |
| 额外 FLOPs（per bbox region） | 0 | 8 × decode_step_cost | — |

### 实际延迟影响

- MCQ 评测（BLINK/VSTAR/HRBench）答案通常 **10–30 个 token**
- 8 LVR 步 ≈ 多生成 8 个 token
- **延迟增加约 25%~80%**（答案越短，相对开销越大）
- Decode step 因 KV cache 复用，绝对耗时远小于 prefill，整体影响可控

---

## 3. 推理机制设计

### 训练 vs 推理

**训练时：**
- **Box-Guided Compression（Teacher）**：用 bbox 区域 vision tokens 生成 8 个 latent targets
- **Dynamic Autoregressive Compression（Student）**：用 LLM 的 8 个连续隐状态 cross-attend 到图像，做 MSE 蒸馏

**推理时：**
- 两者均不调用，完全 Coconut 风格：将 `last_position_hidden_state` 直接作为下一步 input embedding
- 设计意图：通过蒸馏让 LLM 隐状态本身具备视觉 attend 能力，推理时无需任何外部模块

### 核心代码逻辑

```python
# monkey_patch_forward_lvr.py: qwen2_5_mixed_modality_forward_lvr_with_resampler_inference
# NOTE: Do NOT apply box_feature_resampler enhancement to hidden states here.
# Inference: we use last_position_hidden_state as input for next step (Coconut style).
if last_position_hidden_state is not None and lvr_mode_switch is not None:
    inputs_embeds[lvr_mode_switch, -1, :] = last_position_hidden_state[lvr_mode_switch]
```

---

## 4. 总结

| 维度 | 数值 |
|------|------|
| **参数额外开销** | +102.83M = +1.45%（仅占显存，推理无 FLOPs） |
| **显存额外开销** | +196 MB（bf16），占基础模型 13.2 GB 的 1.5% |
| **计算额外开销** | +8 个 KV-cached decode steps per bbox region |
| **实际延迟开销** | ≈ +25%~80% decode 时间（取决于答案长度） |

**结论：** 整体设计效率较高——用 1.5% 的显存和 8 步 decode overhead 换取视觉定位能力的提升。两个 resampler 模块的参数在推理时完全"沉默"，推理路径本质上是带有 8 步隐状态回环的纯 LLM decode。

---

## 5. 附录：参数计算明细

### Box-Guided Compression (51.43M)

| 组件 | 形状 | 参数量 |
|------|------|--------|
| queries | [1, 8, 3584] | 28,672 |
| cross_attn in_proj_weight | [3×3584, 3584] | 38,535,168 |
| cross_attn in_proj_bias | [3×3584] | 10,752 |
| cross_attn out_proj_weight | [3584, 3584] | 12,845,056 |
| cross_attn out_proj_bias | [3584] | 3,584 |
| output_norm | [2×3584] | 7,168 |

### Dynamic Autoregressive Compression (51.40M)

- `q_proj` / `kv_proj`：均为 None（`llm_hidden_size == vision_dim == hidden_size`）
- 结构与 Box-Guided Compression 的 cross_attn + output_norm 相同，无 learnable queries
