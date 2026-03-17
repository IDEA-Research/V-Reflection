# Box-Guided Compression 技术文档

## 概述

Box-Guided Compression 是一个将**可变长度的 bbox 视觉特征**压缩为**固定 8 个 latent token** 的模块，用于 LVR (Latent Visual Reasoning) 训练。

## 动机

原始 LVR 设计中，每个 GT bbox 对应的 token 数量随 bbox 大小和图像分辨率变化（可能是 256、1024 甚至更多）。这带来两个问题：

1. **序列长度不固定**：不同样本的 `<lvr>` token 数量不同，难以批处理
2. **信息冗余**：大 bbox 可能有上千个 token，但实际有用信息可能很少

Box-Guided Compression 通过 **cross-attention** 将可变长特征压缩为固定 8 个 latent，解决上述问题。

## 架构

```
输入: bbox_region_features (num_bboxes, max_N, D)  -- 可变长，padding 对齐
      key_padding_mask (num_bboxes, max_N)         -- True = padding 位置

        ┌─────────────────────────────────────┐
        │  8 Learnable Queries (1, 8, D)      │
        │         ↓                           │
        │  Cross-Attention                    │
        │    Q: queries                       │
        │    K/V: bbox_region_features        │
        │         ↓                           │
        │  LayerNorm                          │
        └─────────────────────────────────────┘

输出: (num_bboxes, 8, D)  -- 固定长度
```

### 核心代码

```python
class BoxGuidedCompression(nn.Module):
    def __init__(self, hidden_size, num_queries=8, num_heads=None, vision_dim=None):
        self.queries = nn.Parameter(torch.randn(1, num_queries, hidden_size) * 0.02)
        self.cross_attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.output_norm = LayerNorm(hidden_size)

    def forward(self, bbox_region_features, key_padding_mask=None):
        q = self.queries.expand(L, -1, -1)  # (num_bboxes, 8, D)
        attn_out, _ = self.cross_attn(q, bbox_region_features, bbox_region_features,
                                       key_padding_mask=key_padding_mask)
        return self.output_norm(attn_out)  # (num_bboxes, 8, D)
```

## 训练流程

### 1. 数据准备

序列模板：`<lvr_start>` + 8×`<lvr>` + `<lvr_end>`

每个 bbox 固定 8 个 latent 槽位，不再依赖 GT bbox 的实际 token 数。

### 2. Forward 流程

```
┌────────────────────────────────────────────────────────────────────────────┐
│  Step 1: 图像编码                                                           │
│    image → ViT → merger → image_embeds (frozen)                            │
│                                                                            │
│  Step 2: 准备 bbox 特征                                                     │
│    lvr_tokens[i] = GT bbox 在 image_embeds 中的索引 (可变长)                │
│    _prepare_bbox_region_features → (num_bboxes, max_N, D), padding_mask    │
│                                                                            │
│  Step 3: Box-Guided Compression 压缩                                       │
│    BoxGuidedCompression(bbox_feats) → fill_embeds (num_bboxes, 8, D)       │
│                                                                            │
│  Step 4: 填充 inputs_embeds                                                 │
│    fill_embeds.detach() → inputs_embeds[<lvr>位置]                          │
│                                                                            │
│  Step 5: LLM Forward                                                        │
│    inputs_embeds → LLM → hidden_states                                     │
└────────────────────────────────────────────────────────────────────────────┘
```

### 3. Loss 计算：双向对称损失

为了**联合训练** resampler 和 LLM，同时**防止作弊**，采用双向对称损失：

```python
resampler_output = self.box_feature_resampler(bbox_feats)  # 有梯度
llm_latent_8 = hidden_states[batch_indices, seq_positions - 1]  # LLM 输出

# 双向损失：各自的梯度互不影响
loss_resampler = MSE(llm_latent_8.detach(), resampler_output)  # 训练 resampler
loss_llm = MSE(llm_latent_8, resampler_output.detach())        # 训练 LLM
loss_lvr_resampler = (loss_resampler + loss_llm) / 2
```

**设计原理**：
- `loss_resampler`：resampler 学习输出接近 LLM 自然会产生的 hidden state
- `loss_llm`：LLM 学习在 `<lvr>` 位置输出接近 resampler 压缩结果的 hidden state
- 两边独立优化，梯度不会互相干扰，无法"作弊"（如 resampler 输出全 0 来骗 loss）

### 4. 总 Loss

```
Total Loss = loss_ce + λ_resampler * loss_lvr_resampler
```

其中 `λ_resampler` 由 `--loss_lvr_resampler_lambda` 控制（默认 0.1）。

## 配置参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--use_box_feature_resampler` | 启用 Box-Guided Compression | False |
| `--num_latent_tokens` | 每个 bbox 的固定 latent 数 | 8 |
| `--loss_lvr_resampler_lambda` | resampler loss 权重 | 0.1 |
| `--latent_end_token` | 是否使用 `<lvr_latent_end>` | False (推荐) |
| `--lvr_head` | 是否使用 LVR head | False (与 resampler 独立) |

## 使用示例

### 训练脚本

```bash
# scripts_release/train/sft_7b_stage1_box_resampler.sh
LVR_HEAD=False
USE_BOX_FEATURE_RESAMPLER=True
NUM_LATENT_TOKENS=8
LATENT_END_TOKEN=False
LOSS_LVR_RESAMPLER_LAMBDA=0.1

deepspeed src/train/train_lvr.py \
    --use_box_feature_resampler $USE_BOX_FEATURE_RESAMPLER \
    --num_latent_tokens $NUM_LATENT_TOKENS \
    --loss_lvr_resampler_lambda $LOSS_LVR_RESAMPLER_LAMBDA \
    --latent_end_token $LATENT_END_TOKEN \
    --lvr_head $LVR_HEAD \
    ...
```

## 文件结构

```
src/
├── model/
│   ├── lvr_heads.py          # BoxGuidedCompression 类定义
│   └── qwen_lvr_model.py     # 模型初始化 (_init_box_feature_resampler)
├── train/
│   ├── train_lvr.py          # 训练入口，配置 resampler 可训练
│   └── monkey_patch_forward_lvr.py  # forward 中的填充和 loss 计算
├── dataset/
│   └── data_utils.py         # 数据模板：固定 8 个 <lvr> token
└── trainer/
    └── lvr_trainer.py        # loss 汇总
```

## 与其他模块的关系

| 模块 | 关系 |
|------|------|
| LVR Head | **独立**。resampler 和 LVR head 可以同时使用，也可以单独使用 |
| `<lvr_latent_end>` | **可选**。resampler 模式下通常不需要 latent_end_token |
| Vision Encoder | **冻结**。resampler 输入来自 frozen vision encoder 的输出 |
| LLM | **联合训练**。通过双向对称损失同时优化 |

## 注意事项

1. **填充阶段使用 `.detach()`**：避免双重计算图
2. **Loss 阶段不完全 detach**：实现联合训练
3. **参数初始化**：queries 使用 `0.02 * randn` 小值初始化
4. **数值稳定性**：loss 计算前对 tensor 进行 clamp，避免 NaN/Inf
