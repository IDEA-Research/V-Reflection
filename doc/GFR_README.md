# Gated Feature Reweighting (GFR) LVR Head

## 概述

Gated Feature Reweighting (GFR) 是一个轻量级的门控特征重加权机制，用于解决 LVRHeadAttention 可能导致的显存不足或 NCCL 内存溢出问题。

## 核心特点

1. **极少的参数量**：仅约 4.2M 参数（对于 7B 模型）
2. **内存高效**：避免显存和 NCCL 内存溢出问题
3. **简单高效**：通过门控机制动态重加权视觉特征
4. **灵活配置**：支持自定义视觉维度、chunk 大小等参数

## 算法流程

1. 使用语言状态生成门控向量（sigmoid 激活）
2. 将门控向量应用到所有视觉 token（逐元素相乘）
3. 对重加权后的特征进行 LayerNorm 归一化
4. 通过平均池化聚合为单一向量
5. 可选：投影回 `hidden_size` 并应用输出归一化

## 使用方法

### 1. 基本使用

在训练脚本中设置：

```bash
LVR_HEAD_TYPE="gated-focus"  # 或 "gfr"
```

### 2. 配置参数

GFR 支持以下参数：

- `gfr_visual_dim`: 视觉特征维度（默认：None，使用 `hidden_size`）
  - 设置为更小的值可以进一步节省参数
  - 例如：`--gfr_visual_dim 1024`
  
- `gfr_chunk_size`: Chunk 大小（默认：None，自动选择，实际默认 512）
  - 用于长序列的内存高效处理
  - 例如：`--gfr_chunk_size 256`
  
- `gfr_use_output_norm`: 是否使用输出归一化（默认：True）
  - 例如：`--gfr_use_output_norm True`

### 3. 训练脚本示例

使用提供的训练脚本：

```bash
sbatch scripts/sbatch/sft_7b_GFR.sh
```

或者自定义参数：

```bash
export GFR_VISUAL_DIM=1024
export GFR_CHUNK_SIZE=512
export GFR_USE_OUTPUT_NORM=True
sbatch scripts/sbatch/sft_7b_GFR.sh
```

### 4. 完整训练命令示例

```bash
deepspeed src/train/train_lvr.py \
    --lvr_head True \
    --lvr_head_type gated-focus \
    --gfr_visual_dim 1024 \
    --gfr_chunk_size 512 \
    --gfr_use_output_norm True \
    --loss_lvr_fct mse \
    --loss_lvr_lambda 0.1 \
    ...
```

## 参数说明

### visual_dim

- **默认值**: `None`（使用 `hidden_size`）
- **说明**: 视觉特征的维度。如果设置为小于 `hidden_size` 的值，可以进一步减少参数量
- **参数数量影响**: `hidden_size * visual_dim`
  - 对于 7B 模型（hidden_size=3584）：
    - `visual_dim=None` (3584): ~12.8M 参数
    - `visual_dim=1024`: ~3.7M 参数
    - `visual_dim=512`: ~1.8M 参数

### chunk_size

- **默认值**: `None`（自动选择，实际默认 512）
- **说明**: 处理长序列时的 chunk 大小。较大的值可能更快但占用更多内存
- **建议**: 
  - 短序列（<512 tokens）: 不需要设置
  - 中等序列（512-2048 tokens）: 256-512
  - 长序列（>2048 tokens）: 128-256

### use_output_norm

- **默认值**: `True`
- **说明**: 是否在输出时应用 LayerNorm 归一化
- **建议**: 保持 `True` 以获得更好的训练稳定性

## 与其他 LVR Head 的对比

| 特性 | GFR | IVR | Attention-Mask |
|------|-----|-----|----------------|
| 参数量 | ~4.2M | 0 (无参数) | ~25.7M |
| 内存占用 | 低 | 最低 | 中等 |
| 训练稳定性 | 高 | 最高 | 中等 |
| 计算复杂度 | 低 | 低 | 中等 |
| 适用场景 | 显存受限 | 显存极度受限 | 显存充足 |

## 实现细节

### 架构

```
Language State (B, H) 
    ↓
Gate Projection (Linear: H → V_dim)
    ↓
Sigmoid → Gate Vector (B, V_dim)
    ↓
Visual Tokens (N, H) → [Project to V_dim] → (N, V_dim)
    ↓
Apply Gate: Visual Tokens * Gate (B, N, V_dim)
    ↓
LayerNorm (B, N, V_dim)
    ↓
Mean Pooling (B, V_dim)
    ↓
[Project to H] → Output Norm → (B, H)
```

### 内存优化

1. **Chunked Processing**: 支持分块处理长序列
2. **在线计算**: 避免存储大型中间张量
3. **参数共享**: 门控向量在所有视觉 token 间共享
4. **Gradient Detachment**: 自动对 image tokens 进行 detach，避免不必要的梯度计算（vision tower 通常被冻结）

## 故障排除

### 显存不足

1. 减小 `gfr_visual_dim`（例如：1024 或 512）
2. 减小 `gfr_chunk_size`（例如：256 或 128）
3. 减小 batch size 或增加 gradient accumulation steps

### NCCL 超时

1. 增加 `NCCL_TIMEOUT` 环境变量
2. 检查网络配置
3. 使用更小的 `gfr_chunk_size`

## 参考文献

基于 Gated Feature Reweighting 机制，这是一种轻量级的特征选择方法，特别适合多模态场景下的视觉-语言交互。

