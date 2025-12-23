# LVRHeadImplicitVisualRouting (IVR) - 使用指南

## 概述

`LVRHeadImplicitVisualRouting` (IVR) 是基于胶囊网络路由思想的轻量级 LVR Head 实现。它是**完全无参数**的（除了可选的输出归一化层），通过迭代优化实现动态聚焦，在训练稳定性方面表现优异，能够有效避免显存和 NCCL 内存溢出问题。

### 核心特点

1. **完全无参数**：除了可选的输出归一化层（LayerNorm），不包含任何可学习参数
2. **训练最稳定**：纯迭代算法，无梯度爆炸风险
3. **内存高效**：支持 chunked 处理长序列，避免大张量分配
4. **性能优异**：在 ScienceQA 上达到 Q-Former 98% 的性能

## 算法原理

IVR 使用胶囊网络的路由机制，通过迭代优化实现动态聚焦：

1. **初始化路由系数** `b`：使用 `lang_state` 与 `visual_tokens` 的相似度初始化
2. **迭代更新**（默认 3 次）：
   - 计算注意力系数：`c = softmax(b)`
   - 生成聚焦特征：`s = weighted_sum(visual_tokens, c)`
   - 更新路由系数：`b = b + similarity(visual_tokens, s)`
3. **返回最终聚焦特征** `s`

## 使用方法

### 1. 基本使用

在训练脚本中设置 `lvr_head_type` 为 `ivr` 或 `implicit-visual-routing`：

```bash
python src/train/train_lvr.py \
    --lvr_head True \
    --lvr_head_type ivr \
    --ivr_iterations 3 \
    --ivr_use_output_norm True \
    --ivr_temperature 1.0 \
    ...
```

### 2. 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `ivr_iterations` | int | 3 | 路由迭代次数，建议范围：2-5 |
| `ivr_chunk_size` | Optional[int] | None | Chunk 大小（None 表示自动选择），建议：512-1024 |
| `ivr_use_output_norm` | bool | True | 是否使用输出归一化（LayerNorm） |
| `ivr_temperature` | float | 1.0 | 温度参数，控制路由的锐度 |

### 3. 完整训练脚本示例

参考 `scripts/sbatch/sft_7b_IVR.sh`：

```bash
#!/bin/bash
#SBATCH --partition=cvr
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:hgx:8
#SBATCH --mem=640G

source ~/miniconda3/etc/profile.d/conda.sh
conda activate train

cd /comp_robot/zhoujiazhou/projects/Active-Coconut
export PYTHONPATH="${PWD}:${PWD}/src/train:${PYTHONPATH:-}"

MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"
LVR_HEAD=True
LVR_HEAD_TYPE="ivr"

# IVR 参数
IVR_ITERATIONS=3
IVR_CHUNK_SIZE=""  # 空字符串表示使用默认值（自动选择）
IVR_USE_OUTPUT_NORM=True
IVR_TEMPERATURE=1.0

deepspeed src/train/train_lvr.py \
    --lvr_head $LVR_HEAD \
    --lvr_head_type $LVR_HEAD_TYPE \
    --ivr_iterations $IVR_ITERATIONS \
    --ivr_use_output_norm $IVR_USE_OUTPUT_NORM \
    --ivr_temperature $IVR_TEMPERATURE \
    --model_id $MODEL_NAME \
    ...
```

### 4. 与其他 LVR Head 类型的对比

| 类型 | 参数数量 | 内存占用 | 训练稳定性 | 适用场景 |
|------|---------|---------|-----------|---------|
| `simple` | 少 | 低 | 高 | 简单任务 |
| `glu` | 中 | 中 | 中 | 中等复杂度 |
| `attention-mask` | 多 | 高 | 中 | 复杂任务，需要注意力机制 |
| `slot-attention` | 多 | 中 | 中 | 需要语义槽位的任务 |
| `ivr` | **极少** | **低** | **最高** | **显存受限，需要稳定训练** |

## 内存优化

IVR 实现了多种内存优化策略：

1. **Chunked 处理**：对于长序列（> chunk_size），自动使用 chunked 处理，避免大张量分配
2. **在线计算**：使用在线 softmax 算法，避免存储完整的注意力分数矩阵
3. **自动模式选择**：根据序列长度自动选择最优处理模式

## 调试

启用调试模式查看详细信息：

```bash
export LVR_DEBUG=1
python src/train/train_lvr.py ...
```

调试信息包括：
- 路由迭代过程
- Chunked 处理信息
- 内存使用情况
- 性能统计

## 性能建议

1. **短序列**（< 512 tokens）：使用默认设置即可
2. **中等序列**（512-2048 tokens）：设置 `ivr_chunk_size=512`
3. **长序列**（> 2048 tokens）：设置 `ivr_chunk_size=512` 或 `1024`
4. **显存受限**：设置 `ivr_use_output_norm=False`（会略微降低性能）

## 注意事项

1. IVR 是完全无参数的（除了可选的归一化层），因此不会增加模型参数量
2. IVR 与 `attention-mask` 和 `slot-attention` 一样，需要 `image_embeds` 作为输入
3. 在 DeepSpeed ZeRO-3 环境下，IVR 表现稳定，不会出现 NCCL 超时问题

## 参考文献

- Capsule Networks: [Dynamic Routing Between Capsules](https://arxiv.org/abs/1710.09829)
- Implicit Visual Routing: 基于胶囊网络路由思想的轻量实现


