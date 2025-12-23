# LVRHeadTransformerBlock 使用说明

## 概述

`LVRHeadTransformerBlock` 是一个新的 LVR Head 实现，它复用 Qwen-7B 预训练的 transformer block 来进行 cross-attention，替代原有的 `LVRHeadAttention`。这个方案的优势包括：

1. **利用预训练知识**：复用 Qwen-7B 的预训练参数，可能比从头训练更稳定
2. **更强的特征融合**：Transformer block 的 MLP 部分可以提供更强的特征融合能力
3. **灵活的冻结策略**：可以选择冻结 attention 或 MLP 部分，减少显存占用和训练参数
4. **显存优化**：支持 chunked processing 处理长序列，避免显存溢出

## 使用方法

### 1. 配置文件中设置

在模型配置文件中，将 `lvr_head_type` 设置为 `'transformer-block'`：

```python
config.lvr_head = True
config.lvr_head_type = 'transformer-block'

# 可选配置参数
config.transformer_block_layer_idx = None  # None 表示使用中间层，也可以指定层索引（0-based）
config.freeze_transformer_attention = False  # 是否冻结 attention 权重
config.freeze_transformer_mlp = False  # 是否冻结 MLP 权重
config.use_chunked_attention = True  # 是否使用 chunked processing
config.transformer_chunk_size = 512  # chunk 大小
config.use_flash_attention = False  # 是否使用 Flash Attention（如果可用）
```

### 2. 参数说明

- **transformer_block_layer_idx**: 
  - `None` (默认): 自动选择中间层（`num_layers // 2`）
  - 整数: 指定要使用的层索引（0-based），例如 `10` 表示使用第 11 层
  
- **freeze_transformer_attention**: 
  - `False` (默认): 允许 fine-tune attention 权重
  - `True`: 冻结 attention 权重，只训练其他部分
  
- **freeze_transformer_mlp**:
  - `False` (默认): 允许 fine-tune MLP 权重
  - `True`: 冻结 MLP 权重，只训练其他部分
  
- **use_chunked_attention**:
  - `True` (默认): 对长序列使用 chunked processing，避免显存溢出
  - `False`: 使用标准 attention（仅适用于短序列）
  
- **transformer_chunk_size**:
  - `512` (默认): chunk 大小，可以根据显存情况调整
  - 较小的值（如 256）可以节省显存，但可能增加计算时间
  - 较大的值（如 1024）可以减少计算时间，但需要更多显存

### 3. 显存优化建议

如果遇到显存不足或 NCCL 内存溢出的问题，可以尝试以下策略：

1. **冻结部分参数**：
   ```python
   config.freeze_transformer_attention = True  # 冻结 attention
   config.freeze_transformer_mlp = True  # 冻结 MLP
   ```

2. **减小 chunk 大小**：
   ```python
   config.transformer_chunk_size = 256  # 或更小
   ```

3. **使用中间层**（通常中间层的参数更稳定）：
   ```python
   config.transformer_block_layer_idx = None  # 自动选择中间层
   ```

4. **启用 Flash Attention**（如果可用）：
   ```python
   config.use_flash_attention = True
   ```

## 工作原理

`LVRHeadTransformerBlock` 的核心思想是：

1. **提取 Transformer Block**：从 Qwen-7B 模型中提取一个 transformer decoder layer（默认使用中间层）

2. **Cross-Attention 改造**：
   - Query: 来自 LLM 的 hidden state `h_t`（经过 input_layernorm）
   - Key/Value: 来自 image embeddings
   - 使用预训练的 attention 权重进行 cross-attention

3. **特征融合**：
   - 经过 cross-attention 后，通过 MLP 进行进一步的特征融合
   - 使用残差连接保持梯度流动

4. **显存优化**：
   - 对于长序列，使用 chunked processing 和 online softmax 算法
   - 避免存储完整的 attention 矩阵，显著减少显存占用

## 与 LVRHeadAttention 的对比

| 特性 | LVRHeadAttention | LVRHeadTransformerBlock |
|------|------------------|------------------------|
| 参数量 | ~25.7M (mlp_ratio=1.0) | 复用预训练参数 |
| 预训练知识 | 无 | 有（来自 Qwen-7B） |
| MLP 融合 | 无 | 有（更强的特征融合） |
| 冻结策略 | 不支持 | 支持（可冻结 attention/MLP） |
| 显存占用 | 中等 | 可通过冻结参数降低 |
| 适用场景 | 通用 | 需要利用预训练知识的场景 |

## 注意事项

1. **层选择**：建议使用中间层（默认），因为中间层通常具有更好的特征表示能力

2. **参数共享**：`LVRHeadTransformerBlock` 直接使用原始 transformer block 的权重（不是复制），这意味着：
   - 如果 `freeze_attention=False` 或 `freeze_mlp=False`，训练会修改原始模型的权重
   - 如果需要保持原始模型不变，需要先复制 transformer block

3. **兼容性**：确保使用的 Qwen 模型版本支持 `model.layers` 属性

4. **调试**：设置环境变量 `LVR_DEBUG=1` 可以启用详细的调试信息

## 示例配置

```python
# 最小显存占用配置
config.lvr_head_type = 'transformer-block'
config.transformer_block_layer_idx = None
config.freeze_transformer_attention = True
config.freeze_transformer_mlp = True
config.use_chunked_attention = True
config.transformer_chunk_size = 256

# 平衡配置（推荐）
config.lvr_head_type = 'transformer-block'
config.transformer_block_layer_idx = None
config.freeze_transformer_attention = False
config.freeze_transformer_mlp = False
config.use_chunked_attention = True
config.transformer_chunk_size = 512

# 最大性能配置（需要足够显存）
config.lvr_head_type = 'transformer-block'
config.transformer_block_layer_idx = None
config.freeze_transformer_attention = False
config.freeze_transformer_mlp = False
config.use_chunked_attention = False  # 不使用 chunked processing
config.use_flash_attention = True
```

