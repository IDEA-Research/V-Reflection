# LVRHeadIntrinsicSimilarity 内存消耗分析

## 问题诊断

虽然 `LVRHeadIntrinsicSimilarity` 本身是"零参数"的（除了可选的 output_norm），但它仍然会**大幅增加内存消耗**，主要原因如下：

### 1. **image_embeds 张量本身的内存占用**

当 `MAX_INSTANCE_PER_BATCH=4` 时，每个 batch 可能包含多个图像实例：
- `image_embeds` 形状：`(batch_size, max_num_tokens, hidden_size)`
- 假设 `hidden_size=4096`, `max_num_tokens=576`（常见值）
- 单个样本内存：`576 * 4096 * 4 bytes = 9.4 MB` (float32)
- 当 `batch_size=4` 时：`4 * 9.4 MB = 37.6 MB`（仅 image_embeds）

**关键问题**：即使 `detach()` 了，张量本身仍然占用内存。`detach()` 只是断开梯度计算图，不会减少内存占用。

### 2. **中间张量的内存消耗**

#### 非 chunked 版本 (`_intrinsic_similarity_3d`)：
```python
# 1. transpose 操作创建新张量
Z_grid.transpose(1, 2)  # (batch_size, hidden_size, max_num_tokens)
# 内存：batch_size * hidden_size * max_num_tokens * 4 bytes

# 2. similarity 矩阵
similarity = torch.bmm(...)  # (batch_size, max_num_tokens)
# 内存：batch_size * max_num_tokens * 4 bytes

# 3. attention_weights 矩阵
attention_weights = F.softmax(similarity, dim=1)  # (batch_size, max_num_tokens)
# 内存：batch_size * max_num_tokens * 4 bytes
```

**总内存峰值**（不包括输入）：
- `batch_size * (hidden_size * max_num_tokens + 2 * max_num_tokens) * 4 bytes`
- 当 `batch_size=4, max_num_tokens=576, hidden_size=4096`：
  - `4 * (4096 * 576 + 2 * 576) * 4 = 37.7 MB`

#### Chunked 版本 (`_intrinsic_similarity_3d_chunked`)：
虽然使用了 chunked 处理，但仍有问题：

```python
# 问题 1: chunk_similarities 列表存储所有 chunk 的结果
chunk_similarities = []
for chunk_idx in range(num_chunks):
    chunk_similarities.append(chunk_similarity)  # 累积存储

# 问题 2: torch.cat 创建完整的 similarity 矩阵
similarity = torch.cat(chunk_similarities, dim=1)  # (batch_size, max_num_tokens)
# 这仍然创建了完整的相似度矩阵！

# 问题 3: attention_weights 仍然是完整的
attention_weights = F.softmax(similarity, dim=1)  # (batch_size, max_num_tokens)
```

**关键问题**：chunked 版本虽然分块计算相似度，但最后仍然 `cat` 成完整矩阵，然后计算 softmax，这**没有真正减少内存峰值**。

### 3. **为什么 MAX_INSTANCE_PER_BATCH 从 4 降到 1 能解决问题？**

- 当 `batch_size` 从 4 降到 1 时，所有中间张量的大小都**线性减少 4 倍**
- 内存消耗与 `batch_size` 成正比
- 例如：`similarity` 矩阵从 `(4, 576)` 降到 `(1, 576)`，内存从 `9.2 KB` 降到 `2.3 KB`

### 4. **真正的内存瓶颈**

1. **image_embeds 本身**：必须保留在内存中（这是输入）
2. **transpose 操作**：创建了 `(batch_size, hidden_size, max_num_tokens)` 的临时张量
3. **完整的 similarity 和 attention_weights 矩阵**：即使 chunked 版本也没有真正避免

## 优化建议

### 方案 1: 真正的在线 Softmax（推荐）

使用在线 softmax 算法，避免存储完整的 similarity 和 attention_weights 矩阵：

```python
def _intrinsic_similarity_3d_online_softmax(
    self,
    h_t: torch.Tensor,
    Z_grid: torch.Tensor,
    image_attention_mask: Optional[torch.Tensor],
    chunk_size: int
) -> torch.Tensor:
    """
    使用在线 softmax 算法，真正避免存储完整的相似度矩阵。
    """
    batch_size = h_t.shape[0]
    max_num_tokens = Z_grid.shape[1]
    hidden_size = h_t.shape[1]
    device = h_t.device
    dtype = h_t.dtype
    
    num_chunks = (max_num_tokens + chunk_size - 1) // chunk_size
    
    Z_grid = Z_grid.detach()
    
    # 在线 softmax 算法
    max_similarity = torch.full((batch_size,), float('-inf'), device=device, dtype=dtype)
    exp_sum = torch.zeros(batch_size, device=device, dtype=dtype)
    weighted_sum = torch.zeros(batch_size, hidden_size, device=device, dtype=dtype)
    
    # 第一遍：计算 max 和 exp_sum
    for chunk_idx in range(num_chunks):
        chunk_start = chunk_idx * chunk_size
        chunk_end = min(chunk_start + chunk_size, max_num_tokens)
        chunk_tokens = Z_grid[:, chunk_start:chunk_end]
        
        chunk_similarity = torch.bmm(
            h_t.unsqueeze(1),
            chunk_tokens.transpose(1, 2)
        ).squeeze(1)
        
        if image_attention_mask is not None:
            chunk_mask = image_attention_mask[:, chunk_start:chunk_end]
            chunk_similarity = chunk_similarity.masked_fill(~chunk_mask, float('-inf'))
        
        chunk_max = chunk_similarity.max(dim=1)[0]
        max_similarity = torch.maximum(max_similarity, chunk_max)
    
    # 第二遍：计算加权和
    for chunk_idx in range(num_chunks):
        chunk_start = chunk_idx * chunk_size
        chunk_end = min(chunk_start + chunk_size, max_num_tokens)
        chunk_tokens = Z_grid[:, chunk_start:chunk_end]
        
        chunk_similarity = torch.bmm(
            h_t.unsqueeze(1),
            chunk_tokens.transpose(1, 2)
        ).squeeze(1)
        
        if image_attention_mask is not None:
            chunk_mask = image_attention_mask[:, chunk_start:chunk_end]
            chunk_similarity = chunk_similarity.masked_fill(~chunk_mask, float('-inf'))
        
        # 在线 softmax
        exp_scores = torch.exp(chunk_similarity - max_similarity.unsqueeze(-1))
        if image_attention_mask is not None:
            exp_scores = exp_scores * chunk_mask.to(dtype)
        
        exp_sum += exp_scores.sum(dim=1)
        
        # 加权求和
        weighted_chunk = torch.bmm(
            exp_scores.unsqueeze(1),
            chunk_tokens
        ).squeeze(1)
        weighted_sum += weighted_chunk
    
    # 归一化
    V_focal = weighted_sum / (exp_sum.unsqueeze(-1) + 1e-8)
    return V_focal
```

### 方案 2: 避免 transpose 操作

使用 `torch.einsum` 或直接矩阵乘法，避免显式的 transpose：

```python
# 当前方式（创建临时张量）
similarity = torch.bmm(
    h_t.unsqueeze(1),  # (batch_size, 1, hidden_size)
    Z_grid.transpose(1, 2)  # (batch_size, hidden_size, max_num_tokens) <- 新张量
).squeeze(1)

# 优化方式（避免 transpose）
similarity = torch.einsum('bh,bnh->bn', h_t, Z_grid)
# 或者
similarity = (h_t.unsqueeze(1) @ Z_grid.transpose(1, 2)).squeeze(1)
# 但 einsum 可能更高效
```

### 方案 3: 更激进的 chunked 处理

强制使用更小的 chunk_size，即使对于"短"序列也使用 chunked 处理。

## 总结

**核心问题**：
1. ✅ 你的猜测是对的：**image_embeds 本身占用大量内存**
2. ❌ 但更严重的是：**中间张量（similarity, attention_weights）也占用大量内存**
3. ❌ **chunked 版本没有真正解决问题**：仍然创建完整的 similarity 和 attention_weights 矩阵

**解决方案**：
- 使用**在线 softmax 算法**（方案 1），这是最有效的优化
- 避免不必要的 transpose 操作（方案 2）
- 强制使用更小的 chunk_size（方案 3）

