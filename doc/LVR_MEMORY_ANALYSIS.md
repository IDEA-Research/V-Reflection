# LVRHeadIntrinsicSimilarity 内存分析

## 问题诊断

虽然 `LVRHeadIntrinsicSimilarity` 本身**参数很少**（只有可选的 `output_norm`），但**前向传播过程中会创建大量中间张量**，导致内存急剧增加。

## 内存瓶颈分析

### 1. 中间张量内存占用

#### 非chunked版本 (`_intrinsic_similarity_2d` / `_intrinsic_similarity_3d`)

```python
# 步骤1: similarity = h_t @ Z_grid^T
similarity: (batch_size, num_image_tokens)  # 例如: (4, 2048) = 8K 个float32 = 32KB

# 步骤2: attention_weights = softmax(similarity)
attention_weights: (batch_size, num_image_tokens)  # 又一个 (4, 2048) = 32KB

# 步骤3: V_focal = attention_weights @ Z_grid
# Z_grid: (batch_size, max_num_tokens, hidden_size) 或 (num_image_tokens, hidden_size)
# 例如: (4, 2048, 4096) = 33.5M float32 = 134MB
```

**内存峰值**：
- `similarity`: `batch_size × num_image_tokens × 4 bytes`
- `attention_weights`: `batch_size × num_image_tokens × 4 bytes`
- `Z_grid`: `batch_size × max_num_tokens × hidden_size × 4 bytes`

**当 batch_size=4, num_tokens=2048, hidden_size=4096 时**：
- similarity: 4 × 2048 × 4 = 32 KB
- attention_weights: 4 × 2048 × 4 = 32 KB
- Z_grid: 4 × 2048 × 4096 × 4 = **134 MB** (主要瓶颈)
- **总计**: ~134 MB (仅这一个head)

#### Chunked版本的缺陷

```python
# _intrinsic_similarity_2d_chunked 或 _intrinsic_similarity_3d_chunked

# 虽然计算similarity时用了chunk:
chunk_similarities = []
for chunk_idx in range(num_chunks):
    chunk_similarity = ...  # (batch_size, chunk_size)
    chunk_similarities.append(chunk_similarity)

# 但最后还是cat成完整张量:
similarity = torch.cat(chunk_similarities, dim=1)  # (batch_size, num_tokens)
attention_weights = F.softmax(similarity, dim=1)  # 需要完整张量！
```

**问题**：chunked版本**并没有真正节省内存**，因为：
1. 所有chunk的similarity最终还是要合并成完整张量
2. `softmax` 需要完整的 `similarity` 张量才能正确归一化
3. `attention_weights` 仍然是完整大小

### 2. Batch Size 的影响

**内存增长是线性的**：
- batch_size=1: `(1, num_tokens)` 的中间张量
- batch_size=4: `(4, num_tokens)` 的中间张量 → **4倍内存**

**实际场景**：
- 如果 `num_tokens=2048`, `hidden_size=4096`
- batch_size=1: Z_grid = 1 × 2048 × 4096 × 4 = 33.5 MB
- batch_size=4: Z_grid = 4 × 2048 × 4096 × 4 = **134 MB** (4倍)

### 3. Image Tokens 的存储

**是的，主要原因是需要把 image tokens 拿过来**：

1. **Z_grid 本身占用大量内存**：
   - 每个image可能有几百到几千个tokens
   - 每个token是 `hidden_size` 维的向量（通常4096）
   - batch_size增加时，内存线性增长

2. **梯度计算**：
   - 虽然3D版本有 `Z_grid.detach()`，但2D版本没有
   - 如果vision tower需要梯度，会进一步增加内存

3. **中间计算**：
   - `torch.matmul(h_t, Z_grid.t())` 需要Z_grid在内存中
   - `torch.matmul(attention_weights, Z_grid)` 又需要Z_grid

## 根本原因

**"零参数"不等于"零内存"**：
- 参数少：只有 `output_norm` 的权重（约 `2 × hidden_size` 个参数）
- 但前向传播需要：
  - 完整的 `image_embeds` 张量
  - 完整的 `similarity` 和 `attention_weights` 张量
  - 这些张量的大小与 `batch_size × num_tokens` 成正比

## 优化建议

### 1. 真正的在线Softmax（推荐）

使用在线softmax算法，避免存储完整的similarity张量：

```python
def _intrinsic_similarity_3d_online_softmax(
    self,
    h_t: torch.Tensor,
    Z_grid: torch.Tensor,
    image_attention_mask: Optional[torch.Tensor],
    chunk_size: int
) -> torch.Tensor:
    """
    使用在线softmax，避免存储完整的similarity张量
    """
    batch_size = h_t.shape[0]
    max_num_tokens = Z_grid.shape[1]
    hidden_size = h_t.shape[1]
    device = h_t.device
    dtype = h_t.dtype
    
    Z_grid = Z_grid.detach()
    num_chunks = (max_num_tokens + chunk_size - 1) // chunk_size
    
    # 在线softmax算法
    max_scores = torch.full((batch_size,), float('-inf'), device=device, dtype=dtype)
    exp_sum = torch.zeros(batch_size, device=device, dtype=dtype)
    weighted_sum = torch.zeros(batch_size, hidden_size, device=device, dtype=dtype)
    
    # 第一遍：计算max和exp_sum
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
        old_max = max_scores
        max_scores = torch.maximum(max_scores, chunk_max)
        
        if chunk_idx > 0:
            exp_adjust = torch.exp(old_max - max_scores)
            exp_sum = exp_sum * exp_adjust
            weighted_sum = weighted_sum * exp_adjust.unsqueeze(-1)
        
        scores_diff = chunk_similarity - max_scores.unsqueeze(-1)
        chunk_exp = torch.exp(scores_diff)
        
        if image_attention_mask is not None:
            chunk_mask = image_attention_mask[:, chunk_start:chunk_end]
            chunk_exp = chunk_exp * chunk_mask.to(dtype)
        
        exp_sum += chunk_exp.sum(dim=1)
        weighted_sum += torch.bmm(chunk_exp.unsqueeze(1), chunk_tokens).squeeze(1)
    
    # 归一化
    eps = 1e-8
    V_focal = weighted_sum / (exp_sum.unsqueeze(-1) + eps)
    
    return V_focal
```

### 2. 强制使用chunked处理

修改 `forward` 方法，强制对batch_size>1的情况使用chunked处理：

```python
# 在forward方法中
if image_embeds.dim() == 3:
    max_num_tokens = image_embeds.shape[1]
    
    # 强制使用chunked处理（即使序列不长）
    # 因为batch_size>1时内存压力更大
    use_chunked = max_num_tokens > 256 or batch_size > 1  # 降低阈值
```

### 3. 添加detach到2D版本

```python
def _intrinsic_similarity_2d(self, h_t, Z_grid):
    Z_grid = Z_grid.detach()  # 添加这行
    similarity = torch.matmul(h_t, Z_grid.t())
    # ...
```

### 4. 减小chunk_size

对于batch_size>1的情况，使用更小的chunk_size：

```python
if batch_size > 1:
    effective_chunk_size = min(self.chunk_size, 256)  # 更小的chunk
else:
    effective_chunk_size = self.chunk_size
```

## 总结

**问题根源**：
1. ✅ **是的，主要原因是需要把image tokens拿过来** - Z_grid占用大量内存
2. ✅ **batch_size增加时，所有中间张量线性增长**
3. ✅ **chunked版本并没有真正节省内存** - 因为softmax需要完整张量

**解决方案**：
- 实现真正的在线softmax算法（避免存储完整similarity）
- 强制batch_size>1时使用chunked处理
- 添加detach减少梯度内存
- 使用更小的chunk_size

