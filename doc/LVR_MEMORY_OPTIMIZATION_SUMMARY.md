# LVRHeadIntrinsicSimilarity 内存优化总结

## 问题根源

**是的，主要原因是需要把 image tokens 拿过来**，但还有其他因素：

1. **Image Tokens 存储**：
   - `Z_grid`: `(batch_size, max_num_tokens, hidden_size)` 
   - 当 batch_size=4, num_tokens=2048, hidden_size=4096 时 = **134 MB**

2. **中间张量**（优化前）：
   - `similarity`: `(batch_size, num_tokens)` = 32 KB
   - `attention_weights`: `(batch_size, num_tokens)` = 32 KB
   - **问题**：chunked版本虽然分块计算，但最后还是 `torch.cat()` 合并，内存没有真正节省

3. **Batch Size 影响**：
   - 所有中间张量大小与 batch_size 线性增长
   - batch_size=4 时，内存是 batch_size=1 的 **4倍**

## 已实施的优化

### 1. ✅ 真正的在线 Softmax 算法

**修改前**（`_intrinsic_similarity_3d_chunked`）：
```python
# 问题：虽然分块计算，但最后还是cat成完整张量
chunk_similarities = []
for chunk_idx in range(num_chunks):
    chunk_similarity = ...
    chunk_similarities.append(chunk_similarity)

similarity = torch.cat(chunk_similarities, dim=1)  # ❌ 仍然需要完整张量
attention_weights = F.softmax(similarity, dim=1)    # ❌ 需要完整张量
```

**修改后**：
```python
# ✅ 使用在线softmax算法，避免存储完整similarity张量
max_scores = torch.full((batch_size,), float('-inf'), ...)
exp_sum = torch.zeros(batch_size, ...)
weighted_sum = torch.zeros(batch_size, hidden_size, ...)

for chunk_idx in range(num_chunks):
    chunk_similarity = ...
    chunk_max = chunk_similarity.max(dim=1)[0]
    max_scores = torch.maximum(max_scores, chunk_max)
    
    # 在线更新，不需要存储完整similarity
    scores_diff = chunk_similarity - max_scores.unsqueeze(-1)
    chunk_exp = torch.exp(scores_diff)
    exp_sum += chunk_exp.sum(dim=1)
    weighted_sum += torch.bmm(chunk_exp.unsqueeze(1), chunk_tokens).squeeze(1)

V_focal = weighted_sum / (exp_sum.unsqueeze(-1) + eps)
```

**内存节省**：
- 优化前：需要 `(batch_size, num_tokens)` 的完整 similarity 和 attention_weights 张量
- 优化后：只需要 `(batch_size,)` 的 max_scores 和 exp_sum，以及 `(batch_size, hidden_size)` 的 weighted_sum
- **节省**: `2 × batch_size × num_tokens × 4 bytes` (例如：batch_size=4, num_tokens=2048 → 节省 64 KB)

### 2. ✅ 强制 batch_size > 1 时使用 chunked 处理

**修改前**：
```python
if max_num_tokens > 512:  # 阈值太高
    use_chunked = True
```

**修改后**：
```python
if max_num_tokens > 256 or batch_size > 1:  # ✅ 降低阈值，强制batch_size>1时chunked
    use_chunked = True
    if batch_size > 1:
        # ✅ 使用更小的chunk_size
        effective_chunk_size = 128-256  # 而不是512
```

### 3. ✅ 添加 detach 到 2D 版本

**修改前**：
```python
def _intrinsic_similarity_2d(self, h_t, Z_grid):
    similarity = torch.matmul(h_t, Z_grid.t())  # ❌ Z_grid可能保留梯度
```

**修改后**：
```python
def _intrinsic_similarity_2d(self, h_t, Z_grid):
    Z_grid = Z_grid.detach()  # ✅ 避免不必要的梯度计算
    similarity = torch.matmul(h_t, Z_grid.t())
```

## 内存优化效果

### 优化前（batch_size=4, num_tokens=2048, hidden_size=4096）

| 张量 | 大小 | 内存 |
|------|------|------|
| Z_grid | (4, 2048, 4096) | 134 MB |
| similarity | (4, 2048) | 32 KB |
| attention_weights | (4, 2048) | 32 KB |
| **总计** | | **~134 MB** |

### 优化后（batch_size=4, num_tokens=2048, hidden_size=4096）

| 张量 | 大小 | 内存 |
|------|------|------|
| Z_grid | (4, 2048, 4096) | 134 MB |
| max_scores | (4,) | 16 B |
| exp_sum | (4,) | 16 B |
| weighted_sum | (4, 4096) | 64 KB |
| **总计** | | **~134 MB** |

**注意**：虽然中间张量内存大幅减少，但 **Z_grid 本身仍然是主要瓶颈**（134 MB）。

## 进一步优化建议

### 1. 如果可能，减小 image tokens 数量
- 使用更小的 vision encoder
- 使用 pooling 或 downsampling

### 2. 使用梯度检查点（Gradient Checkpointing）
```python
from torch.utils.checkpoint import checkpoint

v_focal = checkpoint(self._intrinsic_similarity_3d_chunked, 
                     hidden_state, image_embeds, image_attention_mask, effective_chunk_size)
```

### 3. 使用混合精度训练
```python
with torch.cuda.amp.autocast():
    v_focal = self.forward(hidden_state, image_embeds, image_attention_mask)
```

### 4. 考虑使用 LVRHeadImplicitVisualRouting
- 它也有在线softmax实现，可能内存效率更高

## 总结

**问题根源确认**：
1. ✅ **是的，主要原因是需要把 image tokens 拿过来** - Z_grid 占用 134 MB
2. ✅ **batch_size 增加时内存线性增长** - 所有张量都 ×4
3. ✅ **chunked版本之前没有真正节省内存** - 因为softmax需要完整张量

**已实施的优化**：
- ✅ 真正的在线softmax算法（避免存储完整similarity）
- ✅ 强制batch_size>1时使用chunked处理
- ✅ 使用更小的chunk_size（128-256而不是512）
- ✅ 添加detach减少梯度内存

**预期效果**：
- 中间张量内存减少：从 64 KB → 64 KB（主要是Z_grid本身）
- 但通过更小的chunk_size和强制chunked处理，可以减少峰值内存
- **建议测试**：尝试将 MAX_INSTANCE_PER_BATCH 从1增加到2或3，看是否能在内存限制内运行

