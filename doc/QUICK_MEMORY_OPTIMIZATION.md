# 快速内存优化参考

## 🎯 优化优先级（按性能影响排序）

### ✅ 第一优先级：无性能损失的优化

这些优化**不会影响模型性能**，应该**优先使用**：

1. **减少序列长度**
   ```bash
   LST=2048  # 从 4096 改为 2048
   ```
   - 内存减少：~75%
   - 性能影响：无（只是过滤长序列）

2. **增加梯度累积**
   ```bash
   GRAD_ACCUM_STEPS=32  # 从 16 改为 32
   ```
   - 内存减少：显著
   - 性能影响：无（有效 batch size 不变）

3. **优化 DeepSpeed 配置**
   ```bash
   --deepspeed scripts/zero3_offload_disk_optimized_memory.json
   ```
   - 内存减少：中等
   - 性能影响：几乎无

4. **减少图像 token**
   ```bash
   MAX_TOKEN=2560  # 从 5120 改为 2560
   ```
   - 内存减少：中等
   - 性能影响：轻微（取决于数据分布）

### ⚠️ 第二优先级：轻微性能损失的优化

这些优化**可能轻微影响性能**（< 5%），在阶段 1 不够时使用：

5. **减少 slot_iters**
   ```bash
   SLOT_ITERS=2  # 从 3 改为 2
   ```
   - 内存减少：~33% 槽位初始化计算
   - 性能影响：< 5%

### ⚠️⚠️ 第三优先级：可能影响性能的优化

这些优化**可能明显影响性能**（5-20%），**谨慎使用**：

6. **减少 num_slots**
   ```bash
   NUM_SLOTS=6  # 从 8 改为 6（性能损失 5-10%）
   NUM_SLOTS=4  # 从 8 改为 4（性能损失 10-20%，不推荐）
   ```
   - 内存减少：25-50%
   - 性能影响：5-20%

7. **减少 top_k**（⚠️⚠️⚠️ 不推荐）
   ```bash
   TOP_K=1  # 从 2 改为 1（性能损失 15-25%，不推荐）
   ```
   - 内存减少：~50%
   - 性能影响：15-25%（显著降低表达能力）

## 📋 推荐配置方案

### 方案 A：保守优化（推荐，性能损失 < 5%）

```bash
# 无性能损失的优化
LST=2048
GRAD_ACCUM_STEPS=32
MAX_TOKEN=2560
--deepspeed scripts/zero3_offload_disk_optimized_memory.json

# 轻微性能损失的优化
SLOT_ITERS=2
NUM_SLOTS=8   # 保持不变
TOP_K=2       # 保持不变
```

**预期**：
- ✅ 内存减少：~60-70%
- ✅ 性能损失：< 5%
- ✅ **推荐用于生产环境**

### 方案 B：平衡优化（性能损失 5-10%）

```bash
# 无性能损失的优化
LST=2048
GRAD_ACCUM_STEPS=32
MAX_TOKEN=2560
--deepspeed scripts/zero3_offload_disk_optimized_memory.json

# 轻微性能损失的优化
SLOT_ITERS=2
NUM_SLOTS=6   # 从 8 减少到 6
TOP_K=2       # 保持不变
```

**预期**：
- ✅ 内存减少：~70-75%
- ⚠️ 性能损失：5-10%
- ⚠️ **仅在方案 A 不够时使用**

### 方案 C：激进优化（性能损失 10-20%，不推荐）

```bash
LST=1536
GRAD_ACCUM_STEPS=64
MAX_TOKEN=2048
SLOT_ITERS=2
NUM_SLOTS=4   # 从 8 减少到 4
TOP_K=1       # 从 2 减少到 1（不推荐）
```

**预期**：
- ✅ 内存减少：~80-85%
- ⚠️⚠️ 性能损失：15-25%
- ⚠️⚠️ **仅用于极端内存受限环境**

## 🚀 快速开始

### 步骤 1：应用保守优化

编辑 `scripts/sbatch/sft_7b_SlotAttention.sh`：

```bash
# 修改这些参数
LST=2048
GRAD_ACCUM_STEPS=32
MAX_TOKEN=2560
SLOT_ITERS=2

# 修改 DeepSpeed 配置路径
--deepspeed scripts/zero3_offload_disk_optimized_memory.json
```

### 步骤 2：测试训练

```bash
sbatch scripts/sbatch/sft_7b_SlotAttention.sh
```

### 步骤 3：如果仍然 OOM

尝试平衡优化（方案 B）：
```bash
NUM_SLOTS=6  # 从 8 改为 6
```

### 步骤 4：监控内存

```bash
watch -n 1 nvidia-smi
```

## 📚 详细文档

- **`MEMORY_OPTIMIZATION_GUIDE.md`** - 完整的内存优化指南
- **`SLOT_ATTENTION_PERFORMANCE_TRADEOFF.md`** - 详细的性能权衡分析

## ⚠️ 关键原则

1. ✅ **优先优化不影响性能的参数**
2. ⚠️ **最后才考虑影响性能的参数**
3. 📌 **保持 `top_k=2` 和 `num_slots>=6` 以维持性能**
4. 🔍 **渐进式测试**：先试方案 A，不够再试方案 B

## 💡 性能影响快速参考

| 参数 | 减少值 | 内存节省 | 性能影响 | 推荐度 |
|------|--------|----------|----------|--------|
| `LST` | 4096→2048 | ~75% | 无 | ⭐⭐⭐⭐⭐ |
| `GRAD_ACCUM_STEPS` | 16→32 | 显著 | 无 | ⭐⭐⭐⭐⭐ |
| `MAX_TOKEN` | 5120→2560 | 中等 | 轻微 | ⭐⭐⭐⭐ |
| `SLOT_ITERS` | 3→2 | ~33% | < 5% | ⭐⭐⭐⭐ |
| `NUM_SLOTS` | 8→6 | ~25% | 5-10% | ⭐⭐⭐ |
| `NUM_SLOTS` | 8→4 | ~50% | 10-20% | ⭐⭐ |
| `TOP_K` | 2→1 | ~50% | 15-25% | ⭐ |


