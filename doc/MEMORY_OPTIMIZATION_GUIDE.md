# 内存优化指南 - Slot Attention 训练

## 问题分析

从日志 `stage1_7b_slotattention_b256_lvr0.5_312790.txt` 可以看到：
- **错误位置**：反向传播阶段（backward pass）
- **GPU 内存**：GPU 3 和 GPU 4 都出现 OOM
- **序列长度**：3903, 2771, 3027 tokens（较长）
- **当前配置**：
  - `per_device_train_batch_size=1`
  - `gradient_accumulation_steps=16`
  - `max_packed_tokens=4096`
  - `long_seq_threshold=4096`

## 优化方案

### 1. 减少序列长度（最有效）

**修改 `scripts/sbatch/sft_7b_SlotAttention.sh`**：

```bash
# 从 4096 减少到 2048 或更小
LST=2048  # 或 1536, 1024
MAX_INSTANCE_PER_BATCH=1
MAX_PACKED_TOKENS=$((MAX_INSTANCE_PER_BATCH * LST))
```

**影响**：
- ✅ 显著减少激活值内存（O(seq_len²)）
- ✅ 减少注意力矩阵内存
- ⚠️ 可能过滤掉一些长序列样本

### 2. 优化 DeepSpeed ZeRO-3 配置

**创建新的配置文件 `scripts/zero3_offload_disk_optimized_memory.json`**：

```json
{
  "fp16": {
    "enabled": "auto",
    "loss_scale": 0,
    "loss_scale_window": 1000,
    "initial_scale_power": 16,
    "hysteresis": 2,
    "min_loss_scale": 1
  },
  "bf16": {
    "enabled": "auto"
  },
  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": "auto",
      "betas": "auto",
      "eps": "auto",
      "weight_decay": "auto"
    }
  },
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "offload_param": {
      "device": "cpu",
      "pin_memory": true
    },
    "overlap_comm": true,
    "contiguous_gradients": true,
    "sub_group_size": 1e9,
    "reduce_bucket_size": 2e8,  // 从 5e8 减少到 2e8
    "stage3_prefetch_bucket_size": 2e7,  // 从 5e7 减少到 2e7
    "stage3_param_persistence_threshold": 1e5,  // 明确设置，减少持久化参数
    "stage3_max_live_parameters": 5e8,  // 从 1e9 减少到 5e8
    "stage3_max_reuse_distance": 5e8,  // 从 1e9 减少到 5e8
    "stage3_gather_16bit_weights_on_model_save": false
  },
  "gradient_accumulation_steps": "auto",
  "gradient_clipping": "auto",
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "steps_per_print": 1e5,
  "wall_clock_breakdown": false
}
```

### 3. 增加梯度累积步数

**修改 `scripts/sbatch/sft_7b_SlotAttention.sh`**：

```bash
# 从 16 增加到 32 或更多
GRAD_ACCUM_STEPS=32  # 或 64
BATCH_PER_DEVICE=1  # 保持为 1
```

**影响**：
- ✅ 减少峰值内存（更小的有效 batch size）
- ⚠️ 训练速度可能稍慢

### 4. 优化 Slot Attention 参数（⚠️ 可能影响性能）

**重要**：这些优化可能影响模型性能，应该**最后考虑**。详见 `SLOT_ATTENTION_PERFORMANCE_TRADEOFF.md`。

**保守优化（推荐，性能损失 < 5%）**：

```bash
# 在 sft_7b_SlotAttention.sh 中修改
SLOT_ITERS=2  # 从 3 减少到 2（轻微影响）
NUM_SLOTS=8   # 保持不变
TOP_K=2       # 保持不变
```

**平衡优化（性能损失 5-10%）**：

```bash
SLOT_ITERS=2  # 从 3 减少到 2
NUM_SLOTS=6   # 从 8 减少到 6（可能影响性能）
TOP_K=2       # 保持不变
```

**激进优化（性能损失 10-20%，不推荐）**：

```bash
NUM_SLOTS=4   # 从 8 减少到 4（显著影响性能）
TOP_K=1       # 从 2 减少到 1（显著影响性能）
SLOT_ITERS=2  # 从 3 减少到 2
```

**影响**：
- ✅ 减少 LVR head 的内存占用
- ⚠️ **可能影响模型性能**（详见性能权衡文档）
- 📌 **建议**：优先使用无性能损失的优化（方案 1-3），最后才考虑此优化

### 5. 启用更激进的 CPU Offload

**在训练脚本中添加环境变量**：

```bash
# 在 sft_7b_SlotAttention.sh 中添加
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
```

### 6. 减少图像 token 数量

**修改图像 token 限制**：

```bash
# 在 sft_7b_SlotAttention.sh 中修改
MAX_TOKEN=2560  # 从 5120 减少到 2560
MIN_TOKEN=128
```

### 7. 使用梯度检查点（已启用，但可以优化）

确保 `gradient_checkpointing=True`（已启用），这是正确的。

### 8. 过滤超长序列

**启用序列长度过滤**：

```bash
# 在 sft_7b_SlotAttention.sh 中修改
MAX_SEQ_LENGTH_FILTER=4096  # 过滤超过 4096 tokens 的序列
```

然后在训练命令中添加：
```bash
--max_seq_length_filter $MAX_SEQ_LENGTH_FILTER
```

## 推荐的组合方案

### 方案 A：保守优化（保持性能）

```bash
LST=2048
MAX_PACKED_TOKENS=2048
GRAD_ACCUM_STEPS=32
MAX_TOKEN=2560
# 使用新的 DeepSpeed 配置（减少 bucket size）
```

### 方案 B：激进优化（最大化内存节省）

```bash
LST=1536
MAX_PACKED_TOKENS=1536
GRAD_ACCUM_STEPS=64
MAX_TOKEN=2048
NUM_SLOTS=4
TOP_K=1
# 使用新的 DeepSpeed 配置
```

### 方案 C：平衡优化（推荐，性能损失 < 5%）

```bash
LST=2048
MAX_PACKED_TOKENS=2048
GRAD_ACCUM_STEPS=32
MAX_TOKEN=2560
SLOT_ITERS=2  # 从 3 减少到 2（轻微影响）
NUM_SLOTS=8   # 保持不变（避免性能损失）
TOP_K=2       # 保持
# 使用优化的 DeepSpeed 配置
```

**预期**：
- 内存减少：~60-70%
- 性能损失：< 5%
- **推荐用于生产环境**

### 方案 D：平衡优化（性能损失 5-10%）

```bash
LST=2048
MAX_PACKED_TOKENS=2048
GRAD_ACCUM_STEPS=32
MAX_TOKEN=2560
SLOT_ITERS=2
NUM_SLOTS=6   # 从 8 减少到 6（可能影响性能）
TOP_K=2       # 保持
# 使用优化的 DeepSpeed 配置
```

**预期**：
- 内存减少：~70-75%
- 性能损失：5-10%
- **仅在方案 C 不够时使用**

## 实施步骤

1. **创建优化的 DeepSpeed 配置**：
   ```bash
   cp scripts/zero3_offload_disk_optimized.json \
      scripts/zero3_offload_disk_optimized_memory.json
   # 然后按照上面的配置修改
   ```

2. **修改训练脚本**：
   ```bash
   # 编辑 scripts/sbatch/sft_7b_SlotAttention.sh
   # 应用方案 C 的配置
   ```

3. **更新 DeepSpeed 配置文件路径**：
   ```bash
   --deepspeed scripts/zero3_offload_disk_optimized_memory.json
   ```

4. **重新提交训练任务**

## 监控内存使用

训练时监控 GPU 内存：
```bash
watch -n 1 nvidia-smi
```

如果仍然 OOM，可以进一步：
- 减少 `LST` 到 1024
- 增加 `GRAD_ACCUM_STEPS` 到 64
- 减少 `NUM_SLOTS` 到 4

## 注意事项

1. **序列长度过滤**：减少 `max_packed_tokens` 会过滤掉长序列，可能影响训练数据分布
2. **训练速度**：增加 `gradient_accumulation_steps` 会稍微降低训练速度
3. **模型性能**：减少 slot attention 参数可能影响模型性能，需要权衡
   - 📌 **详见** `SLOT_ATTENTION_PERFORMANCE_TRADEOFF.md` 了解详细影响
   - 📌 **建议**：优先使用无性能损失的优化（方案 1-3），最后才考虑 slot attention 参数优化
4. **DeepSpeed 配置**：减少 bucket size 可能稍微影响通信效率，但能节省内存

## 性能权衡参考

详细的性能影响分析请参考：
- **`SLOT_ATTENTION_PERFORMANCE_TRADEOFF.md`** - 详细的参数性能权衡分析

**关键原则**：
1. ✅ **优先优化不影响性能的参数**（序列长度、梯度累积、DeepSpeed 配置）
2. ⚠️ **最后才考虑影响性能的参数**（slot attention 参数）
3. 📌 **保持 `top_k=2` 和 `num_slots>=6` 以维持性能**

