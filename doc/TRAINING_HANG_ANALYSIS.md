# 训练卡住问题分析

## 问题描述

训练在 step 12 后卡住，最后输出：
```
{'batch_size': 1, 'tokens_per_device': 1033, 'epoch': 0.0}
{'loss_total': 8.152480125427246, 'loss_ce': 8.055577278137207, 'loss_lvr': 0.969028890132904, 'loss_mode_switch': 0.0, 'epoch': 0.0}
  0%|          | 12/2500 [12:59<39:48:02, 57.59s/it]
```

## 可能原因分析

### 1. 磁盘空间不足（最可能）

**症状**：
- 使用 `scripts/zero3_offload.json`（`gather_16bit_weights_on_model_save: true`）
- 每个 checkpoint 约 102GB
- 如果磁盘空间不足，DeepSpeed 的参数同步会卡住

**检查方法**：
```bash
df -h /comp_robot/zhoujiazhou/projects/Active-Coconut
du -sh result/stage1_checkpoints_7b_mlp_ratio0.01/*/checkpoint-*
```

### 2. DeepSpeed ZeRO-3 参数同步问题

**症状**：
- 即使不保存 checkpoint，ZeRO-3 也会进行参数同步
- 如果 checkpoint 目录在 NFS 上，I/O 可能很慢
- 日志中有 NFS 警告：`The cache directory for DeepSpeed Triton autotune, /home/zhoujiazhou/.triton/autotune, appears to be on an NFS system`

**可能原因**：
- NFS 文件系统 I/O 慢
- 磁盘空间不足导致写入失败
- DeepSpeed 的参数同步操作卡住

### 3. 数据加载问题

**症状**：
- 卡在 `tokens_per_device: 1033` 之后
- 可能是数据加载器卡住

**检查方法**：
```bash
# 检查数据加载进程
ps aux | grep dataloader
```

## 解决方案

### 方案 1：使用优化的 DeepSpeed 配置（推荐）

**修改训练脚本**：
```bash
# 在 sft_7b_Focus.sh 中
--deepspeed scripts/zero3_offload_disk_optimized.json \
```

**优点**：
- 减少磁盘使用量 85-88%
- 减少参数同步时间
- 降低磁盘 I/O 压力

### 方案 2：清理磁盘空间

**检查磁盘使用**：
```bash
# 检查磁盘使用情况
df -h /comp_robot/zhoujiazhou/projects/Active-Coconut

# 删除旧的 checkpoint
find result -type d -name "checkpoint-*" -exec du -sh {} \; | sort -h
# 删除不需要的 checkpoint（保留最新的几个）
```

### 方案 3：调整 checkpoint 保存策略

**修改训练脚本**：
```bash
--save_steps 1000 \        # 增加保存间隔
--save_total_limit 3 \    # 减少保留数量
```

### 方案 4：使用本地临时目录（如果可能）

**设置环境变量**：
```bash
# 如果 checkpoint 目录在 NFS 上，考虑使用本地临时目录
export TMPDIR=/tmp/checkpoints
# 或者修改输出目录到本地磁盘
```

## 诊断步骤

### 1. 检查磁盘空间
```bash
./check_disk_and_job.sh 312776
```

### 2. 检查训练进程状态
```bash
# 在计算节点上
squeue -j 312776
ps aux | grep train_lvr
nvidia-smi
```

### 3. 检查是否有磁盘错误
```bash
# 检查系统日志
dmesg | tail -50 | grep -i "disk\|I/O\|error"

# 检查训练日志中的错误
grep -i "error\|exception\|traceback" logs/stage1_7b_mlpmask_b256_lvr0.5_MLPmask_ratio0.05_312776.txt | tail -20
```

### 4. 检查 DeepSpeed 状态
```bash
# 查看 DeepSpeed 日志（如果有）
ls -lh result/stage1_checkpoints_7b_mlp_ratio0.01/*/checkpoint-*/ds_state*
```

## 立即行动建议

1. **检查磁盘空间**：运行 `./check_disk_and_job.sh 312776`
2. **如果磁盘空间不足**：
   - 删除旧的 checkpoint
   - 使用 `zero3_offload_disk_optimized.json`
   - 调整 `save_steps` 和 `save_total_limit`
3. **如果磁盘空间充足**：
   - 检查是否是 NFS I/O 问题
   - 考虑使用本地临时目录
   - 检查数据加载器是否卡住

## 预防措施

1. **使用优化的 DeepSpeed 配置**：
   - `gather_16bit_weights_on_model_save: false`
   - 减少磁盘使用量 85-88%

2. **调整 checkpoint 保存策略**：
   - `save_steps`: 500-1000（根据训练时长）
   - `save_total_limit`: 3-5（只保留必要的 checkpoint）

3. **监控磁盘使用**：
   - 定期检查磁盘空间
   - 设置磁盘使用告警

4. **优化 NFS 性能**（如果使用 NFS）：
   - 使用本地临时目录
   - 设置 `TRITON_CACHE_DIR` 到非 NFS 路径


