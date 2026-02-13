#!/bin/bash
#SBATCH --partition=cvr
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:hgx:1
#SBATCH --mem=128G
#SBATCH --qos=preemptive
#SBATCH --output=/comp_robot/zhoujiazhou/projects/Active-Coconut/logs/dit_sanity_check_%j.txt

# DiT Sanity Check: Teacher Forcing 一步验证法
# 用途: 验证 DiT 训练是否正确, 排查推理全是噪点的问题

source ~/miniconda3/etc/profile.d/conda.sh
conda activate train

cd /comp_robot/zhoujiazhou/projects/Active-Coconut
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

# 可选: 指定 checkpoint 路径 (默认使用 LATENT12 的 checkpoint-2500)
export CHECKPOINT_PATH="${CHECKPOINT_PATH:-/comp_robot/zhoujiazhou/projects/Active-Coconut/result/dit_recon/SFT_dit_recon_steps2500_b4_dit1.0_resampler0.1_acc8_LATENT12/checkpoint-2500}"
export NUM_SAMPLES="${NUM_SAMPLES:-5}"
export DIT_STEPS="${DIT_STEPS:-20}"

echo "=========================================="
echo "DiT Sanity Check"
echo "=========================================="
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Samples: $NUM_SAMPLES"
echo "DiT Steps: $DIT_STEPS"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "=========================================="

python scripts/debug_dit_sanity_check.py

EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo "Sanity check failed with exit code $EXIT_CODE"
    exit $EXIT_CODE
fi

echo ""
echo "=========================================="
echo "Sanity check completed!"
echo "Results saved to: evaluation/dit_sanity_check/"
echo "=========================================="
