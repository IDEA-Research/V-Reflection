#!/bin/bash
# Stage 2 Resampler Attention Visualization
# Outputs per sample: {prefix}_qa.txt (Q&A) + {prefix}.png (combined image)
#   Benchmark: [original image | student attention]; Training: [image+bbox | teacher | student]
# Modes: USE_TRAINING_SET=1 (default) | USE_BENCHMARK_DATASETS=1 | legacy meta
#SBATCH --partition=cvr
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:hgx:1
#SBATCH --mem=80G
#SBATCH --qos=preemptive
#SBATCH --output=logs/visualize_stage2_attention_%j.txt

source ~/miniconda3/etc/profile.d/conda.sh
conda activate train

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="${PWD}:${PWD}/src/train:${PYTHONPATH:-}"
export STAGE2_VIS_ATTENTION=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"
OUTPUT_DIR_TRAIN="${OUTPUT_DIR_TRAIN:-$PROJECT_ROOT/evaluation/results/stage2_attention_vis_train}"
OUTPUT_DIR_VAL="${OUTPUT_DIR_VAL:-$PROJECT_ROOT/evaluation/results/stage2_attention_vis_val}"

# Training set mode (default): uniformly sample from meta
USE_TRAINING_SET="${USE_TRAINING_SET:-0}"
NUM_TRAINING_SAMPLES="${NUM_TRAINING_SAMPLES:-200}"
META_PATH="${META_PATH:-data/meta_data_lvr_sft_stage1.json}"

# Benchmark mode: sample from BLINK, MMVP, Vstar-bench, HRBench4k, HRBench8k, MME-RealWorld-Lite
# For datasets with subsets: 10 samples per subset; for datasets without: 10 total
USE_BENCHMARK_DATASETS="${USE_BENCHMARK_DATASETS:-1}"
SAMPLES_PER_SUBSET="${SAMPLES_PER_SUBSET:-20}"

# Legacy meta mode (specific indices, when both above are 0)
SAMPLE_INDICES="${SAMPLE_INDICES:-0 5 10 15 50 100}"

mkdir -p logs "$OUTPUT_DIR_TRAIN" "$OUTPUT_DIR_VAL"

if [ "$USE_TRAINING_SET" = "1" ]; then
    echo "Running visualization from training set (uniformly ${NUM_TRAINING_SAMPLES} samples)"
    python scripts/visualization/visualize_stage2_attention.py \
        --checkpoint_path "$CHECKPOINT_PATH" \
        --meta_path "$META_PATH" \
        --use_training_set \
        --num_training_samples "$NUM_TRAINING_SAMPLES" \
        --output_dir "$OUTPUT_DIR_TRAIN" \
        --device cuda
elif [ "$USE_BENCHMARK_DATASETS" = "1" ]; then
    # VAL_DATASETS=(BLINK MMVP VSTAR HRBench4K HRBench8K MME-RealWorld-Lite)
    VAL_DATASETS=(MME-RealWorld-Lite)
    echo "Running visualization from Val: one dataset at a time (${SAMPLES_PER_SUBSET} per subset)"
    for ds in "${VAL_DATASETS[@]}"; do
        echo "===== Processing dataset: $ds ====="
        python scripts/visualization/visualize_stage2_attention.py \
            --checkpoint_path "$CHECKPOINT_PATH" \
            --output_dir "$OUTPUT_DIR_VAL" \
            --device cuda \
            --use_benchmark_datasets \
            --benchmark_datasets "$ds" \
            --samples_per_subset "$SAMPLES_PER_SUBSET"
        echo "===== Done: $ds ====="
    done
else
    echo "Running visualization from meta_path (sample_indices)"
    python scripts/visualization/visualize_stage2_attention.py \
        --checkpoint_path "$CHECKPOINT_PATH" \
        --meta_path "$META_PATH" \
        --sample_indices $SAMPLE_INDICES \
        --output_dir "$OUTPUT_DIR_VAL" \
        --device cuda
fi
