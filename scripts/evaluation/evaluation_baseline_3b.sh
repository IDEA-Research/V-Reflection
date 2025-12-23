#!/bin/bash
#SBATCH --partition=cvr
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:hgx:1 # A800
#SBATCH --mem=80G
#SBATCH --qos=preemptive #specify preemptive Q0S
#SBATCH --output=/comp_robot/zhoujiazhou/projects/Active-Coconut/logs/baseline_qwen2.5vl_3b_%j.txt

# ============================================================================
# Configuration - Model Settings
# ============================================================================
CHECKPOINT_PATH="Qwen/Qwen2.5-VL-3B-Instruct"
EVAL_STEP_LIST="${EVAL_STEP_LIST:-4}"
# ============================================================================

# Initialize conda if available
if command -v conda &> /dev/null; then
    # Try to initialize conda
    if [ -f "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" ]; then
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate train 2>/dev/null || true
    elif [ -n "$CONDA_DEFAULT_ENV" ]; then
        # Already in a conda environment
        :
    fi
fi

# Setup working directory and Python path
PROJECT_ROOT="/comp_robot/zhoujiazhou/projects/Active-Coconut"
cd "$PROJECT_ROOT" || exit 1
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# Check if evaluation.py exists
EVAL_SCRIPT="${PROJECT_ROOT}/evaluation/evaluation.py"
if [ ! -f "$EVAL_SCRIPT" ]; then
    echo "Error: evaluation.py not found at $EVAL_SCRIPT"
    exit 1
fi

# Check if calculate_accuracy_by_category.py exists
ACCURACY_SCRIPT="${PROJECT_ROOT}/evaluation/calculate_accuracy_by_category.py"
if [ ! -f "$ACCURACY_SCRIPT" ]; then
    echo "Error: calculate_accuracy_by_category.py not found at $ACCURACY_SCRIPT"
    exit 1
fi

# Set GPU (use single GPU)
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    export CUDA_VISIBLE_DEVICES=0
    echo "Using GPU 0 (default)"
else
    echo "Using GPU(s) from CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
fi

# Run evaluation with base model flag
echo "============================================================================"
echo "Starting baseline evaluation (Qwen2.5-VL without SFT/LVR)..."
echo "Model: $CHECKPOINT_PATH"
echo "============================================================================"
export EVAL_CHECKPOINT_PATH="$CHECKPOINT_PATH"
export EVAL_STEP_LIST="$EVAL_STEP_LIST"
export USE_BASE_MODEL=1  # THIS IS THE KEY: Use base model instead of LVR
python "$EVAL_SCRIPT" "$@"
EVAL_EXIT_CODE=$?

if [ $EVAL_EXIT_CODE -ne 0 ]; then
    echo "Error: Evaluation failed with exit code $EVAL_EXIT_CODE"
    exit $EVAL_EXIT_CODE
fi

echo ""
echo "============================================================================"
echo "Evaluation completed successfully. Calculating accuracy by category..."
echo "============================================================================"

# Generate run_name from checkpoint path (same logic as in evaluation.py)
# run_name is created by joining all path components with underscores
RUN_NAME=$(echo "$CHECKPOINT_PATH" | sed 's/\//_/g')

# Calculate accuracy for blink dataset
BLINK_JSON_PATH="${PROJECT_ROOT}/evaluation/results/blink/decoding_by_steps/${RUN_NAME}/steps004.json"
if [ -f "$BLINK_JSON_PATH" ]; then
    echo ""
    echo "Calculating accuracy for BLINK dataset..."
    echo "File: $BLINK_JSON_PATH"
    python "$ACCURACY_SCRIPT" "$BLINK_JSON_PATH"
    if [ $? -ne 0 ]; then
        echo "Warning: Failed to calculate accuracy for BLINK dataset"
    fi
else
    echo "Warning: BLINK result file not found: $BLINK_JSON_PATH"
fi

# Calculate accuracy for vstar_bench dataset
VSTAR_JSON_PATH="${PROJECT_ROOT}/evaluation/results/vstar_bench/decoding_by_steps/${RUN_NAME}/steps004.json"
if [ -f "$VSTAR_JSON_PATH" ]; then
    echo ""
    echo "Calculating accuracy for VSTAR_BENCH dataset..."
    echo "File: $VSTAR_JSON_PATH"
    python "$ACCURACY_SCRIPT" "$VSTAR_JSON_PATH"
    if [ $? -ne 0 ]; then
        echo "Warning: Failed to calculate accuracy for VSTAR_BENCH dataset"
    fi
else
    echo "Warning: VSTAR_BENCH result file not found: $VSTAR_JSON_PATH"
fi

# Calculate accuracy for MMVP dataset
MMVP_JSON_PATH="${PROJECT_ROOT}/evaluation/results/MMVP/decoding_by_steps/${RUN_NAME}/steps004.json"
if [ -f "$MMVP_JSON_PATH" ]; then
    echo ""
    echo "Calculating accuracy for MMVP dataset..."
    echo "File: $MMVP_JSON_PATH"
    python "$ACCURACY_SCRIPT" "$MMVP_JSON_PATH"
    if [ $? -ne 0 ]; then
        echo "Warning: Failed to calculate accuracy for MMVP dataset"
    fi
else
    echo "Warning: MMVP result file not found: $MMVP_JSON_PATH"
fi

echo ""
echo "============================================================================"
echo "All tasks completed!"
echo "============================================================================"


