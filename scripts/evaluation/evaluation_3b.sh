# ============================================================================
# Evaluation Script for LVR Model
# ============================================================================
# This script runs evaluation on trained LVR models using the evaluation.py script.
# It sets up the proper Python path and environment to avoid import errors.
#
# Usage:
#   bash scripts/run_evaluation.sh
# ============================================================================

# ============================================================================
# Configuration - Model and GPU Settings
# ============================================================================
# Model checkpoint path - Edit this to point to your trained model
CHECKPOINT_PATH="/comp_robot/zhoujiazhou/projects/Active-Coconut/stage1_checkpoints/checkpoint-2000"

# HuggingFace Mirror Configuration
# Set HF_ENDPOINT to use HuggingFace mirror (e.g., https://hf-mirror.com)
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_ENDPOINT

# GPU Settings
GPU_ID="0,1,2,3,4,5,6,7"  # GPUs to use for parallel evaluation
# Set to True to launch separate processes for each GPU (true parallelism)
# Set to False to use only the first GPU in GPU_ID list
USE_MULTI_PROCESS=True  # Launch multiple processes for true parallelism

EVAL_STEP_LIST="${EVAL_STEP_LIST:-4,8,16}"  # Default: 4,8,16
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
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT" || exit 1
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# Check if evaluation.py exists
EVAL_SCRIPT="${PROJECT_ROOT}/evaluation/evaluation.py"
if [ ! -f "$EVAL_SCRIPT" ]; then
    echo "Error: evaluation.py not found at $EVAL_SCRIPT"
    exit 1
fi

# Launch evaluation
if [ "$USE_MULTI_PROCESS" = "True" ] || [ "$USE_MULTI_PROCESS" = "true" ]; then
    # Multi-process mode: launch separate process for each GPU
    if [ -n "$GPU_ID" ]; then
        # Convert comma-separated GPU list to array
        IFS=',' read -ra GPU_ARRAY <<< "$GPU_ID"
        echo "Launching ${#GPU_ARRAY[@]} parallel evaluation processes..."
        echo "GPUs: ${GPU_ARRAY[@]}"
        
        # Launch processes in background
        PIDS=()
        TOTAL_GPUS=${#GPU_ARRAY[@]}
        for i in "${!GPU_ARRAY[@]}"; do
            gpu="${GPU_ARRAY[$i]}"
            echo "Starting process $((i+1))/${TOTAL_GPUS} on GPU $gpu"
            EVAL_PROCESS_ID=$i EVAL_TOTAL_PROCESSES=$TOTAL_GPUS EVAL_CHECKPOINT_PATH="$CHECKPOINT_PATH" EVAL_STEP_LIST="$EVAL_STEP_LIST" HF_ENDPOINT="$HF_ENDPOINT" CUDA_VISIBLE_DEVICES="$gpu" \
                python "$EVAL_SCRIPT" "$@" > "${PROJECT_ROOT}/evaluation/eval_gpu${gpu}.log" 2>&1 &
            PIDS+=($!)
        done
        
        # Wait for all processes to complete
        echo "Waiting for all processes to complete..."
        for pid in "${PIDS[@]}"; do
            wait $pid
            echo "Process $pid completed"
        done
        echo "All evaluation processes completed!"
    else
        echo "Error: GPU_ID is empty. Cannot launch multi-process evaluation."
        exit 1
    fi
else
    # Single process mode: use first GPU or specified GPU
    if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
        if [ -n "$GPU_ID" ]; then
            # Use first GPU from the list
            FIRST_GPU=$(echo "$GPU_ID" | cut -d',' -f1)
            export CUDA_VISIBLE_DEVICES="$FIRST_GPU"
            echo "Using single GPU: $FIRST_GPU"
        else
            export CUDA_VISIBLE_DEVICES=0
            echo "Using GPU 0 (default)"
        fi
    else
        echo "Using GPU(s) from CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
    fi
    
    export EVAL_CHECKPOINT_PATH="$CHECKPOINT_PATH"
    export EVAL_STEP_LIST="$EVAL_STEP_LIST"
    python "$EVAL_SCRIPT" "$@"
fi

