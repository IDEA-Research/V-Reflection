#!/bin/bash
#SBATCH --partition=cvr
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:hgx:1 # A800
#SBATCH --mem=80G
#SBATCH --qos=preemptive #specify preemptive Q0S
#SBATCH --output=/comp_robot/zhoujiazhou/projects/Active-Coconut/logs/evaluation_7b_SFT_%j.txt

# Support both CHECKPOINT_PATH and EVAL_CHECKPOINT_PATH for compatibility
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${EVAL_CHECKPOINT_PATH:-/comp_robot/zhoujiazhou/projects/Active-Coconut/result/stage1_checkpoints_7b_intrinsic-similarity/Stage1_ISG_steps2100_b1_mseLVR0.1-MaxVisToken5120-MinVisToken128/checkpoint-1500}}"
# Ensure EVAL_CHECKPOINT_PATH is set (used by evaluation.py)
export EVAL_CHECKPOINT_PATH="${EVAL_CHECKPOINT_PATH:-$CHECKPOINT_PATH}"

# GPU Settings
# Can be set via environment variable GPU_ID (e.g., "0,1,2,3,4,5,6,7" for 8 GPUs)
# If not set, defaults to single GPU "0"
GPU_ID="${GPU_ID:-0}"  # GPUs to use for parallel evaluation
# Set to True to launch separate processes for each GPU (true parallelism)
# Set to False to use only the first GPU in GPU_ID list
USE_MULTI_PROCESS=True  # Launch multiple processes for true parallelism

EVAL_STEP_LIST="${EVAL_STEP_LIST:-4}"  # Default: 4,8,16
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
            EVAL_PROCESS_ID=$i EVAL_TOTAL_PROCESSES=$TOTAL_GPUS EVAL_CHECKPOINT_PATH="${EVAL_CHECKPOINT_PATH:-$CHECKPOINT_PATH}" EVAL_STEP_LIST="$EVAL_STEP_LIST" HF_ENDPOINT="$HF_ENDPOINT" CUDA_VISIBLE_DEVICES="$gpu" \
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
    
    export EVAL_CHECKPOINT_PATH="${EVAL_CHECKPOINT_PATH:-$CHECKPOINT_PATH}"
    export EVAL_STEP_LIST="$EVAL_STEP_LIST"
    python "$EVAL_SCRIPT" "$@"
fi

