#!/bin/bash
# Single-checkpoint evaluation. Usage: EVAL_CHECKPOINT_PATH=path/to/checkpoint bash scripts_release/evaluation/evaluation_7b.sh

# Support both CHECKPOINT_PATH and EVAL_CHECKPOINT_PATH for compatibility
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${EVAL_CHECKPOINT_PATH:-}}"
# Ensure EVAL_CHECKPOINT_PATH is set (used by evaluation.py)
export EVAL_CHECKPOINT_PATH="${EVAL_CHECKPOINT_PATH:-$CHECKPOINT_PATH}"

if [ -z "$EVAL_CHECKPOINT_PATH" ]; then
    echo "Error: CHECKPOINT_PATH or EVAL_CHECKPOINT_PATH must be set to your checkpoint directory."
    exit 1
fi

# GPU Settings
# Can be set via environment variable GPU_ID (e.g., "0,1,2,3,4,5,6,7" for 8 GPUs)
# If not set, defaults to single GPU "0"
GPU_ID="${GPU_ID:-0}"  # GPUs to use for parallel evaluation
# Set to True to launch separate processes for each GPU (true parallelism)
# Set to False to use only the first GPU in GPU_ID list
USE_MULTI_PROCESS=True  # Launch multiple processes for true parallelism

EVAL_STEP_LIST="${EVAL_STEP_LIST:-4}"  # Default: 4,8,16

# Force re-evaluation: always re-run inference even if result files exist
# Set to 1 to force re-evaluation, 0 to use existing results if available
FORCE_RE_EVALUATE="${FORCE_RE_EVALUATE:-0}"
export FORCE_RE_EVALUATE

# Activation map visualization control
# Set to "1" to enable saving activation maps for LVR heads (gated-focus, intrinsic-similarity)
# Activation maps will be saved to: evaluation/results/{benchmark}/decoding_by_{strategy}/{run_name}/{checkpoint_num}/activation_maps/
# Default: "0" (disabled) - Can be overridden via environment variable
LVR_SAVE_ACTIVATION_MAPS="${LVR_SAVE_ACTIVATION_MAPS:-0}"  # Default: 0 (disabled)
export LVR_SAVE_ACTIVATION_MAPS

# DiT image generation control (for DiT Recon models)
# Set to "1" to enable DiT image generation during inference
DIT_GENERATE_IMAGES="${DIT_GENERATE_IMAGES:-0}"
DIT_SAVE_DIR="${DIT_SAVE_DIR:-}"
DIT_NUM_INFERENCE_STEPS="${DIT_NUM_INFERENCE_STEPS:-20}"
DIT_DEBUG="${DIT_DEBUG:-0}"
export DIT_GENERATE_IMAGES DIT_SAVE_DIR DIT_NUM_INFERENCE_STEPS DIT_DEBUG

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
if [ -n "${PROJECT_ROOT}" ]; then
    :  # Already set by parent
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
fi
export PROJECT_ROOT
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
            EVAL_PROCESS_ID=$i \
            EVAL_TOTAL_PROCESSES=$TOTAL_GPUS \
            EVAL_CHECKPOINT_PATH="${EVAL_CHECKPOINT_PATH:-$CHECKPOINT_PATH}" \
            EVAL_STEP_LIST="$EVAL_STEP_LIST" \
            LVR_SAVE_ACTIVATION_MAPS="$LVR_SAVE_ACTIVATION_MAPS" \
            FORCE_RE_EVALUATE="$FORCE_RE_EVALUATE" \
            DIT_GENERATE_IMAGES="$DIT_GENERATE_IMAGES" \
            DIT_SAVE_DIR="$DIT_SAVE_DIR" \
            DIT_NUM_INFERENCE_STEPS="$DIT_NUM_INFERENCE_STEPS" \
            DIT_DEBUG="$DIT_DEBUG" \
            HF_ENDPOINT="$HF_ENDPOINT" \
            CUDA_VISIBLE_DEVICES="$gpu" \
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
    export LVR_SAVE_ACTIVATION_MAPS="$LVR_SAVE_ACTIVATION_MAPS"
    export FORCE_RE_EVALUATE="$FORCE_RE_EVALUATE"
    python "$EVAL_SCRIPT" "$@"
fi

