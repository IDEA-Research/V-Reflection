#!/bin/bash
#SBATCH --partition=cvr
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:hgx:8 # A800 - Use 8 GPUs for parallel evaluation
#SBATCH --mem=640G
#SBATCH --qos=preemptive #specify preemptive Q0S
#SBATCH --output=/comp_robot/zhoujiazhou/projects/Active-Coconut/logs/evaluation_7b_SFT_%j.txt

# ============================================================================
# Configuration - Model Settings
# ============================================================================
# Base checkpoint directory - can be overridden via environment variable
BASE_CHECKPOINT_DIR="${BASE_CHECKPOINT_DIR:-/comp_robot/zhoujiazhou/projects/Active-Coconut/result/stage1_checkpoints_7b_intrinsic-similarity/Stage1_ISG_steps2100_b1_mseLVR0.1-MaxVisToken5120-MinVisToken128}"

# Auto-detect all checkpoint directories
# If CHECKPOINT_STEPS is not set, automatically find all checkpoints
# DEBUG MODE: Set TEST_SINGLE_CK=true to test only the first checkpoint
TEST_SINGLE_CK="${TEST_SINGLE_CK:-true}"  # Default to true for debugging

if [ -z "${CHECKPOINT_STEPS+x}" ]; then
    echo "Auto-detecting checkpoints in: $BASE_CHECKPOINT_DIR"
    # Find all checkpoint directories recursively
    CHECKPOINT_DIRS=$(find "$BASE_CHECKPOINT_DIR" -type d -name "checkpoint-*" | sort -V)
    if [ -z "$CHECKPOINT_DIRS" ]; then
        echo "Error: No checkpoint directories found in $BASE_CHECKPOINT_DIR"
        exit 1
    fi
    # Extract step numbers from checkpoint paths
    CHECKPOINT_STEPS=()
    while IFS= read -r checkpoint_dir; do
        step=$(basename "$checkpoint_dir" | sed 's/checkpoint-//')
        CHECKPOINT_STEPS+=("$step")
    done <<< "$CHECKPOINT_DIRS"
    echo "Found checkpoints: ${CHECKPOINT_STEPS[*]}"
    
    # DEBUG MODE: Only test the first checkpoint
    if [ "$TEST_SINGLE_CK" = "true" ] || [ "$TEST_SINGLE_CK" = "True" ] || [ "$TEST_SINGLE_CK" = "1" ]; then
        if [ ${#CHECKPOINT_STEPS[@]} -gt 0 ]; then
            FIRST_CK="${CHECKPOINT_STEPS[0]}"
            CHECKPOINT_STEPS=("$FIRST_CK")
            echo ""
            echo "============================================================================"
            echo "DEBUG MODE: Testing only first checkpoint: checkpoint-${FIRST_CK}"
            echo "Set TEST_SINGLE_CK=false to test all checkpoints"
            echo "============================================================================"
            echo ""
        fi
    fi
else
    # Use manually specified checkpoint steps
    CHECKPOINT_STEPS=(${CHECKPOINT_STEPS[@]})
    
    # DEBUG MODE: Only test the first checkpoint if TEST_SINGLE_CK is true
    if [ "$TEST_SINGLE_CK" = "true" ] || [ "$TEST_SINGLE_CK" = "True" ] || [ "$TEST_SINGLE_CK" = "1" ]; then
        if [ ${#CHECKPOINT_STEPS[@]} -gt 0 ]; then
            FIRST_CK="${CHECKPOINT_STEPS[0]}"
            CHECKPOINT_STEPS=("$FIRST_CK")
            echo ""
            echo "============================================================================"
            echo "DEBUG MODE: Testing only first checkpoint: checkpoint-${FIRST_CK}"
            echo "Set TEST_SINGLE_CK=false to test all checkpoints"
            echo "============================================================================"
            echo ""
        fi
    fi
fi

# Evaluation step list
EVAL_STEP_LIST="4"

# Dataset configuration for training
# Options: "default" (viscot_sroie_dude) or "viscot_full" (Visual_cot full dataset)
DATASET_CONFIG="${DATASET_CONFIG:-default}"  # Default: default

# Available dataset configurations:
# - "default": Uses meta_data_lvr_sft_stage1.json (viscot_sroie_dude_lvr_formatted.json)
# - "viscot_full": Uses meta_data_lvr_sft_stage1_viscot_full.json (viscot_363k_lvr_formatted.json with coco/openimages)
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

# Set GPU - Use all available GPUs for parallel evaluation
# Can be overridden via environment variable CUDA_VISIBLE_DEVICES
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
    echo "Using all 8 GPUs: $CUDA_VISIBLE_DEVICES"
else
    echo "Using GPU(s) from CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
fi

# Set GPU_ID for evaluation script (comma-separated list)
GPU_ID="${GPU_ID:-$CUDA_VISIBLE_DEVICES}"
export GPU_ID

# ============================================================================
# Step 1: Pre-convert all DeepSpeed checkpoints
# ============================================================================
echo ""
echo "============================================================================"
echo "Step 1: Pre-converting DeepSpeed Checkpoints"
echo "============================================================================"
echo "This will convert all DeepSpeed Zero-3 checkpoints to standard format"
echo "to avoid slow conversion during evaluation."
echo ""

# Convert only the checkpoints we're going to test (for debugging, only convert first checkpoint)
for step in "${CHECKPOINT_STEPS[@]}"; do
    checkpoint_path=$(find "$BASE_CHECKPOINT_DIR" -type d -name "checkpoint-${step}" | head -n 1)
    if [ -z "$checkpoint_path" ]; then
        checkpoint_path="${BASE_CHECKPOINT_DIR}/checkpoint-${step}"
    fi
    
    if [ ! -d "$checkpoint_path" ]; then
        echo "Warning: Checkpoint directory not found: $checkpoint_path"
        continue
    fi
    
    # Check if already converted
    MODEL_FILE="${checkpoint_path}/pytorch_model.bin"
    SAFETENSORS_FILE="${checkpoint_path}/model.safetensors"
    SAFETENSORS_SHARDED=$(find "$checkpoint_path" -maxdepth 1 -name "model-*-of-*.safetensors" | head -n 1)
    SAFETENSORS_INDEX="${checkpoint_path}/model.safetensors.index.json"
    
    if [ -f "$MODEL_FILE" ] || [ -f "$SAFETENSORS_FILE" ] || [ -n "$SAFETENSORS_SHARDED" ] || [ -f "$SAFETENSORS_INDEX" ]; then
        echo "✓ Checkpoint-${step} already converted"
        continue
    fi
    
    # Check if DeepSpeed checkpoint exists
    GLOBAL_STEP_DIR=$(find "$checkpoint_path" -maxdepth 1 -type d -name "global_step*" | head -n 1)
    if [ -z "$GLOBAL_STEP_DIR" ]; then
        echo "⚠ Warning: No DeepSpeed checkpoint found in checkpoint-${step}"
        continue
    fi
    
    echo "Converting checkpoint-${step}..."
    ZERO_TO_FP32="${checkpoint_path}/zero_to_fp32.py"
    ORIGINAL_DIR="${PWD}"
    
    if [ -f "$ZERO_TO_FP32" ]; then
        cd "$checkpoint_path"
        python "$ZERO_TO_FP32" . . --safe_serialization >/dev/null 2>&1 || \
        python "$ZERO_TO_FP32" . . >/dev/null 2>&1 || {
            echo "  ✗ Failed to convert checkpoint-${step}"
            cd "$ORIGINAL_DIR"
            continue
        }
        cd "$ORIGINAL_DIR"
        echo "  ✓ Checkpoint-${step} converted successfully"
    else
        echo "  ⚠ Warning: zero_to_fp32.py not found for checkpoint-${step}"
    fi
done

echo ""
echo "✓ Checkpoint conversion completed for test checkpoints!"

echo ""
echo "============================================================================"
echo "Step 2: Starting Evaluation"
echo "============================================================================"
echo ""

# Function to run evaluation for a single checkpoint
run_evaluation() {
    local checkpoint_step=$1
    # Try to find checkpoint directory (could be directly under BASE_CHECKPOINT_DIR or in subdirectories)
    local checkpoint_path=$(find "$BASE_CHECKPOINT_DIR" -type d -name "checkpoint-${checkpoint_step}" | head -n 1)
    
    # Fallback to old path format if not found
    if [ -z "$checkpoint_path" ]; then
        checkpoint_path="${BASE_CHECKPOINT_DIR}/checkpoint-${checkpoint_step}"
    fi
    
    echo ""
    echo "============================================================================"
    echo "Starting evaluation for checkpoint-${checkpoint_step}..."
    echo "============================================================================"
    echo "Dataset Configuration: $DATASET_CONFIG"
    echo "Checkpoint Path: $checkpoint_path"
    echo "Evaluation Steps: $EVAL_STEP_LIST"
    
    # Check if checkpoint exists
    if [ ! -d "$checkpoint_path" ]; then
        echo "Warning: Checkpoint directory not found: $checkpoint_path"
        echo "Skipping checkpoint-${checkpoint_step}..."
        return 1
    fi
    
    # Check if checkpoint is in DeepSpeed format and needs conversion
    # NOTE: Checkpoint conversion is VERY SLOW (5-10 minutes per checkpoint)
    # It's recommended to convert all checkpoints in advance after training completes
    GLOBAL_STEP_DIR=$(find "$checkpoint_path" -maxdepth 1 -type d -name "global_step*" | head -n 1)
    MODEL_FILE="${checkpoint_path}/pytorch_model.bin"
    SAFETENSORS_FILE="${checkpoint_path}/model.safetensors"
    # Check for sharded safetensors files (model-00001-of-00007.safetensors, etc.)
    SAFETENSORS_SHARDED=$(find "$checkpoint_path" -maxdepth 1 -name "model-*-of-*.safetensors" | head -n 1)
    SAFETENSORS_INDEX="${checkpoint_path}/model.safetensors.index.json"
    
    # If DeepSpeed checkpoint exists and standard model files don't exist, convert it
    # Check for single file, sharded files, or index file
    if [ -n "$GLOBAL_STEP_DIR" ] && \
       [ ! -f "$MODEL_FILE" ] && \
       [ ! -f "$SAFETENSORS_FILE" ] && \
       [ -z "$SAFETENSORS_SHARDED" ] && \
       [ ! -f "$SAFETENSORS_INDEX" ]; then
        echo "WARNING: Converting DeepSpeed checkpoint to standard format for checkpoint-${checkpoint_step}..."
        echo "This may take 5-10 minutes. Consider converting checkpoints in advance."
        ZERO_TO_FP32="${checkpoint_path}/zero_to_fp32.py"
        ORIGINAL_DIR="${PWD}"
        
        if [ -f "$ZERO_TO_FP32" ]; then
            # Convert using zero_to_fp32.py
            # The script expects: checkpoint_dir (containing global_step* subdir) and output_dir
            cd "$checkpoint_path"
            python "$ZERO_TO_FP32" . . --safe_serialization || {
                echo "Warning: Failed to convert checkpoint using zero_to_fp32.py with safe_serialization, trying without..."
                python "$ZERO_TO_FP32" . . || {
                    echo "Error: Failed to convert DeepSpeed checkpoint for checkpoint-${checkpoint_step}"
                    cd "$ORIGINAL_DIR"
                    return 1
                }
            }
            cd "$ORIGINAL_DIR"
            echo "Checkpoint conversion completed for checkpoint-${checkpoint_step}."
        else
            echo "Warning: zero_to_fp32.py not found in checkpoint directory. Attempting to use DeepSpeed's built-in converter..."
            # Try using DeepSpeed's converter if available
            python -c "from deepspeed.checkpoint import convert_zero_checkpoint_to_fp32_state_dict; import os; convert_zero_checkpoint_to_fp32_state_dict('$GLOBAL_STEP_DIR', '$checkpoint_path')" || {
                echo "Error: Could not convert DeepSpeed checkpoint for checkpoint-${checkpoint_step}. Please convert manually."
                return 1
            }
        fi
    fi
    
    # Run evaluation using evaluation_7b.sh script which supports multi-GPU parallel evaluation
    # Set both CHECKPOINT_PATH and EVAL_CHECKPOINT_PATH for compatibility
    export CHECKPOINT_PATH="$checkpoint_path"
    export EVAL_CHECKPOINT_PATH="$checkpoint_path"
    export EVAL_STEP_LIST="$EVAL_STEP_LIST"
    export DATASET_CONFIG="$DATASET_CONFIG"
    # Use evaluation_7b.sh which supports multi-GPU parallel processing
    bash "${PROJECT_ROOT}/scripts/evaluation/evaluation_7b.sh" "$@"
    local eval_exit_code=$?
    
    if [ $eval_exit_code -ne 0 ]; then
        echo "Error: Evaluation failed for checkpoint-${checkpoint_step} with exit code $eval_exit_code"
        return $eval_exit_code
    fi
    
    echo ""
    echo "============================================================================"
    echo "Evaluation completed for checkpoint-${checkpoint_step}. Calculating accuracy by category..."
    echo "============================================================================"
    
    # Generate run_name from checkpoint path
    # evaluation.py generates run_name from full checkpoint path
    # Format: comp_robot_zhoujiazhou_projects_Active-Coconut_result_..._checkpoint-XXX
    local run_name=$(echo "$checkpoint_path" | sed 's|/|_|g' | sed 's|^_||')
    
    # Calculate accuracy for blink dataset
    local blink_json_path="${PROJECT_ROOT}/evaluation/results/blink/decoding_by_steps/${run_name}/steps004.json"
    if [ -f "$blink_json_path" ]; then
        echo ""
        echo "Calculating accuracy for BLINK dataset (checkpoint-${checkpoint_step})..."
        echo "File: $blink_json_path"
        python "$ACCURACY_SCRIPT" "$blink_json_path"
        if [ $? -ne 0 ]; then
            echo "Warning: Failed to calculate accuracy for BLINK dataset (checkpoint-${checkpoint_step})"
        fi
    else
        echo "Warning: BLINK result file not found: $blink_json_path"
    fi
    
    # Calculate accuracy for vstar_bench dataset
    local vstar_json_path="${PROJECT_ROOT}/evaluation/results/vstar_bench/decoding_by_steps/${run_name}/steps004.json"
    if [ -f "$vstar_json_path" ]; then
        echo ""
        echo "Calculating accuracy for VSTAR_BENCH dataset (checkpoint-${checkpoint_step})..."
        echo "File: $vstar_json_path"
        python "$ACCURACY_SCRIPT" "$vstar_json_path"
        if [ $? -ne 0 ]; then
            echo "Warning: Failed to calculate accuracy for VSTAR_BENCH dataset (checkpoint-${checkpoint_step})"
        fi
    else
        echo "Warning: VSTAR_BENCH result file not found: $vstar_json_path"
    fi
    
    # Calculate accuracy for MMVP dataset
    local mmvp_json_path="${PROJECT_ROOT}/evaluation/results/MMVP/decoding_by_steps/${run_name}/steps004.json"
    if [ -f "$mmvp_json_path" ]; then
        echo ""
        echo "Calculating accuracy for MMVP dataset (checkpoint-${checkpoint_step})..."
        echo "File: $mmvp_json_path"
        python "$ACCURACY_SCRIPT" "$mmvp_json_path"
        if [ $? -ne 0 ]; then
            echo "Warning: Failed to calculate accuracy for MMVP dataset (checkpoint-${checkpoint_step})"
        fi
    else
        echo "Warning: MMVP result file not found: $mmvp_json_path"
    fi
    
    echo ""
    echo "============================================================================"
    echo "Completed evaluation for checkpoint-${checkpoint_step}!"
    echo "============================================================================"
    
    return 0
}

# Main execution: loop through all checkpoints
echo "============================================================================"
echo "Batch Evaluation Script for Stage1 Checkpoints"
echo "============================================================================"
echo "Base Checkpoint Directory: $BASE_CHECKPOINT_DIR"
echo "Checkpoints to evaluate: ${CHECKPOINT_STEPS[@]}"
echo "Evaluation Steps: $EVAL_STEP_LIST"
echo "Dataset Configuration: $DATASET_CONFIG"
echo "============================================================================"

# Track results
failed_checkpoints=()
successful_checkpoints=()

# Loop through each checkpoint step
for step in "${CHECKPOINT_STEPS[@]}"; do
    echo ""
    echo "############################################################################"
    echo "# Processing checkpoint-${step} ($(date))"
    echo "############################################################################"
    
    if run_evaluation "$step" "$@"; then
        successful_checkpoints+=("$step")
        echo "✓ Successfully completed evaluation for checkpoint-${step}"
    else
        failed_checkpoints+=("$step")
        echo "✗ Failed evaluation for checkpoint-${step}"
    fi
    
    echo ""
done

# Summary
echo ""
echo "============================================================================"
echo "Batch Evaluation Summary"
echo "============================================================================"
echo "Total checkpoints processed: ${#CHECKPOINT_STEPS[@]}"
echo "Successful: ${#successful_checkpoints[@]}"
if [ ${#successful_checkpoints[@]} -gt 0 ]; then
    echo "  - ${successful_checkpoints[*]}"
fi
echo "Failed: ${#failed_checkpoints[@]}"
if [ ${#failed_checkpoints[@]} -gt 0 ]; then
    echo "  - ${failed_checkpoints[*]}"
fi
echo "============================================================================"

# Exit with error if any checkpoint failed
if [ ${#failed_checkpoints[@]} -gt 0 ]; then
    echo "Warning: Some checkpoints failed evaluation. Please check the logs above."
    exit 1
else
    echo "All checkpoints evaluated successfully!"
    exit 0
fi

