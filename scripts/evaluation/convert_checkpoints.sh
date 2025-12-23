#!/bin/bash
#SBATCH --partition=cvr
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:hgx:1 # A800 - Only need 1 GPU for conversion
#SBATCH --mem=80G
#SBATCH --qos=preemptive
#SBATCH --output=/comp_robot/zhoujiazhou/projects/Active-Coconut/logs/convert_checkpoints_%j.txt

# ============================================================================
# Pre-convert DeepSpeed Checkpoints Script
# ============================================================================
# This script converts all DeepSpeed Zero-3 checkpoints to standard format
# in advance, so evaluation can run much faster without conversion overhead.
#
# What are "shards"?
# - DeepSpeed Zero-3 splits model parameters across multiple GPUs (shards)
# - With 8 GPUs, the model is split into 8 shards (one per GPU)
# - Each shard contains a portion of the model weights
# - Conversion merges all 8 shards into a single complete model file
#
# Usage:
#   bash scripts/convert_checkpoints.sh
#   Or set BASE_CHECKPOINT_DIR environment variable:
#   BASE_CHECKPOINT_DIR="result/SFT_7b_intrinsic-similarity/intrinsic-similarity_b1_acc8_mseLVR0.1" bash scripts/convert_checkpoints.sh
# ============================================================================

source ~/miniconda3/etc/profile.d/conda.sh
conda activate train

cd /comp_robot/zhoujiazhou/projects/Active-Coconut
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

# Base checkpoint directory - can be set via environment variable
BASE_CHECKPOINT_DIR="${BASE_CHECKPOINT_DIR:-result/SFT_7b_intrinsic-similarity/intrinsic-similarity_b1_acc8_mseLVR0.1}"

# Convert to absolute path
if [[ ! "$BASE_CHECKPOINT_DIR" = /* ]]; then
    BASE_CHECKPOINT_DIR="${PWD}/${BASE_CHECKPOINT_DIR}"
fi

echo "============================================================================"
echo "Pre-converting DeepSpeed Checkpoints"
echo "============================================================================"
echo "Base Checkpoint Directory: $BASE_CHECKPOINT_DIR"
echo ""

# Find all checkpoint directories
CHECKPOINT_DIRS=$(find "$BASE_CHECKPOINT_DIR" -maxdepth 1 -type d -name "checkpoint-*" | sort -V)

if [ -z "$CHECKPOINT_DIRS" ]; then
    echo "Error: No checkpoint directories found in $BASE_CHECKPOINT_DIR"
    exit 1
fi

# Extract checkpoint steps
CHECKPOINT_STEPS=()
while IFS= read -r checkpoint_dir; do
    step=$(basename "$checkpoint_dir" | sed 's/checkpoint-//')
    CHECKPOINT_STEPS+=("$step")
done <<< "$CHECKPOINT_DIRS"

echo "Found ${#CHECKPOINT_STEPS[@]} checkpoints: ${CHECKPOINT_STEPS[*]}"
echo ""

# Function to convert a single checkpoint
convert_checkpoint() {
    local checkpoint_step=$1
    local checkpoint_path="${BASE_CHECKPOINT_DIR}/checkpoint-${checkpoint_step}"
    
    echo "============================================================================"
    echo "Processing checkpoint-${checkpoint_step}..."
    echo "Checkpoint Path: $checkpoint_path"
    echo "============================================================================"
    
    if [ ! -d "$checkpoint_path" ]; then
        echo "Warning: Checkpoint directory not found: $checkpoint_path"
        return 1
    fi
    
    # Check if checkpoint is in DeepSpeed format
    GLOBAL_STEP_DIR=$(find "$checkpoint_path" -maxdepth 1 -type d -name "global_step*" | head -n 1)
    MODEL_FILE="${checkpoint_path}/pytorch_model.bin"
    SAFETENSORS_FILE="${checkpoint_path}/model.safetensors"
    # Check for sharded safetensors files (model-00001-of-00007.safetensors, etc.)
    SAFETENSORS_SHARDED=$(find "$checkpoint_path" -maxdepth 1 -name "model-*-of-*.safetensors" | head -n 1)
    SAFETENSORS_INDEX="${checkpoint_path}/model.safetensors.index.json"
    
    # Check if already converted (check for single file, sharded files, or index)
    if [ -f "$MODEL_FILE" ] || [ -f "$SAFETENSORS_FILE" ] || [ -n "$SAFETENSORS_SHARDED" ] || [ -f "$SAFETENSORS_INDEX" ]; then
        echo "✓ Checkpoint-${checkpoint_step} already converted (found model file)"
        return 0
    fi
    
    # Check if DeepSpeed checkpoint exists
    if [ -z "$GLOBAL_STEP_DIR" ]; then
        echo "⚠ Warning: No DeepSpeed checkpoint found in checkpoint-${checkpoint_step}"
        echo "  (No global_step* directory found)"
        return 1
    fi
    
    echo "Converting DeepSpeed checkpoint (this may take 5-10 minutes)..."
    echo "  Source: $GLOBAL_STEP_DIR"
    echo "  Target: $checkpoint_path"
    
    ZERO_TO_FP32="${checkpoint_path}/zero_to_fp32.py"
    ORIGINAL_DIR="${PWD}"
    
    # Try multiple conversion methods
    CONVERSION_SUCCESS=false
    
    # Method 1: Try DeepSpeed's built-in converter (handles various formats better)
    echo "  Attempting conversion using DeepSpeed's built-in converter..."
    CONVERT_OUTPUT=$(python -c "
from deepspeed.checkpoint import convert_zero_checkpoint_to_fp32_state_dict
import os
import sys
try:
    convert_zero_checkpoint_to_fp32_state_dict('$GLOBAL_STEP_DIR', '$checkpoint_path')
    print('SUCCESS')
except Exception as e:
    print(f'ERROR: {e}', file=sys.stderr)
    sys.exit(1)
" 2>&1)
    CONVERT_EXIT=$?
    
    if [ $CONVERT_EXIT -eq 0 ]; then
        # Check if conversion actually succeeded by verifying output files
        SAFETENSORS_SHARDED_CHECK=$(find "$checkpoint_path" -maxdepth 1 -name "model-*-of-*.safetensors" | head -n 1)
        if [ -f "$MODEL_FILE" ] || [ -f "$SAFETENSORS_FILE" ] || [ -n "$SAFETENSORS_SHARDED_CHECK" ] || [ -f "${checkpoint_path}/model.safetensors.index.json" ]; then
            echo "  ✓ Conversion completed using DeepSpeed built-in converter"
            CONVERSION_SUCCESS=true
        else
            echo "  DeepSpeed converter ran but no model files found"
        fi
    else
        echo "  DeepSpeed converter failed: $(echo "$CONVERT_OUTPUT" | grep "ERROR" | head -1)"
    fi
    
    # Method 2: Try zero_to_fp32.py script if Method 1 failed
    if [ "$CONVERSION_SUCCESS" = false ] && [ -f "$ZERO_TO_FP32" ]; then
        echo "  Trying zero_to_fp32.py script..."
        cd "$checkpoint_path"
        
        # Try with --safe_serialization first
        if python "$ZERO_TO_FP32" . . --safe_serialization >/dev/null 2>&1; then
            CONVERSION_SUCCESS=true
            echo "  ✓ Conversion completed using zero_to_fp32.py with --safe_serialization"
        # Try without --safe_serialization
        elif python "$ZERO_TO_FP32" . . >/dev/null 2>&1; then
            CONVERSION_SUCCESS=true
            echo "  ✓ Conversion completed using zero_to_fp32.py"
        else
            # Check the actual error
            ERROR_OUTPUT=$(python "$ZERO_TO_FP32" . . 2>&1 | tail -5)
            echo "  ✗ zero_to_fp32.py failed"
            if echo "$ERROR_OUTPUT" | grep -q "Expected 1.*but found 8"; then
                echo "  Error: Optimizer states are sharded (8 files) but script expects 1 merged file"
                echo "  This checkpoint format is incompatible with zero_to_fp32.py"
                echo "  Note: Evaluation only needs model weights, not optimizer states."
            else
                echo "  Error details:"
                echo "$ERROR_OUTPUT" | head -3
            fi
        fi
        cd "$ORIGINAL_DIR"
    fi
    
    if [ "$CONVERSION_SUCCESS" = false ]; then
        echo "  ✗ Warning: All conversion methods failed for checkpoint-${checkpoint_step}"
        echo "  This checkpoint may have incompatible format or missing files."
        echo "  Evaluation will be skipped for this checkpoint."
        return 1
    fi
    
    # Verify conversion succeeded
    # Check for single file, sharded safetensors files, or index file
    SAFETENSORS_SHARDED=$(find "$checkpoint_path" -maxdepth 1 -name "model-*-of-*.safetensors" | head -n 1)
    SAFETENSORS_INDEX="${checkpoint_path}/model.safetensors.index.json"
    
    if [ -f "$MODEL_FILE" ] || [ -f "$SAFETENSORS_FILE" ] || [ -n "$SAFETENSORS_SHARDED" ] || [ -f "$SAFETENSORS_INDEX" ]; then
        if [ -n "$SAFETENSORS_SHARDED" ]; then
            SHARD_COUNT=$(find "$checkpoint_path" -maxdepth 1 -name "model-*-of-*.safetensors" | wc -l)
            echo "  ✓ Verified: Model file created successfully (sharded format: $SHARD_COUNT shards)"
        elif [ -f "$SAFETENSORS_INDEX" ]; then
            echo "  ✓ Verified: Model file created successfully (sharded format with index)"
        else
            echo "  ✓ Verified: Model file created successfully"
        fi
        return 0
    else
        echo "  ✗ Warning: Conversion completed but model file not found"
        echo "  Checking directory contents:"
        ls -lh "$checkpoint_path" | grep -E "(model|pytorch)" | head -10 || echo "  No model files found"
        return 1
    fi
}

# Convert all checkpoints
SUCCESS_COUNT=0
FAIL_COUNT=0
SKIP_COUNT=0

for step in "${CHECKPOINT_STEPS[@]}"; do
    echo ""
    if convert_checkpoint "$step"; then
        # Check for converted files (single file, sharded files, or index)
        CHECKPOINT_DIR="${BASE_CHECKPOINT_DIR}/checkpoint-${step}"
        if [ -f "${CHECKPOINT_DIR}/pytorch_model.bin" ] || \
           [ -f "${CHECKPOINT_DIR}/model.safetensors" ] || \
           [ -n "$(find "$CHECKPOINT_DIR" -maxdepth 1 -name "model-*-of-*.safetensors" | head -n 1)" ] || \
           [ -f "${CHECKPOINT_DIR}/model.safetensors.index.json" ]; then
            ((SUCCESS_COUNT++))
        else
            ((SKIP_COUNT++))
        fi
    else
        ((FAIL_COUNT++))
    fi
    echo ""
done

# Summary
echo ""
echo "============================================================================"
echo "Conversion Summary"
echo "============================================================================"
echo "Total checkpoints: ${#CHECKPOINT_STEPS[@]}"
echo "Successfully converted: $SUCCESS_COUNT"
echo "Already converted (skipped): $SKIP_COUNT"
echo "Failed: $FAIL_COUNT"
echo "============================================================================"

if [ $FAIL_COUNT -gt 0 ]; then
    echo "Warning: Some checkpoints failed conversion. Please check the logs above."
    exit 1
else
    echo "All checkpoints converted successfully!"
    echo "You can now run evaluation without conversion overhead."
    exit 0
fi

