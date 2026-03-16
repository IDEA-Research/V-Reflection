#!/bin/bash
#SBATCH --partition=cvr
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:hgx:8 # A800 - 8 GPUs for parallel evaluation
#SBATCH --mem=640G
#SBATCH --qos=preemptive #specify preemptive Q0S
#SBATCH --output=/comp_robot/zhoujiazhou/projects/Active-Coconut/logs/baseline_qwen2.5vl_7b_%j.txt

# ============================================================================
# Configuration - Model Settings
# ============================================================================
CHECKPOINT_PATH="Qwen/Qwen2.5-VL-7B-Instruct"
# Baseline: no LVR steps (EVAL_STEP_LIST not used)
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

# Set GPU - Use all 8 GPUs for parallel evaluation (like evaluation_7b.sh)
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
fi
GPU_ID="${GPU_ID:-$CUDA_VISIBLE_DEVICES}"
export GPU_ID

# Run evaluation with base model flag (MMVP, BLINK, VSTAR, MMMU, MMStar, POPE, refCOCO, refCOCO+, refCOCOg, ReasonSeg)
echo "============================================================================"
echo "Starting baseline evaluation (Qwen2.5-VL without SFT/LVR)..."
echo "Model: $CHECKPOINT_PATH"
echo "============================================================================"
export EVAL_CHECKPOINT_PATH="$CHECKPOINT_PATH"
export EVAL_BENCHMARKS="refCOCO, refCOCO+, refCOCOg, ReasonSeg"
export USE_BASE_MODEL=1  # THIS IS THE KEY: Use base model instead of LVR

# Multi-process: launch 8 parallel processes (one per GPU)
IFS=',' read -ra GPU_ARRAY <<< "$GPU_ID"
echo "Launching ${#GPU_ARRAY[@]} parallel evaluation processes on GPUs: ${GPU_ARRAY[*]}"
PIDS=()
TOTAL_GPUS=${#GPU_ARRAY[@]}
for i in "${!GPU_ARRAY[@]}"; do
    gpu="${GPU_ARRAY[$i]}"
    echo "Starting process $((i+1))/${TOTAL_GPUS} on GPU $gpu"
    EVAL_PROCESS_ID=$i EVAL_TOTAL_PROCESSES=$TOTAL_GPUS \
    CUDA_VISIBLE_DEVICES="$gpu" \
        python "$EVAL_SCRIPT" "$@" > "${PROJECT_ROOT}/evaluation/eval_gpu${gpu}.log" 2>&1 &
    PIDS+=($!)
done
echo "Waiting for all processes to complete..."
EVAL_EXIT_CODE=0
for pid in "${PIDS[@]}"; do
    wait $pid || EVAL_EXIT_CODE=1
done

if [ $EVAL_EXIT_CODE -ne 0 ]; then
    echo "Error: Evaluation failed with exit code $EVAL_EXIT_CODE"
    exit $EVAL_EXIT_CODE
fi

echo ""
echo "============================================================================"
echo "Evaluation completed successfully. Calculating accuracy by category..."
echo "============================================================================"

# Generate run_name from checkpoint path (same logic as in evaluation.py load_model_and_processor)
# For HuggingFace paths like Qwen/Qwen2.5-VL-7B-Instruct: run_name = basename(dirname) = Qwen
if [[ "$CHECKPOINT_PATH" == *"/"* ]]; then
    RUN_NAME=$(basename "$(dirname "$CHECKPOINT_PATH")")
else
    RUN_NAME=$(basename "$CHECKPOINT_PATH")
fi

# Baseline: single output file per benchmark (no steps)
echo ""
echo "========== Baseline Results =========="

# Calculate accuracy for MMVP dataset
MMVP_JSON_PATH="${PROJECT_ROOT}/evaluation/results/MMVP/decoding_by_baseline/${RUN_NAME}/baseline.json"
if [ -f "$MMVP_JSON_PATH" ]; then
    echo ""
    echo "Calculating accuracy for MMVP dataset..."
    echo "File: $MMVP_JSON_PATH"
    python "$ACCURACY_SCRIPT" "$MMVP_JSON_PATH"
    [ $? -ne 0 ] && echo "Warning: Failed to calculate accuracy for MMVP dataset"
else
    echo "Warning: MMVP result file not found: $MMVP_JSON_PATH"
fi

# Calculate accuracy for blink dataset
BLINK_JSON_PATH="${PROJECT_ROOT}/evaluation/results/blink/decoding_by_baseline/${RUN_NAME}/baseline.json"
if [ -f "$BLINK_JSON_PATH" ]; then
    echo ""
    echo "Calculating accuracy for BLINK dataset..."
    echo "File: $BLINK_JSON_PATH"
    python "$ACCURACY_SCRIPT" "$BLINK_JSON_PATH"
    [ $? -ne 0 ] && echo "Warning: Failed to calculate accuracy for BLINK dataset"
else
    echo "Warning: BLINK result file not found: $BLINK_JSON_PATH"
fi

# Calculate accuracy for vstar_bench dataset
VSTAR_JSON_PATH="${PROJECT_ROOT}/evaluation/results/vstar_bench/decoding_by_baseline/${RUN_NAME}/baseline.json"
if [ -f "$VSTAR_JSON_PATH" ]; then
    echo ""
    echo "Calculating accuracy for VSTAR_BENCH dataset..."
    echo "File: $VSTAR_JSON_PATH"
    python "$ACCURACY_SCRIPT" "$VSTAR_JSON_PATH"
    [ $? -ne 0 ] && echo "Warning: Failed to calculate accuracy for VSTAR_BENCH dataset"
else
    echo "Warning: VSTAR_BENCH result file not found: $VSTAR_JSON_PATH"
fi

# Calculate accuracy for MMMU dataset
MMMU_JSON_PATH="${PROJECT_ROOT}/evaluation/results/MMMU/decoding_by_baseline/${RUN_NAME}/baseline.json"
if [ -f "$MMMU_JSON_PATH" ]; then
    echo ""
    echo "Calculating accuracy for MMMU dataset..."
    echo "File: $MMMU_JSON_PATH"
    python "$ACCURACY_SCRIPT" "$MMMU_JSON_PATH"
    [ $? -ne 0 ] && echo "Warning: Failed to calculate accuracy for MMMU dataset"
else
    echo "Warning: MMMU result file not found: $MMMU_JSON_PATH"
fi

# Calculate accuracy for MMStar dataset
MMSTAR_JSON_PATH="${PROJECT_ROOT}/evaluation/results/MMStar/decoding_by_baseline/${RUN_NAME}/baseline.json"
if [ -f "$MMSTAR_JSON_PATH" ]; then
    echo ""
    echo "Calculating accuracy for MMStar dataset..."
    echo "File: $MMSTAR_JSON_PATH"
    python "$ACCURACY_SCRIPT" "$MMSTAR_JSON_PATH"
    [ $? -ne 0 ] && echo "Warning: Failed to calculate accuracy for MMStar dataset"
else
    echo "Warning: MMStar result file not found: $MMSTAR_JSON_PATH"
fi

# Calculate accuracy for POPE dataset (yes/no format)
POPE_JSON_PATH="${PROJECT_ROOT}/evaluation/results/POPE/decoding_by_baseline/${RUN_NAME}/baseline.json"
if [ -f "$POPE_JSON_PATH" ]; then
    echo ""
    echo "Calculating accuracy for POPE dataset..."
    echo "File: $POPE_JSON_PATH"
    python3 -c "
import json
path = '$POPE_JSON_PATH'
with open(path) as f:
    data = json.load(f)
total = correct = 0
for item in data:
    if 'accuracy_by_category' in item or 'overall_accuracy' in item:
        continue
    pred = item.get('prediction', [])
    pred = pred[0] if isinstance(pred, list) and pred else (pred if isinstance(pred, str) else '')
    label = str(item.get('label', '')).strip().lower()
    text = pred.lower()
    idx_yes, idx_no = text.find('yes'), text.find('no')
    given = 'yes' if (idx_yes >= 0 and (idx_no < 0 or idx_yes < idx_no)) else ('no' if idx_no >= 0 else None)
    if given is not None:
        total += 1
        if given == label:
            correct += 1
print(f'  Overall: {correct}/{total} = {100*correct/total:.2f}%' if total > 0 else '  No valid samples')
"
    [ $? -ne 0 ] && echo "Warning: Failed to calculate accuracy for POPE dataset"
else
    echo "Warning: POPE result file not found: $POPE_JSON_PATH"
fi

# Calculate accuracy for MME dataset (yes/no format)
MME_JSON_PATH="${PROJECT_ROOT}/evaluation/results/MME/decoding_by_baseline/${RUN_NAME}/baseline.json"
if [ -f "$MME_JSON_PATH" ]; then
    echo ""
    echo "Calculating accuracy for MME dataset..."
    echo "File: $MME_JSON_PATH"
    python3 -c "
import json
path = '$MME_JSON_PATH'
with open(path) as f:
    data = json.load(f)
total = correct = 0
for item in data:
    if 'accuracy_by_category' in item or 'overall_accuracy' in item:
        continue
    pred = item.get('prediction', [])
    pred = pred[0] if isinstance(pred, list) and pred else (pred if isinstance(pred, str) else '')
    label = str(item.get('label', '')).strip().lower()
    text = pred.lower()
    idx_yes, idx_no = text.find('yes'), text.find('no')
    given = 'yes' if (idx_yes >= 0 and (idx_no < 0 or idx_yes < idx_no)) else ('no' if idx_no >= 0 else None)
    if given is not None:
        total += 1
        if given == label:
            correct += 1
print(f'  Overall: {correct}/{total} = {100*correct/total:.2f}%' if total > 0 else '  No valid samples')
"
    [ $? -ne 0 ] && echo "Warning: Failed to calculate accuracy for MME dataset"
else
    echo "Warning: MME result file not found: $MME_JSON_PATH"
fi

# refCOCO, refCOCO+, refCOCOg, ReasonSeg (REC, IoU>0.5 accuracy)
for REC_NAME in refCOCO "refCOCO+" refCOCOg ReasonSeg; do
    REC_JSON_PATH="${PROJECT_ROOT}/evaluation/results/${REC_NAME}/decoding_by_baseline/${RUN_NAME}/baseline.json"
    if [ -f "$REC_JSON_PATH" ]; then
        echo ""
        echo "Results for ${REC_NAME} dataset (REC, IoU>0.5)..."
        echo "File: $REC_JSON_PATH"
        python3 -c "
import json
path = '$REC_JSON_PATH'
with open(path) as f:
    data = json.load(f)
if data and isinstance(data[0], dict) and 'overall_accuracy' in data[0]:
    s = data[0]
    print(f\"  Overall: {s.get('overall_correct', '?')}/{s.get('overall_total', '?')} = {s.get('overall_accuracy', 0):.2f}%\")
    by_cat = s.get('accuracy_by_category', {})
    for cat, v in sorted(by_cat.items()):
        print(f\"  {cat}: {v.get('correct', 0)}/{v.get('total', 0)} = {v.get('accuracy', 0):.2f}%\")
else:
    print('  No summary found')
"
        [ $? -ne 0 ] && echo "Warning: Failed to print results for ${REC_NAME}"
    else
        echo "Warning: ${REC_NAME} result file not found: $REC_JSON_PATH"
    fi
done

echo ""
echo "============================================================================"
echo "All tasks completed!"
echo "============================================================================"


