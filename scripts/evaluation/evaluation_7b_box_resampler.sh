#!/bin/bash
#SBATCH --partition=cvr
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:hgx:8 # A800 - Use 8 GPUs for parallel evaluation
#SBATCH --mem=640G
#SBATCH --qos=preemptive #specify preemptive Q0S
#SBATCH --output=/comp_robot/zhoujiazhou/projects/Active-Coconut/logs/evaluation_7b_box_resampler_%j.txt

# ============================================================================
# Evaluation Script for Box Resampler Model
# Tests all checkpoints with step4 and step8 to compare LVR thinking steps
# ============================================================================

# Base checkpoint directory for box_resampler model (parent dir containing checkpoint-*)
BASE_CHECKPOINT_DIR="${BASE_CHECKPOINT_DIR:-/comp_robot/zhoujiazhou/projects/Active-Coconut/result/box_resampler/SFT_box_resampler_steps2500_b4_resampler0.5_acc8_latent8_lr5e-6}"

# Only test specific checkpoint(s). Set to "2500" for ck-2500 only; leave empty to auto-detect all
CHECKPOINT_STEPS="${CHECKPOINT_STEPS:-}"

# Only test specific benchmarks. Comma-separated, e.g. "MathVision,MathVista,VisuLogic,EMMA". Empty = all
EVAL_BENCHMARKS="${EVAL_BENCHMARKS:-BLINK, MMVP, VSTAR, POPE}"
export EVAL_BENCHMARKS

# Auto-detect all checkpoint directories (only when CHECKPOINT_STEPS is empty)
if [ -z "${CHECKPOINT_STEPS}" ]; then
    echo "Auto-detecting checkpoints in: $BASE_CHECKPOINT_DIR"
    CHECKPOINT_DIRS=$(find "$BASE_CHECKPOINT_DIR" -type d -name "checkpoint-*" | sort -V)
    if [ -z "$CHECKPOINT_DIRS" ]; then
        echo "Error: No checkpoint directories found in $BASE_CHECKPOINT_DIR"
        exit 1
    fi
    CHECKPOINT_STEPS=()
    while IFS= read -r checkpoint_dir; do
        step=$(basename "$checkpoint_dir" | sed 's/checkpoint-//')
        CHECKPOINT_STEPS+=("$step")
    done <<< "$CHECKPOINT_DIRS"
    echo "Found checkpoints: ${CHECKPOINT_STEPS[*]}"
else
    CHECKPOINT_STEPS=(${CHECKPOINT_STEPS[@]})
fi

EVAL_STEP_LIST="${EVAL_STEP_LIST:-8}"

DATASET_CONFIG="${DATASET_CONFIG:-default}"
LVR_SAVE_ACTIVATION_MAPS="${LVR_SAVE_ACTIVATION_MAPS:-0}"

# ============================================================================

# Initialize conda
if command -v conda &> /dev/null; then
    if [ -f "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" ]; then
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate train 2>/dev/null || true
    fi
fi

PROJECT_ROOT="/comp_robot/zhoujiazhou/projects/Active-Coconut"
cd "$PROJECT_ROOT" || exit 1
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

EVAL_SCRIPT="${PROJECT_ROOT}/evaluation/evaluation.py"
ACCURACY_SCRIPT="${PROJECT_ROOT}/evaluation/calculate_accuracy_by_category.py"
MERGE_SCRIPT="${PROJECT_ROOT}/evaluation/merge_process_results.py"

if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
fi
GPU_ID="${GPU_ID:-$CUDA_VISIBLE_DEVICES}"
export GPU_ID

# ============================================================================
# Step 1: Pre-convert all DeepSpeed checkpoints
# ============================================================================
echo ""
echo "============================================================================"
echo "Step 1: Pre-converting DeepSpeed Checkpoints"
echo "============================================================================"

for step in "${CHECKPOINT_STEPS[@]}"; do
    checkpoint_path=$(find "$BASE_CHECKPOINT_DIR" -type d -name "checkpoint-${step}" | head -n 1)
    if [ -z "$checkpoint_path" ]; then
        checkpoint_path="${BASE_CHECKPOINT_DIR}/checkpoint-${step}"
    fi
    
    if [ ! -d "$checkpoint_path" ]; then
        echo "Warning: Checkpoint directory not found: $checkpoint_path"
        continue
    fi
    
    MODEL_FILE="${checkpoint_path}/pytorch_model.bin"
    SAFETENSORS_FILE="${checkpoint_path}/model.safetensors"
    SAFETENSORS_SHARDED=$(find "$checkpoint_path" -maxdepth 1 -name "model-*-of-*.safetensors" | head -n 1)
    SAFETENSORS_INDEX="${checkpoint_path}/model.safetensors.index.json"
    
    if [ -f "$MODEL_FILE" ] || [ -f "$SAFETENSORS_FILE" ] || [ -n "$SAFETENSORS_SHARDED" ] || [ -f "$SAFETENSORS_INDEX" ]; then
        echo "✓ Checkpoint-${step} already converted"
        continue
    fi
    
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
    fi
done

echo "✓ Checkpoint conversion completed!"

# ============================================================================
# Function to calculate and print accuracy from JSON result file
# ============================================================================
calculate_accuracy() {
    local json_file=$1
    local dataset_name=$2
    local show_categories=${3:-1}  # Default: show categories
    
    if [ ! -f "$json_file" ]; then
        echo "  [Result] $dataset_name: File not found"
        return 1
    fi
    
    # Use Python to calculate accuracy with category breakdown
    python3 << EOF
import json
from collections import defaultdict

def extract_answer(pred, is_mcq=True):
    given_answer = pred.split('<answer>')[-1].split('</answer')[0].strip()
    if is_mcq:
        if " " in given_answer:
            given_answer = given_answer.split(" ")[0]
        if len(given_answer) > 1:
            given_answer = given_answer[0]
        return given_answer.upper()
    return given_answer

try:
    with open('$json_file', 'r') as f:
        data = json.load(f)
    
    # Category stats
    category_stats = defaultdict(lambda: {'total': 0, 'correct': 0})
    total = correct = 0
    
    for item in data:
        if 'prediction' not in item or 'label' not in item:
            continue
        if 'accuracy_by_category' in item or 'overall_accuracy' in item:
            continue
            
        pred = item['prediction'][0] if isinstance(item['prediction'], list) else item['prediction']
        label = str(item['label'])
        category = item.get('category', 'Unknown')
        is_mcq = len(label) == 1 and label.upper() in 'ABCDE'
        
        given_answer = extract_answer(pred, is_mcq)
        
        category_stats[category]['total'] += 1
        if is_mcq:
            match = given_answer == label.upper()
        else:
            match = given_answer.lower().strip() == label.lower().strip()
        if match:
            correct += 1
            category_stats[category]['correct'] += 1
        total += 1
    
    if total > 0:
        acc = 100 * correct / total
        print(f"  [Result] $dataset_name: {correct}/{total} = {acc:.2f}%")
        
        # Print category breakdown if enabled
        if $show_categories and len(category_stats) > 1:
            print(f"    {'Category':<35} {'Acc':>8}")
            print(f"    {'-'*45}")
            for cat in sorted(category_stats.keys()):
                stats = category_stats[cat]
                cat_acc = 100 * stats['correct'] / stats['total'] if stats['total'] > 0 else 0
                cat_display = cat[:32] + '...' if len(cat) > 35 else cat
                print(f"    {cat_display:<35} {cat_acc:>6.2f}%")
    else:
        print(f"  [Result] $dataset_name: No valid samples")
except Exception as e:
    print(f"  [Result] $dataset_name: Error - {e}")
EOF
}

# ============================================================================
# Step 2: Run Evaluation
# ============================================================================
echo ""
echo "============================================================================"
echo "Step 2: Starting Evaluation for Box Resampler Model"
echo "============================================================================"
echo "Base Directory: $BASE_CHECKPOINT_DIR"
echo "Checkpoints: ${CHECKPOINT_STEPS[*]}"
echo "Evaluation Steps: $EVAL_STEP_LIST"
echo "============================================================================"

# Store all results for final summary
declare -A ALL_RESULTS

run_evaluation() {
    local checkpoint_step=$1
    local checkpoint_path=$(find "$BASE_CHECKPOINT_DIR" -type d -name "checkpoint-${checkpoint_step}" | head -n 1)
    
    if [ -z "$checkpoint_path" ]; then
        checkpoint_path="${BASE_CHECKPOINT_DIR}/checkpoint-${checkpoint_step}"
    fi
    
    echo ""
    echo "============================================================================"
    echo "Evaluating checkpoint-${checkpoint_step}..."
    echo "============================================================================"
    
    if [ ! -d "$checkpoint_path" ]; then
        echo "Warning: Checkpoint directory not found: $checkpoint_path"
        return 1
    fi
    
    export CHECKPOINT_PATH="$checkpoint_path"
    export EVAL_CHECKPOINT_PATH="$checkpoint_path"
    export EVAL_STEP_LIST="$EVAL_STEP_LIST"
    export DATASET_CONFIG="$DATASET_CONFIG"
    export LVR_SAVE_ACTIVATION_MAPS="$LVR_SAVE_ACTIVATION_MAPS"
    
    bash "${PROJECT_ROOT}/scripts/evaluation/evaluation_7b.sh" "$@"
    local eval_exit_code=$?
    
    if [ $eval_exit_code -ne 0 ]; then
        echo "Error: Evaluation failed for checkpoint-${checkpoint_step}"
        return $eval_exit_code
    fi
    
    # Calculate and print results
    echo ""
    echo "============================================================================"
    echo "Results for checkpoint-${checkpoint_step}:"
    echo "============================================================================"
    
    # Get run_name for result paths
    local result_prefix="/comp_robot/zhoujiazhou/projects/Active-Coconut/result"
    local run_name=""
    if [[ "$checkpoint_path" == "$result_prefix"* ]]; then
        local relative_path="${checkpoint_path#$result_prefix/}"
        run_name=$(dirname "$relative_path")
    else
        run_name=$(basename "$(dirname "$checkpoint_path")")
    fi
    
    # Parse EVAL_STEP_LIST
    IFS=',' read -ra STEP_ARRAY <<< "$EVAL_STEP_LIST"
    
    for eval_step in "${STEP_ARRAY[@]}"; do
        eval_step=$(echo "$eval_step" | xargs)
        echo ""
        echo "--- Step $eval_step ---"
        
        # BLINK results
        local blink_json="${PROJECT_ROOT}/evaluation/results/blink/decoding_by_steps/${run_name}/ck-${checkpoint_step}-step${eval_step}.json"
        calculate_accuracy "$blink_json" "BLINK"
        
        # Store result for summary
        if [ -f "$blink_json" ]; then
            local acc=$(python3 -c "
import json
with open('$blink_json') as f:
    data = json.load(f)
total = correct = 0
for item in data:
    if 'prediction' not in item or 'label' not in item: continue
    pred = item['prediction'][0] if isinstance(item['prediction'], list) else item['prediction']
    label = item['label']
    given = pred.split('<answer>')[-1].split('</answer')[0].strip()
    if ' ' in given: given = given.split(' ')[0]
    if len(given) > 1: given = given[0]
    if given == label: correct += 1
    total += 1
print(f'{100*correct/total:.2f}' if total > 0 else 'N/A')
" 2>/dev/null)
            ALL_RESULTS["ck${checkpoint_step}_step${eval_step}_blink"]="$acc"
        fi
        
        # VSTAR results
        local vstar_json="${PROJECT_ROOT}/evaluation/results/vstar_bench/decoding_by_steps/${run_name}/ck-${checkpoint_step}-step${eval_step}.json"
        calculate_accuracy "$vstar_json" "VSTAR"
        
        # MMVP results
        local mmvp_json="${PROJECT_ROOT}/evaluation/results/MMVP/decoding_by_steps/${run_name}/ck-${checkpoint_step}-step${eval_step}.json"
        calculate_accuracy "$mmvp_json" "MMVP"

        # MathVision, MathVista, VisuLogic, EMMA results
        calculate_accuracy "${PROJECT_ROOT}/evaluation/results/MathVision/decoding_by_steps/${run_name}/ck-${checkpoint_step}-step${eval_step}.json" "MathVision" 0
        calculate_accuracy "${PROJECT_ROOT}/evaluation/results/MathVista/decoding_by_steps/${run_name}/ck-${checkpoint_step}-step${eval_step}.json" "MathVista" 0
        calculate_accuracy "${PROJECT_ROOT}/evaluation/results/VisuLogic/decoding_by_steps/${run_name}/ck-${checkpoint_step}-step${eval_step}.json" "VisuLogic" 0
        calculate_accuracy "${PROJECT_ROOT}/evaluation/results/EMMA/decoding_by_steps/${run_name}/ck-${checkpoint_step}-step${eval_step}.json" "EMMA" 0
    done
    
    echo ""
    echo "✓ Completed evaluation for checkpoint-${checkpoint_step}"
    return 0
}

# Track results
failed_checkpoints=()
successful_checkpoints=()

for step in "${CHECKPOINT_STEPS[@]}"; do
    echo ""
    echo "############################################################################"
    echo "# Processing checkpoint-${step} ($(date))"
    echo "############################################################################"
    
    if run_evaluation "$step" "$@"; then
        successful_checkpoints+=("$step")
    else
        failed_checkpoints+=("$step")
    fi
done

# ============================================================================
# Step 2.5: Merge orphaned _process*.json files (when Process 0 merge failed)
# ============================================================================
echo ""
echo "============================================================================"
echo "Step 2.5: Merging orphaned process result files"
echo "============================================================================"

result_prefix="/comp_robot/zhoujiazhou/projects/Active-Coconut/result"
if [[ "$BASE_CHECKPOINT_DIR" == "$result_prefix"* ]]; then
    run_name="${BASE_CHECKPOINT_DIR#$result_prefix/}"
else
    run_name=$(basename "$(dirname "$BASE_CHECKPOINT_DIR")")
fi
run_name="${run_name%/}"

IFS=',' read -ra STEP_ARRAY <<< "$EVAL_STEP_LIST"
DATASETS=("blink:BLINK" "vstar_bench:VSTAR" "MMVP:MMVP" "MathVision:MathVision" "MathVista:MathVista" "VisuLogic:VisuLogic" "EMMA:EMMA")
merge_count=0

for ck in "${CHECKPOINT_STEPS[@]}"; do
    for eval_step in "${STEP_ARRAY[@]}"; do
        eval_step=$(echo "$eval_step" | xargs)
        for ds_entry in "${DATASETS[@]}"; do
            ds_dir="${ds_entry%%:*}"
            ds_name="${ds_entry##*:}"
            result_dir="${PROJECT_ROOT}/evaluation/results/${ds_dir}/decoding_by_steps/${run_name}"
            final_file="${result_dir}/ck-${ck}-step${eval_step}.json"
            if [ ! -f "$final_file" ]; then
                process_pattern="${result_dir}/ck-${ck}-step${eval_step}_process*.json"
                if ls $process_pattern 1>/dev/null 2>&1; then
                    echo "  [Merge] ${ds_name} ck-${ck} step${eval_step}: merging orphaned process files..."
                    if python3 "$MERGE_SCRIPT" "$result_dir" "$ds_name" "$ck" "$eval_step"; then
                        ((merge_count++)) || true
                    fi
                fi
            fi
        done
    done
done

if [ "$merge_count" -gt 0 ]; then
    echo "  Merged $merge_count result set(s)"
else
    echo "  No orphaned process files found"
fi
echo ""

# ============================================================================
# Final Summary - Generate comprehensive results table
# ============================================================================
echo ""
echo "============================================================================"
echo "Box Resampler Evaluation Summary"
echo "============================================================================"
echo "Total checkpoints: ${#CHECKPOINT_STEPS[@]}"
echo "Successful: ${#successful_checkpoints[@]} - ${successful_checkpoints[*]}"
echo "Failed: ${#failed_checkpoints[@]} - ${failed_checkpoints[*]}"
echo ""

# Generate comprehensive results table using Python
python3 << 'SUMMARY_EOF'
import json
import os
from collections import defaultdict

def extract_answer(pred):
    given_answer = pred.split('<answer>')[-1].split('</answer')[0].strip()
    if " " in given_answer:
        given_answer = given_answer.split(" ")[0]
    if len(given_answer) > 1:
        given_answer = given_answer[0]
    return given_answer

def calculate_stats(json_file):
    """Calculate overall and per-category accuracy from a result file."""
    if not os.path.exists(json_file):
        return None, {}
    
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        category_stats = defaultdict(lambda: {'total': 0, 'correct': 0})
        total = correct = 0
        
        for item in data:
            if 'prediction' not in item or 'label' not in item:
                continue
            if 'accuracy_by_category' in item:
                continue
            
            pred = item['prediction'][0] if isinstance(item['prediction'], list) else item['prediction']
            label = item['label']
            category = item.get('category', 'Unknown')
            
            given_answer = extract_answer(pred)
            
            category_stats[category]['total'] += 1
            if given_answer == label:
                correct += 1
                category_stats[category]['correct'] += 1
            total += 1
        
        overall = 100 * correct / total if total > 0 else None
        cat_results = {}
        for cat, stats in category_stats.items():
            cat_results[cat] = 100 * stats['correct'] / stats['total'] if stats['total'] > 0 else None
        
        return overall, cat_results
    except:
        return None, {}

# Configuration
project_root = os.environ.get('PROJECT_ROOT', '/comp_robot/zhoujiazhou/projects/Active-Coconut')
base_dir = os.environ.get('BASE_CHECKPOINT_DIR', '')
eval_steps = os.environ.get('EVAL_STEP_LIST', '4,8').split(',')
eval_steps = [s.strip() for s in eval_steps]

# Get run_name from base_dir
result_prefix = "/comp_robot/zhoujiazhou/projects/Active-Coconut/result"
if base_dir.startswith(result_prefix):
    run_name = base_dir[len(result_prefix)+1:]
else:
    run_name = os.path.basename(base_dir)

# Find all checkpoints
checkpoints = []
if os.path.exists(base_dir):
    for d in sorted(os.listdir(base_dir)):
        if d.startswith('checkpoint-'):
            ck_num = d.replace('checkpoint-', '')
            checkpoints.append(ck_num)

if not checkpoints:
    print("No checkpoints found.")
    exit(0)

# Datasets to evaluate
datasets = {
    'BLINK': f'{project_root}/evaluation/results/blink/decoding_by_steps/{run_name}',
    'VSTAR': f'{project_root}/evaluation/results/vstar_bench/decoding_by_steps/{run_name}',
    'MMVP': f'{project_root}/evaluation/results/MMVP/decoding_by_steps/{run_name}',
    'MathVision': f'{project_root}/evaluation/results/MathVision/decoding_by_steps/{run_name}',
    'MathVista': f'{project_root}/evaluation/results/MathVista/decoding_by_steps/{run_name}',
    'VisuLogic': f'{project_root}/evaluation/results/VisuLogic/decoding_by_steps/{run_name}',
    'EMMA': f'{project_root}/evaluation/results/EMMA/decoding_by_steps/{run_name}',
}

# Collect all results
all_results = {}  # {ck: {step: {dataset: {overall: x, categories: {}}}}}
all_categories = {ds: set() for ds in datasets}

for ck in checkpoints:
    all_results[ck] = {}
    for step in eval_steps:
        all_results[ck][step] = {}
        for ds_name, ds_path in datasets.items():
            json_file = f'{ds_path}/ck-{ck}-step{step}.json'
            overall, cats = calculate_stats(json_file)
            all_results[ck][step][ds_name] = {'overall': overall, 'categories': cats}
            all_categories[ds_name].update(cats.keys())

# Print comprehensive table
print("=" * 100)
print("COMPREHENSIVE RESULTS TABLE")
print("=" * 100)

for ds_name in datasets:
    print(f"\n{'='*100}")
    print(f"Dataset: {ds_name}")
    print(f"{'='*100}")
    
    categories = sorted(all_categories[ds_name])
    
    # Header
    header = f"{'Checkpoint':<12}"
    for step in eval_steps:
        header += f" | Step{step:<6}"
    print(header)
    print("-" * len(header.expandtabs()))
    
    # Overall accuracy row
    row = f"{'Overall':<12}"
    for step in eval_steps:
        values = []
        for ck in checkpoints:
            val = all_results[ck][step][ds_name]['overall']
            values.append(f"{val:.1f}" if val is not None else "N/A")
        row += f" | {'/'.join(values):<{6+len(checkpoints)*5}}"
    print(row)
    print(f"  (ck: {' / '.join(checkpoints)})")
    print("-" * 80)
    
    # Per-category rows
    for cat in categories:
        cat_display = cat[:25] + '..' if len(cat) > 27 else cat
        row = f"{cat_display:<27}"
        for step in eval_steps:
            values = []
            for ck in checkpoints:
                cats = all_results[ck][step][ds_name]['categories']
                val = cats.get(cat)
                values.append(f"{val:.1f}" if val is not None else "-")
            row += f" | {'/'.join(values):<{6+len(checkpoints)*5}}"
        print(row)

# Print summary table (compact)
print(f"\n{'='*100}")
print("SUMMARY TABLE (Overall Accuracy)")
print(f"{'='*100}")

# Header with all combinations
header = f"{'Checkpoint':<12}"
for ds_name in datasets:
    for step in eval_steps:
        header += f" | {ds_name[:5]}_s{step}"
print(header)
print("-" * len(header.expandtabs()))

for ck in checkpoints:
    row = f"ck-{ck:<9}"
    for ds_name in datasets:
        for step in eval_steps:
            val = all_results[ck][step][ds_name]['overall']
            val_str = f"{val:.2f}" if val is not None else "N/A"
            row += f" | {val_str:>8}"
    print(row)

print("=" * 100)
SUMMARY_EOF

echo ""
if [ ${#failed_checkpoints[@]} -gt 0 ]; then
    exit 1
else
    echo "All checkpoints evaluated successfully!"
    exit 0
fi
