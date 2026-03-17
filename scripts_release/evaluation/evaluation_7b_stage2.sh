#!/bin/bash
# Evaluation for Stage 2 Distillation model. Multiple checkpoints.
# Usage: bash scripts_release/evaluation/evaluation_7b_stage2.sh

# ============================================================================
# Evaluation Script for Stage 2 Distillation Model
# Tests all checkpoints with step4 and step8 to compare LVR thinking steps
# Stage 2: DynamicAutoregressiveResampler (Student) + frozen BoxFeatureResampler (Teacher)
# Inference uses box_feature_resampler for <lvr> fill (same as Stage 1)
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Base checkpoint directory for stage2_distillation model
# Set BASE_CHECKPOINT_DIR or run from project root; default uses relative path
BASE_CHECKPOINT_DIR="${BASE_CHECKPOINT_DIR:-result/stage2_distillation/SFT_stage2_distillation_steps2500_b4_LVR0.1_resampler0.5_attnTransfer1.0_acc8_latent8_lr1e-6}"
[[ "$BASE_CHECKPOINT_DIR" != /* ]] && BASE_CHECKPOINT_DIR="${PROJECT_ROOT}/${BASE_CHECKPOINT_DIR}"

# Default: ck-1500, BLINK, MMVP, VSTAR, HRBench4K, HRBench8K, MME-RealWorld-Lite
CHECKPOINT_STEPS="${CHECKPOINT_STEPS:-1500}"
EVAL_BENCHMARKS="${EVAL_BENCHMARKS:-HRBench8K}"

# Auto-detect all checkpoint directories (when CHECKPOINT_STEPS unset; default above is 1500)
if [ -z "${CHECKPOINT_STEPS+x}" ]; then
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

EVAL_STEP_LIST="${EVAL_STEP_LIST:-4,8,16,32}"


DATASET_CONFIG="${DATASET_CONFIG:-default}"
# Benchmarks to evaluate: BLINK, VSTAR, MMVP, MMStar, HRBench4K, HRBench8K, MME-RealWorld-Lite, HallBench, MMHal, CRPE, POPE
# Set EVAL_BENCHMARKS to override (e.g. "HRBench8K" or "HRBench4K,HRBench8K,MME-RealWorld-Lite")
# Note: HRBench8K uses 1h merge timeout (8K images); set EVAL_MERGE_WAIT_TIMEOUT to override
export BASE_CHECKPOINT_DIR
export EVAL_BENCHMARKS
LVR_SAVE_ACTIVATION_MAPS="${LVR_SAVE_ACTIVATION_MAPS:-0}"

# ============================================================================

# Initialize conda
if command -v conda &> /dev/null; then
    if [ -f "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" ]; then
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate train 2>/dev/null || true
    fi
fi

export PROJECT_ROOT
cd "$PROJECT_ROOT" || exit 1
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

EVAL_SCRIPT="${PROJECT_ROOT}/evaluation/evaluation.py"
ACCURACY_SCRIPT="${PROJECT_ROOT}/evaluation/calculate_accuracy_by_category.py"

if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
fi
GPU_ID="${GPU_ID:-$CUDA_VISIBLE_DEVICES}"
# HRBench8K: use 4 GPUs to avoid OOM (8K images + 7B model per process)
if [[ "$EVAL_BENCHMARKS" == *"HRBench8K"* ]]; then
    GPU_ID="0,1,2,3"
    echo "HRBench8K detected: using 4 GPUs to avoid OOM (was 8)"
fi
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
        echo "Checkpoint-${step} already converted (skip)"
        continue
    fi

    GLOBAL_STEP_DIR=$(find "$checkpoint_path" -maxdepth 1 -type d -name "global_step*" | head -n 1)
    if [ -z "$GLOBAL_STEP_DIR" ]; then
        echo "Warning: No DeepSpeed checkpoint found in checkpoint-${step}"
        continue
    fi

    echo "Converting checkpoint-${step}..."
    ZERO_TO_FP32="${checkpoint_path}/zero_to_fp32.py"
    ORIGINAL_DIR="${PWD}"

    if [ -f "$ZERO_TO_FP32" ]; then
        cd "$checkpoint_path"
        python "$ZERO_TO_FP32" . . --safe_serialization >/dev/null 2>&1 || \
        python "$ZERO_TO_FP32" . . >/dev/null 2>&1 || {
            echo "  Failed to convert checkpoint-${step}"
            cd "$ORIGINAL_DIR"
            continue
        }
        cd "$ORIGINAL_DIR"
        echo "  Checkpoint-${step} converted"
    fi
done

echo "Checkpoint conversion completed!"

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

def extract_answer(pred):
    given_answer = pred.split('<answer>')[-1].split('</answer')[0].strip()
    if " " in given_answer:
        given_answer = given_answer.split(" ")[0]
    if len(given_answer) > 1:
        given_answer = given_answer[0]
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
        label = item['label']
        category = item.get('category', 'Unknown')

        given_answer = extract_answer(pred)

        category_stats[category]['total'] += 1
        if given_answer == label:
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

# Benchmarks with 'correct' field in JSON (HallBench yes/no, MMHal, refCOCO/REC IoU)
# suffix: "IoU>0.5" for REC, "yes/no" for HallBench/MMHal
calculate_rec_accuracy() {
    local json_file=$1
    local dataset_name=$2
    local show_categories=${3:-1}
    local suffix="${4:-IoU>0.5}"  # default REC; use "yes/no" for HallBench

    if [ ! -f "$json_file" ]; then
        echo "  [Result] $dataset_name: File not found"
        return 1
    fi

    python3 << EOF
import json
path = '$json_file'
suffix = '$suffix'
with open(path) as f:
    data = json.load(f)
if data and isinstance(data[0], dict) and 'overall_accuracy' in data[0]:
    s = data[0]
    total = s.get('overall_total', 0)
    correct = s.get('overall_correct', 0)
    acc = s.get('overall_accuracy', 0)
    print(f"  [Result] $dataset_name: {correct}/{total} = {acc:.2f}% ({suffix})")
    # HallBench: also show Question Pair Acc (official metric)
    if 'question_pair_accuracy' in s:
        pc, pt = s.get('question_pair_correct', 0), s.get('question_pair_total', 0)
        print(f"    Question Pair Acc: {pc}/{pt} = {s['question_pair_accuracy']:.2f}%")
    if $show_categories:
        by_cat = s.get('accuracy_by_category', {})
        for cat, v in sorted(by_cat.items()):
            print(f"    {cat}: {v.get('correct', 0)}/{v.get('total', 0)} = {v.get('accuracy', 0):.2f}%")
else:
    print(f"  [Result] $dataset_name: No summary found")
EOF
}

# MME-RealWorld-Lite: read from summary, show Perception/Reasoning + Overall
calculate_mme_realworld_lite_accuracy() {
    local json_file=$1

    if [ ! -f "$json_file" ]; then
        echo "  [Result] MME-RealWorld-Lite: File not found"
        return 1
    fi

    python3 << EOF
import json
path = '$json_file'
try:
    with open(path) as f:
        data = json.load(f)
    if not data or not isinstance(data[0], dict) or 'overall_accuracy' not in data[0]:
        print("  [Result] MME-RealWorld-Lite: No summary found")
    else:
        s = data[0]
        total = s.get('overall_total', 0)
        correct = s.get('overall_correct', 0)
        acc = s.get('overall_accuracy', 0)
        print(f"  [Result] MME-RealWorld-Lite: {correct}/{total} = {acc:.2f}% (Overall)")
        pr = s.get('accuracy_by_perception_reasoning', {})
        if pr:
            for cat in ['Perception', 'Reasoning']:
                if cat in pr:
                    v = pr[cat]
                    print(f"    {cat}: {v.get('correct', 0)}/{v.get('total', 0)} = {v.get('accuracy', 0):.2f}%")
        else:
            by_cat = s.get('accuracy_by_category', {})
            for cat, v in sorted(by_cat.items()):
                print(f"    {cat}: {v.get('correct', 0)}/{v.get('total', 0)} = {v.get('accuracy', 0):.2f}%")
except Exception as e:
    print(f"  [Result] MME-RealWorld-Lite: Error - {e}")
EOF
}

# POPE: yes/no format - extract yes/no from full prediction text
calculate_yesno_accuracy() {
    local json_file=$1
    local dataset_name=$2
    local show_categories=${3:-1}

    if [ ! -f "$json_file" ]; then
        echo "  [Result] $dataset_name: File not found"
        return 1
    fi

    python3 << EOF
import json
path = '$json_file'
with open(path) as f:
    data = json.load(f)
total = correct = 0
category_stats = {}
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
        cat = item.get('category', 'Unknown')
        category_stats[cat] = category_stats.get(cat, {'t': 0, 'c': 0})
        category_stats[cat]['t'] += 1
        if given == label:
            category_stats[cat]['c'] += 1
if total > 0:
    print(f"  [Result] $dataset_name: {correct}/{total} = {100*correct/total:.2f}% (yes/no)")
    if $show_categories and len(category_stats) > 1:
        for cat in sorted(category_stats.keys()):
            s = category_stats[cat]
            print(f"    {cat}: {s['c']}/{s['t']} = {100*s['c']/s['t']:.2f}%")
else:
    print(f"  [Result] $dataset_name: No valid samples")
EOF
}

# ============================================================================
# Step 2: Run Evaluation
# ============================================================================
echo ""
echo "============================================================================"
echo "Step 2: Starting Evaluation for Stage 2 Distillation Model"
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
    export EVAL_BENCHMARKS="$EVAL_BENCHMARKS"
    export LVR_SAVE_ACTIVATION_MAPS="$LVR_SAVE_ACTIVATION_MAPS"
    export FORCE_RE_EVALUATE="${FORCE_RE_EVALUATE:-1}"
    # HRBench8K (800 samples, 8K images): 1h merge timeout; evaluation.py uses per-benchmark defaults
    export EVAL_MERGE_WAIT_TIMEOUT="${EVAL_MERGE_WAIT_TIMEOUT:-3600}"

    bash "${PROJECT_ROOT}/scripts_release/evaluation/evaluation_7b.sh" "$@"
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

    # Get run_name for result paths (must match evaluation.py: result_prefix = PROJECT_ROOT/result)
    local result_prefix="${PROJECT_ROOT}/result"
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

        # HRBench4K, HRBench8K, MME-RealWorld-Lite
        calculate_accuracy "${PROJECT_ROOT}/evaluation/results/HRBench4K/decoding_by_steps/${run_name}/ck-${checkpoint_step}-step${eval_step}.json" "HRBench4K" 0
        calculate_accuracy "${PROJECT_ROOT}/evaluation/results/HRBench8K/decoding_by_steps/${run_name}/ck-${checkpoint_step}-step${eval_step}.json" "HRBench8K" 0
        calculate_mme_realworld_lite_accuracy "${PROJECT_ROOT}/evaluation/results/MME-RealWorld-Lite/decoding_by_steps/${run_name}/ck-${checkpoint_step}-step${eval_step}.json"
    done

    echo ""
    echo "Done: checkpoint-${checkpoint_step}"
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
# Final Summary - Generate comprehensive results table
# ============================================================================
echo ""
echo "============================================================================"
echo "Stage 2 Distillation Evaluation Summary"
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

def calculate_stats(json_file, is_rec=False, is_yesno=False, read_pair_acc=False, use_mme_pr=False):
    """Calculate overall and per-category accuracy from a result file.
    is_rec: True for REC benchmarks (refCOCO, etc.) which use 'correct' field.
    is_yesno: True for POPE/MME which use yes/no extraction from full text.
    read_pair_acc: True for HallBench - read question_pair_accuracy from summary.
    use_mme_pr: True for MME-RealWorld-Lite - read accuracy_by_perception_reasoning from summary.
    """
    if not os.path.exists(json_file):
        return None, {}

    try:
        with open(json_file, 'r') as f:
            data = json.load(f)

        # MME-RealWorld-Lite: read from summary (Perception/Reasoning + Overall)
        if use_mme_pr and data and isinstance(data[0], dict) and 'overall_accuracy' in data[0]:
            s = data[0]
            overall = s.get('overall_accuracy')
            pr = s.get('accuracy_by_perception_reasoning', {})
            cat_results = {cat: pr[cat].get('accuracy') for cat in ['Perception', 'Reasoning'] if cat in pr}
            return (overall, cat_results)

        # HallBench: read pre-computed question_pair_accuracy from summary
        pair_acc = None
        if read_pair_acc and data and isinstance(data[0], dict) and 'question_pair_accuracy' in data[0]:
            pair_acc = data[0].get('question_pair_accuracy')

        category_stats = defaultdict(lambda: {'total': 0, 'correct': 0})
        total = correct = 0

        for item in data:
            if 'accuracy_by_category' in item or 'overall_accuracy' in item:
                continue

            if is_rec and 'correct' in item:
                # REC format: use pre-computed correct field
                match = bool(item['correct'])
                category = item.get('category', 'Unknown')
            elif is_yesno:
                if 'prediction' not in item or 'label' not in item:
                    continue
                pred = item['prediction'][0] if isinstance(item['prediction'], list) else item['prediction']
                label = item['label']
                category = item.get('category', 'Unknown')
                text = (pred or '').lower()
                idx_yes, idx_no = text.find('yes'), text.find('no')
                given = 'yes' if (idx_yes >= 0 and (idx_no < 0 or idx_yes < idx_no)) else ('no' if idx_no >= 0 else None)
                if given is None:
                    continue  # skip when cannot extract yes/no
                match = given == str(label).strip().lower()
            else:
                if 'prediction' not in item or 'label' not in item:
                    continue
                pred = item['prediction'][0] if isinstance(item['prediction'], list) else item['prediction']
                label = item['label']
                category = item.get('category', 'Unknown')
                given_answer = extract_answer(pred)
                match = given_answer == label

            category_stats[category]['total'] += 1
            if match:
                correct += 1
                category_stats[category]['correct'] += 1
            total += 1

        overall = 100 * correct / total if total > 0 else None
        cat_results = {}
        for cat, stats in category_stats.items():
            cat_results[cat] = 100 * stats['correct'] / stats['total'] if stats['total'] > 0 else None

        if pair_acc is not None:
            return (overall, cat_results, pair_acc)
        return (overall, cat_results)
    except:
        return None, {}

# Configuration
project_root = os.environ.get('PROJECT_ROOT', os.getcwd())
base_dir = os.environ.get('BASE_CHECKPOINT_DIR', '')
eval_steps = os.environ.get('EVAL_STEP_LIST', '4,8').split(',')
eval_steps = [s.strip() for s in eval_steps]

# Get run_name from base_dir (must match evaluation.py)
result_prefix = os.path.join(project_root, "result")
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

# Datasets to evaluate: BLINK, MMVP, VSTAR, HRBench4K, HRBench8K, MME-RealWorld-Lite
datasets = {
    'BLINK': ('blink', '', False),
    'VSTAR': ('vstar_bench', '', False),
    'MMVP': ('MMVP', '', False),
    'HRBench4K': ('HRBench4K', '', False),
    'HRBench8K': ('HRBench8K', '', False),
    'MME-RealWorld-Lite': ('MME-RealWorld-Lite', '', 'mme_pr'),
}

# Collect all results
all_results = {}  # {ck: {step: {dataset: {overall: x, categories: {}}}}}
all_categories = {ds: set() for ds in datasets}

for ck in checkpoints:
    all_results[ck] = {}
    for step in eval_steps:
        all_results[ck][step] = {}
        for ds_name, (dir_name, suffix, fmt) in datasets.items():
            json_file = f'{project_root}/evaluation/results/{dir_name}/decoding_by_steps/{run_name}/ck-{ck}-step{step}{suffix}.json'
            is_rec = fmt is True or fmt == 'correct'
            is_yesno = fmt == 'yesno'
            read_pair = (ds_name == 'HallBench')
            use_mme_pr = (fmt == 'mme_pr')
            res = calculate_stats(json_file, is_rec=is_rec, is_yesno=is_yesno, read_pair_acc=read_pair, use_mme_pr=use_mme_pr)
            overall, cats = res[0], res[1]
            extra = {'pair_acc': res[2]} if len(res) > 2 else {}
            all_results[ck][step][ds_name] = {'overall': overall, 'categories': cats, **extra}
            all_categories[ds_name].update(cats.keys())

# Print comprehensive table
print("=" * 100)
print("COMPREHENSIVE RESULTS TABLE (Stage 2 Distillation)")
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
    # HallBench: Question Pair Acc (official metric)
    if ds_name == 'HallBench' and any('pair_acc' in all_results[ck][step][ds_name] for ck in checkpoints for step in eval_steps):
        row = f"{'Pair Acc':<12}"
        for step in eval_steps:
            values = []
            for ck in checkpoints:
                val = all_results[ck][step][ds_name].get('pair_acc')
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
