import sys
import os
import numpy as np
import torch
import string
import csv
import glob
import re
import json

# Get project root directory (assuming evaluation.py is in evaluation/ subdirectory)
# and add it to Python path so imports from src module work
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from datasets import load_dataset, load_from_disk, DownloadConfig
from tqdm import tqdm
from src.model.qwen_lvr_model import QwenWithLVR
from transformers import AutoTokenizer, AutoProcessor, AutoConfig, Qwen2_5_VLForConditionalGeneration

from qwen_vl_utils import process_vision_info
from PIL import Image
from huggingface_hub import hf_hub_download

from src.train.monkey_patch_forward_lvr import replace_qwen2_5_with_mixed_modality_forward_lvr
from src.constants import DEFAULT_IM_END_TOKEN

# STEP_LIST can be set via environment variable EVAL_STEP_LIST (comma-separated, e.g., "4,8,16")
# If not set, defaults to [4,8,16]
# When USE_BASE_MODEL=1, STEP_LIST is forced to [1] (single run, no LVR steps)
USE_BASE_MODEL = os.environ.get('USE_BASE_MODEL', '0') == '1'
STEP_LIST_ENV = os.environ.get('EVAL_STEP_LIST', '4,8,16')
if USE_BASE_MODEL:
    STEP_LIST = [1]  # Baseline: no step concept, single run
else:
    STEP_LIST = [int(x.strip()) for x in STEP_LIST_ENV.split(',') if x.strip()]

# Multi-process evaluation support
# EVAL_PROCESS_ID: process index (0-based)
# EVAL_TOTAL_PROCESSES: total number of processes
EVAL_PROCESS_ID = int(os.environ.get('EVAL_PROCESS_ID', '0'))
EVAL_TOTAL_PROCESSES = int(os.environ.get('EVAL_TOTAL_PROCESSES', '1'))

# Force re-evaluation: if set to '1', always re-run inference even if result files exist
FORCE_RE_EVALUATE = os.environ.get('FORCE_RE_EVALUATE', '0') == '1'

# HRBench image resolution: use default (same as other benchmarks)
# Set EVAL_HRBENCH8K_MAX_PIXELS=25645056 to preserve more 8K detail (may affect accuracy)
EVAL_HRBENCH8K_MAX_PIXELS = int(os.environ.get('EVAL_HRBENCH8K_MAX_PIXELS', '12845056'))
EVAL_HRBENCH8K_MIN_PIXELS = int(os.environ.get('EVAL_HRBENCH8K_MIN_PIXELS', '3136'))
# HRBench4K: 4K images ~8M px fit in default 12.8M, but use same for consistency if desired
EVAL_HRBENCH4K_MAX_PIXELS = int(os.environ.get('EVAL_HRBENCH4K_MAX_PIXELS', '12845056'))
EVAL_HRBENCH4K_MIN_PIXELS = int(os.environ.get('EVAL_HRBENCH4K_MIN_PIXELS', '3136'))

# Multi-process: write temp file every N samples (so Process 0 can merge when each has >= N)
EVAL_INCREMENTAL_WRITE_INTERVAL = int(os.environ.get('EVAL_INCREMENTAL_WRITE_INTERVAL', '100'))
# Partial merge: only include process files that have at least this many samples
EVAL_MERGE_MIN_SAMPLES_PER_PROCESS = int(os.environ.get('EVAL_MERGE_MIN_SAMPLES_PER_PROCESS', '100'))

# ==== Config ====

# Dataset cache directory
DATASETS_DIR = "/comp_robot/zhoujiazhou/Datasets"
os.makedirs(DATASETS_DIR, exist_ok=True)

# Results output directory (relative to project root)
RESULTS_DIR = os.path.join(PROJECT_ROOT, "evaluation", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

CHKPT_PATHS = [os.environ.get('EVAL_CHECKPOINT_PATH')]

DATASET_CONFIG = {
    'blink': {
        "loader": lambda gen_w_head,run_name,decoding_strategy,checkpoint_num: load_blink_dataset(gen_w_head,run_name,decoding_strategy,checkpoint_num),
        "evaluator": lambda model,proc,data,img_dir,out_dir,ds_name,decoding_strategy,checkpoint_num: evaluate_blink(model, proc, data, img_dir, out_dir, ds_name, decoding_strategy, checkpoint_num),
    },
    "vstar": {
        "loader": lambda gen_w_head,run_name,decoding_strategy,checkpoint_num: load_vstar_dataset(gen_w_head,run_name,decoding_strategy,checkpoint_num),
        "evaluator": lambda model,proc,data,img_dir,out_dir,ds_name,decoding_strategy,checkpoint_num: evaluate_vstar(model, proc, data, img_dir, out_dir, ds_name, decoding_strategy, checkpoint_num),
    },
    "MMVP": {
        "loader": lambda gen_w_head,run_name,decoding_strategy,checkpoint_num: load_mmvp_dataset(gen_w_head,run_name,decoding_strategy,checkpoint_num),
        "evaluator": lambda model,proc,data,img_dir,out_dir,ds_name,decoding_strategy,checkpoint_num: evaluate_mmvp(model, proc, data, img_dir, out_dir, ds_name, decoding_strategy, checkpoint_num),
    },
    "MathVision": {
        "loader": lambda gen_w_head,run_name,decoding_strategy,checkpoint_num: load_mathvision_dataset(gen_w_head,run_name,decoding_strategy,checkpoint_num),
        "evaluator": lambda model,proc,data,img_dir,out_dir,ds_name,decoding_strategy,checkpoint_num: evaluate_mathvision(model, proc, data, img_dir, out_dir, ds_name, decoding_strategy, checkpoint_num),
    },
    "MathVista": {
        "loader": lambda gen_w_head,run_name,decoding_strategy,checkpoint_num: load_mathvista_dataset(gen_w_head,run_name,decoding_strategy,checkpoint_num),
        "evaluator": lambda model,proc,data,img_dir,out_dir,ds_name,decoding_strategy,checkpoint_num: evaluate_mathvista(model, proc, data, img_dir, out_dir, ds_name, decoding_strategy, checkpoint_num),
    },
    "VisuLogic": {
        "loader": lambda gen_w_head,run_name,decoding_strategy,checkpoint_num: load_visulogic_dataset(gen_w_head,run_name,decoding_strategy,checkpoint_num),
        "evaluator": lambda model,proc,data,img_dir,out_dir,ds_name,decoding_strategy,checkpoint_num: evaluate_visulogic(model, proc, data, img_dir, out_dir, ds_name, decoding_strategy, checkpoint_num),
    },
    "EMMA": {
        "loader": lambda gen_w_head,run_name,decoding_strategy,checkpoint_num: load_emma_dataset(gen_w_head,run_name,decoding_strategy,checkpoint_num),
        "evaluator": lambda model,proc,data,img_dir,out_dir,ds_name,decoding_strategy,checkpoint_num: evaluate_emma(model, proc, data, img_dir, out_dir, ds_name, decoding_strategy, checkpoint_num),
    },
    "MMMU": {
        "loader": lambda gen_w_head,run_name,decoding_strategy,checkpoint_num: load_mmmu_dataset(gen_w_head,run_name,decoding_strategy,checkpoint_num),
        "evaluator": lambda model,proc,data,img_dir,out_dir,ds_name,decoding_strategy,checkpoint_num: evaluate_mmmu(model, proc, data, img_dir, out_dir, ds_name, decoding_strategy, checkpoint_num),
    },
    "MMStar": {
        "loader": lambda gen_w_head,run_name,decoding_strategy,checkpoint_num: load_mmstar_dataset(gen_w_head,run_name,decoding_strategy,checkpoint_num),
        "evaluator": lambda model,proc,data,img_dir,out_dir,ds_name,decoding_strategy,checkpoint_num: evaluate_mmstar(model, proc, data, img_dir, out_dir, ds_name, decoding_strategy, checkpoint_num),
    },
    "POPE": {
        "loader": lambda gen_w_head,run_name,decoding_strategy,checkpoint_num: load_pope_dataset(gen_w_head,run_name,decoding_strategy,checkpoint_num),
        "evaluator": lambda model,proc,data,img_dir,out_dir,ds_name,decoding_strategy,checkpoint_num: evaluate_pope(model, proc, data, img_dir, out_dir, ds_name, decoding_strategy, checkpoint_num),
    },
    "MME": {
        "loader": lambda gen_w_head,run_name,decoding_strategy,checkpoint_num: load_mme_dataset(gen_w_head,run_name,decoding_strategy,checkpoint_num),
        "evaluator": lambda model,proc,data,img_dir,out_dir,ds_name,decoding_strategy,checkpoint_num: evaluate_mme(model, proc, data, img_dir, out_dir, ds_name, decoding_strategy, checkpoint_num),
    },
    "refCOCO": {
        "loader": lambda gen_w_head,run_name,decoding_strategy,checkpoint_num: load_refcoco_dataset(gen_w_head,run_name,decoding_strategy,checkpoint_num),
        "evaluator": lambda model,proc,data,img_dir,out_dir,ds_name,decoding_strategy,checkpoint_num: evaluate_refcoco(model, proc, data, img_dir, out_dir, ds_name, decoding_strategy, checkpoint_num),
    },
    "refCOCO+": {
        "loader": lambda gen_w_head,run_name,decoding_strategy,checkpoint_num: load_refcoco_plus_dataset(gen_w_head,run_name,decoding_strategy,checkpoint_num),
        "evaluator": lambda model,proc,data,img_dir,out_dir,ds_name,decoding_strategy,checkpoint_num: evaluate_refcoco_plus(model, proc, data, img_dir, out_dir, ds_name, decoding_strategy, checkpoint_num),
    },
    "refCOCOg": {
        "loader": lambda gen_w_head,run_name,decoding_strategy,checkpoint_num: load_refcocog_dataset(gen_w_head,run_name,decoding_strategy,checkpoint_num),
        "evaluator": lambda model,proc,data,img_dir,out_dir,ds_name,decoding_strategy,checkpoint_num: evaluate_refcocog(model, proc, data, img_dir, out_dir, ds_name, decoding_strategy, checkpoint_num),
    },
    "ReasonSeg": {
        "loader": lambda gen_w_head,run_name,decoding_strategy,checkpoint_num: load_reasonseg_dataset(gen_w_head,run_name,decoding_strategy,checkpoint_num),
        "evaluator": lambda model,proc,data,img_dir,out_dir,ds_name,decoding_strategy,checkpoint_num: evaluate_reasonseg(model, proc, data, img_dir, out_dir, ds_name, decoding_strategy, checkpoint_num),
    },
    "HRBench4K": {
        "loader": lambda gen_w_head,run_name,decoding_strategy,checkpoint_num: load_hrbench4k_dataset(gen_w_head,run_name,decoding_strategy,checkpoint_num),
        "evaluator": lambda model,proc,data,img_dir,out_dir,ds_name,decoding_strategy,checkpoint_num: evaluate_hrbench4k(model, proc, data, img_dir, out_dir, ds_name, decoding_strategy, checkpoint_num),
    },
    "HRBench8K": {
        "loader": lambda gen_w_head,run_name,decoding_strategy,checkpoint_num: load_hrbench8k_dataset(gen_w_head,run_name,decoding_strategy,checkpoint_num),
        "evaluator": lambda model,proc,data,img_dir,out_dir,ds_name,decoding_strategy,checkpoint_num: evaluate_hrbench8k(model, proc, data, img_dir, out_dir, ds_name, decoding_strategy, checkpoint_num),
    },
    "MME-RealWorld-Lite": {
        "loader": lambda gen_w_head,run_name,decoding_strategy,checkpoint_num: load_mme_realworld_lite_dataset(gen_w_head,run_name,decoding_strategy,checkpoint_num),
        "evaluator": lambda model,proc,data,img_dir,out_dir,ds_name,decoding_strategy,checkpoint_num: evaluate_mme_realworld_lite(model, proc, data, img_dir, out_dir, ds_name, decoding_strategy, checkpoint_num),
    },
    "HallBench": {
        "loader": lambda gen_w_head,run_name,decoding_strategy,checkpoint_num: load_hallbench_dataset(gen_w_head,run_name,decoding_strategy,checkpoint_num),
        "evaluator": lambda model,proc,data,img_dir,out_dir,ds_name,decoding_strategy,checkpoint_num: evaluate_hallbench(model, proc, data, img_dir, out_dir, ds_name, decoding_strategy, checkpoint_num),
    },
    "MMHal": {
        "loader": lambda gen_w_head,run_name,decoding_strategy,checkpoint_num: load_mmhal_dataset(gen_w_head,run_name,decoding_strategy,checkpoint_num),
        "evaluator": lambda model,proc,data,img_dir,out_dir,ds_name,decoding_strategy,checkpoint_num: evaluate_mmhal(model, proc, data, img_dir, out_dir, ds_name, decoding_strategy, checkpoint_num),
    },
    "CRPE": {
        "loader": lambda gen_w_head,run_name,decoding_strategy,checkpoint_num: load_crpe_dataset(gen_w_head,run_name,decoding_strategy,checkpoint_num),
        "evaluator": lambda model,proc,data,img_dir,out_dir,ds_name,decoding_strategy,checkpoint_num: evaluate_crpe(model, proc, data, img_dir, out_dir, ds_name, decoding_strategy, checkpoint_num),
    },
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== Core utilities ====

def _extract_answer_from_response(response: str) -> str:
    """Extract answer from <answer>...</answer> block."""
    given_answer = response.split('<answer>')[-1]
    given_answer = given_answer.split('</answer')[0].strip()
    return given_answer

def accuracy_reward(response: str, ground_truth: str) -> float:
    """MCQ-style: extract first A/B/C/D/E (handles (A), (E), A, etc.)."""
    given_answer = _extract_answer_from_response(response)
    # Fallback: if no <answer> block or empty, search in full response (align with lmms-eval)
    if not given_answer or not given_answer.strip():
        given_answer = response
    # Use regex to find first option letter (handles "(E)", "E", "(A) Option" etc.)
    m = re.search(r'[ABCDE]', given_answer, re.IGNORECASE)
    given_letter = m.group(0).upper() if m else ''
    return given_letter == str(ground_truth).strip().upper()

def accuracy_reward_math(response: str, ground_truth: str, precision: int = None) -> float:
    """Free-form math: normalize and compare. Handles float rounding if precision given."""
    given_answer = _extract_answer_from_response(response)
    given_answer = re.sub(r'\s+', ' ', given_answer).strip()
    gt = str(ground_truth).strip()
    if precision is not None and precision >= 0:
        try:
            given_val = float(given_answer.replace(',', ''))
            gt_val = float(gt.replace(',', ''))
            return abs(given_val - gt_val) < 10 ** (-precision)
        except (ValueError, TypeError):
            pass
    return given_answer.lower() == gt.lower()


def accuracy_reward_yesno(response: str, ground_truth: str) -> float:
    """Yes/No style: extract yes or no from response (case-insensitive)."""
    text = response.lower()
    gt = str(ground_truth).strip().lower()
    idx_yes = text.find('yes')
    idx_no = text.find('no')
    if idx_yes >= 0 and (idx_no < 0 or idx_yes < idx_no):
        given = 'yes'
    elif idx_no >= 0:
        given = 'no'
    else:
        return False
    return given == gt


def _parse_bbox_from_response(response: str):
    """
    Parse bounding box from model response.
    Supports formats:
    - [x1, y1, x2, y2] literal list in output
    - {"bbox_2d": [x1,y1,x2,y2]}, {"bbox": [x1,y1,x2,y2]}
    - <answer bbox_2d="[x1,y1,x2,y2]"> or <answer bbox="[x1,y1,x2,y2]"> attribute format
    Returns (x1, y1, x2, y2) or None if parse fails.
    Uses the same regex as the official Qwen refCOCO script.
    """
    def _coords_from_four(nums):
        x1, y1, x2, y2 = float(nums[0]), float(nums[1]), float(nums[2]), float(nums[3])
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1
        return (x1, y1, x2, y2)

    # 1. Prefer bbox from bbox_2d / bbox attribute (XML or JSON key)
    # Handles: <answer bbox_2d="[x,y,x2,y2]"> and {"bbox_2d": [x,y,x2,y2]}
    attr_pattern = re.search(r'bbox(?:_2d)?\s*[=:]\s*["\s]*\[(\d+)[,\s]+(\d+)[,\s]+(\d+)[,\s]+(\d+)\]', response)
    if attr_pattern:
        return _coords_from_four(attr_pattern.groups())

    # 2. Any [x, y, x2, y2] pattern (official Qwen script uses this regex)
    bbox_pattern = r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]'
    matches = re.findall(bbox_pattern, response)
    if matches:
        return _coords_from_four(matches[0])

    # 3. Fallback: find first 4 numbers anywhere in text
    numbers = re.findall(r'\d+\.?\d*', response)
    if len(numbers) >= 4:
        return _coords_from_four(numbers[:4])

    return None


def _rec_pred_to_absolute(bbox_pred, img_width, img_height, model_input_width=None, model_input_height=None):
    """
    Convert predicted bbox to absolute pixel coords in original image space.
    When model_input_width/height given (from image_grid_thw): Qwen outputs in resized space,
    scale pred_model -> pred_orig = pred * (img_orig / model_input).
    Otherwise: handle [0,1] normalized, [0,1000] normalized, or absolute (clip only).
    """
    if bbox_pred is None or img_width <= 0 or img_height <= 0:
        return None
    x1, y1, x2, y2 = [float(v) for v in bbox_pred[:4]]
    coords = np.array([x1, y1, x2, y2], dtype=np.float64)
    if model_input_width and model_input_height and model_input_width > 0 and model_input_height > 0:
        coords[0::2] = coords[0::2] * (img_width / model_input_width)
        coords[1::2] = coords[1::2] * (img_height / model_input_height)
    else:
        max_val = float(np.max(np.abs(coords)))
        img_max = max(img_width, img_height)
        if max_val <= 1.5:
            coords[0::2] *= img_width
            coords[1::2] *= img_height
        elif max_val <= 1000 and max_val > img_max * 1.15:
            coords[0::2] = coords[0::2] / 1000.0 * img_width
            coords[1::2] = coords[1::2] / 1000.0 * img_height
    coords[0] = np.clip(coords[0], 0, img_width)
    coords[2] = np.clip(coords[2], 0, img_width)
    coords[1] = np.clip(coords[1], 0, img_height)
    coords[3] = np.clip(coords[3], 0, img_height)
    if coords[2] < coords[0]:
        coords[0], coords[2] = coords[2], coords[0]
    if coords[3] < coords[1]:
        coords[1], coords[3] = coords[3], coords[1]
    return tuple(coords.tolist())


def _compute_iou(bbox_pred, bbox_gt, img_width=None, img_height=None, model_input_width=None, model_input_height=None):
    """
    Compute IoU between two boxes. Both in [x1, y1, x2, y2] format.
    bbox_pred is converted to absolute coords via _rec_pred_to_absolute when img size given.
    model_input_*: Qwen model input size (image_grid_thw), for resized->original conversion.
    """
    if bbox_pred is None:
        return 0.0
    if img_width and img_height:
        bbox_pred = _rec_pred_to_absolute(
            bbox_pred, img_width, img_height,
            model_input_width=model_input_width, model_input_height=model_input_height
        )
        if bbox_pred is None:
            return 0.0
    x1_p, y1_p, x2_p, y2_p = bbox_pred
    x1_g, y1_g, x2_g, y2_g = bbox_gt
    xi1 = max(x1_p, x1_g)
    yi1 = max(y1_p, y1_g)
    xi2 = min(x2_p, x2_g)
    yi2 = min(y2_p, y2_g)
    inter_w = max(0, xi2 - xi1)
    inter_h = max(0, yi2 - yi1)
    inter = inter_w * inter_h
    area_p = (x2_p - x1_p) * (y2_p - y1_p)
    area_g = (x2_g - x1_g) * (y2_g - y1_g)
    union = area_p + area_g - inter
    return inter / union if union > 0 else 0.0


def _coco_bbox_to_xyxy(bbox):
    """Convert COCO [x, y, w, h] to [x1, y1, x2, y2]."""
    x, y, w, h = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
    return (x, y, x + w, y + h)


def calculate_and_add_accuracy_summary(data: list, json_file_path: str):
    """
    Calculate accuracy by category and add summary to the beginning of the data list.
    Similar to calculate_accuracy_by_category.py logic.
    """
    from collections import defaultdict
    
    # Skip if already has summary (accuracy_by_category or accuracy_by_perception_reasoning for MME-RealWorld-Lite)
    if len(data) > 0 and isinstance(data[0], dict):
        first = data[0]
        if 'overall_accuracy' in first or 'accuracy_by_category' in first or 'accuracy_by_perception_reasoning' in first:
            return data
    
    # Calculate accuracy by category
    category_stats = defaultdict(lambda: {'total': 0, 'correct': 0})
    pr_stats = defaultdict(lambda: {'total': 0, 'correct': 0})  # Perception/Reasoning for MME-RealWorld-Lite
    
    for item in data:
        # Skip summary objects
        if 'accuracy_by_category' in item or 'overall_accuracy' in item:
            continue
            
        category = item.get('category', 'Unknown')
        prediction = item.get('prediction', [])
        label = item.get('label', '')
        
        # Extract prediction string
        if isinstance(prediction, list) and len(prediction) > 0:
            prediction_str = prediction[0]
        elif isinstance(prediction, str):
            prediction_str = prediction
        else:
            prediction_str = ''
        
        # REC (referring expression comprehension): use pre-computed "correct" if present
        if 'correct' in item:
            match = bool(item['correct'])
        else:
            given_answer = _extract_answer_from_response(prediction_str)
            label_str = str(label).strip().lower()
            is_mcq = len(label_str) == 1 and label_str.upper() in 'ABCDE'
            is_yesno = label_str in ('yes', 'no')
            if is_mcq:
                m = re.search(r'[ABCDE]', given_answer, re.IGNORECASE)
                given_letter = m.group(0).upper() if m else ''
                match = given_letter == str(label).strip().upper()
            elif is_yesno:
                match = accuracy_reward_yesno(prediction_str, label)
            else:
                given_answer = re.sub(r'\s+', ' ', given_answer).strip()
                match = given_answer.lower() == label_str
        
        # Count by category
        category_stats[category]['total'] += 1
        if match:
            category_stats[category]['correct'] += 1

        # MME-RealWorld-Lite: also count by Perception/Reasoning (id format: "Perception/..." or "Reasoning/...")
        sample_id = str(item.get('id', ''))
        if '/' in sample_id:
            top_level = sample_id.split('/')[0].strip().lower()
            if top_level in ('perception', 'reasoning'):
                pr_cat = 'Perception' if top_level == 'perception' else 'Reasoning'
                pr_stats[pr_cat]['total'] += 1
                if match:
                    pr_stats[pr_cat]['correct'] += 1
    
    # MME-RealWorld-Lite: use Perception/Reasoning for overall, omit accuracy_by_category
    if pr_stats:
        pr_correct = sum(stats['correct'] for stats in pr_stats.values())
        pr_total = sum(stats['total'] for stats in pr_stats.values())
        total_samples = pr_total
        overall_accuracy = (pr_correct / pr_total * 100) if pr_total > 0 else 0.0
        accuracy_by_perception_reasoning = {}
        for pr_cat, stats in pr_stats.items():
            t, c = stats['total'], stats['correct']
            accuracy_by_perception_reasoning[pr_cat] = {
                'total': t, 'correct': c,
                'accuracy': (c / t * 100) if t > 0 else 0.0
            }
        accuracy_summary = {
            'accuracy_by_perception_reasoning': accuracy_by_perception_reasoning,
            'overall_accuracy': overall_accuracy,
            'overall_correct': pr_correct,
            'overall_total': pr_total
        }
    else:
        # Other benchmarks: use accuracy_by_category
        results = {}
        for category, stats in category_stats.items():
            total = stats['total']
            correct = stats['correct']
            accuracy = (correct / total * 100) if total > 0 else 0.0
            results[category] = {
                'total': total,
                'correct': correct,
                'accuracy': accuracy
            }
        total_correct = sum(stats['correct'] for stats in results.values())
        total_samples = sum(stats['total'] for stats in results.values())
        overall_accuracy = (total_correct / total_samples * 100) if total_samples > 0 else 0.0
        accuracy_summary = {
            'accuracy_by_category': results,
            'overall_accuracy': overall_accuracy,
            'overall_correct': total_correct,
            'overall_total': total_samples
        }
        # CRPE: also compute CRPE_relation (excluding exist: predicate + subject + object only)
        if 'exist' in results:
            relation_cats = ['predicate', 'subject', 'object']
            relation_correct = sum(results[cat]['correct'] for cat in relation_cats if cat in results)
            relation_total = sum(results[cat]['total'] for cat in relation_cats if cat in results)
            if relation_total > 0:
                accuracy_summary['CRPE_relation_accuracy'] = relation_correct / relation_total * 100
                accuracy_summary['CRPE_relation_correct'] = relation_correct
                accuracy_summary['CRPE_relation_total'] = relation_total

    # HallBench: compute Question Pair Acc (official metric)
    # Pair key: (category, subcategory, set_id, pair_question_id). Pair correct iff all questions in pair correct.
    if total_samples > 0 and any('set_id' in x and 'pair_question_id' in x for x in data if isinstance(x, dict)):
        from collections import defaultdict
        pair_correctness = defaultdict(list)  # key -> [correct, correct, ...]
        for item in data:
            if 'accuracy_by_category' in item or 'overall_accuracy' in item:
                continue
            key = (item.get('category', ''), item.get('subcategory', ''), str(item.get('set_id', '')), str(item.get('pair_question_id', '')))
            pair_correctness[key].append(bool(item.get('correct', False)))
        pair_correct = sum(1 for corr_list in pair_correctness.values() if all(corr_list))
        pair_total = len(pair_correctness)
        question_pair_acc = (pair_correct / pair_total * 100) if pair_total > 0 else 0.0
        accuracy_summary['question_pair_correct'] = pair_correct
        accuracy_summary['question_pair_total'] = pair_total
        accuracy_summary['question_pair_accuracy'] = question_pair_acc
    
    # Add summary to the beginning
    return [accuracy_summary] + data

def _get_dataset_chunk(dataset, total_processes, process_id):
    """Split dataset for multi-process evaluation. Returns the chunk for this process."""
    if total_processes <= 1:
        return dataset
    dataset_size = len(dataset)
    chunk_size = dataset_size // total_processes
    start_idx = process_id * chunk_size
    end_idx = dataset_size if process_id == total_processes - 1 else start_idx + chunk_size
    # Use .select() for HuggingFace Dataset, or slice for list
    if hasattr(dataset, 'select'):
        chunk = dataset.select(range(start_idx, end_idx))
    else:
        chunk = dataset[start_idx:end_idx]
    print(f"[Process {process_id}/{total_processes}] Processing samples {start_idx}-{end_idx-1} (total: {dataset_size})")
    return chunk


# Expected total samples per benchmark (for merge validation). 8 processes => each should output total/8.
# HR-Bench: 4K/8K = image resolution, each has 800 samples (HuggingFace DreamMr/HR-Bench)
# MME-RealWorld-Lite: 1919 items in JSON (after img exists filter)
# HallBench: 951 image + 178 non_image = 1129
BENCHMARK_EXPECTED_TOTAL = {
    'hrbench8k': 800,            # 8 processes => 100 each
    'hrbench4k': 800,            # 8 processes => 100 each
    'mme-realworld-lite': 1919,  # 8 processes => ~240 each
    'mme_realworld_lite': 1919,
    'hallbench': 1129,           # 8 processes => ~141 each
}


def _get_merge_wait_timeout(benchmark_name=""):
    """Get merge wait timeout in seconds. Large benchmarks (HRBench8K, MME-RealWorld-Lite) need longer."""
    timeout_env = os.environ.get('EVAL_MERGE_WAIT_TIMEOUT', '')
    if timeout_env:
        try:
            return int(timeout_env)
        except ValueError:
            pass
    # HRBench8K (8K resolution images): 1 hour - inference slower than 4K
    # HRBench4K, MME-RealWorld-Lite: 30 min
    # Default: 10 min for smaller benchmarks
    bn = str(benchmark_name or '').lower().replace('-', '_')
    if bn == 'hrbench8k':
        return 3600  # 1 hour for HRBench8K (8K images)
    if bn in ('hrbench4k', 'mme-realworld-lite', 'mme_realworld_lite'):
        return 1800  # 30 min
    return 600  # 10 min default


def _count_samples_in_file(filepath):
    """Count result samples (items with id) in a JSON file."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        return len([x for x in data if isinstance(x, dict) and ('id' in x or 'prediction' in x)])
    except Exception:
        return 0


def _merge_multiprocess_results(out_file, total_processes, process_id, benchmark_name="", expected_total_samples=None):
    """
    Handle multi-process result merging.
    - Process 0: wait for all temp files, validate each has expected samples (e.g. 1000 for HRBench8K),
      then merge. Only merge when all 8 processes have output their full chunk.
    - Other processes: wait briefly.
    - On timeout: merge partial results (preserve progress) instead of discarding.
    
    IMPORTANT: Only Process 0 cleans up old temp files (before evaluation starts,
    the caller should call _cleanup_old_temp_files). This avoids a race condition
    where fast processes write temp files that are then deleted by slower processes.
    """
    if total_processes <= 1:
        return
    import time
    all_temp_files = [out_file.replace('.json', f'_process{i}.json') for i in range(total_processes)]
    
    # Resolve expected per-process count
    expected_per_process = None
    if expected_total_samples is not None:
        expected_per_process = expected_total_samples // total_processes
    elif benchmark_name:
        key = str(benchmark_name).lower().replace('-', '_')
        total = BENCHMARK_EXPECTED_TOTAL.get(key)
        if total is not None:
            expected_per_process = total // total_processes
    
    if process_id == 0:
        max_wait_time = _get_merge_wait_timeout(benchmark_name)
        check_interval = 2
        waited_time = 0
        all_exist = False
        all_valid = False
        
        while waited_time < max_wait_time:
            all_exist = all(os.path.exists(f) for f in all_temp_files)
            if all_exist and expected_per_process is not None:
                # Validate each file has expected sample count (e.g. 1000 for HRBench8K)
                counts = [_count_samples_in_file(f) for f in all_temp_files]
                all_valid = all(c >= expected_per_process for c in counts)
                if all_valid:
                    break
                # All files exist but some incomplete (process crashed) - no point waiting, merge partial
                label = f"{benchmark_name}: " if benchmark_name else ""
                print(f"[Process 0] {label}Some files incomplete: counts={counts} (need >={expected_per_process} each). Merging partial.")
                break
            elif all_exist:
                all_valid = True
                break
            time.sleep(check_interval)
            waited_time += check_interval
            if waited_time % 10 == 0:
                existing_count = sum(os.path.exists(f) for f in all_temp_files)
                label = f"{benchmark_name}: " if benchmark_name else ""
                print(f"[Process 0] {label}Waiting for all processes... ({existing_count}/{total_processes} temp files found, waited {waited_time}s)")
        
        # Collect temp files that exist (for full or partial merge)
        existing_temp_files = [f for f in all_temp_files if os.path.exists(f)]
        
        if all_exist and all_valid:
            label = f"{benchmark_name} " if benchmark_name else ""
            print(f"[Process 0] Merging {label}results from {total_processes} processes (each has >={expected_per_process or '?'} samples)...")
        elif all_exist and not all_valid:
            counts = [_count_samples_in_file(f) for f in existing_temp_files]
            label = f"{benchmark_name} " if benchmark_name else ""
            print(f"[Process 0] Timeout: some files incomplete (counts={counts}, need >={expected_per_process} each). Merging PARTIAL results.")
        elif not all_exist:
            existing_count = len(existing_temp_files)
            label = f"{benchmark_name} " if benchmark_name else ""
            print(f"[Process 0] Timeout after {max_wait_time}s: only {existing_count}/{total_processes} processes completed")
            print(f"[Process 0] Merging PARTIAL {label}results (missing: {[os.path.basename(f) for f in all_temp_files if f not in existing_temp_files]})")
        
        if existing_temp_files:
            merged_result = []
            for temp_file in existing_temp_files:
                try:
                    with open(temp_file, 'r') as f:
                        merged_result.extend(json.load(f))
                except Exception as e:
                    print(f"[Process 0] Warning: Failed to read {temp_file}: {e}")
            # Sort by id to maintain order (id can be int or str like "val_Counting_1")
            def _sort_key(x):
                if 'id' not in x:
                    return ''
                vid = x.get('id')
                return str(vid) if vid is not None else ''
            merged_result.sort(key=_sort_key)
            merged_result = calculate_and_add_accuracy_summary(merged_result, out_file)
            json.dump(merged_result, open(out_file, 'w+'), indent=2)
            result_count = len([x for x in merged_result if 'id' in x])
            for temp_file in existing_temp_files:
                try:
                    os.remove(temp_file)
                except Exception:
                    pass
            if not all_exist:
                print(f"[Process 0] WARNING: Partial merge saved {result_count} results to {out_file}. Re-run with FORCE_RE_EVALUATE=1 to complete.")
            else:
                print(f"[Process 0] Merged {result_count} {label}results into {out_file}")
        else:
            label = f"{benchmark_name} " if benchmark_name else ""
            print(f"[Process 0] Error: No temp files found for {label}merge")
    else:
        time.sleep(1)


def _cleanup_old_temp_files(out_file):
    """Clean up leftover temp files from previous failed runs.
    MUST only be called by Process 0 to avoid race conditions.
    Only deletes files older than 5 minutes to avoid race: Process 0 does merge
    so it may start a benchmark later than Process 1-7, who could have already
    written their temp files. Deleting those would cause merge failure."""
    if EVAL_PROCESS_ID != 0 or EVAL_TOTAL_PROCESSES <= 1:
        return
    import time
    stale_threshold = 300  # seconds - only delete files from previous runs
    now = time.time()
    temp_pattern = out_file.replace('.json', '_process*.json')
    for old_temp_file in glob.glob(temp_pattern):
        try:
            if os.path.getmtime(old_temp_file) < now - stale_threshold:
                os.remove(old_temp_file)
                print(f"Cleaned up old temp file: {old_temp_file}")
        except Exception as e:
            print(f"Warning: Failed to remove old temp file {old_temp_file}: {e}")


def _get_out_file(out_dir, checkpoint_num, steps, decoding_strategy, task_instruction):
    """Generate output file path for a given checkpoint/step."""
    if decoding_strategy == "baseline":
        filename = "baseline.json"
    elif checkpoint_num:
        filename = f"ck-{checkpoint_num}-step{steps}.json"
    else:
        filename = f"{decoding_strategy}{steps:03d}.json"
    if not (len(task_instruction) > 0 and task_instruction[0] != ' '):
        base_name = filename.replace('.json', '')
        filename = f"{base_name}_noTaskInstruction.json"
    return os.path.join(out_dir, filename)


def _get_temp_out_file(out_file):
    """Get the temp output file path for this process."""
    if EVAL_TOTAL_PROCESSES > 1:
        return out_file.replace('.json', f'_process{EVAL_PROCESS_ID}.json')
    return out_file


def get_task_instruction(bench_name):
    # HRBench: use same format as other MCQ (model trained with this); lmms-eval uses "Answer the option letter directly"
    if bench_name in ("vstar", "mmvp", "blink", "visulogic", "emma", "mmmu", "mmstar", "hrbench4k", "hrbench8k", "mme_realworld_lite", "crpe"):
        return "\nAnswer with the option's letter from the given choices directly."
    elif bench_name in ("mathvision", "mathvista"):
        return "\nProvide your final answer within <answer></answer> tags."
    elif bench_name in ("pope", "mme"):
        return "\nAnswer with yes or no only."
    elif bench_name in ("refcoco", "refcoco_plus", "refcocog", "reasonseg"):
        return ""
    elif bench_name in ("hallbench", "mmhal"):
        return "\nProvide your answer concisely."
    else:
        raise ValueError(f"Unknown benchmark: {bench_name}")

def _get_hrbench_image_kwargs(benchmark_name):
    """Get min_pixels/max_pixels for HRBench to preserve high-res (align with lmms-eval)."""
    bn = str(benchmark_name or '').lower().replace('-', '_')
    if bn == 'hrbench8k':
        return {"min_pixels": EVAL_HRBENCH8K_MIN_PIXELS, "max_pixels": EVAL_HRBENCH8K_MAX_PIXELS}
    if bn == 'hrbench4k':
        return {"min_pixels": EVAL_HRBENCH4K_MIN_PIXELS, "max_pixels": EVAL_HRBENCH4K_MAX_PIXELS}
    return {}


def create_messages(img_path, question, benchmark_name=None):
    """Create chat messages. For HRBench, pass min_pixels/max_pixels to preserve high-res images."""
    img_kwargs = _get_hrbench_image_kwargs(benchmark_name)

    def _make_image_content(img):
        content = {"type": "image", "image": img}
        content.update(img_kwargs)
        return content

    if not isinstance(img_path, list):
        messages = [
            {
                "role": "user",
                "content": [
                    _make_image_content(img_path),
                    {"type": "text", "text": question},
                ],
            }
        ]
    else:
        vision_content = [_make_image_content(ip) for ip in img_path]
        vision_content.append({"type": "text", "text": question})
        messages = [{"role": "user", "content": vision_content}]
    return messages

def load_model_and_processor(chkpt_pth):
    # Determine run_name for organizing result directories.
    # Priority: EVAL_BASE_RUN_NAME env var > strip known result prefix > derive from path
    base_run_name = os.environ.get('EVAL_BASE_RUN_NAME')
    if base_run_name:
        checkpoint_name = os.path.basename(chkpt_pth)
        run_name = f"{base_run_name}/{checkpoint_name}"
    else:
        # Strip the project result prefix to get a relative run_name
        result_prefix = os.path.join(PROJECT_ROOT, "result")
        norm_path = os.path.normpath(chkpt_pth)
        norm_prefix = os.path.normpath(result_prefix)
        if norm_path.startswith(norm_prefix + os.sep):
            relative_path = norm_path[len(norm_prefix) + 1:]
        else:
            # Fallback: use parent directory name as run_name
            relative_path = os.path.basename(os.path.dirname(chkpt_pth))
        # Remove trailing checkpoint-N directory from run_name
        parts = relative_path.split(os.sep)
        if parts and re.match(r'checkpoint-\d+', parts[-1]):
            parts = parts[:-1]
        run_name = '/'.join(parts) if parts else os.path.basename(chkpt_pth)

    # Determine device configuration (single GPU only)
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0")
    
    if USE_BASE_MODEL:
        # Load base Qwen2.5-VL model without LVR modifications
        print("Loading base Qwen2.5-VL model (no LVR)...")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            chkpt_pth,
            trust_remote_code=True,
            torch_dtype="auto",
            attn_implementation="flash_attention_2",
            low_cpu_mem_usage=True,
        )
    else:
        # Load LVR model with monkey patch
        print("Loading LVR model...")
        config = AutoConfig.from_pretrained(chkpt_pth)
        # Pass additional config parameters to ensure inference mode matches training mode
        use_box_feature_resampler = getattr(config, 'use_box_feature_resampler', False)
        use_stage2_distillation = getattr(config, 'use_stage2_distillation', False)
        replace_qwen2_5_with_mixed_modality_forward_lvr(
            inference_mode=True,
            coconut=True,
            use_box_feature_resampler=use_box_feature_resampler,
            use_stage2_distillation=use_stage2_distillation,
        )
        
        model = QwenWithLVR.from_pretrained(
            chkpt_pth,
            config=config,
            trust_remote_code=True,
            torch_dtype="auto",
            attn_implementation="flash_attention_2",
            low_cpu_mem_usage=True,
        )
        
        # Explicitly bind the monkey-patched forward to the model instance
        # This is needed because class-level monkey patching may not propagate to already-loaded classes
        import types
        from src.train.monkey_patch_forward_lvr import qwen2_5_mixed_modality_forward_lvr_with_resampler_inference
        model.forward = types.MethodType(qwen2_5_mixed_modality_forward_lvr_with_resampler_inference, model)
        type(model).forward = qwen2_5_mixed_modality_forward_lvr_with_resampler_inference
        
        # Enable activation map visualization if requested
        save_activation_maps = os.environ.get('LVR_SAVE_ACTIVATION_MAPS', '0') == '1'
        if save_activation_maps and hasattr(model, 'lvr_head'):
            lvr_head_type_name = type(model.lvr_head).__name__
            if lvr_head_type_name in ['LVRHeadGatedFocus', 'LVRHeadIntrinsicSimilarity'] and hasattr(model.lvr_head, 'save_activation_maps'):
                model.lvr_head.save_activation_maps = True
    
    # Move model to device manually
    model = model.to(device)
    
    # Re-bind forward after to(device) in case it was reset
    if not USE_BASE_MODEL and (getattr(config, 'use_box_feature_resampler', False) or getattr(config, 'use_dit_reconstruction', False)):
        import types
        from src.train.monkey_patch_forward_lvr import qwen2_5_mixed_modality_forward_lvr_with_resampler_inference
        model.forward = types.MethodType(qwen2_5_mixed_modality_forward_lvr_with_resampler_inference, model)
        type(model).forward = qwen2_5_mixed_modality_forward_lvr_with_resampler_inference

    processor = AutoProcessor.from_pretrained(chkpt_pth)

    return model, processor, run_name

def run_inference(model, processor, img_path, text, steps, decoding_strategy, 
                  sample_idx=None, activation_map_save_dir=None, benchmark_name=None):
    messages = create_messages(img_path, text, benchmark_name=benchmark_name)
    text_formatted = processor.apply_chat_template( messages, tokenize=False, add_generation_prompt=True)

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text_formatted],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    # Ensure inputs are on the same device as the model
    model_device = next(model.parameters()).device
    inputs = {k: v.to(model_device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    
    # Store visualization context in model for LVR head to access
    save_activation_maps = os.environ.get('LVR_SAVE_ACTIVATION_MAPS', '0') == '1'
    if save_activation_maps and hasattr(model, 'lvr_head') and hasattr(model.lvr_head, 'save_activation_maps'):
        if model.lvr_head.save_activation_maps:
            # Store visualization context as model attributes
            model._lvr_activation_map_save_dir = activation_map_save_dir
            model._lvr_sample_idx = sample_idx
            model._lvr_step_idx = steps
            model._lvr_image_grid_thw = inputs.get('image_grid_thw', None)

    # DiT reconstruction: set sample index so forward can save generated images (DIT_SAVE_DIR)
    # Use the real sample_idx from the evaluation loop so filenames match the dataset index.
    dit_save_dir = os.environ.get('DIT_SAVE_DIR', '')
    if dit_save_dir:
        model._dit_sample_idx = sample_idx if sample_idx is not None else 0
        model._dit_benchmark_name = benchmark_name or ''
        # Reset LVR hidden buffer for new sample to ensure clean state
        if hasattr(model, '_dit_lvr_hidden_buffer'):
            model._dit_lvr_hidden_buffer = {}
            model._dit_lvr_step_counter = {}

    # Set up stopping criteria: stop at <|im_end|> token
    # This is compatible with all modes (base model, LVR model, attention isolation mode)
    im_end_token_id = processor.tokenizer.convert_tokens_to_ids(DEFAULT_IM_END_TOKEN)
    if im_end_token_id is None or im_end_token_id == processor.tokenizer.unk_token_id:
        # Fallback: use eos_token_id if im_end_token_id is not found
        im_end_token_id = processor.tokenizer.eos_token_id
    
    with torch.no_grad():
        if USE_BASE_MODEL:
            # Base model: use standard generation without LVR parameters
            generated_ids = model.generate(
                **inputs, 
                max_new_tokens=512,
                eos_token_id=im_end_token_id,
                pad_token_id=processor.tokenizer.pad_token_id
            )
        else:
            # LVR model: use custom generation with decoding_strategy and lvr_steps
            lvr_steps = [steps]
            
            generated_ids = model.generate(
                **inputs, 
                max_new_tokens=512,
                decoding_strategy=decoding_strategy,
                lvr_steps=lvr_steps,
                eos_token_id=im_end_token_id,
                pad_token_id=processor.tokenizer.pad_token_id
            )
        
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs['input_ids'], generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        
        # Post-process: stop at <|im_end|> token if present, and clean up repeated special tokens
        # This is compatible with all modes:
        # - Base model: only cleans up if excessive tokens appear (shouldn't happen normally)
        # - LVR model: cleans up normal output
        # - Attention isolation mode: cleans up excessive repeated tokens
        cleaned_outputs = []
        for text in output_text:
            # Stop at <|im_end|> if present (handles cases where model generates beyond stop token)
            if DEFAULT_IM_END_TOKEN in text:
                text = text.split(DEFAULT_IM_END_TOKEN)[0] + DEFAULT_IM_END_TOKEN
            
            # Remove excessive repeated special tokens
            # Only removes consecutive duplicates (separated by whitespace), safe for all modes
            # Remove repeated <|lvr_start|> patterns (only if consecutive)
            text = re.sub(r'(<\|lvr_start\|>\s*)+', '<|lvr_start|>', text)
            # Remove repeated <|im_start|> patterns (only if consecutive)
            text = re.sub(r'(<\|im_start\|>\s*)+', '<|im_start|>', text)
            
            cleaned_outputs.append(text)
        output_text = cleaned_outputs
    
    # Clear visualization context after inference
    if save_activation_maps and hasattr(model, '_lvr_activation_map_save_dir'):
        delattr(model, '_lvr_activation_map_save_dir')
        if hasattr(model, '_lvr_sample_idx'):
            delattr(model, '_lvr_sample_idx')
        if hasattr(model, '_lvr_step_idx'):
            delattr(model, '_lvr_step_idx')
        if hasattr(model, '_lvr_image_grid_thw'):
            delattr(model, '_lvr_image_grid_thw')
    # Clean up DiT inference state after each sample
    for attr in ('_dit_sample_idx', '_dit_benchmark_name', '_dit_lvr_hidden_buffer', '_dit_lvr_step_counter'):
        if hasattr(model, attr):
            delattr(model, attr)

    rec_benchmarks = ('refcoco', 'refcoco_plus', 'refcocog', 'reasonseg')
    if benchmark_name and benchmark_name.lower() in rec_benchmarks:
        grid = inputs.get('image_grid_thw')
        if grid is not None and hasattr(grid, '__len__') and len(grid) > 0:
            return output_text, grid
    return output_text

# ==== Evaluation ====

def evaluate_vstar(
        model, 
        processor, 
        dataset,
        image_dir,
        out_dir,
        ds_name,
        decoding_strategy="steps",
        checkpoint_num=None,
        ):
    print(f"Evaluating VSTAR with decoding strategy: {decoding_strategy}")
    os.makedirs(out_dir, exist_ok=True)
    
    save_activation_maps = os.environ.get('LVR_SAVE_ACTIVATION_MAPS', '0') == '1'
    activation_map_dir = None
    if save_activation_maps:
        activation_map_dir = os.path.join(out_dir, "activation_maps")
        os.makedirs(activation_map_dir, exist_ok=True)
    
    task_instruction = get_task_instruction("vstar")
    step2results_category = {}
    step2results_overall = {}
    dataset_chunk = _get_dataset_chunk(dataset, EVAL_TOTAL_PROCESSES, EVAL_PROCESS_ID)

    for steps in STEP_LIST:
        step2results_category[steps] = {}
        out_file = _get_out_file(out_dir, checkpoint_num, steps, decoding_strategy, task_instruction)
        _cleanup_old_temp_files(out_file)
        temp_out_file = _get_temp_out_file(out_file)
        
        total, correct = 0, 0
        res_by_category = {}
        result = []
        
        for idx, dat in enumerate(tqdm(dataset_chunk, desc=f"Evaluating Vstar, decoding by steps={steps}")):
            img_path = dat['image']
            if isinstance(img_path, str):
                if os.path.isabs(img_path) and os.path.exists(img_path):
                    pass
                elif image_dir is not None and os.path.exists(os.path.join(image_dir, img_path)):
                    img_path = os.path.join(image_dir, img_path)
                else:
                    try:
                        vstar_cache_dir = os.path.join(DATASETS_DIR, "vstar_bench")
                        import time
                        max_retries = 3
                        download_success = False
                        for attempt in range(max_retries):
                            try:
                                img_path = hf_hub_download(
                                    repo_id="craigwu/vstar_bench", filename=img_path,
                                    cache_dir=vstar_cache_dir, repo_type="dataset", local_files_only=False)
                                if os.path.exists(img_path):
                                    download_success = True
                                    break
                            except Exception as download_error:
                                if attempt < max_retries - 1:
                                    wait_time = (attempt + 1) * 2
                                    print(f"  Retry {attempt+1}/{max_retries} for {dat['image']} (waiting {wait_time}s)...")
                                    time.sleep(wait_time)
                                else:
                                    raise download_error
                        if not download_success:
                            raise Exception("Download failed after retries")
                    except Exception as e:
                        print(f"Error: Could not download image {dat['image']}: {e}")
                        print(f"  Skipping this sample...")
                        continue

            text = dat['text'] + task_instruction
            sample_idx = dat.get('question_id', idx)
            outputs = run_inference(model, processor, img_path, text, steps, decoding_strategy,
                                   sample_idx=sample_idx, activation_map_save_dir=activation_map_dir,
                                   benchmark_name='vstar')

            res = {
                'id': dat['question_id'],
                'prediction': outputs,
                'label': dat['label'],
                'category': dat['category']
            }
            result.append(res)
            
            if dat['category'] not in res_by_category:
                res_by_category[dat['category']] = {"total": 0, "correct": 0}
            res_by_category[dat['category']]["total"] += 1
            
            if accuracy_reward(outputs[0], dat['label']):
                correct += 1
                res_by_category[dat['category']]["correct"] += 1
            total += 1
        
        if EVAL_TOTAL_PROCESSES == 1:
            result = calculate_and_add_accuracy_summary(result, temp_out_file)
        json.dump(result, open(temp_out_file, 'w+'), indent=2)
        step2results_category[steps] = res_by_category
        step2results_overall[steps] = {"total": total, "correct": correct}
        
        _merge_multiprocess_results(out_file, EVAL_TOTAL_PROCESSES, EVAL_PROCESS_ID, "VSTAR")
        print(f"Steps: {steps} - Accuracy: {correct}/{total} = {correct/total*100:.2f}")

    print("Overall accuracy by steps:")
    print(",".join([f"{items['correct']/items['total']*100:.2f}" for items in step2results_overall.values()]))
    for category in ["direct_attributes", "relative_position"]:
        print(f"Category: {category}")
        res = []
        for steps in step2results_category:
            res_by_category = step2results_category[steps]
            if category in res_by_category:
                cat_total = res_by_category[category]["total"]
                cat_correct = res_by_category[category]["correct"]
                res.append(cat_correct / cat_total)
        print(",".join([f"{items*100:.2f}" for items in res]))
    return

def evaluate_mmvp(
        model, 
        processor, 
        dataset,
        image_dir,
        out_dir,
        ds_name,
        decoding_strategy="steps",
        checkpoint_num=None,
        ):
    print(f"Evaluating MMVP with decoding strategy: {decoding_strategy}")
    os.makedirs(out_dir, exist_ok=True)
    task_instruction = get_task_instruction("mmvp")

    save_activation_maps = os.environ.get('LVR_SAVE_ACTIVATION_MAPS', '0') == '1'
    activation_map_dir = None
    if save_activation_maps:
        activation_map_dir = os.path.join(out_dir, "activation_maps")
        os.makedirs(activation_map_dir, exist_ok=True)

    dataset_chunk = _get_dataset_chunk(dataset, EVAL_TOTAL_PROCESSES, EVAL_PROCESS_ID)

    for steps in STEP_LIST:
        out_file = _get_out_file(out_dir, checkpoint_num, steps, decoding_strategy, task_instruction)
        _cleanup_old_temp_files(out_file)
        temp_out_file = _get_temp_out_file(out_file)
        
        total, correct = 0, 0
        result = []
        
        for idx, dat in enumerate(tqdm(dataset_chunk, desc=f"Evaluating MMVP, decoding by steps={steps}")):
            text = dat['query'].replace('(a)', 'A.').replace('(b)', 'B.') + task_instruction
            sample_idx = dat.get('question_id', idx)
            outputs = run_inference(model, processor, dat['image'], text, steps, decoding_strategy,
                                   sample_idx=sample_idx, activation_map_save_dir=activation_map_dir,
                                   benchmark_name='mmvp')
            
            label = dat['label']
            if label in ['(a)', '(b)']:
                label = label.strip().upper()[1]
            
            res = {
                'id': dat['question_id'],
                'prediction': outputs,
                'label': label
            }
            result.append(res)
            if accuracy_reward(outputs[0], label):
                correct += 1
            total += 1
        
        if EVAL_TOTAL_PROCESSES == 1:
            result = calculate_and_add_accuracy_summary(result, temp_out_file)
        json.dump(result, open(temp_out_file, 'w+'), indent=2)
        
        _merge_multiprocess_results(out_file, EVAL_TOTAL_PROCESSES, EVAL_PROCESS_ID, "MMVP")
        print(f"Steps: {steps} - Accuracy: {correct}/{total} = {correct/total*100:.2f}")
    return {"dummy_metric": 0}

def evaluate_blink(
        model, 
        processor, 
        dataset,
        image_dir,
        out_dir,
        ds_name,
        decoding_strategy="steps",
        checkpoint_num=None,
        ):
    print(f"Evaluating BLINK with decoding strategy: {decoding_strategy}")
    os.makedirs(out_dir, exist_ok=True)
    task_instruction = get_task_instruction("blink")
    step2results_category = {}
    step2results_overall = {}

    save_activation_maps = os.environ.get('LVR_SAVE_ACTIVATION_MAPS', '0') == '1'
    activation_map_dir = None
    if save_activation_maps:
        activation_map_dir = os.path.join(out_dir, "activation_maps")
        os.makedirs(activation_map_dir, exist_ok=True)

    dataset_chunk = _get_dataset_chunk(dataset, EVAL_TOTAL_PROCESSES, EVAL_PROCESS_ID)

    for steps in STEP_LIST:
        step2results_category[steps] = {}
        out_file = _get_out_file(out_dir, checkpoint_num, steps, decoding_strategy, task_instruction)
        _cleanup_old_temp_files(out_file)
        temp_out_file = _get_temp_out_file(out_file)
        
        total, correct = 0, 0
        res_by_category = {}
        result = []
        
        for idx, dat in enumerate(tqdm(dataset_chunk, desc=f"Evaluating BLINK, decoding by steps={steps}")):
            img_path = dat['image']
            text = dat['query'] + task_instruction
            sample_idx = dat.get('question_id', idx)
            outputs = run_inference(model, processor, img_path, text, steps, decoding_strategy,
                                   sample_idx=sample_idx, activation_map_save_dir=activation_map_dir,
                                   benchmark_name='blink')

            res = {
                'id': dat['question_id'],
                'prediction': outputs,
                'label': dat['label'],
                'category': dat['category']
            }
            result.append(res)
            
            if dat['category'] not in res_by_category:
                res_by_category[dat['category']] = {"total": 0, "correct": 0}
            res_by_category[dat['category']]["total"] += 1
            
            if accuracy_reward(outputs[0], dat['label']):
                correct += 1
                res_by_category[dat['category']]["correct"] += 1
            total += 1
        
        if EVAL_TOTAL_PROCESSES == 1:
            result = calculate_and_add_accuracy_summary(result, temp_out_file)
        json.dump(result, open(temp_out_file, 'w+'), indent=2)
        step2results_category[steps] = res_by_category
        step2results_overall[steps] = {"total": total, "correct": correct}
        
        _merge_multiprocess_results(out_file, EVAL_TOTAL_PROCESSES, EVAL_PROCESS_ID, "BLINK")
        print(f"Steps: {steps} - Accuracy: {correct}/{total} = {correct/total*100:.2f}")

    print("Overall accuracy by steps:")
    print(",".join([f"{items['correct']/items['total']*100:.2f}" for items in step2results_overall.values()]))
    for category in ['Counting', 'IQ_Test', 'Jigsaw', 'Multi-view_Reasoning', 'Object_Localization',
                     'Relative_Depth', 'Relative_Reflectance', 'Semantic_Correspondence',
                     'Spatial_Relation', 'Visual_Correspondence', 'Visual_Similarity']:
        res = []
        for steps in step2results_category:
            res_by_category = step2results_category[steps]
            if category in res_by_category:
                cat_total = res_by_category[category]["total"]
                cat_correct = res_by_category[category]["correct"]
                res.append(cat_correct / cat_total)
        print(category + ',' + ",".join([f"{items*100:.2f}" for items in res]))
    return

"""Data Loaders"""
# ==== Example dataset loader stubs ====
def load_vstar_dataset(gen_w_head,run_name,decoding_strategy,checkpoint_num=None):
    ds_name = "vstar"
    # Load dataset from HuggingFace and save to specified directory
    vstar_cache_dir = os.path.join(DATASETS_DIR, "vstar_bench")
    os.makedirs(vstar_cache_dir, exist_ok=True)
    ds = load_dataset("craigwu/vstar_bench", cache_dir=vstar_cache_dir)
    # Try to use images from dataset if available, otherwise use image_dir
    image_dir = None  # Images should come from HuggingFace dataset

    out_dir = os.path.join(RESULTS_DIR, "vstar_bench", f"decoding_by_{decoding_strategy}", run_name)
    # Don't add checkpoint_num as subdirectory - it will be included in filename instead

    return ds['test'],image_dir,out_dir,ds_name

def load_mmvp_dataset(gen_w_head,run_name,decoding_strategy,checkpoint_num=None):
    ds_name = "mmvp"
    mmvp_cache_dir = os.path.join(DATASETS_DIR, "MMVP")
    os.makedirs(mmvp_cache_dir, exist_ok=True)
    
    # Load CSV file
    csv_file = os.path.join(mmvp_cache_dir, "Questions.csv")
    if not os.path.exists(csv_file):
        csv_file = hf_hub_download(
            repo_id="MMVP/MMVP",
            filename="Questions.csv",
            cache_dir=mmvp_cache_dir,
            repo_type="dataset"
        )
    
    # Find image files: MMVP on HuggingFace uses subfolder "MMVP Images" (with space)
    search_paths = [
        os.path.join(mmvp_cache_dir, "**", "*.jpg"),
        os.path.join(mmvp_cache_dir, "MMVP Images", "*.jpg"),
        os.path.join(mmvp_cache_dir, "MMVP Images", "**", "*.jpg"),
        os.path.join(os.path.expanduser("~/.cache/huggingface/hub"), "datasets--MMVP--MMVP", "**", "*.jpg"),
    ]
    all_jpg_files = []
    for pattern in search_paths:
        all_jpg_files.extend(glob.glob(pattern, recursive=True))
        if all_jpg_files:
            break
    
    # Map filename number to file path (e.g., "1.jpg" -> path)
    filename_to_path = {}
    for jpg_file in all_jpg_files:
        match = re.match(r'^(\d+)\.jpg$', os.path.basename(jpg_file), re.IGNORECASE)
        if match:
            filename_to_path[int(match.group(1))] = jpg_file
    
    data = []
    if filename_to_path:
        # Load data from CSV and map to image paths (file path)
        with open(csv_file, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                idx = int(row["Index"])
                image_path = filename_to_path.get(idx)
                if not image_path or not os.path.exists(image_path):
                    continue
                data.append({
                    "question_id": idx,
                    'image': image_path,
                    "query": row["Question"] + '\nOptions:\n' + row["Options"],
                    "label": row["Correct Answer"]
                })
        image_dir = os.path.dirname(list(filename_to_path.values())[0])
    else:
        # No .jpg on disk (e.g. only arrow cache): load via HuggingFace dataset to get PIL images
        try:
            ds = load_dataset("MMVP/MMVP", cache_dir=mmvp_cache_dir, trust_remote_code=True)
            train_ds = ds["train"]
        except Exception as e:
            raise RuntimeError(
                f"MMVP: no images found under {mmvp_cache_dir} (including 'MMVP Images/') and load_dataset failed: {e}. "
                "Either place MMVP images in DATASETS_DIR/MMVP/MMVP Images/1.jpg ... 300.jpg, or ensure MMVP/MMVP is downloadable."
            ) from e
        with open(csv_file, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                idx = int(row["Index"])
                if idx < 1 or idx > len(train_ds):
                    continue
                # Dataset rows are 0-based; CSV Index is 1-based
                pil_image = train_ds[idx - 1]["image"]
                data.append({
                    "question_id": idx,
                    "image": pil_image,
                    "query": row["Question"] + '\nOptions:\n' + row["Options"],
                    "label": row["Correct Answer"]
                })
        image_dir = None
        print(f"MMVP: loaded {len(data)} samples from HuggingFace dataset (PIL images, no .jpg on disk)")
    
    if not data:
        raise RuntimeError(
            f"MMVP: loaded 0 samples. CSV has rows but no images found. "
            f"Ensure images exist under {mmvp_cache_dir}/MMVP Images/ as 1.jpg ... 300.jpg, or use load_dataset('MMVP/MMVP')."
        )
    
    # Set output directory
    out_dir = os.path.join(RESULTS_DIR, "MMVP", f"decoding_by_{decoding_strategy}", run_name)
    return data, image_dir, out_dir, ds_name

from datasets import get_dataset_config_names
def load_blink_dataset(gen_w_head,run_name,decoding_strategy,checkpoint_num=None):
    ds_name = "blink"
    # Load BLINK dataset from local cache or HuggingFace
    blink_cache_dir = os.path.join(DATASETS_DIR, "BLINK")
    os.makedirs(blink_cache_dir, exist_ok=True)
    configs = [ 'Counting','IQ_Test', 'Jigsaw', 'Relative_Reflectance', 'Spatial_Relation']
    all_datasets = {}
    
    dataset_hash = "a3666eb249237ba3d5eca8db21176cc47967e040"
    
    for config in configs:
        # Check if dataset exists in local cache
        val_path = os.path.join(blink_cache_dir, f"BLINK-Benchmark___blink", config, "0.0.0", dataset_hash)
        val_file = os.path.join(val_path, "blink-val.arrow")
        
        if os.path.exists(val_file):
            # Dataset exists locally, use load_dataset with DownloadConfig to avoid network requests
            print(f"Loading {config} from local cache: {val_file}")
            try:
                # Use DownloadConfig with local_files_only=True to force using only cached data, no network requests
                download_config = DownloadConfig(local_files_only=True)
                all_datasets[config] = load_dataset(
                    "BLINK-Benchmark/BLINK", 
                    config, 
                    cache_dir=blink_cache_dir,
                    download_config=download_config  # Force offline mode - only use cached data
                )
                print(f"Successfully loaded {config} from local cache")
            except Exception as e:
                print(f"Error: Failed to load {config} from local cache: {e}")
                print(f"Please ensure the dataset is properly cached at: {val_path}")
                raise
        else:
            # Dataset not in cache, load from HuggingFace
            print(f"Warning: {config} not found in local cache, loading from HuggingFace...")
            all_datasets[config] = load_dataset("BLINK-Benchmark/BLINK", config, cache_dir=blink_cache_dir)
    # ds = load_dataset("BLINK-Benchmark/BLINK")
    image_dir = None


    processed_data = []
    for config in all_datasets:
        ds = all_datasets[config]['val']
        for dat in ds:
            idx = dat["idx"]
            choices = dat["choices"]
            letters = string.ascii_uppercase 
            paired = list(zip(letters, choices))
            option_string = ""
            for letter, choice in paired:
                option_string += f"{letter}. {choice}\n"
            if len(dat['answer']) >1:
                ans = dat['answer'][1].upper()
            else:
                ans = dat['answer'][0].upper()
            images = []
            for k in ['image_1','image_2','image_3','image_4']:
                if k in dat and dat[k] is not None:
                    images.append(dat[k])
            question = dat['question'] + "\nOptions:\n" + option_string
            buffer = {
                "question_id": idx,
                "image": images,
                "query": question,
                "label": ans,
                "category": config
            }
            processed_data.append(buffer)

    out_dir = os.path.join(RESULTS_DIR, "blink", f"decoding_by_{decoding_strategy}", run_name)
    # Don't add checkpoint_num as subdirectory - it will be included in filename instead

    return processed_data,image_dir,out_dir,ds_name


def load_mathvision_dataset(gen_w_head, run_name, decoding_strategy, checkpoint_num=None):
    """Load MathVision (Wang et al, 2024) testmini from local Datasets."""
    ds_name = "mathvision"
    base_dir = os.path.join(DATASETS_DIR, "MathVision")
    parquet_path = os.path.join(base_dir, "data", "testmini-00000-of-00001-f8ff70fcb2f29b1d.parquet")
    if not os.path.exists(parquet_path):
        parquet_path = os.path.join(base_dir, "data", "test-00000-of-00001-3532b8d3f1b4047a.parquet")
    import pandas as pd
    df = pd.read_parquet(parquet_path)
    image_dir = os.path.join(base_dir, "images")
    data = []
    for _, row in df.iterrows():
        img_path = row.get('image', '')
        if isinstance(img_path, str) and not os.path.isabs(img_path):
            img_path = os.path.join(base_dir, img_path)
        if not os.path.exists(img_path):
            decoded = row.get('decoded_image')
            if isinstance(decoded, dict) and 'bytes' in decoded:
                import tempfile
                tmp = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
                tmp.write(decoded['bytes'])
                tmp.close()
                img_path = tmp.name
            else:
                continue
        question = str(row.get('question', '')).replace('<image1>', '')
        options = row.get('options')
        if options is not None and len(options) > 0:
            opt_str = "\n".join([f"{chr(65+i)}. {o}" for i, o in enumerate(options)])
            question = question + "\nOptions:\n" + opt_str
        data.append({
            'question_id': row.get('id', len(data)),
            'image': img_path,
            'query': question,
            'label': str(row.get('answer', '')),
            'category': str(row.get('subject', 'Unknown')),
            'question_type': 'multi_choice' if options is not None and len(options) > 0 else 'free_form',
        })
    out_dir = os.path.join(RESULTS_DIR, "MathVision", f"decoding_by_{decoding_strategy}", run_name)
    return data, image_dir, out_dir, ds_name


def load_mathvista_dataset(gen_w_head, run_name, decoding_strategy, checkpoint_num=None):
    """Load MathVista (Lu et al, 2024) testmini from local Datasets."""
    ds_name = "mathvista"
    base_dir = os.path.join(DATASETS_DIR, "MathVista")
    parquet_path = os.path.join(base_dir, "data", "testmini-00000-of-00001-725687bf7a18d64b.parquet")
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"MathVista testmini not found at {parquet_path}")
    import pandas as pd
    df = pd.read_parquet(parquet_path)
    image_dir = os.path.join(base_dir, "images")
    data = []
    for _, row in df.iterrows():
        img_path = row.get('image', '')
        if isinstance(img_path, str):
            if not os.path.isabs(img_path):
                img_path = os.path.join(base_dir, img_path)
            if not os.path.exists(img_path):
                decoded = row.get('decoded_image')
                if isinstance(decoded, dict) and 'bytes' in decoded:
                    import tempfile
                    tmp = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
                    tmp.write(decoded['bytes'])
                    tmp.close()
                    img_path = tmp.name
                else:
                    continue
        query = row.get('query', row.get('question', ''))
        choices = row.get('choices')
        if choices is not None and len(choices) > 0:
            opt_str = "\n".join([f"{chr(65+i)}. {c}" for i, c in enumerate(choices)])
            query = query + "\nOptions:\n" + opt_str
        data.append({
            'question_id': row.get('pid', len(data)),
            'image': img_path,
            'query': query,
            'label': str(row.get('answer', '')),
            'category': str(row.get('metadata', {}).get('task', 'Unknown')) if isinstance(row.get('metadata'), dict) else 'Unknown',
            'question_type': str(row.get('question_type', 'free_form')),
            'precision': row.get('precision'),
        })
    out_dir = os.path.join(RESULTS_DIR, "MathVista", f"decoding_by_{decoding_strategy}", run_name)
    return data, image_dir, out_dir, ds_name


def load_visulogic_dataset(gen_w_head, run_name, decoding_strategy, checkpoint_num=None):
    """Load VisuLogic (Xu et al, 2025) from local Datasets."""
    ds_name = "visulogic"
    base_dir = os.path.join(DATASETS_DIR, "VisuLogic")
    jsonl_path = os.path.join(base_dir, "data.jsonl")
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(f"VisuLogic data.jsonl not found at {jsonl_path}")
    data = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            img_path = item.get('image_path', '')
            if not os.path.isabs(img_path):
                img_path = os.path.join(base_dir, img_path)
            if not os.path.exists(img_path):
                continue
            question = item.get('question', '')
            data.append({
                'question_id': item.get('id', len(data)),
                'image': img_path,
                'query': question,
                'label': str(item.get('label', '')).upper(),
                'category': item.get('tag', 'Unknown'),
            })
    out_dir = os.path.join(RESULTS_DIR, "VisuLogic", f"decoding_by_{decoding_strategy}", run_name)
    return data, None, out_dir, ds_name


def load_emma_dataset(gen_w_head, run_name, decoding_strategy, checkpoint_num=None):
    """Load EMMA from local Datasets (Math, Physics, Chemistry, Coding)."""
    ds_name = "emma"
    base_dir = os.path.join(DATASETS_DIR, "EMMA")
    import pandas as pd
    data = []
    for subj in ['Math', 'Physics', 'Chemistry', 'Coding']:
        subj_dir = os.path.join(base_dir, subj)
        parquet_files = glob.glob(os.path.join(subj_dir, "*.parquet"))
        for pf in parquet_files:
            df = pd.read_parquet(pf)
            for _, row in df.iterrows():
                img_path = None
                for i in range(1, 6):
                    img = row.get(f'image_{i}')
                    if img is not None and isinstance(img, dict) and 'bytes' in img:
                        import tempfile
                        tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
                        tmp.write(img['bytes'])
                        tmp.close()
                        img_path = tmp.name
                        break
                if img_path is None:
                    continue
                question = str(row.get('question', '')).replace('<image_1>', '').replace('<image_2>', '').replace('<image_3>', '').replace('<image_4>', '').replace('<image_5>', '')
                options = row.get('options')
                if options is not None and len(options) > 0:
                    opt_list = list(options)
                    if all(isinstance(o, str) and len(o) == 1 for o in opt_list):
                        opt_str = ", ".join(opt_list)
                    else:
                        opt_str = "\n".join([f"{chr(65+i)}. {o}" for i, o in enumerate(opt_list)])
                    question = question + "\nOptions: " + opt_str
                data.append({
                    'question_id': row.get('pid', f"{subj}_{len(data)}"),
                    'image': img_path,
                    'query': question,
                    'label': str(row.get('answer', '')).upper(),
                    'category': subj,
                })
    out_dir = os.path.join(RESULTS_DIR, "EMMA", f"decoding_by_{decoding_strategy}", run_name)
    return data, None, out_dir, ds_name


def load_mmmu_dataset(gen_w_head, run_name, decoding_strategy, checkpoint_num=None):
    """Load MMMU from local /comp_robot/zhoujiazhou/Datasets/MMMU."""
    ds_name = "mmmu"
    base_dir = os.path.join(DATASETS_DIR, "MMMU")
    import pandas as pd
    from io import BytesIO
    data = []
    subject_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and not d.startswith('.')]
    for subj in subject_dirs:
        subj_dir = os.path.join(base_dir, subj)
        test_files = glob.glob(os.path.join(subj_dir, "test-*.parquet"))
        for pf in test_files:
            df = pd.read_parquet(pf)
            for _, row in df.iterrows():
                images = []
                for i in range(1, 8):
                    img_col = row.get(f'image_{i}')
                    if img_col is not None and isinstance(img_col, dict) and 'bytes' in img_col:
                        images.append(Image.open(BytesIO(img_col['bytes'])).convert('RGB'))
                if not images:
                    continue
                options = row.get('options')
                if options is not None and len(options) > 0:
                    opt_str = "\n".join([f"{chr(65+i)}. {o}" for i, o in enumerate(list(options))])
                    question = str(row.get('question', '')) + "\nOptions:\n" + opt_str
                else:
                    question = str(row.get('question', ''))
                answer = str(row.get('answer', '')).strip().upper()
                if len(answer) == 1 and answer in 'ABCD':
                    pass
                elif answer.isdigit() and 0 <= int(answer) < 4:
                    answer = chr(65 + int(answer))
                data.append({
                    'question_id': row.get('id', f"{subj}_{len(data)}"),
                    'image': images[0] if len(images) == 1 else images,
                    'query': question,
                    'label': answer,
                    'category': subj,
                })
    out_dir = os.path.join(RESULTS_DIR, "MMMU", f"decoding_by_{decoding_strategy}", run_name)
    return data, None, out_dir, ds_name


def load_mmstar_dataset(gen_w_head, run_name, decoding_strategy, checkpoint_num=None):
    """Load MMStar from local /comp_robot/zhoujiazhou/Datasets/MMStar."""
    ds_name = "mmstar"
    import pandas as pd
    from io import BytesIO
    parquet_path = os.path.join(DATASETS_DIR, "MMStar", "mmstar.parquet")
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"MMStar parquet not found: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    data = []
    for _, row in df.iterrows():
        img_raw = row.get('image')
        if img_raw is None:
            continue
        if isinstance(img_raw, dict) and 'bytes' in img_raw:
            img = Image.open(BytesIO(img_raw['bytes'])).convert('RGB')
        else:
            img = Image.open(BytesIO(img_raw)).convert('RGB')
        question = str(row.get('question', ''))
        answer = str(row.get('answer', '')).strip().upper()
        if len(answer) > 1:
            answer = answer[0]
        data.append({
            'question_id': row.get('index', len(data)),
            'image': img,
            'query': question,
            'label': answer,
            'category': str(row.get('category', 'Unknown')),
        })
    out_dir = os.path.join(RESULTS_DIR, "MMStar", f"decoding_by_{decoding_strategy}", run_name)
    return data, None, out_dir, ds_name


def load_pope_dataset(gen_w_head, run_name, decoding_strategy, checkpoint_num=None):
    """Load POPE from local /comp_robot/zhoujiazhou/Datasets/POPE."""
    ds_name = "pope"
    import pandas as pd
    from io import BytesIO
    data_dir = os.path.join(DATASETS_DIR, "POPE", "data")
    parquet_files = glob.glob(os.path.join(data_dir, "*.parquet"))
    data = []
    for pf in parquet_files:
        df = pd.read_parquet(pf)
        for _, row in df.iterrows():
            img_col = row.get('image')
            if img_col is None:
                continue
            if isinstance(img_col, dict) and 'bytes' in img_col:
                img = Image.open(BytesIO(img_col['bytes'])).convert('RGB')
            else:
                img = Image.open(BytesIO(img_col)).convert('RGB')
            data.append({
                'question_id': row.get('question_id', len(data)),
                'image': img,
                'query': str(row.get('question', '')),
                'label': str(row.get('answer', '')).strip().lower(),
                'category': str(row.get('category', 'Unknown')),
            })
    out_dir = os.path.join(RESULTS_DIR, "POPE", f"decoding_by_{decoding_strategy}", run_name)
    return data, None, out_dir, ds_name


def load_mme_dataset(gen_w_head, run_name, decoding_strategy, checkpoint_num=None):
    """Load MME from local /comp_robot/zhoujiazhou/Datasets/MME."""
    ds_name = "mme"
    import pandas as pd
    from io import BytesIO
    data_dir = os.path.join(DATASETS_DIR, "MME", "data")
    parquet_files = glob.glob(os.path.join(data_dir, "*.parquet"))
    data = []
    for pf in parquet_files:
        df = pd.read_parquet(pf)
        for _, row in df.iterrows():
            img_col = row.get('image')
            if img_col is None:
                continue
            if isinstance(img_col, dict) and 'bytes' in img_col:
                img = Image.open(BytesIO(img_col['bytes'])).convert('RGB')
            else:
                img = Image.open(BytesIO(img_col)).convert('RGB')
            answer = str(row.get('answer', '')).strip().lower()
            data.append({
                'question_id': row.get('question_id', len(data)),
                'image': img,
                'query': str(row.get('question', '')),
                'label': answer,
                'category': str(row.get('category', 'Unknown')),
            })
    out_dir = os.path.join(RESULTS_DIR, "MME", f"decoding_by_{decoding_strategy}", run_name)
    return data, None, out_dir, ds_name


def _load_hrbench_dataset(parquet_name, ds_name, run_name, decoding_strategy, checkpoint_num):
    """Load HRBench 4K or 8K from parquet. Images are base64-encoded in parquet."""
    import pandas as pd
    import base64
    from io import BytesIO
    base_dir = os.path.join(DATASETS_DIR, "HRBench")
    parquet_path = os.path.join(base_dir, parquet_name)
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"HRBench parquet not found: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    data = []
    for _, row in df.iterrows():
        img_raw = row.get('image')
        if img_raw is None:
            continue
        try:
            if isinstance(img_raw, str):
                img_bytes = base64.b64decode(img_raw)
                img = Image.open(BytesIO(img_bytes)).convert('RGB')
            elif isinstance(img_raw, bytes):
                img = Image.open(BytesIO(img_raw)).convert('RGB')
            else:
                continue
        except Exception:
            continue
        question = str(row.get('question', ''))
        options = []
        for opt in ['A', 'B', 'C', 'D']:
            val = row.get(opt)
            if val is not None and str(val).strip():
                options.append(f"{opt}. {val}")
        if options:
            question = question + "\nOptions:\n" + "\n".join(options)
        answer = str(row.get('answer', '')).strip().upper()
        if len(answer) > 1:
            answer = answer[0]
        data.append({
            'question_id': row.get('index', len(data)),
            'image': img,
            'query': question,
            'label': answer,
            'category': str(row.get('category', 'Unknown')),
        })
    out_dir = os.path.join(RESULTS_DIR, ds_name, f"decoding_by_{decoding_strategy}", run_name)
    return data, None, out_dir, ds_name.lower().replace('-', '_')


def load_hrbench4k_dataset(gen_w_head, run_name, decoding_strategy, checkpoint_num=None):
    """Load HRBench 4K from local /comp_robot/zhoujiazhou/Datasets/HRBench."""
    return _load_hrbench_dataset("hr_bench_4k.parquet", "HRBench4K", run_name, decoding_strategy, checkpoint_num)


def load_hrbench8k_dataset(gen_w_head, run_name, decoding_strategy, checkpoint_num=None):
    """Load HRBench 8K from local /comp_robot/zhoujiazhou/Datasets/HRBench."""
    return _load_hrbench_dataset("hr_bench_8k.parquet", "HRBench8K", run_name, decoding_strategy, checkpoint_num)


def load_mme_realworld_lite_dataset(gen_w_head, run_name, decoding_strategy, checkpoint_num=None):
    """Load MME-RealWorld-Lite from local /comp_robot/zhoujiazhou/Datasets/MME-RealWorld-Lite."""
    ds_name = "mme_realworld_lite"
    import zipfile
    base_dir = os.path.join(DATASETS_DIR, "MME-RealWorld-Lite")
    zip_path = os.path.join(base_dir, "data.zip")
    data_dir = os.path.join(base_dir, "data")
    imgs_dir = os.path.join(data_dir, "imgs")
    json_path = os.path.join(data_dir, "MME-RealWorld-Lite.json")
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"MME-RealWorld-Lite data.zip not found: {zip_path}")
    if not os.path.exists(data_dir):
        print("Extracting MME-RealWorld-Lite data.zip (first time only)...")
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(base_dir)
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"MME-RealWorld-Lite JSON not found: {json_path}")
    with open(json_path, 'r') as f:
        items = json.load(f)
    data = []
    for item in items:
        img_name = item.get('Image', '')
        if not img_name:
            continue
        img_path = os.path.join(imgs_dir, img_name)
        if not os.path.exists(img_path):
            continue
        text = str(item.get('Text', ''))
        choices = item.get('Answer choices', [])
        if choices:
            opt_str = "\n".join(choices)
            text = text + "\nOptions:\n" + opt_str
        gt = str(item.get('Ground truth', '')).strip().upper()
        if len(gt) > 1:
            gt = gt[0]
        category = str(item.get('Subtask', item.get('Category', 'Unknown')))
        data.append({
            'question_id': item.get('Question_id', len(data)),
            'image': img_path,
            'query': text,
            'label': gt,
            'category': category,
        })
    out_dir = os.path.join(RESULTS_DIR, "MME-RealWorld-Lite", f"decoding_by_{decoding_strategy}", run_name)
    return data, imgs_dir, out_dir, ds_name


def load_hallbench_dataset(gen_w_head, run_name, decoding_strategy, checkpoint_num=None):
    """Load HallBench (HallusionBench) from local /comp_robot/zhoujiazhou/Datasets/HallBench.
    Loads both image (951) and non_image (178) splits for full 1129 questions.
    Preserves set_id, figure_id, subcategory, visual_input for Question Pair Acc."""
    ds_name = "hallbench"
    from io import BytesIO
    import pandas as pd
    base_dir = os.path.join(DATASETS_DIR, "HallBench")
    data_dir = os.path.join(base_dir, "data")
    data = []
    global_idx = 0

    for parquet_name in ("image-00000-of-00001.parquet", "non_image-00000-of-00001.parquet"):
        parquet_path = os.path.join(data_dir, parquet_name)
        if not os.path.exists(parquet_path):
            continue
        df = pd.read_parquet(parquet_path)
        for _, row in df.iterrows():
            img_col = row.get('image')
            if img_col is not None and isinstance(img_col, dict) and 'bytes' in img_col:
                img = Image.open(BytesIO(img_col['bytes'])).convert('RGB')
            else:
                # non_image: no image, use placeholder (1x1 white) for VLM
                img = Image.new('RGB', (1, 1), color=(255, 255, 255))
            gt = str(row.get('gt_answer', '')).strip()
            data.append({
                'question_id': global_idx,  # unique index for merge/sort
                'pair_question_id': str(row.get('question_id', '')),  # original qid for pair grouping
                'image': img,
                'query': str(row.get('question', '')),
                'label': gt,
                'category': str(row.get('category', 'Unknown')),
                'subcategory': str(row.get('subcategory', '')),
                'set_id': str(row.get('set_id', '')),
                'figure_id': str(row.get('figure_id', '')),
                'visual_input': str(row.get('visual_input', '')),
            })
            global_idx += 1

    out_dir = os.path.join(RESULTS_DIR, "HallBench", f"decoding_by_{decoding_strategy}", run_name)
    return data, None, out_dir, ds_name


def load_mmhal_dataset(gen_w_head, run_name, decoding_strategy, checkpoint_num=None):
    """Load MMHal-Bench from local /comp_robot/zhoujiazhou/Datasets/MMHal-Bench."""
    ds_name = "mmhal"
    base_dir = os.path.join(DATASETS_DIR, "MMHal-Bench")
    json_path = os.path.join(base_dir, "response_template.json")
    imgs_dir = os.path.join(base_dir, "images")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"MMHal-Bench response_template.json not found: {json_path}")
    if not os.path.exists(imgs_dir):
        raise FileNotFoundError(f"MMHal-Bench images dir not found: {imgs_dir}. Extract test_data.zip if needed.")
    with open(json_path, 'r') as f:
        items = json.load(f)
    data = []
    for idx, item in enumerate(items):
        url = item.get('image_src', '')
        if not url:
            continue
        fname = url.split('/')[-1]
        img_path = os.path.join(imgs_dir, fname)
        if not os.path.exists(img_path):
            continue
        data.append({
            'question_id': item.get('image_id', idx),
            'image': img_path,
            'query': str(item.get('question', '')),
            'label': str(item.get('gt_answer', '')).strip(),
            'category': str(item.get('question_type', 'Unknown')),
            'image_content': item.get('image_content', []),
        })
    out_dir = os.path.join(RESULTS_DIR, "MMHal", f"decoding_by_{decoding_strategy}", run_name)
    return data, imgs_dir, out_dir, ds_name


def load_crpe_dataset(gen_w_head, run_name, decoding_strategy, checkpoint_num=None):
    """Load CRPE from local /comp_robot/zhoujiazhou/Datasets/CRPE.
    Image paths: coco/val2017/*.jpg (requires COCO) or abnormal_images/*.jpg.
    COCO val2017 is resolved in order: COCO_VAL2017_DIR env, DATASETS_DIR/coco/val2017,
    or DATASETS_DIR/Visual_cot/images/coco/val2017.
    """
    ds_name = "crpe"
    base_dir = os.path.join(DATASETS_DIR, "CRPE")
    _coco_candidates = [
        os.environ.get('COCO_VAL2017_DIR'),
        os.path.join(DATASETS_DIR, "coco", "val2017"),
        os.path.join(DATASETS_DIR, "Visual_cot", "images", "coco", "val2017"),
    ]
    coco_dir = None
    for p in _coco_candidates:
        if p and os.path.isdir(p):
            coco_dir = p
            break
    if coco_dir is None:
        coco_dir = _coco_candidates[1]  # use default for path construction; images will be skipped
    data = []
    for jsonl_name in ["crpe_exist.jsonl", "crpe_relation.jsonl"]:
        jsonl_path = os.path.join(base_dir, jsonl_name)
        if not os.path.exists(jsonl_path):
            continue
        with open(jsonl_path, 'r') as f:
            for line in f:
                item = json.loads(line)
                img_rel = item.get('image', '')
                if not img_rel:
                    continue
                if img_rel.startswith('coco/'):
                    img_path = os.path.join(coco_dir, img_rel.replace('coco/val2017/', ''))
                else:
                    img_path = os.path.join(base_dir, img_rel)
                if not os.path.exists(img_path):
                    continue
                text = str(item.get('text', ''))
                correct = str(item.get('correct_option', '')).strip().upper()
                if len(correct) > 1:
                    correct = correct[0]
                data.append({
                    'question_id': item.get('question_id', len(data)),
                    'image': img_path,
                    'query': text,
                    'label': correct,
                    'category': str(item.get('category', 'Unknown')),
                })
    if not data:
        raise FileNotFoundError(
            f"CRPE: No valid samples. Ensure crpe_exist.jsonl and crpe_relation.jsonl exist, and images are available. "
            f"For coco/val2017 images, set COCO_VAL2017_DIR or place COCO at {coco_dir}. "
            f"abnormal_images are under {base_dir}/abnormal_images/."
        )
    out_dir = os.path.join(RESULTS_DIR, "CRPE", f"decoding_by_{decoding_strategy}", run_name)
    return data, base_dir, out_dir, ds_name


def _remove_leading_articles(text):
    """Remove leading 'a', 'an', 'the' from referring expression (Qwen refCOCO eval)."""
    if not isinstance(text, str):
        return str(text)
    return re.sub(r'^(a|an|the)\s+', '', text.strip(), flags=re.IGNORECASE)


def _load_refcoco_style_dataset(base_dir, ds_name, run_name, decoding_strategy, checkpoint_num, splits=("val",)):
    """Load refCOCO-style dataset (refCOCO, refCOCO+, refCOCOg) from local parquet."""
    from io import BytesIO
    import pandas as pd
    data = []
    data_dir = os.path.join(base_dir, "data")
    for split in splits:
        pattern = os.path.join(data_dir, f"{split}-*.parquet")
        files = sorted(glob.glob(pattern))
        for pf in files:
            df = pd.read_parquet(pf)
            for _, row in df.iterrows():
                img_col = row.get("image")
                if img_col is None:
                    continue
                if isinstance(img_col, dict) and "bytes" in img_col:
                    img = Image.open(BytesIO(img_col["bytes"])).convert("RGB")
                else:
                    continue
                answer = row.get("answer")
                if answer is None or (hasattr(answer, "__len__") and len(answer) == 0):
                    continue
                ref_expr = answer[0] if hasattr(answer, "__len__") and not isinstance(answer, str) else str(answer)
                ref_expr = _remove_leading_articles(ref_expr)
                bbox = row.get("bbox")
                if bbox is None or (hasattr(bbox, "__len__") and len(bbox) != 4):
                    continue
                bbox = np.array(bbox, dtype=np.float64).tolist() if hasattr(bbox, "tolist") else list(bbox)
                data.append({
                    "question_id": row.get("question_id", len(data)),
                    "image": img,
                    "query": f"Locate {ref_expr} in this image and output the bbox coordinates in JSON format.",
                    "label": bbox,
                    "category": split,
                    "bbox_fmt": "coco",
                    "img_width": img.size[0],
                    "img_height": img.size[1],
                })
    out_dir = os.path.join(RESULTS_DIR, ds_name.replace("_", ""), f"decoding_by_{decoding_strategy}", run_name)
    return data, None, out_dir, ds_name


def load_refcoco_dataset(gen_w_head, run_name, decoding_strategy, checkpoint_num=None):
    ds_name = "refcoco"
    base_dir = os.path.join(DATASETS_DIR, "refCOCO")
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"refCOCO not found at {base_dir}. Please download to /comp_robot/zhoujiazhou/Datasets/")
    return _load_refcoco_style_dataset(base_dir, "refCOCO", run_name, decoding_strategy, checkpoint_num, splits=("val", "test", "testA", "testB"))


def load_refcoco_plus_dataset(gen_w_head, run_name, decoding_strategy, checkpoint_num=None):
    ds_name = "refcoco_plus"
    base_dir = os.path.join(DATASETS_DIR, "refCOCO+")
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"refCOCO+ not found at {base_dir}. Please download to /comp_robot/zhoujiazhou/Datasets/")
    return _load_refcoco_style_dataset(base_dir, "refCOCO+", run_name, decoding_strategy, checkpoint_num, splits=("val", "testA", "testB"))


def load_refcocog_dataset(gen_w_head, run_name, decoding_strategy, checkpoint_num=None):
    ds_name = "refcocog"
    base_dir = os.path.join(DATASETS_DIR, "refCOCOg")
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"refCOCOg not found at {base_dir}. Please download to /comp_robot/zhoujiazhou/Datasets/")
    return _load_refcoco_style_dataset(base_dir, "refCOCOg", run_name, decoding_strategy, checkpoint_num, splits=("val", "test"))


def load_reasonseg_dataset(gen_w_head, run_name, decoding_strategy, checkpoint_num=None):
    """Load ReasonSeg from local /comp_robot/zhoujiazhou/Datasets/ReasonSeg."""
    ds_name = "reasonseg"
    from io import BytesIO
    import pandas as pd
    base_dir = os.path.join(DATASETS_DIR, "ReasonSeg")
    data = []
    for split in ("val", "test"):
        split_dir = os.path.join(base_dir, split)
        parquet_dir = os.path.join(split_dir, "data")
        if not os.path.isdir(parquet_dir):
            continue
        for pf in sorted(glob.glob(os.path.join(parquet_dir, "*.parquet"))):
            df = pd.read_parquet(pf)
            for _, row in df.iterrows():
                img_col = row.get("image")
                if img_col is None:
                    continue
                if isinstance(img_col, dict) and "bytes" in img_col:
                    img = Image.open(BytesIO(img_col["bytes"])).convert("RGB")
                else:
                    continue
                text = _remove_leading_articles(row.get("text", ""))
                if not text:
                    continue
                mask = row.get("mask")
                img_h, img_w = int(row.get("img_height", 0)), int(row.get("img_width", 0))
                bbox = None
                if mask is not None and img_h and img_w:
                    try:
                        if hasattr(mask, "__len__") and len(mask) == img_h:
                            mask_arr = np.stack([np.asarray(m) for m in mask])
                        else:
                            mask_arr = np.asarray(mask)
                        if hasattr(mask_arr, "shape") and mask_arr.ndim >= 2:
                            ys, xs = np.where(mask_arr)
                            if len(ys) > 0:
                                bbox = [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]
                    except Exception:
                        pass
                if bbox is None:
                    continue
                data.append({
                    "question_id": row.get("ann_id", row.get("image_id", len(data))),
                    "image": img,
                    "query": f"Locate {text} in this image and output the bbox coordinates in JSON format.",
                    "label": bbox,
                    "category": split,
                    "bbox_fmt": "xyxy",
                    "img_width": img_w,
                    "img_height": img_h,
                })
    out_dir = os.path.join(RESULTS_DIR, "ReasonSeg", f"decoding_by_{decoding_strategy}", run_name)
    return data, None, out_dir, ds_name


def _evaluate_rec_benchmark(model, processor, dataset, image_dir, out_dir, ds_name, decoding_strategy, checkpoint_num, bench_label, iou_threshold=0.5):
    """Generic REC evaluator: IoU > threshold = correct."""
    print(f"Evaluating {bench_label} (REC, IoU>{iou_threshold}) with decoding strategy: {decoding_strategy}")
    os.makedirs(out_dir, exist_ok=True)
    task_instruction = get_task_instruction(ds_name)
    dataset_chunk = _get_dataset_chunk(dataset, EVAL_TOTAL_PROCESSES, EVAL_PROCESS_ID)
    for steps in STEP_LIST:
        out_file = _get_out_file(out_dir, checkpoint_num, steps, decoding_strategy, task_instruction)
        _cleanup_old_temp_files(out_file)
        temp_out_file = _get_temp_out_file(out_file)
        total, correct = 0, 0
        result = []
        for idx, dat in enumerate(tqdm(dataset_chunk, desc=f"Evaluating {bench_label}, steps={steps}")):
            img = dat["image"]
            text = dat["query"] + task_instruction
            sample_idx = dat.get("question_id", idx)
            run_ret = run_inference(model, processor, img, text, steps, decoding_strategy,
                                   sample_idx=sample_idx, benchmark_name=ds_name)
            model_in_h = model_in_w = None
            if isinstance(run_ret, tuple):
                outputs, image_grid_thw = run_ret
                if image_grid_thw is not None and len(image_grid_thw) > 0:
                    g0 = image_grid_thw[0]
                    model_in_h = (int(g0[1].item()) if hasattr(g0[1], 'item') else int(g0[1])) * 14
                    model_in_w = (int(g0[2].item()) if hasattr(g0[2], 'item') else int(g0[2])) * 14
            else:
                outputs = run_ret
            label_bbox = dat["label"]
            bbox_fmt = dat.get("bbox_fmt", "coco")
            if bbox_fmt == "coco":
                gt_xyxy = _coco_bbox_to_xyxy(label_bbox[:4])
            else:
                gt_xyxy = tuple(float(x) for x in label_bbox[:4])
            pred_bbox = _parse_bbox_from_response(outputs[0])
            img_w = dat.get("img_width")
            img_h = dat.get("img_height")
            if img_w is None or img_h is None:
                img_obj = dat.get("image")
                if hasattr(img_obj, "size") and img_obj.size:
                    img_w, img_h = img_obj.size[0], img_obj.size[1]
            if pred_bbox is not None:
                iou = _compute_iou(pred_bbox, gt_xyxy, img_width=img_w, img_height=img_h,
                                   model_input_width=model_in_w, model_input_height=model_in_h)
                match = iou >= iou_threshold
            else:
                match = False
            result.append({
                "id": dat["question_id"],
                "prediction": outputs,
                "label": label_bbox,
                "category": dat.get("category", "Unknown"),
                "correct": match,
            })
            if match:
                correct += 1
            total += 1
        if EVAL_TOTAL_PROCESSES == 1:
            result = calculate_and_add_accuracy_summary(result, temp_out_file)
        json.dump(result, open(temp_out_file, "w+"), indent=2)
        _merge_multiprocess_results(out_file, EVAL_TOTAL_PROCESSES, EVAL_PROCESS_ID, bench_label)
        print(f"Steps: {steps} - Accuracy (IoU>0.5): {correct}/{total} = {100*correct/total:.2f}%" if total > 0 else f"Steps: {steps} - No samples")


def evaluate_refcoco(model, processor, dataset, image_dir, out_dir, ds_name, decoding_strategy="steps", checkpoint_num=None):
    return _evaluate_rec_benchmark(model, processor, dataset, image_dir, out_dir, "refcoco", decoding_strategy, checkpoint_num, "refCOCO")


def evaluate_refcoco_plus(model, processor, dataset, image_dir, out_dir, ds_name, decoding_strategy="steps", checkpoint_num=None):
    return _evaluate_rec_benchmark(model, processor, dataset, image_dir, out_dir, "refcoco_plus", decoding_strategy, checkpoint_num, "refCOCO+")


def evaluate_refcocog(model, processor, dataset, image_dir, out_dir, ds_name, decoding_strategy="steps", checkpoint_num=None):
    return _evaluate_rec_benchmark(model, processor, dataset, image_dir, out_dir, "refcocog", decoding_strategy, checkpoint_num, "refCOCOg")


def evaluate_reasonseg(model, processor, dataset, image_dir, out_dir, ds_name, decoding_strategy="steps", checkpoint_num=None):
    return _evaluate_rec_benchmark(model, processor, dataset, image_dir, out_dir, "reasonseg", decoding_strategy, checkpoint_num, "ReasonSeg")


def _evaluate_mcq_benchmark(model, processor, dataset, image_dir, out_dir, ds_name, decoding_strategy, checkpoint_num, bench_label):
    """Generic MCQ evaluator (MMMU, MMStar)."""
    print(f"Evaluating {bench_label} with decoding strategy: {decoding_strategy}")
    os.makedirs(out_dir, exist_ok=True)
    task_instruction = get_task_instruction(ds_name)
    dataset_chunk = _get_dataset_chunk(dataset, EVAL_TOTAL_PROCESSES, EVAL_PROCESS_ID)
    for steps in STEP_LIST:
        out_file = _get_out_file(out_dir, checkpoint_num, steps, decoding_strategy, task_instruction)
        _cleanup_old_temp_files(out_file)
        temp_out_file = _get_temp_out_file(out_file)
        total, correct = 0, 0
        result = []
        for idx, dat in enumerate(tqdm(dataset_chunk, desc=f"Evaluating {bench_label}, steps={steps}")):
            img = dat['image']
            text = dat['query'] + task_instruction
            sample_idx = dat.get('question_id', idx)
            outputs = run_inference(model, processor, img, text, steps, decoding_strategy,
                                   sample_idx=sample_idx, benchmark_name=ds_name)
            label = dat['label']
            result.append({'id': dat['question_id'], 'prediction': outputs, 'label': label, 'category': dat.get('category', 'Unknown')})
            if accuracy_reward(outputs[0], label):
                correct += 1
            total += 1
        if EVAL_TOTAL_PROCESSES == 1:
            result = calculate_and_add_accuracy_summary(result, temp_out_file)
        json.dump(result, open(temp_out_file, 'w+'), indent=2)
        _merge_multiprocess_results(out_file, EVAL_TOTAL_PROCESSES, EVAL_PROCESS_ID, bench_label, expected_total_samples=len(dataset))
        print(f"Steps: {steps} - Accuracy: {correct}/{total} = {correct/total*100:.2f}" if total > 0 else f"Steps: {steps} - No samples")


def _evaluate_yesno_benchmark(model, processor, dataset, image_dir, out_dir, ds_name, decoding_strategy, checkpoint_num, bench_label):
    """Generic Yes/No evaluator (POPE, MME)."""
    print(f"Evaluating {bench_label} with decoding strategy: {decoding_strategy}")
    os.makedirs(out_dir, exist_ok=True)
    task_instruction = get_task_instruction(ds_name)
    dataset_chunk = _get_dataset_chunk(dataset, EVAL_TOTAL_PROCESSES, EVAL_PROCESS_ID)
    for steps in STEP_LIST:
        out_file = _get_out_file(out_dir, checkpoint_num, steps, decoding_strategy, task_instruction)
        _cleanup_old_temp_files(out_file)
        temp_out_file = _get_temp_out_file(out_file)
        total, correct = 0, 0
        result = []
        for idx, dat in enumerate(tqdm(dataset_chunk, desc=f"Evaluating {bench_label}, steps={steps}")):
            img = dat['image']
            text = dat['query'] + task_instruction
            sample_idx = dat.get('question_id', idx)
            outputs = run_inference(model, processor, img, text, steps, decoding_strategy,
                                   sample_idx=sample_idx, benchmark_name=ds_name)
            label = dat['label']
            result.append({'id': dat['question_id'], 'prediction': outputs, 'label': label, 'category': dat.get('category', 'Unknown')})
            if accuracy_reward_yesno(outputs[0], label):
                correct += 1
            total += 1
        if EVAL_TOTAL_PROCESSES == 1:
            result = calculate_and_add_accuracy_summary(result, temp_out_file)
        json.dump(result, open(temp_out_file, 'w+'), indent=2)
        _merge_multiprocess_results(out_file, EVAL_TOTAL_PROCESSES, EVAL_PROCESS_ID, bench_label, expected_total_samples=len(dataset))
        print(f"Steps: {steps} - Accuracy: {correct}/{total} = {correct/total*100:.2f}" if total > 0 else f"Steps: {steps} - No samples")


def evaluate_mmmu(model, processor, dataset, image_dir, out_dir, ds_name, decoding_strategy="steps", checkpoint_num=None):
    return _evaluate_mcq_benchmark(model, processor, dataset, image_dir, out_dir, ds_name, decoding_strategy, checkpoint_num, "MMMU")


def evaluate_mmstar(model, processor, dataset, image_dir, out_dir, ds_name, decoding_strategy="steps", checkpoint_num=None):
    return _evaluate_mcq_benchmark(model, processor, dataset, image_dir, out_dir, ds_name, decoding_strategy, checkpoint_num, "MMStar")


def evaluate_pope(model, processor, dataset, image_dir, out_dir, ds_name, decoding_strategy="steps", checkpoint_num=None):
    return _evaluate_yesno_benchmark(model, processor, dataset, image_dir, out_dir, ds_name, decoding_strategy, checkpoint_num, "POPE")


def evaluate_mme(model, processor, dataset, image_dir, out_dir, ds_name, decoding_strategy="steps", checkpoint_num=None):
    return _evaluate_yesno_benchmark(model, processor, dataset, image_dir, out_dir, ds_name, decoding_strategy, checkpoint_num, "MME")


def evaluate_hrbench4k(model, processor, dataset, image_dir, out_dir, ds_name, decoding_strategy="steps", checkpoint_num=None):
    return _evaluate_mcq_benchmark(model, processor, dataset, image_dir, out_dir, ds_name, decoding_strategy, checkpoint_num, "HRBench4K")


def evaluate_hrbench8k(model, processor, dataset, image_dir, out_dir, ds_name, decoding_strategy="steps", checkpoint_num=None):
    return _evaluate_mcq_benchmark(model, processor, dataset, image_dir, out_dir, ds_name, decoding_strategy, checkpoint_num, "HRBench8K")


def evaluate_mme_realworld_lite(model, processor, dataset, image_dir, out_dir, ds_name, decoding_strategy="steps", checkpoint_num=None):
    return _evaluate_mcq_benchmark(model, processor, dataset, image_dir, out_dir, ds_name, decoding_strategy, checkpoint_num, "MME-RealWorld-Lite")


def _evaluate_hallbench_benchmark(model, processor, dataset, image_dir, out_dir, ds_name, decoding_strategy, checkpoint_num, bench_label):
    """HallBench: gt_answer is 0/1 (No/Yes). Extract yes/no from model output."""
    print(f"Evaluating {bench_label} with decoding strategy: {decoding_strategy}")
    os.makedirs(out_dir, exist_ok=True)
    task_instruction = get_task_instruction(ds_name)
    dataset_chunk = _get_dataset_chunk(dataset, EVAL_TOTAL_PROCESSES, EVAL_PROCESS_ID)
    for steps in STEP_LIST:
        out_file = _get_out_file(out_dir, checkpoint_num, steps, decoding_strategy, task_instruction)
        _cleanup_old_temp_files(out_file)
        temp_out_file = _get_temp_out_file(out_file)
        total, correct = 0, 0
        result = []
        for idx, dat in enumerate(tqdm(dataset_chunk, desc=f"Evaluating {bench_label}, steps={steps}")):
            img = dat['image']
            text = dat['query'] + task_instruction
            sample_idx = dat.get('question_id', idx)
            outputs = run_inference(model, processor, img, text, steps, decoding_strategy,
                                   sample_idx=sample_idx, benchmark_name=ds_name)
            label = str(dat['label']).strip()
            text_lower = (outputs[0] or '').lower()
            idx_yes, idx_no = text_lower.find('yes'), text_lower.find('no')
            given = 'yes' if (idx_yes >= 0 and (idx_no < 0 or idx_yes < idx_no)) else ('no' if idx_no >= 0 else None)
            gt_01 = '1' if label == '1' else '0'
            pred_01 = '1' if given == 'yes' else ('0' if given == 'no' else None)
            match = pred_01 == gt_01 if pred_01 is not None else False
            result.append({
                'id': dat['question_id'],
                'prediction': outputs,
                'label': label,
                'category': dat.get('category', 'Unknown'),
                'correct': match,
                'subcategory': dat.get('subcategory', ''),
                'set_id': dat.get('set_id', ''),
                'figure_id': dat.get('figure_id', ''),
                'visual_input': dat.get('visual_input', ''),
                'pair_question_id': dat.get('pair_question_id', ''),
            })
            if match:
                correct += 1
            total += 1
        if EVAL_TOTAL_PROCESSES == 1:
            result = calculate_and_add_accuracy_summary(result, temp_out_file)
        json.dump(result, open(temp_out_file, 'w+'), indent=2)
        _merge_multiprocess_results(out_file, EVAL_TOTAL_PROCESSES, EVAL_PROCESS_ID, bench_label, expected_total_samples=len(dataset))
        print(f"Steps: {steps} - Accuracy: {correct}/{total} = {correct/total*100:.2f}" if total > 0 else f"Steps: {steps} - No samples")


def _evaluate_mmhal_benchmark(model, processor, dataset, image_dir, out_dir, ds_name, decoding_strategy, checkpoint_num, bench_label):
    """MMHal: Free-form QA. Use normalized substring match (gt in pred or pred in gt)."""
    print(f"Evaluating {bench_label} with decoding strategy: {decoding_strategy}")
    os.makedirs(out_dir, exist_ok=True)
    task_instruction = get_task_instruction(ds_name)
    dataset_chunk = _get_dataset_chunk(dataset, EVAL_TOTAL_PROCESSES, EVAL_PROCESS_ID)
    for steps in STEP_LIST:
        out_file = _get_out_file(out_dir, checkpoint_num, steps, decoding_strategy, task_instruction)
        _cleanup_old_temp_files(out_file)
        temp_out_file = _get_temp_out_file(out_file)
        total, correct = 0, 0
        result = []
        for idx, dat in enumerate(tqdm(dataset_chunk, desc=f"Evaluating {bench_label}, steps={steps}")):
            img = dat['image']
            text = dat['query'] + task_instruction
            sample_idx = dat.get('question_id', idx)
            outputs = run_inference(model, processor, img, text, steps, decoding_strategy,
                                   sample_idx=sample_idx, benchmark_name=ds_name)
            pred = _extract_answer_from_response(outputs[0] or '')
            gt = re.sub(r'\s+', ' ', str(dat['label']).strip().lower())
            pred_norm = re.sub(r'\s+', ' ', pred.strip().lower())
            match = (gt in pred_norm or pred_norm in gt) if gt and pred_norm else False
            result.append({'id': dat['question_id'], 'prediction': outputs, 'label': dat['label'], 'category': dat.get('category', 'Unknown'), 'correct': match})
            if match:
                correct += 1
            total += 1
        if EVAL_TOTAL_PROCESSES == 1:
            result = calculate_and_add_accuracy_summary(result, temp_out_file)
        json.dump(result, open(temp_out_file, 'w+'), indent=2)
        _merge_multiprocess_results(out_file, EVAL_TOTAL_PROCESSES, EVAL_PROCESS_ID, bench_label, expected_total_samples=len(dataset))
        print(f"Steps: {steps} - Accuracy: {correct}/{total} = {correct/total*100:.2f}" if total > 0 else f"Steps: {steps} - No samples")


def evaluate_hallbench(model, processor, dataset, image_dir, out_dir, ds_name, decoding_strategy="steps", checkpoint_num=None):
    return _evaluate_hallbench_benchmark(model, processor, dataset, image_dir, out_dir, ds_name, decoding_strategy, checkpoint_num, "HallBench")


def evaluate_mmhal(model, processor, dataset, image_dir, out_dir, ds_name, decoding_strategy="steps", checkpoint_num=None):
    return _evaluate_mmhal_benchmark(model, processor, dataset, image_dir, out_dir, ds_name, decoding_strategy, checkpoint_num, "MMHal")


def evaluate_crpe(model, processor, dataset, image_dir, out_dir, ds_name, decoding_strategy="steps", checkpoint_num=None):
    return _evaluate_mcq_benchmark(model, processor, dataset, image_dir, out_dir, ds_name, decoding_strategy, checkpoint_num, "CRPE")


def evaluate_mathvision(model, processor, dataset, image_dir, out_dir, ds_name, decoding_strategy="steps", checkpoint_num=None):
    """Evaluate MathVision benchmark."""
    return _evaluate_math_benchmark(model, processor, dataset, out_dir, ds_name, decoding_strategy, checkpoint_num, "MathVision")


def evaluate_mathvista(model, processor, dataset, image_dir, out_dir, ds_name, decoding_strategy="steps", checkpoint_num=None):
    """Evaluate MathVista benchmark."""
    return _evaluate_math_benchmark(model, processor, dataset, out_dir, ds_name, decoding_strategy, checkpoint_num, "MathVista")


def _evaluate_math_benchmark(model, processor, dataset, out_dir, ds_name, decoding_strategy, checkpoint_num, bench_label):
    """Generic evaluator for MathVision/MathVista (MCQ + free-form)."""
    print(f"Evaluating {bench_label} with decoding strategy: {decoding_strategy}")
    os.makedirs(out_dir, exist_ok=True)
    task_instruction = get_task_instruction(ds_name)
    dataset_chunk = _get_dataset_chunk(dataset, EVAL_TOTAL_PROCESSES, EVAL_PROCESS_ID)
    for steps in STEP_LIST:
        out_file = _get_out_file(out_dir, checkpoint_num, steps, decoding_strategy, task_instruction)
        _cleanup_old_temp_files(out_file)
        temp_out_file = _get_temp_out_file(out_file)
        total, correct = 0, 0
        result = []
        for idx, dat in enumerate(tqdm(dataset_chunk, desc=f"Evaluating {bench_label}, steps={steps}")):
            img_path = dat['image']
            text = dat['query'] + task_instruction
            sample_idx = dat.get('question_id', idx)
            outputs = run_inference(model, processor, img_path, text, steps, decoding_strategy,
                                   sample_idx=sample_idx, benchmark_name=ds_name)
            label = dat['label']
            qtype = dat.get('question_type', 'free_form')
            is_mcq = qtype == 'multi_choice' or (len(str(label)) == 1 and str(label).upper() in 'ABCDE')
            if is_mcq:
                match = accuracy_reward(outputs[0], label)
            else:
                prec = dat.get('precision')
                prec = int(prec) if prec is not None and not (isinstance(prec, float) and (prec != prec or prec < 0)) else None
                match = accuracy_reward_math(outputs[0], label, precision=prec)
            result.append({'id': dat['question_id'], 'prediction': outputs, 'label': label, 'category': dat.get('category', 'Unknown')})
            if match:
                correct += 1
            total += 1
        if EVAL_TOTAL_PROCESSES == 1:
            result = calculate_and_add_accuracy_summary(result, temp_out_file)
        json.dump(result, open(temp_out_file, 'w+'), indent=2)
        _merge_multiprocess_results(out_file, EVAL_TOTAL_PROCESSES, EVAL_PROCESS_ID, bench_label)
        print(f"Steps: {steps} - Accuracy: {correct}/{total} = {correct/total*100:.2f}" if total > 0 else f"Steps: {steps} - No samples")


def evaluate_visulogic(model, processor, dataset, image_dir, out_dir, ds_name, decoding_strategy="steps", checkpoint_num=None):
    """Evaluate VisuLogic benchmark (MCQ)."""
    print(f"Evaluating VisuLogic with decoding strategy: {decoding_strategy}")
    os.makedirs(out_dir, exist_ok=True)
    task_instruction = get_task_instruction("visulogic")
    dataset_chunk = _get_dataset_chunk(dataset, EVAL_TOTAL_PROCESSES, EVAL_PROCESS_ID)
    for steps in STEP_LIST:
        out_file = _get_out_file(out_dir, checkpoint_num, steps, decoding_strategy, task_instruction)
        _cleanup_old_temp_files(out_file)
        temp_out_file = _get_temp_out_file(out_file)
        total, correct = 0, 0
        result = []
        for idx, dat in enumerate(tqdm(dataset_chunk, desc=f"Evaluating VisuLogic, steps={steps}")):
            text = dat['query'] + task_instruction
            sample_idx = dat.get('question_id', idx)
            outputs = run_inference(model, processor, dat['image'], text, steps, decoding_strategy, benchmark_name='visulogic')
            label = dat['label']
            result.append({'id': dat['question_id'], 'prediction': outputs, 'label': label, 'category': dat.get('category', 'Unknown')})
            if accuracy_reward(outputs[0], label):
                correct += 1
            total += 1
        if EVAL_TOTAL_PROCESSES == 1:
            result = calculate_and_add_accuracy_summary(result, temp_out_file)
        json.dump(result, open(temp_out_file, 'w+'), indent=2)
        _merge_multiprocess_results(out_file, EVAL_TOTAL_PROCESSES, EVAL_PROCESS_ID, "VisuLogic")
        print(f"Steps: {steps} - Accuracy: {correct}/{total} = {correct/total*100:.2f}" if total > 0 else "")


def evaluate_emma(model, processor, dataset, image_dir, out_dir, ds_name, decoding_strategy="steps", checkpoint_num=None):
    """Evaluate EMMA benchmark (MCQ)."""
    print(f"Evaluating EMMA with decoding strategy: {decoding_strategy}")
    os.makedirs(out_dir, exist_ok=True)
    task_instruction = get_task_instruction("emma")
    dataset_chunk = _get_dataset_chunk(dataset, EVAL_TOTAL_PROCESSES, EVAL_PROCESS_ID)
    for steps in STEP_LIST:
        out_file = _get_out_file(out_dir, checkpoint_num, steps, decoding_strategy, task_instruction)
        _cleanup_old_temp_files(out_file)
        temp_out_file = _get_temp_out_file(out_file)
        total, correct = 0, 0
        result = []
        for idx, dat in enumerate(tqdm(dataset_chunk, desc=f"Evaluating EMMA, steps={steps}")):
            text = dat['query'] + task_instruction
            sample_idx = dat.get('question_id', idx)
            outputs = run_inference(model, processor, dat['image'], text, steps, decoding_strategy, benchmark_name='emma')
            label = dat['label']
            result.append({'id': dat['question_id'], 'prediction': outputs, 'label': label, 'category': dat.get('category', 'Unknown')})
            if accuracy_reward(outputs[0], label):
                correct += 1
            total += 1
        if EVAL_TOTAL_PROCESSES == 1:
            result = calculate_and_add_accuracy_summary(result, temp_out_file)
        json.dump(result, open(temp_out_file, 'w+'), indent=2)
        _merge_multiprocess_results(out_file, EVAL_TOTAL_PROCESSES, EVAL_PROCESS_ID, "EMMA")
        print(f"Steps: {steps} - Accuracy: {correct}/{total} = {correct/total*100:.2f}" if total > 0 else "")


# ==== Main evaluation ====

def main():
    DECODING_STRATEGY = "baseline" if USE_BASE_MODEL else "steps"  # baseline: no LVR steps
    for checkpoint_dir in CHKPT_PATHS:
        model, processor, run_name = load_model_and_processor(checkpoint_dir)
        
        # Extract checkpoint number from checkpoint directory path
        checkpoint_num = None
        if checkpoint_dir:
            checkpoint_name = os.path.basename(checkpoint_dir)
            match = re.search(r'checkpoint-(\d+)', checkpoint_name)
            if match:
                checkpoint_num = match.group(1)
        
        # For base model, gen_w_head is False; for LVR model, get from config
        if USE_BASE_MODEL:
            gen_w_head = False
        else:
            gen_w_head = model.config.lvr_head

        # EVAL_BENCHMARKS: comma-separated list to run. Default: BLINK, MMVP, VSTAR, HRBench4K, HRBench8K, MME-RealWorld-Lite
        eval_benchmarks = os.environ.get('EVAL_BENCHMARKS', 'BLINK, MMVP, VSTAR, HRBench4K, HRBench8K, MME-RealWorld-Lite')
        if eval_benchmarks:
            bench_set = {b.strip().lower() for b in eval_benchmarks.split(',') if b.strip()}
            config_items = [(k, v) for k, v in DATASET_CONFIG.items() if k.lower() in bench_set]
        else:
            config_items = list(DATASET_CONFIG.items())
        for bench_name, cfg in config_items:
            dataset, image_dir, out_dir, ds_name = cfg["loader"](gen_w_head, run_name, decoding_strategy=DECODING_STRATEGY, checkpoint_num=checkpoint_num)
            cfg["evaluator"](model, processor, dataset, image_dir, out_dir, ds_name, decoding_strategy=DECODING_STRATEGY, checkpoint_num=checkpoint_num)

if __name__ == "__main__":
    main()
