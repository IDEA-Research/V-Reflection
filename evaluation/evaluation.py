import sys
import os
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
STEP_LIST_ENV = os.environ.get('EVAL_STEP_LIST', '4,8,16')
STEP_LIST = [int(x.strip()) for x in STEP_LIST_ENV.split(',') if x.strip()]

# USE_BASE_MODEL controls whether to use base Qwen2.5-VL or LVR model
USE_BASE_MODEL = os.environ.get('USE_BASE_MODEL', '0') == '1'

# Multi-process evaluation support
# EVAL_PROCESS_ID: process index (0-based)
# EVAL_TOTAL_PROCESSES: total number of processes
EVAL_PROCESS_ID = int(os.environ.get('EVAL_PROCESS_ID', '0'))
EVAL_TOTAL_PROCESSES = int(os.environ.get('EVAL_TOTAL_PROCESSES', '1'))

# Force re-evaluation: if set to '1', always re-run inference even if result files exist
FORCE_RE_EVALUATE = os.environ.get('FORCE_RE_EVALUATE', '0') == '1'

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
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== Core utilities ====

def accuracy_reward(response: str, ground_truth: str) -> float:
    # content_match = re.search(r"<answer>(.*?)</answer>", response)
    # given_answer = content_match.group(1).strip() if content_match else response.strip()
    given_answer = response.split('<answer>')[-1]
    given_answer = given_answer.split('</answer')[0].strip()
    if " " in given_answer:
        given_answer = given_answer.split(" ")[0]
    if len(given_answer) >1:
        given_answer = given_answer[0]
    return given_answer == ground_truth

def calculate_and_add_accuracy_summary(data: list, json_file_path: str):
    """
    Calculate accuracy by category and add summary to the beginning of the data list.
    Similar to calculate_accuracy_by_category.py logic.
    """
    from collections import defaultdict
    
    # Skip if already has summary
    if len(data) > 0 and ('accuracy_by_category' in data[0] or 'overall_accuracy' in data[0]):
        return data
    
    # Calculate accuracy by category
    category_stats = defaultdict(lambda: {'total': 0, 'correct': 0})
    
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
        
        # Extract answer using same logic as accuracy_reward
        given_answer = prediction_str.split('<answer>')[-1]
        given_answer = given_answer.split('</answer')[0].strip()
        if " " in given_answer:
            given_answer = given_answer.split(" ")[0]
        if len(given_answer) > 1:
            given_answer = given_answer[0]
        
        # Count
        category_stats[category]['total'] += 1
        if given_answer == label:
            category_stats[category]['correct'] += 1
    
    # Calculate accuracy percentages
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
    
    # Calculate overall accuracy
    total_correct = sum(stats['correct'] for stats in results.values())
    total_samples = sum(stats['total'] for stats in results.values())
    overall_accuracy = (total_correct / total_samples * 100) if total_samples > 0 else 0.0
    
    # Create summary object
    accuracy_summary = {
        'accuracy_by_category': results,
        'overall_accuracy': overall_accuracy,
        'overall_correct': total_correct,
        'overall_total': total_samples
    }
    
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


def _merge_multiprocess_results(out_file, total_processes, process_id, benchmark_name=""):
    """
    Handle multi-process result merging.
    - Process 0: wait for all temp files, merge, clean up.
    - Other processes: wait briefly.
    
    IMPORTANT: Only Process 0 cleans up old temp files (before evaluation starts,
    the caller should call _cleanup_old_temp_files). This avoids a race condition
    where fast processes write temp files that are then deleted by slower processes.
    """
    if total_processes <= 1:
        return
    import time
    all_temp_files = [out_file.replace('.json', f'_process{i}.json') for i in range(total_processes)]
    
    if process_id == 0:
        max_wait_time = 300
        check_interval = 2
        waited_time = 0
        all_exist = False
        
        while waited_time < max_wait_time:
            all_exist = all(os.path.exists(f) for f in all_temp_files)
            if all_exist:
                break
            time.sleep(check_interval)
            waited_time += check_interval
            if waited_time % 10 == 0:
                existing_count = sum(os.path.exists(f) for f in all_temp_files)
                label = f"{benchmark_name}: " if benchmark_name else ""
                print(f"[Process 0] {label}Waiting for all processes... ({existing_count}/{total_processes} temp files found, waited {waited_time}s)")
        
        if all_exist:
            label = f"{benchmark_name} " if benchmark_name else ""
            print(f"[Process 0] Merging {label}results from {total_processes} processes...")
            merged_result = []
            for temp_file in all_temp_files:
                try:
                    with open(temp_file, 'r') as f:
                        merged_result.extend(json.load(f))
                except Exception as e:
                    print(f"[Process 0] Warning: Failed to read {temp_file}: {e}")
            # Sort by id to maintain order
            merged_result.sort(key=lambda x: (
                int(x.get('id', 0)) if isinstance(x.get('id'), (str, int)) else 0
            ) if 'id' in x else -1)
            merged_result = calculate_and_add_accuracy_summary(merged_result, out_file)
            json.dump(merged_result, open(out_file, 'w+'), indent=2)
            for temp_file in all_temp_files:
                try:
                    os.remove(temp_file)
                except Exception:
                    pass
            result_count = len([x for x in merged_result if 'id' in x])
            print(f"[Process 0] Merged {result_count} {label}results into {out_file}")
        else:
            existing_count = sum(os.path.exists(f) for f in all_temp_files)
            label = f"{benchmark_name} " if benchmark_name else ""
            print(f"[Process 0] Error: Not all {label}temp files exist after {max_wait_time}s ({existing_count}/{total_processes} found)")
            print(f"[Process 0] Missing files: {[f for f in all_temp_files if not os.path.exists(f)]}")
            # Clean up existing temp files even if merge failed
            for temp_file in all_temp_files:
                if os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                        print(f"[Process 0] Removed {temp_file}")
                    except Exception:
                        pass
    else:
        time.sleep(1)


def _cleanup_old_temp_files(out_file):
    """Clean up leftover temp files from previous failed runs.
    MUST only be called by Process 0 to avoid race conditions."""
    if EVAL_PROCESS_ID != 0 or EVAL_TOTAL_PROCESSES <= 1:
        return
    temp_pattern = out_file.replace('.json', '_process*.json')
    for old_temp_file in glob.glob(temp_pattern):
        try:
            os.remove(old_temp_file)
            print(f"Cleaned up old temp file: {old_temp_file}")
        except Exception as e:
            print(f"Warning: Failed to remove old temp file {old_temp_file}: {e}")


def _get_out_file(out_dir, checkpoint_num, steps, decoding_strategy, task_instruction):
    """Generate output file path for a given checkpoint/step."""
    if checkpoint_num:
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
    if bench_name == "vstar":
        return "\nAnswer with the option's letter from the given choices directly."
    elif bench_name == "mmvp":
        return "\nAnswer with the option's letter from the given choices directly."
    elif bench_name == "blink":
        return "\nAnswer with the option's letter from the given choices directly."
    else:
        raise ValueError(f"Unknown benchmark: {bench_name}")

def create_messages(img_path,question):

    if not isinstance(img_path, list):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": img_path,
                    },
                    {"type": "text", "text": question},
                ],
            }
        ]
    else:
        vision_content = []
        for ip in img_path:
            vision_content.append({
                        "type": "image",
                        "image": ip,
                    })
        vision_content.append({"type": "text", "text": question})
        messages = [
            {
                "role": "user",
                "content": vision_content,
            }
        ]
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
        use_dit_reconstruction = getattr(config, 'use_dit_reconstruction', False)
        replace_qwen2_5_with_mixed_modality_forward_lvr(
            inference_mode=True,
            lvr_head=config.lvr_head,
            use_box_feature_resampler=use_box_feature_resampler,
            use_dit_reconstruction=use_dit_reconstruction
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
    messages = create_messages(img_path,text)
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
# ==== Main evaluation ====

def main():
    DECODING_STRATEGY = "steps"  # or "latent"
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

        for bench_name, cfg in DATASET_CONFIG.items():
            dataset, image_dir, out_dir, ds_name= cfg["loader"](gen_w_head,run_name,decoding_strategy=DECODING_STRATEGY,checkpoint_num=checkpoint_num)
            cfg["evaluator"](model, processor, dataset, image_dir, out_dir, ds_name, decoding_strategy=DECODING_STRATEGY, checkpoint_num=checkpoint_num)

if __name__ == "__main__":
    main()
