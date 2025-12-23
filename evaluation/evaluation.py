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

from datasets import load_dataset, load_from_disk
from tqdm import tqdm
from src.model.qwen_lvr_model import QwenWithLVR
from transformers import AutoTokenizer, AutoProcessor, AutoConfig, Qwen2_5_VLForConditionalGeneration

from qwen_vl_utils import process_vision_info
from PIL import Image
from huggingface_hub import hf_hub_download

from src.train.monkey_patch_forward_lvr import replace_qwen2_5_with_mixed_modality_forward_lvr

# STEP_LIST can be set via environment variable EVAL_STEP_LIST (comma-separated, e.g., "4,8,16")
# If not set, defaults to [4,8,16]
STEP_LIST_ENV = os.environ.get('EVAL_STEP_LIST', '4,8,16')
STEP_LIST = [int(x.strip()) for x in STEP_LIST_ENV.split(',') if x.strip()]

# USE_BASE_MODEL controls whether to use base Qwen2.5-VL or LVR model
USE_BASE_MODEL = os.environ.get('USE_BASE_MODEL', '0') == '1'
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
        "loader": lambda gen_w_head,run_name,decoding_strategy: load_blink_dataset(gen_w_head,run_name,decoding_strategy),
        "evaluator": lambda model,proc,data,img_dir,out_dir,ds_name,decoding_strategy: evaluate_blink(model, proc, data, img_dir, out_dir, ds_name, decoding_strategy),
    },
    "vstar": {
        "loader": lambda gen_w_head,run_name,decoding_strategy: load_vstar_dataset(gen_w_head,run_name,decoding_strategy),
        "evaluator": lambda model,proc,data,img_dir,out_dir,ds_name,decoding_strategy: evaluate_vstar(model, proc, data, img_dir, out_dir, ds_name, decoding_strategy),
    },
    "MMVP": {
        "loader": lambda gen_w_head,run_name,decoding_strategy: load_mmvp_dataset(gen_w_head,run_name,decoding_strategy),
        "evaluator": lambda model,proc,data,img_dir,out_dir,ds_name,decoding_strategy: evaluate_mmvp(model, proc, data, img_dir, out_dir, ds_name, decoding_strategy),
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
    # Check if base run name is set via environment variable
    # If set, use it as base and append checkpoint name as subdirectory
    base_run_name = os.environ.get('EVAL_BASE_RUN_NAME')
    if base_run_name:
        # Extract checkpoint name from path (e.g., checkpoint-300)
        checkpoint_name = os.path.basename(chkpt_pth)
        run_name = f"{base_run_name}/{checkpoint_name}"
    else:
        # Original behavior: use full path
        details_list = chkpt_pth.split('/')[0:]
        run_name = '_'.join(details_list)

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
        replace_qwen2_5_with_mixed_modality_forward_lvr(inference_mode=True,lvr_head=config.lvr_head)
        
        model = QwenWithLVR.from_pretrained(
            chkpt_pth,
            config=config,
            trust_remote_code=True,
            torch_dtype="auto",
            attn_implementation="flash_attention_2",
            low_cpu_mem_usage=True,
        )
    
    # Move model to device manually
    model = model.to(device)

    processor = AutoProcessor.from_pretrained(chkpt_pth)

    return model, processor, run_name

def run_inference(model, processor,img_path,text,steps,decoding_strategy):
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

    with torch.no_grad():
        if USE_BASE_MODEL:
            # Base model: use standard generation without LVR parameters
            generated_ids = model.generate(**inputs, max_new_tokens=512)
        else:
            # LVR model: use custom generation with decoding_strategy and lvr_steps
            lvr_steps = [steps]
            generated_ids = model.generate(**inputs, max_new_tokens=512,decoding_strategy=decoding_strategy,lvr_steps=lvr_steps)
        
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs['input_ids'], generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
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
        ):
    print(f"Evaluating VSTAR with decoding strategy: {decoding_strategy}")
    os.makedirs(out_dir,exist_ok=True)
    task_instruction = get_task_instruction("vstar")
    step2results_category = {}
    step2results_overall = {}
    for steps in STEP_LIST:
        step2results_category[steps] = {}
        if len(task_instruction)>0 and task_instruction[0] != ' ':
            out_file = os.path.join(out_dir,f"{decoding_strategy}{steps:03d}.json")
        else:
            out_file = os.path.join(out_dir,f"{decoding_strategy}{steps:03d}_noTaskInstruction.json")
        total, correct = 0, 0
        res_by_category = {}
        if os.path.exists(out_file):
            # run_details = "-".join(out_file.split('/')[-3:])
            # print(f"result file existed, loading results for evaluation for {run_details}:")
            with open(out_file, "r") as f:
                result = json.load(f)
            # Recompute accuracy
            for res in result:
                if "category" not in res:
                    # Try to infer category from id if available
                    if "id" in res:
                        res["category"] = "direct_attributes" if int(res["id"]) <= 114 else "relative_position"
                    else:
                        # Skip entries without both category and id
                        print(f"Warning: Skipping result entry - missing both 'category' and 'id' fields")
                        continue
                if res["category"] not in res_by_category:
                    res_by_category[res["category"]] = {"total":0,"correct":0}
                if accuracy_reward(res["prediction"][0], res["label"]):
                    correct += 1
                    res_by_category[res["category"]]["correct"] += 1
                total += 1
                res_by_category[res["category"]]["total"] += 1
            step2results_category[steps] = res_by_category
            step2results_overall[steps] = {"total": total, "correct": correct}
        else:
            result = []
            dataset_chunk = dataset
            
            for dat in tqdm(dataset_chunk,desc=f"Evaluating Vstar, decoding by steps={steps}"):
                img_path = dat['image']
                # Handle VSTAR dataset images - download from HuggingFace if needed
                if isinstance(img_path, str):
                    # Check if it's an absolute path that exists
                    if os.path.isabs(img_path) and os.path.exists(img_path):
                        # Already a valid absolute path
                        pass
                    elif image_dir is not None and os.path.exists(os.path.join(image_dir, img_path)):
                        # Try image_dir first
                        img_path = os.path.join(image_dir, img_path)
                    else:
                        # Try to download from HuggingFace hub
                        # Add retry logic for network issues
                        try:
                            vstar_cache_dir = os.path.join(DATASETS_DIR, "vstar_bench")
                            import time
                            max_retries = 3
                            download_success = False
                            
                            for attempt in range(max_retries):
                                try:
                                    # Try to download image file from HuggingFace
                                    img_path = hf_hub_download(
                                        repo_id="craigwu/vstar_bench",
                                        filename=img_path,
                                        cache_dir=vstar_cache_dir,
                                        repo_type="dataset",
                                        local_files_only=False
                                    )
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
                # If img_path is already a PIL Image, use it directly
                text = dat['text'] + task_instruction
                outputs = run_inference(model, processor,img_path,text,steps,decoding_strategy)

                res = {
                    'id': dat['question_id'],
                    'prediction': outputs,
                    'label': dat['label'],
                    'category': dat['category']
                }
                result.append(res)
                if accuracy_reward(outputs[0], dat['label']):
                    correct += 1
                total += 1
            json.dump(result,open(out_file,'w+'),indent=2)
        print(f"Steps: {steps} - Accuracy: {correct}/{total} = {correct/total*100:.2f}")
    # print overall
    print("Overall accuracy by steps:")
    print(",".join([f"{items['correct']/items['total']*100:.2f}" for items in step2results_overall.values()]))
    # print by category
    for category in ["direct_attributes","relative_position"]:
        print(f"Category: {category}")
        res  = []
        for steps in step2results_category:
            res_by_category = step2results_category[steps]
            if category in res_by_category:
                cat_total = res_by_category[category]["total"]
                cat_correct = res_by_category[category]["correct"]
                res.append(cat_correct/cat_total)
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
        ):
    print(f"Evaluating MMVP with decoding strategy: {decoding_strategy}")
    os.makedirs(out_dir,exist_ok=True)
    task_instruction = get_task_instruction("mmvp")

    for steps in STEP_LIST:
        if len(task_instruction)>0 and task_instruction[0] != ' ':
            out_file = os.path.join(out_dir,f"{decoding_strategy}{steps:03d}.json")
        else:
            out_file = os.path.join(out_dir,f"{decoding_strategy}{steps:03d}_noTaskInstruction.json")
        total, correct = 0, 0
        if os.path.exists(out_file):
            run_details = "-".join(out_file.split('/')[-3:])
            # print(f"result file existed, loading results for evaluation for {run_details}:")
            with open(out_file, "r") as f:
                result = json.load(f)
            # Recompute accuracy
            for res in result:
                # Skip summary objects that don't have prediction/label fields
                if 'prediction' not in res or 'label' not in res:
                    continue
                if accuracy_reward(res["prediction"][0], res["label"]):
                    correct += 1
                total += 1
        else:
            result = []
            dataset_chunk = dataset
            
            for dat in tqdm(dataset_chunk,desc=f"Evaluating MMVP, decoding by steps={steps}"):
                text = dat['query'].replace('(a)','A.').replace('(b)','B.') + task_instruction
                outputs = run_inference(model, processor, dat['image'], text, steps, decoding_strategy)
                
                # Convert label format: (a) -> A, (b) -> B
                label = dat['label']
                if label in ['(a)','(b)']:
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
        print(f"Steps: {steps} - Accuracy: {correct}/{total} = {correct/total*100:.2f}")
        json.dump(result,open(out_file,'w+'),indent=2)
    return {"dummy_metric": 0}

def evaluate_blink(
        model, 
        processor, 
        dataset,
        image_dir,
        out_dir,
        ds_name,
        decoding_strategy="steps",
        ):
    print(f"Evaluating BLINK with decoding strategy: {decoding_strategy}")
    os.makedirs(out_dir,exist_ok=True)
    task_instruction = get_task_instruction("blink")
    step2results_category = {}
    step2results_overall = {}
    for steps in STEP_LIST:
        step2results_category[steps] = {}
        if len(task_instruction)>0 and task_instruction[0] != ' ':
            out_file = os.path.join(out_dir,f"{decoding_strategy}{steps:03d}.json")
        else:
            out_file = os.path.join(out_dir,f"{decoding_strategy}{steps:03d}_noTaskInstruction.json")
        total, correct = 0, 0
        res_by_category = {}
        if os.path.exists(out_file):
            with open(out_file, "r") as f:
                result = json.load(f)
            # Recompute accuracy
            for res in result:
                # Skip entries without category field (from old result files)
                if "category" not in res:
                    print(f"Warning: Skipping result entry {res.get('id', 'unknown')} - missing 'category' field")
                    continue
                if res["category"] not in res_by_category:
                    res_by_category[res["category"]] = {"total":0,"correct":0}
                if accuracy_reward(res["prediction"][0], res["label"]):
                    correct += 1
                    res_by_category[res["category"]]["correct"] += 1
                total += 1
                res_by_category[res["category"]]["total"] += 1
            step2results_category[steps] = res_by_category
            step2results_overall[steps] = {"total": total, "correct": correct}
        else:
            result = []
            dataset_chunk = dataset
            
            for dat in tqdm(dataset_chunk,desc=f"Evaluating BLINK, decoding by steps={steps}"):
                img_path = dat['image']
                text = dat['query'] + task_instruction
                outputs = run_inference(model, processor,img_path,text,steps,decoding_strategy)

                res = {
                    'id': dat['question_id'],
                    'prediction': outputs,
                    'label': dat['label'],
                    'category': dat['category']
                }
                result.append(res)
                if accuracy_reward(outputs[0], dat['label']):
                    correct += 1
                total += 1
            json.dump(result,open(out_file,'w+'),indent=2)
        print(f"Steps: {steps} - Accuracy: {correct}/{total} = {correct/total*100:.2f}")
    # print overall
    print("Overall accuracy by steps:")
    print(",".join([f"{items['correct']/items['total']*100:.2f}" for items in step2results_overall.values()]))
    # print by category
    for category in [ 'Counting','IQ_Test', 'Jigsaw', 'Multi-view_Reasoning', 'Object_Localization', 'Relative_Depth', 'Relative_Reflectance', 'Semantic_Correspondence', 'Spatial_Relation', 'Visual_Correspondence', 'Visual_Similarity']:
        # print(f"Category: {category}")
        res  = []
        for steps in step2results_category:
            res_by_category = step2results_category[steps]
            if category in res_by_category:
                cat_total = res_by_category[category]["total"]
                cat_correct = res_by_category[category]["correct"]
                res.append(cat_correct/cat_total)
        print(category+','+",".join([f"{items*100:.2f}" for items in res]))
    return

"""Data Loaders"""
# ==== Example dataset loader stubs ====
def load_vstar_dataset(gen_w_head,run_name,decoding_strategy):
    ds_name = "vstar"
    # Load dataset from HuggingFace and save to specified directory
    vstar_cache_dir = os.path.join(DATASETS_DIR, "vstar_bench")
    os.makedirs(vstar_cache_dir, exist_ok=True)
    ds = load_dataset("craigwu/vstar_bench", cache_dir=vstar_cache_dir)
    # Try to use images from dataset if available, otherwise use image_dir
    image_dir = None  # Images should come from HuggingFace dataset

    out_dir = os.path.join(RESULTS_DIR, "vstar_bench", f"decoding_by_{decoding_strategy}", run_name)

    return ds['test'],image_dir,out_dir,ds_name

def load_mmvp_dataset(gen_w_head,run_name,decoding_strategy):
    ds_name = "mmvp"
    mmvp_cache_dir = os.path.join(DATASETS_DIR, "MMVP")
    os.makedirs(mmvp_cache_dir, exist_ok=True)
    
    # Load CSV file
    csv_file = os.path.join(mmvp_cache_dir, "Questions.csv")
    if not os.path.exists(csv_file):
        from huggingface_hub import hf_hub_download
        csv_file = hf_hub_download(
            repo_id="MMVP/MMVP",
            filename="Questions.csv",
            cache_dir=mmvp_cache_dir,
            repo_type="dataset"
        )
    
    # Find image files - search in cache dir and HuggingFace hub cache
    search_paths = [
        os.path.join(mmvp_cache_dir, "**", "*.jpg"),
        os.path.join(os.path.expanduser("~/.cache/huggingface/hub"), "datasets--MMVP--MMVP", "**", "*.jpg")
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
    
    # Load data from CSV and map to image paths
    data = []
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
    
    # Determine image_dir (not used but kept for compatibility)
    image_dir = os.path.dirname(list(filename_to_path.values())[0]) if filename_to_path else None
    
    # Set output directory
    out_dir = os.path.join(RESULTS_DIR, "MMVP", f"decoding_by_{decoding_strategy}", run_name)

    return data, image_dir, out_dir, ds_name

from datasets import get_dataset_config_names
def load_blink_dataset(gen_w_head,run_name,decoding_strategy):
    ds_name = "blink"
    # Load BLINK dataset from local cache or HuggingFace
    blink_cache_dir = os.path.join(DATASETS_DIR, "BLINK")
    os.makedirs(blink_cache_dir, exist_ok=True)
    configs = [ 'Counting','IQ_Test', 'Jigsaw', 'Relative_Reflectance', 'Spatial_Relation']
    all_datasets = {}
    
    # Set offline mode to prevent network requests if dataset is already cached
    original_offline = os.environ.get('HF_DATASETS_OFFLINE', '0')
    dataset_hash = "a3666eb249237ba3d5eca8db21176cc47967e040"
    
    for config in configs:
        # Check if dataset exists in local cache
        val_path = os.path.join(blink_cache_dir, f"BLINK-Benchmark___blink", config, "0.0.0", dataset_hash)
        val_file = os.path.join(val_path, "blink-val.arrow")
        
        if os.path.exists(val_file):
            # Dataset exists locally, load directly from arrow file to avoid network requests
            print(f"Loading {config} from local cache: {val_file}")
            try:
                from datasets import Dataset
                import pyarrow as pa
                # Load validation split
                table = pa.ipc.open_file(val_file).read_all()
                val_dataset = Dataset.from_arrow(table)
                
                # Try to load test split if exists
                test_file = os.path.join(val_path, "blink-test.arrow")
                if os.path.exists(test_file):
                    test_table = pa.ipc.open_file(test_file).read_all()
                    test_dataset = Dataset.from_arrow(test_table)
                    all_datasets[config] = {'val': val_dataset, 'test': test_dataset}
                else:
                    all_datasets[config] = {'val': val_dataset}
            except Exception as e:
                print(f"Warning: Failed to load {config} from arrow file: {e}")
                print(f"Falling back to HuggingFace load_dataset...")
                # Fallback to HuggingFace load_dataset
                os.environ['HF_DATASETS_OFFLINE'] = '1'  # Force offline mode
                try:
                    all_datasets[config] = load_dataset(
                        "BLINK-Benchmark/BLINK", 
                        config, 
                        cache_dir=blink_cache_dir,
                        download_mode="reuse_cache_if_exists"
                    )
                finally:
                    os.environ['HF_DATASETS_OFFLINE'] = original_offline
        else:
            # Dataset not in cache, load from HuggingFace
            print(f"Loading {config} from HuggingFace...")
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

    return processed_data,image_dir,out_dir,ds_name
# ==== Main evaluation ====

def main():
    DECODING_STRATEGY = "steps"  # or "latent"
    for checkpoint_dir in CHKPT_PATHS:
        model, processor, run_name = load_model_and_processor(checkpoint_dir)
        
        # For base model, gen_w_head is False; for LVR model, get from config
        if USE_BASE_MODEL:
            gen_w_head = False
            print("Using base Qwen2.5-VL model (no LVR head)")
        else:
            gen_w_head = model.config.lvr_head
            print(f"Using LVR model (lvr_head={gen_w_head})")

        print("\n" + "="*128 + "\nEvaluating model:\n" +  f"{checkpoint_dir} \n"+"="*128 + "\n")
        for bench_name, cfg in DATASET_CONFIG.items():
            print("<"*64 + f" {bench_name} evaluation " + ">"*64)
            dataset, image_dir, out_dir, ds_name= cfg["loader"](gen_w_head,run_name,decoding_strategy=DECODING_STRATEGY)
            cfg["evaluator"](model, processor, dataset, image_dir, out_dir, ds_name, decoding_strategy=DECODING_STRATEGY)

if __name__ == "__main__":
    main()
