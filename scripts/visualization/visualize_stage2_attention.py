#!/usr/bin/env python3
"""
Visualize Stage 2 Resampler attention maps.

Outputs per sample:
- {prefix}_qa.txt: Question and Answer (unchanged)
- {prefix}.png: Combined image
  - Benchmark mode (BLINK, MMVP, VSTAR, POPE): [original image | student attention]
  - Training set mode: [image+bbox | teacher attention | student attention]
  No titles on images.

Usage:
    # From training set (uniformly sample 100):
    python scripts/visualization/visualize_stage2_attention.py \
        --checkpoint_path result/stage2_distillation/.../checkpoint-2500 \
        --use_training_set --num_training_samples 100 \
        --output_dir evaluation/results/stage2_attention_vis

    # From meta (viscot-style with bboxes, specific indices):
    STAGE2_VIS_ATTENTION=1 python scripts/visualization/visualize_stage2_attention.py \
        --checkpoint_path result/stage2_distillation/.../checkpoint-2500 \
        --meta_path data/meta_data_lvr_sft_stage1.json \
        --sample_indices 0 5 10 \
        --output_dir evaluation/results/stage2_attention_vis

    # From Val datasets (BLINK, MMVP, VSTAR, HRBench4K, HRBench8K, MME-RealWorld-Lite, 10 per subset):
    python scripts/visualization/visualize_stage2_attention.py \
        --checkpoint_path ... --use_benchmark_datasets \
        --samples_per_subset 10 --output_dir evaluation/results/stage2_attention_vis
"""

import argparse
import json
import os
import re
import sys
import tempfile
from io import BytesIO

import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from qwen_vl_utils import process_vision_info

# Add project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.dataset.data_utils import get_image_info, map_image_path
from src.train.monkey_patch_forward_lvr import replace_qwen2_5_with_mixed_modality_forward_lvr
from transformers import AutoConfig, AutoProcessor
from src.model.qwen_lvr_model import QwenWithLVR


def bbox_to_token_idxs(bboxes, image_grid_thw):
    """Convert bbox (normalized xyxy) to token indices. Same logic as packed dataset."""
    if image_grid_thw is None or image_grid_thw.shape[0] == 0:
        return None
    _, h, w = image_grid_thw[0].tolist()
    H2, W2 = h // 2, w // 2
    token_idxs = []
    for bbox in bboxes:
        b = bbox
        while isinstance(b, (list, tuple)) and len(b) == 1:
            b = b[0]
        x0, y0, x1, y1 = [float(v) for v in b[:4]]
        if max(x0, y0, x1, y1) > 1.0:
            x0, y0, x1, y1 = x0 / w, y0 / h, x1 / w, y1 / h
        x0_grid = max(0, min(int(np.floor(x0 * w)), w - 1))
        x1_grid = max(0, min(int(np.ceil(x1 * w)), w))
        y0_grid = max(0, min(int(np.floor(y0 * h)), h - 1))
        y1_grid = max(0, min(int(np.ceil(y1 * h)), h))
        x0_token = x0_grid // 2
        x1_token = (x1_grid + 1) // 2
        y0_token = y0_grid // 2
        y1_token = (y1_grid + 1) // 2
        idxs = [int(yy * W2 + xx) for yy in range(y0_token, y1_token) for xx in range(x0_token, x1_token)]
        token_idxs.append(idxs)
    return token_idxs


def load_sample(meta_path, sample_idx, processor, image_folder, min_pixel=3136, max_pixel=12845056):
    """Load one sample: image, bboxes, pixel_values, image_grid_thw, lvr_tokens."""
    with open(meta_path) as f:
        meta = json.load(f)
    all_data = []
    for cfg in meta:
        with open(cfg["data_path"]) as df:
            data = json.load(df)
        folder = cfg.get("image_folder", image_folder)
        for item in data:
            item["_image_folder"] = folder
            item["_ds_name"] = cfg.get("ds_name", "")
            all_data.append(item)
    if sample_idx >= len(all_data):
        raise IndexError(f"sample_idx {sample_idx} >= len {len(all_data)}")
    item = all_data[sample_idx]
    image_folder = item["_image_folder"]
    image_files = item.get("image", item.get("images", []))
    if isinstance(image_files, str):
        image_files = [image_files]
    image_path = map_image_path(image_files[0], image_folder, item.get("dataset"))
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    image_info = get_image_info(image_path, min_pixel, max_pixel, None, None)
    if image_info is None:
        raise ValueError(f"get_image_info returned None for {image_path}")
    image_inputs = [image_info]
    inputs = processor(text=[""], images=image_inputs, videos=None, padding=False, do_resize=False, return_tensors="pt")
    pixel_values = inputs["pixel_values"]
    image_grid_thw = inputs["image_grid_thw"]
    bboxes = item.get("bboxes", [])
    lvr_token_idxs_list = bbox_to_token_idxs(bboxes, image_grid_thw)
    if not lvr_token_idxs_list or any(len(g) == 0 for g in lvr_token_idxs_list):
        raise ValueError(f"Empty bbox tokens for sample {sample_idx}")
    img_pil = Image.open(image_path).convert("RGB")
    return {
        "img_pil": img_pil,
        "bboxes": bboxes,
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
        "image_inputs": image_inputs,
        "lvr_tokens": [torch.tensor(g, dtype=torch.long) for g in lvr_token_idxs_list],
        "item": item,
    }


def load_training_samples_uniform(meta_path, num_samples, processor, image_folder, min_pixel=3136, max_pixel=12845056):
    """Load samples uniformly from training set (meta). Returns [(ds_name, sample_rank, sample_dict), ...]."""
    with open(meta_path) as f:
        meta = json.load(f)
    all_data = []
    for cfg in meta:
        with open(cfg["data_path"]) as df:
            data = json.load(df)
        folder = cfg.get("image_folder", image_folder)
        for item in data:
            item["_image_folder"] = folder
            item["_ds_name"] = cfg.get("ds_name", "")
            all_data.append(item)
    n = len(all_data)
    if n == 0:
        return []
    indices = np.linspace(0, n - 1, min(num_samples, n), dtype=int)
    indices = np.unique(indices)
    result = []
    for sample_rank, global_idx in enumerate(indices):
        try:
            sample = load_sample(meta_path, int(global_idx), processor, image_folder, min_pixel, max_pixel)
            ds_name = sample["item"].get("_ds_name", "unknown")
            result.append((ds_name, sample_rank, sample))
        except Exception as e:
            print(f"Warning: Skip sample {global_idx}: {e}")
            continue
    return result


def _resolve_benchmark_image(img, ds_name):
    """Resolve image to PIL or local path. Handles VSTAR HF paths, dict with bytes, etc."""
    if img is None:
        return None
    if isinstance(img, list):
        img = img[0] if img else None
    if img is None:
        return None
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    if isinstance(img, dict):
        if "bytes" in img:
            return Image.open(BytesIO(img["bytes"])).convert("RGB")
        if "path" in img and os.path.exists(img["path"]):
            return Image.open(img["path"]).convert("RGB")
    if isinstance(img, str):
        if os.path.exists(img):
            return Image.open(img).convert("RGB")
        # VSTAR: image path may be relative to HF repo, need to download
        if ds_name == "VSTAR":
            try:
                from huggingface_hub import hf_hub_download
                from evaluation.evaluation import DATASETS_DIR
                vstar_cache = os.path.join(DATASETS_DIR, "vstar_bench")
                local_path = hf_hub_download(
                    repo_id="craigwu/vstar_bench",
                    filename=img,
                    cache_dir=vstar_cache,
                    repo_type="dataset",
                    local_files_only=False,
                )
                if os.path.exists(local_path):
                    return Image.open(local_path).convert("RGB")
            except Exception as e:
                print(f"Warning: VSTAR image download failed for {img}: {e}")
    return None


def _pil_or_path_to_image_inputs(img, processor, min_pixel=3136, max_pixel=12845056):
    """Convert PIL or path to image_inputs for processor."""
    if isinstance(img, Image.Image):
        img = img.convert("RGB")
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            img.save(f.name)
            path = f.name
        try:
            info = get_image_info(path, min_pixel, max_pixel, None, None)
            if info is None:
                return None
            return [info]
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass
    elif isinstance(img, str) and os.path.exists(img):
        info = get_image_info(img, min_pixel, max_pixel, None, None)
        return [info] if info else None
    return None


def _safe_subset_name(name):
    """Sanitize subset/category name for use in file prefix (e.g. Perception/Existence -> Perception_Existence)."""
    if not name or name == "all":
        return "all"
    return re.sub(r"[^\w\-]", "_", str(name).strip())


def _load_hrbench_lazy(parquet_name, ds_name):
    """Load HRBench parquet without decoding images. Returns list of dicts with metadata + image_raw (bytes).
    Decode image only when needed to avoid OOM (8K images ~100MB each, 1000 = 100GB)."""
    import pandas as pd
    import base64
    from evaluation.evaluation import DATASETS_DIR
    base_dir = os.path.join(DATASETS_DIR, "HRBench")
    parquet_path = os.path.join(base_dir, parquet_name)
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"HRBench parquet not found: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    rows = []
    for _, row in df.iterrows():
        img_raw = row.get("image")
        if img_raw is None:
            continue
        question = str(row.get("question", ""))
        options = []
        for opt in ["A", "B", "C", "D"]:
            val = row.get(opt)
            if val is not None and str(val).strip():
                options.append(f"{opt}. {val}")
        if options:
            question = question + "\nOptions:\n" + "\n".join(options)
        answer = str(row.get("answer", "")).strip().upper()
        if len(answer) > 1:
            answer = answer[0]
        rows.append({
            "question_id": row.get("index", len(rows)),
            "image_raw": img_raw,
            "query": question,
            "label": answer,
            "category": str(row.get("category", "Unknown")),
        })
    return rows


def _decode_hrbench_image(img_raw):
    """Decode HRBench base64/bytes image to PIL."""
    import base64
    from io import BytesIO
    if isinstance(img_raw, str):
        img_bytes = base64.b64decode(img_raw)
    elif isinstance(img_raw, bytes):
        img_bytes = img_raw
    else:
        return None
    return Image.open(BytesIO(img_bytes)).convert("RGB")


def load_benchmark_samples(datasets, samples_per_subset, processor):
    """Load samples from BLINK, MMVP, VSTAR, HRBench4K, HRBench8K, MME-RealWorld-Lite.
    For datasets with subsets (category): sample samples_per_subset from each subset.
    Returns [(dataset_name, subset_name, local_idx, sample_dict), ...]."""
    from collections import defaultdict
    from evaluation.evaluation import (
        load_blink_dataset,
        load_mmvp_dataset,
        load_vstar_dataset,
        load_mme_realworld_lite_dataset,
    )
    loaders = {
        "BLINK": lambda: load_blink_dataset(False, "vis", "steps", None),
        "MMVP": lambda: load_mmvp_dataset(False, "vis", "steps", None),
        "VSTAR": lambda: load_vstar_dataset(False, "vis", "steps", None),
        "MME-RealWorld-Lite": lambda: load_mme_realworld_lite_dataset(False, "vis", "steps", None),
    }
    hrbench_parquets = {"HRBench4K": "hr_bench_4k.parquet", "HRBench8K": "hr_bench_8k.parquet"}
    result = []
    for ds_name in datasets:
        if ds_name not in loaders and ds_name not in hrbench_parquets:
            print(f"Warning: Unknown dataset {ds_name}, skipping")
            continue
        try:
            if ds_name in hrbench_parquets:
                data = _load_hrbench_lazy(hrbench_parquets[ds_name], ds_name)
                use_hrbench_lazy = True
            else:
                data, _, _, _ = loaders[ds_name]()
                use_hrbench_lazy = False
        except Exception as e:
            print(f"Warning: Failed to load {ds_name}: {e}")
            continue
        n = len(data)
        if n == 0:
            print(f"Warning: {ds_name} has 0 samples")
            continue
        # Group by category (subset); datasets without category go to "all"
        by_subset = defaultdict(list)
        for i in range(n):
            dat = data[i]
            cat = dat.get("category", "all")
            if cat is None or (isinstance(cat, str) and cat.strip() == ""):
                cat = "all"
            by_subset[str(cat)].append((i, dat))
        total_loaded = 0
        for subset_name, pairs in sorted(by_subset.items()):
            n_sub = len(pairs)
            k = min(samples_per_subset, n_sub)
            indices = np.linspace(0, n_sub - 1, k, dtype=int) if n_sub > 0 else []
            indices = np.unique(indices)
            for local_idx, sub_idx in enumerate(indices):
                global_idx, dat = pairs[int(sub_idx)]
                if use_hrbench_lazy:
                    img_raw = dat.get("image_raw")
                    img_pil = _decode_hrbench_image(img_raw) if img_raw else None
                else:
                    img_raw = dat.get("image")
                    img_pil = _resolve_benchmark_image(img_raw, ds_name)
                if img_pil is None:
                    continue
                image_inputs = _pil_or_path_to_image_inputs(img_pil, processor)
                if image_inputs is None:
                    continue
                query = dat.get("query", dat.get("text", dat.get("question", "")))
                answer = dat.get("label", dat.get("answer", dat.get("correct_answer", "")))
                if isinstance(answer, (list, tuple)):
                    answer = str(answer)
                elif answer is None:
                    answer = ""
                inputs = processor(
                    text=[""], images=image_inputs, videos=None, padding=False,
                    do_resize=False, return_tensors="pt"
                )
                bboxes = [[0.0, 0.0, 1.0, 1.0]]  # full image for MCQ/yes-no
                image_grid_thw = inputs["image_grid_thw"]
                lvr_token_idxs_list = bbox_to_token_idxs(bboxes, image_grid_thw)
                if not lvr_token_idxs_list or any(len(g) == 0 for g in lvr_token_idxs_list):
                    continue
                sample = {
                    "img_pil": img_pil,
                    "bboxes": bboxes,
                    "pixel_values": inputs["pixel_values"],
                    "image_grid_thw": image_grid_thw,
                    "image_inputs": image_inputs,
                    "lvr_tokens": [torch.tensor(g, dtype=torch.long) for g in lvr_token_idxs_list],
                    "item": {
                        "conversations": [
                            {"from": "human", "value": "<image>\n" + query},
                            {"from": "gpt", "value": "<lvr>\n<answer>placeholder</answer>"},
                        ],
                        "query": query,
                        "answer": answer,
                    },
                }
                subset_safe = _safe_subset_name(subset_name)
                result.append((ds_name, subset_safe, local_idx, sample))
                total_loaded += 1
            print(f"  Loaded {len(indices)} from {ds_name}/{subset_name} (subset has {n_sub} samples)")
        print(f"  Total from {ds_name}: {total_loaded} samples")
    return result


def build_input_for_forward(sample, processor, model, num_latent_tokens=8):
    """Build input_ids, labels, etc. for model forward with LVR template."""
    from src.dataset.data_utils import llava_to_openai_lvr

    item = sample["item"]
    conversations = item.get("conversations", [])
    lvr_token_idxs_list = [t.tolist() if isinstance(t, torch.Tensor) else t for t in sample["lvr_tokens"]]
    sources = llava_to_openai_lvr(
        conversations, is_video=False, lvr_token_idxs_list=lvr_token_idxs_list,
        latent_end_token=False, fixed_num_of_lvr_tokens=num_latent_tokens, use_fixed_num_lvr_tokens=False
    )
    text = ""
    for s in sources:
        role = s.get("role", "")
        content = s.get("content", "")
        if role == "user":
            text += content
        elif role == "assistant":
            text += content
    text_formatted = processor.apply_chat_template(
        [{"role": "user", "content": text}], tokenize=False, add_generation_prompt=False
    )
    inputs = processor(
        text=[text_formatted],
        images=sample["image_inputs"],
        videos=None,
        padding=True,
        do_resize=False,
        return_tensors="pt",
    )
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids))
    labels = input_ids.clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "pixel_values": inputs["pixel_values"],
        "image_grid_thw": inputs["image_grid_thw"],
        "lvr_tokens": sample["lvr_tokens"],
    }


def attention_to_2d_bbox(attn_weights, lvr_tokens_flat, key_pad_mask, token_grid_h, token_grid_w):
    """
    Map Teacher (BoxFeatureResampler) attention to 2D grid.
    attn_weights: (1, N) - one query's attention over N bbox tokens (same order as lvr_tokens_flat).
    lvr_tokens_flat: bbox token indices in same order as bbox_feats (row-major).
    """
    attn = attn_weights.numpy().flatten()  # (N,)
    h, w = token_grid_h, token_grid_w
    grid = np.zeros((h, w), dtype=np.float32)
    for idx, token_idx in enumerate(lvr_tokens_flat):
        if key_pad_mask is not None and idx < key_pad_mask.shape[1] and key_pad_mask[0, idx]:
            continue
        y, x = int(token_idx) // w, int(token_idx) % w
        if 0 <= y < h and 0 <= x < w and idx < len(attn):
            grid[y, x] = attn[idx]
    return grid


def attention_to_2d_full(attn_weights, image_attention_mask, token_grid_h, token_grid_w):
    """
    Map Student (DynamicAutoregressiveResampler) attention to 2D grid.
    attn_weights: (1, 1, Seq_Len) - one latent step's attention over full image tokens (row-major).
    """
    attn = attn_weights.numpy().flatten()  # (Seq_Len,)
    h, w = token_grid_h, token_grid_w
    grid = np.zeros((h, w), dtype=np.float32)
    for j in range(min(len(attn), h * w)):
        if image_attention_mask is not None and j < image_attention_mask.shape[1] and not image_attention_mask[0, j]:
            continue
        y, x = j // w, j % w
        if 0 <= y < h and 0 <= x < w:
            grid[y, x] = attn[j]
    return grid


def _extract_answer_from_conversations(conversations):
    """Extract answer from gpt message: <lvr>\\n<answer>xxx</answer>."""
    for c in (conversations or []):
        if c.get("from") == "gpt":
            val = c.get("value", "")
            m = re.search(r"<answer>(.*?)</answer>", val, re.DOTALL)
            return m.group(1).strip() if m else val.replace("<lvr>\n", "").strip()
    return ""


def _get_student_attn_grid(vis_buffer, img_arr, token_grid_h, token_grid_w):
    """Average student attention over 8 steps, return normalized grid for overlay."""
    student_attn = vis_buffer["student_attn"]
    image_attention_mask = vis_buffer.get("image_attention_mask")
    grids = []
    for step in range(min(8, student_attn.shape[1])):
        attn_step = student_attn[0:1, step:step+1, :]
        grid = attention_to_2d_full(attn_step, image_attention_mask, token_grid_h, token_grid_w)
        grids.append(grid)
    grid = np.mean(grids, axis=0)
    g_min, g_max = grid.min(), grid.max()
    if g_max > g_min:
        grid = (grid - g_min) / (g_max - g_min)
    grid_pil = Image.fromarray((np.clip(grid, 0, 1) * 255).astype(np.uint8))
    return np.array(grid_pil.resize((img_arr.shape[1], img_arr.shape[0]), Image.BILINEAR)) / 255.0


def _get_teacher_attn_grid(vis_buffer, img_arr, token_grid_h, token_grid_w):
    """Average teacher attention over 8 queries, return normalized grid for overlay."""
    teacher_attn = vis_buffer["teacher_attn"]
    lvr_tokens = vis_buffer["lvr_tokens"]
    key_pad_mask = vis_buffer.get("key_pad_mask")
    lvr_flat = []
    for t in lvr_tokens:
        arr = t.numpy() if isinstance(t, torch.Tensor) else np.array(t)
        lvr_flat.extend(arr.tolist())
    lvr_flat = np.array(lvr_flat)
    grids = []
    for q in range(min(8, teacher_attn.shape[1])):
        attn_q = teacher_attn[0, q]
        grid = attention_to_2d_bbox(attn_q.unsqueeze(0), lvr_flat, key_pad_mask, token_grid_h, token_grid_w)
        grids.append(grid)
    grid = np.mean(grids, axis=0)
    g_min, g_max = grid.min(), grid.max()
    if g_max > g_min:
        grid = (grid - g_min) / (g_max - g_min)
    grid_pil = Image.fromarray((np.clip(grid, 0, 1) * 255).astype(np.uint8))
    return np.array(grid_pil.resize((img_arr.shape[1], img_arr.shape[0]), Image.BILINEAR)) / 255.0


def save_visualizations(sample_idx, sample, vis_buffer, output_dir, file_prefix=None, question=None, answer=None, mode="training", preserve_resolution=False):
    """
    Save Q&A and combined visualization.
    mode: "benchmark" -> [original image | student attention] in one image, no titles
    mode: "training" -> [image+bbox | teacher | student] in one image, no titles
    preserve_resolution: if True, output PNG matches original image resolution (for HRBench 4K/8K).
    """
    os.makedirs(output_dir, exist_ok=True)
    prefix = file_prefix if file_prefix is not None else f"sample_{sample_idx}"
    img_pil = sample["img_pil"]
    bboxes = sample["bboxes"]
    img_arr = np.array(img_pil)
    img_h, img_w = img_arr.shape[:2]
    _, h, w = vis_buffer["image_grid_thw"][0].tolist()
    token_grid_h, token_grid_w = h // 2, w // 2

    # (1) Save question and answer to text file (unchanged)
    if question is not None or answer is not None:
        qa_path = os.path.join(output_dir, f"{prefix}_qa.txt")
        with open(qa_path, "w", encoding="utf-8") as f:
            if question:
                f.write(f"Question:\n{question}\n\n")
            if answer:
                f.write(f"Answer:\n{answer}\n")
        print(f"  Saved Q&A to {qa_path}")

    # (2) Combined image: bbox overlay + attention map(s), no titles
    n_cols = 2 if mode == "benchmark" else 3
    if preserve_resolution:
        # Match original image resolution (e.g. HRBench8K 7680x4320); output = (n_cols*img_w) x img_h px
        dpi = 150
        fig_w = (n_cols * img_w) / dpi
        fig_h = img_h / dpi
        fig, axes = plt.subplots(1, n_cols, figsize=(fig_w, fig_h))
    else:
        fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 6))

    # Left: original image (benchmark) or image overlay with bbox (training)
    axes[0].imshow(img_arr)
    if mode != "benchmark":
        for bbox in bboxes:
            b = bbox
            while isinstance(b, (list, tuple)) and len(b) == 1:
                b = b[0]
            x0, y0, x1, y1 = [float(v) for v in b[:4]]
            if max(x0, y0, x1, y1) <= 1.0:
                x0, y0, x1, y1 = x0 * img_w, y0 * img_h, x1 * img_w, y1 * img_h
            rect = plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="red", linewidth=2)
            axes[0].add_patch(rect)
    axes[0].axis("off")

    # Right (benchmark) or Middle+Right (training): attention maps
    student_grid = _get_student_attn_grid(vis_buffer, img_arr, token_grid_h, token_grid_w)
    if mode == "benchmark":
        axes[1].imshow(img_arr)
        axes[1].imshow(student_grid, alpha=0.5, cmap="jet", vmin=0, vmax=1)
        axes[1].axis("off")
    else:
        teacher_grid = _get_teacher_attn_grid(vis_buffer, img_arr, token_grid_h, token_grid_w)
        axes[1].imshow(img_arr)
        axes[1].imshow(teacher_grid, alpha=0.5, cmap="jet", vmin=0, vmax=1)
        axes[1].axis("off")
        axes[2].imshow(img_arr)
        axes[2].imshow(student_grid, alpha=0.5, cmap="jet", vmin=0, vmax=1)
        axes[2].axis("off")

    plt.subplots_adjust(wspace=0.02, hspace=0)
    plt.savefig(os.path.join(output_dir, f"{prefix}.png"), bbox_inches="tight", dpi=150)
    plt.close()

    print(f"Saved visualization for {prefix} to {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--meta_path", type=str, default="data/meta_data_lvr_sft_stage1.json")
    parser.add_argument("--sample_indices", type=int, nargs="+", default=[0, 5, 10])
    parser.add_argument("--output_dir", type=str, default="evaluation/results/stage2_attention_vis")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--use_benchmark_datasets", action="store_true",
                        help="Load from BLINK, MMVP, VSTAR, HRBench4K, HRBench8K, MME-RealWorld-Lite")
    parser.add_argument("--benchmark_datasets", type=str, nargs="+",
                        default=["BLINK", "MMVP", "VSTAR", "HRBench4K", "HRBench8K", "MME-RealWorld-Lite"])
    parser.add_argument("--samples_per_subset", type=int, default=10,
                        help="For datasets with subsets: sample this many per subset; otherwise per dataset")
    parser.add_argument("--use_training_set", action="store_true", help="Uniformly sample from training set (meta_path)")
    parser.add_argument("--num_training_samples", type=int, default=100)
    args = parser.parse_args()

    output_dir = os.path.join(PROJECT_ROOT, args.output_dir) if not os.path.isabs(args.output_dir) else args.output_dir

    print("Loading model...")
    config = AutoConfig.from_pretrained(args.checkpoint_path)
    config.use_box_feature_resampler = True
    config.use_stage2_distillation = True
    replace_qwen2_5_with_mixed_modality_forward_lvr(
        inference_mode=False,
        lvr_head=False,
        use_box_feature_resampler=True,
        use_stage2_distillation=True,
    )
    model = QwenWithLVR.from_pretrained(
        args.checkpoint_path,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        low_cpu_mem_usage=True,
    )
    processor = AutoProcessor.from_pretrained(args.checkpoint_path)
    model = model.to(args.device)
    model.eval()

    os.environ["STAGE2_VIS_ATTENTION"] = "1"

    if args.use_training_set:
        meta_path = os.path.join(PROJECT_ROOT, args.meta_path) if not os.path.isabs(args.meta_path) else args.meta_path
        with open(meta_path) as f:
            meta = json.load(f)
        image_folder = meta[0].get("image_folder", "/comp_robot/zhoujiazhou/Datasets/Visual_cot/images")
        print(f"Uniformly sampling {args.num_training_samples} from training set: {meta_path}")
        samples = load_training_samples_uniform(meta_path, args.num_training_samples, processor, image_folder)
        print(f"Loaded {len(samples)} samples total")
        for ds_name, sample_rank, sample in samples:
            try:
                print(f"Processing {ds_name}_{sample_rank:03d}...")
                batch = build_input_for_forward(sample, processor, model)
                batch = {k: v.to(args.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                if isinstance(batch["lvr_tokens"][0], torch.Tensor):
                    batch["lvr_tokens"] = [t.to(args.device) for t in batch["lvr_tokens"]]

                with torch.no_grad():
                    _ = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"],
                        pixel_values=batch["pixel_values"],
                        image_grid_thw=batch["image_grid_thw"],
                        lvr_tokens=batch["lvr_tokens"],
                    )

                if not hasattr(model, "_stage2_vis_buffer"):
                    print(f"Warning: No _stage2_vis_buffer for {ds_name}_{sample_rank:03d}")
                    continue
                vis_buffer = model._stage2_vis_buffer
                file_prefix = f"{ds_name}_{sample_rank:03d}"
                convs = sample["item"].get("conversations", [])
                question = None
                for c in convs:
                    if c.get("from") == "human":
                        question = c.get("value", "").replace("<image>\n", "").strip()
                        break
                answer = _extract_answer_from_conversations(convs)
                save_visualizations(
                    sample_rank, sample, vis_buffer, output_dir,
                    file_prefix=file_prefix, question=question, answer=answer, mode="training",
                    preserve_resolution=True,
                )
            except Exception as e:
                print(f"Error processing {ds_name}_{sample_rank:03d}: {e}")
                import traceback
                traceback.print_exc()
    elif args.use_benchmark_datasets:
        print(f"Loading from Val datasets: {args.benchmark_datasets}, {args.samples_per_subset} per subset")
        samples = load_benchmark_samples(args.benchmark_datasets, args.samples_per_subset, processor)
        print(f"Loaded {len(samples)} samples total")
        for ds_name, subset_name, local_idx, sample in samples:
            file_prefix = f"{ds_name}_{subset_name}_{local_idx:02d}" if subset_name != "all" else f"{ds_name}_{local_idx:02d}"
            try:
                print(f"Processing {file_prefix}...")
                batch = build_input_for_forward(sample, processor, model)
                batch = {k: v.to(args.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                if isinstance(batch["lvr_tokens"][0], torch.Tensor):
                    batch["lvr_tokens"] = [t.to(args.device) for t in batch["lvr_tokens"]]

                with torch.no_grad():
                    _ = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"],
                        pixel_values=batch["pixel_values"],
                        image_grid_thw=batch["image_grid_thw"],
                        lvr_tokens=batch["lvr_tokens"],
                    )

                if not hasattr(model, "_stage2_vis_buffer"):
                    print(f"Warning: No _stage2_vis_buffer for {file_prefix}")
                    continue
                vis_buffer = model._stage2_vis_buffer
                question = sample["item"].get("query", "")
                answer = sample["item"].get("answer", "")
                save_visualizations(
                    local_idx, sample, vis_buffer, output_dir,
                    file_prefix=file_prefix, question=question, answer=answer, mode="benchmark",
                    preserve_resolution=True,
                )
            except Exception as e:
                print(f"Error processing {file_prefix}: {e}")
                import traceback
                traceback.print_exc()
    else:
        meta_path = os.path.join(PROJECT_ROOT, args.meta_path) if not os.path.isabs(args.meta_path) else args.meta_path
        with open(meta_path) as f:
            meta = json.load(f)
        image_folder = meta[0].get("image_folder", "/comp_robot/zhoujiazhou/Datasets/Visual_cot/images")

        for sample_idx in args.sample_indices:
            try:
                print(f"Processing sample {sample_idx}...")
                sample = load_sample(meta_path, sample_idx, processor, image_folder)
                batch = build_input_for_forward(sample, processor, model)
                batch = {k: v.to(args.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                if isinstance(batch["lvr_tokens"][0], torch.Tensor):
                    batch["lvr_tokens"] = [t.to(args.device) for t in batch["lvr_tokens"]]

                with torch.no_grad():
                    _ = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"],
                        pixel_values=batch["pixel_values"],
                        image_grid_thw=batch["image_grid_thw"],
                        lvr_tokens=batch["lvr_tokens"],
                    )

                if not hasattr(model, "_stage2_vis_buffer"):
                    print(f"Warning: No _stage2_vis_buffer for sample {sample_idx} (no LVR in batch?)")
                    continue
                vis_buffer = model._stage2_vis_buffer
                ds_name = sample["item"].get("_ds_name", "unknown")
                file_prefix = f"{ds_name}_{sample_idx:03d}"
                question = None
                convs = sample["item"].get("conversations", [])
                for c in convs:
                    if c.get("from") == "human":
                        question = c.get("value", "").replace("<image>\n", "").strip()
                        break
                answer = _extract_answer_from_conversations(convs)
                save_visualizations(
                    sample_idx, sample, vis_buffer, output_dir,
                    file_prefix=file_prefix, question=question, answer=answer, mode="training",
                    preserve_resolution=True,
                )
            except Exception as e:
                print(f"Error processing sample {sample_idx}: {e}")
                import traceback
                traceback.print_exc()

    print(f"Done. Outputs in {output_dir}")


if __name__ == "__main__":
    main()
