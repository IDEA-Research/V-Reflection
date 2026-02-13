"""
    This file is adapted from InternVL [https://github.com/OpenGVLab/InternVL/tree/main]
    We adapted and simplified the PackedDataset and IterableSupervisedDataset for our LVR finetuning
"""


import copy
import os
from typing import Dict
import torch
import transformers
import ujson as json
from torch.utils.data import Dataset, IterableDataset

from functools import partial

from src.params import DataArguments
from src.constants import (
    IGNORE_INDEX,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    SYSTEM_MESSAGE,
    VISION_START_TOKEN,
    VISION_END_TOKEN,
    LVR_TOKEN,
    LVR_PLACEHOLDER,
)
from transformers import TrainingArguments

from .data_utils import get_image_info, llava_to_openai_lvr, pad_sequence, map_image_path
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional
import math

# DiT reconstruction: crop bbox region and resize, normalize to [-1, 1]
# Default crop size is now controlled via data_args.dit_crop_size (default 128)
DIT_CROP_SIZE = 128


def crop_bbox_and_resize(pil_img: Image.Image, bbox: List[float], size: int = DIT_CROP_SIZE) -> torch.Tensor:
    """Crop image by normalized bbox [x_min, y_min, x_max, y_max], resize to size x size, return (3, size, size) in [-1, 1]."""
    w, h = pil_img.size
    x_min, y_min, x_max, y_max = bbox
    if max(x_min, y_min, x_max, y_max) > 1.0:
        x_min, y_min = x_min / w, y_min / h
        x_max, y_max = x_max / w, y_max / h
    x_min, y_min = max(0, x_min), max(0, y_min)
    x_max, y_max = min(1, x_max), min(1, y_max)
    x1, y1 = int(x_min * w), int(y_min * h)
    x2, y2 = int(x_max * w), int(y_max * h)
    if x2 <= x1:
        x2 = x1 + 1
    if y2 <= y1:
        y2 = y1 + 1
    crop = pil_img.crop((x1, y1, x2, y2))
    crop = crop.resize((size, size), Image.BICUBIC)
    arr = np.array(crop)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    arr = arr[:, :, :3].astype(np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    t = torch.from_numpy(arr).permute(2, 0, 1).float()
    return t

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

  
def make_packed_supervised_data_module_lvr(model_id, processor, data_args, training_args: TrainingArguments, latent_end_token=False, fixed_num_of_lvr_tokens=None):

    """Make dataset and collator for supervised fine-tuning."""

    data_rank = dist.get_rank()
    data_world_size = dist.get_world_size()
    if fixed_num_of_lvr_tokens is None:
        fixed_num_of_lvr_tokens = getattr(data_args, 'fixed_num_of_lvr_tokens', None)

    # we assume meta data
    meta_data = json.load(open(data_args.data_path))

    datasets = []
    total_data_len = 0
    for meta in meta_data:
        iterable_sft_dataset = IterableSupervisedDatasetLVR(
            data_path=meta['data_path'],
            image_folder=meta['image_folder'],
            ds_name=meta['ds_name'],
            processor=processor,
            data_args=data_args,
            model_id=model_id,
            data_rank=data_rank,
            data_world_size=data_world_size,
            distributed_mode = training_args.enable_data_packing,    # set for packed dataset
            random_seed=data_args.random_seed,
            latent_end_token=latent_end_token,
            fixed_num_of_lvr_tokens=fixed_num_of_lvr_tokens,
            max_packed_tokens=training_args.max_packed_tokens,  # For dynamic resolution adjustment
        )
        datasets.append(iterable_sft_dataset)
        total_data_len += len(iterable_sft_dataset)

    packed_train_dataset = PackedDataset(
        tokenizer=processor.tokenizer,
        datasets=datasets,
        # Get rank and world size from Hugging Face Trainer arguments
        data_rank=data_rank,
        data_world_size=data_world_size,
        # --- Configure your packing parameters ---
        max_packed_tokens=training_args.max_packed_tokens,
        max_buffer_size=100,
        # long_seq_cut=training_args.long_seq_cut,
        long_seq_threshold=training_args.long_seq_threshold,
        # Limiting the number of training data per device to avoid exploded tokens_per_device when pairing long with short
        max_instance_per_batch=training_args.max_instance_per_batch,
        allow_overflow=False,  # Disable overflow to prevent OOM from extremely long sequences
    )

    data_collator = PackedDataCollatorForSupervisedDatasetLVR(
        pad_token_id=processor.tokenizer.pad_token_id
    )

    return dict(
        train_dataset=packed_train_dataset,
        eval_dataset=None,
        data_collator=data_collator,), total_data_len

import torch
from torch.utils.data import IterableDataset

class IterableSupervisedDatasetLVR(Dataset):
    """
    An iterable version of your dataset that streams one processed sample at a time.
    This will be the input to PackedDataset.
    """
    def __init__(
        self,
        data_path: str,
        image_folder: str,
        processor: transformers.ProcessorMixin,
        data_args: DataArguments,
        ds_name: str,
        model_id,
        data_rank=0,
        data_world_size=1,
        distributed_mode = True,    # set for packed dataset
        random_seed=None,
        latent_end_token=False,
        fixed_num_of_lvr_tokens=None,
        max_packed_tokens=4096,  # Add max_packed_tokens for dynamic resolution adjustment
    ):
        super().__init__()
        if isinstance(data_path, str):
            self.raw_data = json.load(open(data_path, "r"))
        else:
            self.raw_data = data_path

        self.model_id = model_id
        self.processor = processor
        self.data_args = data_args
        self.image_folder = image_folder
        self.fixed_num_of_lvr_tokens = fixed_num_of_lvr_tokens
        self.image_min_pixel = data_args.image_min_pixels
        self.image_max_pixel = data_args.image_max_pixels
        self.video_min_pixel = data_args.video_min_pixels
        self.video_max_pixel = data_args.video_max_pixels
        self.image_resized_w = data_args.image_resized_width
        self.image_resized_h = data_args.image_resized_height
        self.video_resized_w = data_args.video_resized_width
        self.video_resized_h = data_args.video_resized_height
        self.ds_name = ds_name
        self.fps = data_args.fps
        
        # For dynamic resolution adjustment to avoid <lvr> truncation
        self.max_packed_tokens = max_packed_tokens
        # Minimum image resolution to avoid quality degradation
        self.min_image_max_pixel = 500000  # ~700x700 minimum

        self.data_world_size = data_world_size
        self.worker_id = None
        self.distributed_mode = distributed_mode
        self.worker_distributed = False
        self._state_dict = {}

        self.random_seed = None
        if random_seed:
            logger.info(f"{self.ds_name} is Shuffled!")
            self.random_seed = random_seed
            self.rng = np.random.default_rng(seed=self.random_seed)
            self.rng.shuffle(self.raw_data)

        # int latent_end_token mode, a latent end token wil be appended to the selected lvr tokens
        self.latent_end_token = latent_end_token
        # DiT crop size: controlled via data_args, default 128
        self.dit_crop_size = getattr(data_args, 'dit_crop_size', DIT_CROP_SIZE)

    def __len__(self):
        return len(self.raw_data)

    def bbox_to_token_idxs(self, bboxes, image_grid_thw):
        """
            This function is intended for Qwen-VL series only.
            The target visual tokens is computed based on image_grid_thw,
            which is the best estimation

            image_grid_thw is a 2D tensor with a single item
            
            Returns:
                token_idxs: List of token indices for each bbox, or None if any bbox is invalid

        """
        if image_grid_thw is None or image_grid_thw.shape[0] == 0:
            logger.warning(
                f"[{self.ds_name}] bbox_to_token_idxs: empty image_grid_thw (shape={getattr(image_grid_thw, 'shape', None)}). "
                "Returning None so caller skips sample."
            )
            return None
        _, h, w = image_grid_thw[0].tolist()
        token_idxs = []
        H2, W2 = h // 2, w // 2
        
        for bbox_idx, bbox in enumerate(bboxes):
            # Validate bbox format
            try:
                if not isinstance(bbox, (list, tuple, np.ndarray)):
                    logger.warning(
                        f"[{self.ds_name}] Invalid bbox[{bbox_idx}] type: {type(bbox)}, value: {bbox}. "
                        f"Expected list/tuple/array of 4 values. Skipping this sample."
                    )
                    return None  # Return None to indicate invalid bbox, caller should skip
                
                # Convert to list if the whole bbox is a numpy array
                if isinstance(bbox, np.ndarray):
                    bbox = bbox.tolist()
                
                # CRITICAL FIX: Handle nested bbox format like [[x0, y0, x1, y1]]
                # Keep unwrapping single-element containers
                while isinstance(bbox, (list, tuple, np.ndarray)) and len(bbox) == 1 and isinstance(bbox[0], (list, tuple, np.ndarray)):
                    bbox = bbox[0]
                
                # Now robustly coerce bbox to a flat array of numeric values
                bbox_arr = np.array(bbox, dtype=np.float64).reshape(-1)
                if bbox_arr.size != 4:
                    logger.warning(
                        f"[{self.ds_name}] Invalid bbox[{bbox_idx}] flattened size: {bbox_arr.size}, "
                        f"value: {bbox}. Expected 4 values [x0, y0, x1, y1]. Skipping this sample."
                    )
                    return None  # Caller should skip this sample
                
                # Extract scalar coordinates; using .item() guarantees 0-d scalars
                x0, y0, x1, y1 = [float(v.item() if isinstance(v, np.generic) else v) for v in bbox_arr]
                
            except (ValueError, TypeError) as e:
                logger.warning(
                    f"[{self.ds_name}] Error unpacking/coercing bbox[{bbox_idx}]: {bbox}, error: {e}. "
                    f"Skipping this sample."
                )
                return None  # Return None to indicate invalid bbox, caller should skip

            # Scale to 14by14 grid
            x0_grid = max(0, min(int(np.floor(x0 * w)), w-1))
            x1_grid = max(0, min(int(np.ceil (x1 * w)), w))
            y0_grid = max(0, min(int(np.floor(y0 * h)), h-1))
            y1_grid = max(0, min(int(np.ceil (y1 * h)), h))


            # Map to 28by28 grid
            x0_token = x0_grid // 2
            x1_token = (x1_grid + 1) // 2
            y0_token = y0_grid // 2
            y1_token = (y1_grid + 1) // 2

            idxs = [
                int(yy * W2 + xx)
                for yy in range(y0_token, y1_token)
                for xx in range(x0_token, x1_token)
            ]

            token_idxs.append(idxs)

        return token_idxs

    def _enable_worker_distributed(self):
        if (
            self.distributed_mode
            and not self.worker_distributed
            and self.worker_id is not None
        ):
            self.worker_distributed = True
            self.raw_data = self.raw_data[self.worker_id::self.num_workers]
            logger.info(f'worker_distributed is enabled, {self.num_workers=}, {len(self.raw_data)=}')

    def __iter__(self):
        self._enable_worker_distributed()
        start_idx = 0
        assert self.worker_state_key is not None
        if self.worker_state_key in self._state_dict and len(self._state_dict[self.worker_state_key]) > 0:
            start_idx = self._state_dict[self.worker_state_key]['current_idx']

            self._state_dict.pop(self.worker_state_key)


        if self.worker_id == 0:
            logger.info(
                f'[{self.ds_name}] [Worker id {self.worker_id}] '
                f'begin to iter with {start_idx=}'
        )

        for i in range(start_idx,len(self.raw_data)):


            sources = self.raw_data[i]

            is_video = False

            processor = self.processor

            videos = None
            grid_key = "image_grid_thw"
            pixel_key = "pixel_values"
            
            image_files = sources["image"]
            image_folder = self.image_folder

            # Normalize image_files to a flat list of strings
            if isinstance(image_files, str):
                image_files = [image_files]
            elif isinstance(image_files, list):
                # Flatten nested lists if any
                flattened = []
                for item in image_files:
                    if isinstance(item, str):
                        flattened.append(item)
                    elif isinstance(item, list):
                        # Handle nested lists
                        flattened.extend([x for x in item if isinstance(x, str)])
                    else:
                        # Skip non-string items
                        logger.warning(f"[{self.ds_name}] Skipping non-string image item: {type(item)}")
                image_files = flattened
            else:
                logger.warning(f"[{self.ds_name}] Unexpected image_files type: {type(image_files)}, converting to list")
                image_files = [str(image_files)] if image_files is not None else []

            images = []
            skipped_images = []

            # Get dataset name from sources if available
            dataset_name = sources.get('dataset', None)
            
            # Helper function to load images with given max_pixel
            def load_images_with_resolution(max_pixel):
                imgs = []
                for image_file in image_files:
                    if not isinstance(image_file, str):
                        continue
                    mapped_path = map_image_path(image_file, image_folder, dataset_name)
                    image_info = get_image_info(mapped_path, self.image_min_pixel, max_pixel, self.image_resized_w, self.image_resized_h)
                    if image_info is not None:
                        imgs.append(image_info)
                return imgs
            
            # First pass: load images with default resolution
            current_max_pixel = self.image_max_pixel
            images = load_images_with_resolution(current_max_pixel)
            
            # Track skipped images for logging
            for image_file in image_files:
                if not isinstance(image_file, str):
                    logger.warning(f"[{self.ds_name}] Skipping non-string image_file: {type(image_file)}")
                    continue
                mapped_path = map_image_path(image_file, image_folder, dataset_name)
                if mapped_path is None or not os.path.exists(mapped_path):
                    display_path = mapped_path if mapped_path is not None else image_file
                    skipped_images.append(display_path)
                    logger.warning(f"[{self.ds_name}] Skipping missing image - original: {image_file}, searched path: {display_path}")
            
            # Skip this sample if no valid images found (required for 8-card: some workers get shards with all missing images)
            if len(images) == 0:
                logger.warning(
                    f"[{self.ds_name}] Skipping sample {i} - all images missing: {skipped_images}. "
                    f"worker_id={getattr(self, 'worker_id', '?')}. Required to avoid IndexError in bbox_to_token_idxs with empty image_grid_thw."
                )
                continue

            # Extract LVR tokens and estimate sequence length
            image_grid_thw = processor(text=[""], images=images, videos=videos, padding=False, do_resize=False, return_tensors='pt')['image_grid_thw']
            
            # === DYNAMIC RESOLUTION ADJUSTMENT ===
            # Estimate sequence length: image_tokens + text_tokens + lvr_tokens
            # image_tokens ≈ T * H/2 * W/2 for Qwen2-VL
            _, h, w = image_grid_thw[0].tolist()
            estimated_image_tokens = h * w // 4  # Qwen2-VL uses 2x2 merging
            num_bboxes = len(sources.get('bboxes', []))
            num_lvr_tokens = num_bboxes * (self.fixed_num_of_lvr_tokens or 8)  # Default 8 tokens per bbox
            # Estimate text tokens (system + question + answer) ~300-500 tokens typically
            estimated_text_tokens = 500
            estimated_total = estimated_image_tokens + num_lvr_tokens + estimated_text_tokens
            
            # If sequence would exceed 90% of max_packed_tokens, reduce image resolution
            max_allowed = int(self.max_packed_tokens * 0.85)  # 85% threshold for safety
            resolution_reduced = False
            
            while estimated_total > max_allowed and current_max_pixel > self.min_image_max_pixel:
                # Calculate target image tokens
                target_image_tokens = max_allowed - num_lvr_tokens - estimated_text_tokens
                if target_image_tokens < 500:  # Minimum image tokens
                    break
                
                # Reduce resolution by ratio (pixel count is proportional to token count)
                reduction_ratio = target_image_tokens / estimated_image_tokens
                new_max_pixel = int(current_max_pixel * reduction_ratio * 0.9)  # Extra 10% reduction for safety
                new_max_pixel = max(new_max_pixel, self.min_image_max_pixel)
                
                if new_max_pixel >= current_max_pixel:
                    break  # Cannot reduce further
                
                if self.worker_id == 0 or self.worker_id is None:
                    logger.info(
                        f"[{self.ds_name}] 📐 Dynamic resolution adjustment: "
                        f"estimated_total={estimated_total} > max_allowed={max_allowed}, "
                        f"reducing max_pixel from {current_max_pixel} to {new_max_pixel}"
                    )
                
                current_max_pixel = new_max_pixel
                images = load_images_with_resolution(current_max_pixel)
                
                if len(images) == 0:
                    break
                
                image_grid_thw = processor(text=[""], images=images, videos=videos, padding=False, do_resize=False, return_tensors='pt')['image_grid_thw']
                _, h, w = image_grid_thw[0].tolist()
                estimated_image_tokens = h * w // 4
                estimated_total = estimated_image_tokens + num_lvr_tokens + estimated_text_tokens
                resolution_reduced = True
            
            if resolution_reduced and (self.worker_id == 0 or self.worker_id is None):
                logger.info(
                    f"[{self.ds_name}] ✅ Resolution reduced: final estimated_total={estimated_total}, "
                    f"final max_pixel={current_max_pixel}, image_grid_thw={image_grid_thw[0].tolist()}"
                )
            
            # === SKIP OVERLONG SAMPLES ===
            # If still too long after reducing to minimum resolution, skip this sample
            if estimated_total > max_allowed:
                logger.warning(
                    f"[{self.ds_name}] ⏭️ Skipping overlong sample {i}: "
                    f"estimated_total={estimated_total} > max_allowed={max_allowed} "
                    f"even after reducing to min_pixel={current_max_pixel}. "
                    f"image_grid_thw={image_grid_thw[0].tolist()}, num_bboxes={num_bboxes}"
                )
                continue
            
            lvr_token_idxs_list = self.bbox_to_token_idxs(sources['bboxes'], image_grid_thw)
            
            # Skip this sample if bbox_to_token_idxs returns None (invalid bbox detected)
            if lvr_token_idxs_list is None:
                logger.warning(
                    f"[{self.ds_name}] Skipping sample {i} - invalid bbox format detected. "
                    f"bboxes={sources.get('bboxes', [])}"
                )
                continue
            
            # === ALIGN LVR TOKENS WITH <lvr> PLACEHOLDERS ===
            # For viscot_x 数据集，常见情况是 bboxes 有多个框，但 prompt 里只放了 1 个 LVR_PLACEHOLDER。
            # replace_lvr_tokens 在这种情况下只会用前 N 个 lvr_token_idxs_list 元素生成 <lvr> token，
            # 但我们这里的 lvr_tokens 仍然包含所有 bbox，对应关系就会错位，造成数量不匹配。
            #
            # 这里按原始 conversations 里出现的 LVR_PLACEHOLDER 次数来裁剪 lvr_token_idxs_list，
            # 保证：
            #   len(lvr_token_idxs_list_used) == num_placeholders
            # 从而使 input_ids 里的 <lvr> 个数和 lvr_tokens 的总长度一致。
            if self.fixed_num_of_lvr_tokens is None:
                num_placeholders = 0
                raw_convs = self.raw_data[i].get('conversations', [])
                for conv in raw_convs:
                    value = conv.get('value', '')
                    if isinstance(value, str):
                        num_placeholders += value.count(LVR_PLACEHOLDER)
                if num_placeholders == 0:
                    # 没有 LVR_PLACEHOLDER，则不应该有任何 lvr_tokens
                    logger.warning(
                        f"[{self.ds_name}] Sample {i} has {len(lvr_token_idxs_list)} bboxes "
                        f"but no LVR_PLACEHOLDER in conversations. "
                        f"Clearing lvr_token_idxs_list to avoid mismatch."
                    )
                    lvr_token_idxs_list = []
                elif len(lvr_token_idxs_list) > num_placeholders:
                    logger.warning(
                        f"[{self.ds_name}] Sample {i} has {len(lvr_token_idxs_list)} bbox groups "
                        f"but only {num_placeholders} LVR_PLACEHOLDER occurrences. "
                        f"Using first {num_placeholders} groups to align with prompts."
                    )
                    lvr_token_idxs_list = lvr_token_idxs_list[:num_placeholders]
                elif len(lvr_token_idxs_list) < num_placeholders:
                    # 占位符比 bbox 多，这样很难保证一一对应，直接跳过样本以避免崩溃
                    logger.warning(
                        f"[{self.ds_name}] Sample {i} has only {len(lvr_token_idxs_list)} bbox groups "
                        f"but {num_placeholders} LVR_PLACEHOLDER occurrences. "
                        f"Skipping this sample to avoid lvr_tokens mismatch."
                    )
                    continue
            
            # CRITICAL: Debug log immediately after bbox_to_token_idxs returns
            # Check for empty groups in lvr_token_idxs_list
            for bbox_idx, token_idxs in enumerate(lvr_token_idxs_list):
                if len(token_idxs) == 0:
                    worker_id_str = getattr(self, 'worker_id', 'unknown')
                    logger.error(
                        f"[{self.ds_name}] ❌ CRITICAL: bbox_to_token_idxs returned empty token_idxs "
                        f"for bbox_idx={bbox_idx}, bbox={sources.get('bboxes', [])[bbox_idx] if bbox_idx < len(sources.get('bboxes', [])) else 'unknown'}, "
                        f"image_grid_thw={image_grid_thw[0].tolist()}, "
                        f"data_idx={i}, worker_id={worker_id_str}. "
                        f"This should NEVER happen after our fixes!"
                    )
                    print(f"[BBOX_RETURNED_EMPTY] bbox_idx={bbox_idx}, bbox={sources.get('bboxes', [])[bbox_idx] if bbox_idx < len(sources.get('bboxes', [])) else 'unknown'}, "
                          f"image_grid_thw={image_grid_thw[0].tolist()}, data_idx={i}, worker_id={worker_id_str}, "
                          f"lvr_token_idxs_list={lvr_token_idxs_list}", flush=True)
            
            # Validate lvr_token_idxs_list - check for empty token lists
            # TEMPORARILY DISABLED: Skip logic disabled for testing
            if len(lvr_token_idxs_list) == 0:
                if self.worker_id == 0:
                    logger.warning(
                        f"[{self.ds_name}] TEMPORARILY DISABLED: Empty lvr_token_idxs_list for data_idx={i}, "
                        f"bboxes={sources.get('bboxes', [])}, "
                        f"image_grid_thw={image_grid_thw[0].tolist()}. "
                        f"Would skip but continuing anyway."
                    )
                # continue  # TEMPORARILY DISABLED
                # Create empty list to continue processing
                lvr_token_idxs_list = [[]]

            # Save bboxes before sources is reassigned to a list
            bboxes = sources.get('bboxes', [])
            
            sources = copy.deepcopy(llava_to_openai_lvr(
                sources['conversations'], is_video=is_video, lvr_token_idxs_list=lvr_token_idxs_list,
                latent_end_token=self.latent_end_token, fixed_num_of_lvr_tokens=self.fixed_num_of_lvr_tokens,
            ))

            all_input_ids = [] 
            all_labels = []
            all_pixel_values = []
            all_image_grid_thw = []
            all_second_gird = []

            # Qwen2-VL uses a default system message so I've added this.
            if len(SYSTEM_MESSAGE) > 0:
                system_message = f"{DEFAULT_IM_START_TOKEN}system\n{SYSTEM_MESSAGE}{DEFAULT_IM_END_TOKEN}\n"
                system_message_input_ids = processor.tokenizer(system_message, add_special_tokens=False, return_tensors='pt')['input_ids']
                system_labels = torch.full_like(system_message_input_ids, IGNORE_INDEX) 
                
                all_input_ids.append(system_message_input_ids.squeeze(0))
                all_labels.append(system_labels.squeeze(0))

            for _, j in enumerate(range(0, len(sources), 2)):
                user_input = sources[j]
                gpt_response = sources[j + 1]

                user_input = f"{DEFAULT_IM_START_TOKEN}{user_input['role']}\n{user_input['content']}{DEFAULT_IM_END_TOKEN}\n{DEFAULT_IM_START_TOKEN}{gpt_response['role']}\n"
                gpt_response = f"{gpt_response['content']}{DEFAULT_IM_END_TOKEN}\n"
                
                if DEFAULT_IMAGE_TOKEN in user_input:
                    inputs = processor(text=[user_input], images=images, videos=videos, padding=False, do_resize=False, return_tensors='pt')
                    prompt_input_ids = inputs['input_ids']
                    all_pixel_values.append(inputs[pixel_key])
                    all_image_grid_thw.append(inputs[grid_key])

                else:
                    prompt_input_ids = processor.tokenizer(user_input, add_special_tokens=False, padding=False, return_tensors='pt')['input_ids']

                # filling the response with bboxes
                
                response_input_ids = processor.tokenizer(gpt_response, add_special_tokens=False, padding=False, return_tensors='pt')['input_ids']

                input_ids = torch.cat([prompt_input_ids, response_input_ids], dim=1).squeeze(0)
                labels = torch.cat(
                    [
                        torch.tensor([IGNORE_INDEX] * len(prompt_input_ids[0])),  
                        response_input_ids.squeeze(0),
                    ],
                    dim=0,
                )

                all_input_ids.append(input_ids)
                all_labels.append(labels)
            
            # There is no need for eos or bos tokens in the input_ids
            # Qwen2-VL does not use them
            input_ids = torch.cat(all_input_ids, dim=0).to(torch.long)
            labels = torch.cat(all_labels, dim=0).to(torch.long)

            # eos_token_id = processor.tokenizer.convert_tokens_to_ids(DEFAULT_IM_END_TOKEN)
            # input_ids, labels = truncate_sequence(input_ids, labels, self.max_length, eos_token_id)

            attention_mask = (input_ids > -1000000).to(torch.long)

            '''Again, we only allow 1-image-multi-area at the moment'''
            lvr_tokens = []
            empty_lvr_tokens_found = False
            
            # Debug: Log lvr_token_idxs_list before creating tensors
            worker_id_str = getattr(self, 'worker_id', 'unknown')
            # print(f"[BEFORE_CREATE_LVR_TOKENS] data_idx={i}, worker_id={worker_id_str}, "
            #       f"lvr_token_idxs_list_length={len(lvr_token_idxs_list)}, "
            #       f"lvr_token_idxs_list={[[len(g) for g in lvr_token_idxs_list]]}, "
            #       f"image_grid_thw={image_grid_thw[0].tolist()}", flush=True)
            
            for group_idx, group in enumerate(lvr_token_idxs_list):
                if len(group) == 0:
                    empty_lvr_tokens_found = True
                    # Always log, not just worker_id == 0, to catch all cases
                    logger.error(
                        f"[{self.ds_name}] ❌ CRITICAL: Empty lvr_token_idxs_list[{group_idx}] for bbox {bboxes[group_idx] if group_idx < len(bboxes) else 'unknown'}, "
                        f"image_grid_thw={image_grid_thw[0].tolist()}, "
                        f"data_idx={i}, worker_id={self.worker_id}. This will cause NaN loss!"
                    )
                    print(f"[EMPTY_GROUP_DETECTED] group_idx={group_idx}, data_idx={i}, worker_id={worker_id_str}, "
                          f"bbox={bboxes[group_idx] if group_idx < len(bboxes) else 'unknown'}, "
                          f"image_grid_thw={image_grid_thw[0].tolist()}", flush=True)
                
                token_tensor = torch.tensor(group, dtype=torch.int)
                # print(f"[CREATE_LVR_TOKEN] group_idx={group_idx}, group_length={len(group)}, "
                #       f"tensor_shape={token_tensor.shape}, tensor_numel={token_tensor.numel()}, "
                #       f"empty={token_tensor.numel() == 0}, data_idx={i}, worker_id={worker_id_str}", flush=True)
                lvr_tokens.append(token_tensor)
            
            # CRITICAL: Double-check after creating tensors - skip if ANY lvr_tokens are empty
            # This prevents NaN loss from empty lvr_tokens
            for token_idx, token_tensor in enumerate(lvr_tokens):
                if token_tensor.numel() == 0:
                    empty_lvr_tokens_found = True
                    break
            
            # TEMPORARILY DISABLED: Skip logic disabled for testing
            if empty_lvr_tokens_found:
                # Always log but don't skip, regardless of worker_id
                logger.error(
                    f"[{self.ds_name}] ❌ TEMPORARILY DISABLED: Empty lvr_tokens detected for data_idx={i}, worker_id={self.worker_id}! "
                    f"Bboxes: {bboxes}, "
                    f"image_grid_thw={image_grid_thw[0].tolist()}, "
                    f"lvr_token_idxs_list lengths: {[len(g) for g in lvr_token_idxs_list]}, "
                    f"lvr_tokens numel: {[t.numel() for t in lvr_tokens]}. "
                    f"Would skip but continuing anyway - may cause NaN loss."
                )
                print(f"[SKIP_SAMPLE_DISABLED] data_idx={i}, worker_id={self.worker_id}, bboxes={bboxes}, "
                      f"image_grid_thw={image_grid_thw[0].tolist()}, "
                      f"lvr_tokens_empty=True", flush=True)
                # continue  # TEMPORARILY DISABLED - Skip this sample immediately
                # Replace empty tensors with a dummy token [0] to avoid downstream errors
                for idx, token_tensor in enumerate(lvr_tokens):
                    if token_tensor.numel() == 0:
                        lvr_tokens[idx] = torch.tensor([0], dtype=torch.int)
                        logger.warning(f"[{self.ds_name}] Replaced empty lvr_tokens[{idx}] with dummy token [0]")

            cropped_bbox_images = None
            if len(images) > 0 and len(bboxes) > 0 and len(bboxes) == len(lvr_token_idxs_list):
                crops = []
                img = images[0]
                for bbox in bboxes:
                    b = bbox
                    while isinstance(b, (list, tuple)) and len(b) == 1 and isinstance(b[0], (list, tuple)):
                        b = b[0]
                    if isinstance(b, (list, tuple)) and len(b) >= 4:
                        crops.append(crop_bbox_and_resize(img, list(b)[:4], size=self.dit_crop_size))
                if crops:
                    cropped_bbox_images = torch.stack(crops, dim=0)

            data_dict = dict(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                lvr_tokens=lvr_tokens,
            )
            if cropped_bbox_images is not None:
                data_dict['cropped_bbox_images'] = cropped_bbox_images

            if pixel_key and grid_key:
                pixel_values = torch.cat(all_pixel_values, dim=0)
                image_thw = torch.cat(all_image_grid_thw, dim=0)
                data_dict[pixel_key] = pixel_values
                data_dict[grid_key] = image_thw

            if len(all_second_gird) > 0:
                second_gird = all_second_gird
                data_dict["second_per_grid_ts"] = second_gird

            # Add image_flags, which is required by the packer
            # data_dict['image_flags'] = torch.ones(data_dict['pixel_values'].shape[0], dtype=torch.long)
            data_dict['input_lengths'] = torch.tensor([input_ids.size(0)])
            # data_dict['question_id'] = self.raw_data[i]['question_id']
            
            # Add debug information for troubleshooting
            # Extract question and answer from conversations
            debug_question = ""
            debug_answer = ""
            if 'conversations' in self.raw_data[i]:
                convs = self.raw_data[i]['conversations']
                for conv in convs:
                    if conv.get('from') == 'human' or conv.get('role') == 'user':
                        debug_question = conv.get('value', '')[:200]  # Limit length
                    elif conv.get('from') == 'gpt' or conv.get('role') == 'assistant':
                        debug_answer = conv.get('value', '')[:200]  # Limit length
            
            # Extract image paths
            debug_image_paths = image_files if isinstance(image_files, list) else [image_files]
            # Extract bboxes
            debug_bboxes = self.raw_data[i].get('bboxes', [])
            # Extract data index
            debug_data_idx = i
            
            data_dict['_debug_question'] = debug_question
            data_dict['_debug_answer'] = debug_answer
            data_dict['_debug_image_paths'] = debug_image_paths
            data_dict['_debug_bboxes'] = debug_bboxes
            data_dict['_debug_data_idx'] = debug_data_idx
            
            # CRITICAL VALIDATION: Check consistency between <lvr> tokens in input_ids and lvr_tokens.
            # NOTE:
            #   - In variable-length LVR mode (fixed_num_of_lvr_tokens is None), each <lvr> corresponds to ONE visual token index,
            #     so we can safely require:
            #         num_lvr_in_input_ids == total_lvr_tokens.
            #   - In fixed-length latent mode (self.fixed_num_of_lvr_tokens > 0, e.g. BoxFeatureResampler / DiT recon),
            #     each bbox uses `fixed_num_of_lvr_tokens` <lvr> slots, while lvr_tokens stores a *set* of visual
            #     token indices per bbox. In this case, the only meaningful check is:
            #         num_lvr_in_input_ids == fixed_num_of_lvr_tokens * num_bboxes
            #     where num_bboxes == len(lvr_tokens). Comparing num_lvr_in_input_ids with the *sum* of all
            #     visual token indices (total_lvr_tokens) is incorrect and will always "mismatch".
            lvr_token_id = self.processor.tokenizer.convert_tokens_to_ids(LVR_TOKEN)
            num_lvr_in_input_ids = (input_ids == lvr_token_id).sum().item()

            if self.fixed_num_of_lvr_tokens:
                # Fixed-N latent mode (BoxFeatureResampler / DiT recon): check per-bbox slot count.
                if num_lvr_in_input_ids % self.fixed_num_of_lvr_tokens != 0:
                    logger.error(
                        f"[{self.ds_name}] ❌ CRITICAL: lvr_tokens slot mismatch in fixed-N mode! "
                        f"input_ids has {num_lvr_in_input_ids} <lvr> tokens which is not divisible by "
                        f"fixed_num_of_lvr_tokens={self.fixed_num_of_lvr_tokens}. "
                        f"data_idx={i}, worker_id={self.worker_id}, "
                        f"image_files={image_files[:2] if len(image_files) > 2 else image_files}, "
                        f"bboxes={bboxes[:2] if len(bboxes) > 2 else bboxes}, "
                        f"image_grid_thw={image_grid_thw[0].tolist() if image_grid_thw is not None and len(image_grid_thw) > 0 else 'None'}. "
                        f"SKIPPING this sample to prevent forward crash."
                    )
                    continue

                expected_num_bboxes = num_lvr_in_input_ids // self.fixed_num_of_lvr_tokens
                actual_num_bboxes = len(lvr_tokens)
                if actual_num_bboxes != expected_num_bboxes:
                    logger.error(
                        f"[{self.ds_name}] ❌ CRITICAL: lvr_tokens bbox count mismatch in fixed-N mode! "
                        f"input_ids has {num_lvr_in_input_ids} <lvr> tokens -> "
                        f"expected {expected_num_bboxes} bbox groups "
                        f"(fixed_num_of_lvr_tokens={self.fixed_num_of_lvr_tokens}), "
                        f"but lvr_tokens has {actual_num_bboxes} groups. "
                        f"data_idx={i}, worker_id={self.worker_id}, "
                        f"image_files={image_files[:2] if len(image_files) > 2 else image_files}, "
                        f"bboxes={bboxes[:2] if len(bboxes) > 2 else bboxes}, "
                        f"image_grid_thw={image_grid_thw[0].tolist() if image_grid_thw is not None and len(image_grid_thw) > 0 else 'None'}. "
                        f"SKIPPING this sample to prevent forward crash."
                    )
                    continue
            else:
                # Variable-length LVR mode: original strict 1:1 check.
                total_lvr_tokens = sum(
                    t.numel() if isinstance(t, torch.Tensor) else len(t) for t in lvr_tokens
                )
                if num_lvr_in_input_ids != total_lvr_tokens:
                    logger.error(
                        f"[{self.ds_name}] ❌ CRITICAL: lvr_tokens count mismatch! "
                        f"input_ids has {num_lvr_in_input_ids} <lvr> tokens but lvr_tokens has {total_lvr_tokens} tokens. "
                        f"data_idx={i}, worker_id={self.worker_id}, "
                        f"image_files={image_files[:2] if len(image_files) > 2 else image_files}, "
                        f"bboxes={bboxes[:2] if len(bboxes) > 2 else bboxes}, "
                        f"image_grid_thw={image_grid_thw[0].tolist() if image_grid_thw is not None and len(image_grid_thw) > 0 else 'None'}. "
                        f"SKIPPING this sample to prevent forward crash."
                    )
                    continue  # Skip this problematic sample
            
            # Instead of returning, we yield the processed dictionary
            yield data_dict


"""
    Below is a adaptation of 
    https://github.com/OpenGVLab/InternVL/blob/main/internvl_chat/internvl/train/dataset_packed.py
    We adopted a greedy data packing logic
"""

# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import bisect
import copy
import logging
from collections import defaultdict
from typing import List, Union

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import IterableDataset, get_worker_info


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class PackedDataset(IterableDataset):
    def __init__(
        self,
        tokenizer,
        data_rank,
        data_world_size,
        datasets: List,
        dataset_weight: List[int] = None,
        num_images_expected: int = 6,
        max_packed_tokens: int = 4096,
        max_buffer_size: int = 100,
        log_freq: int = 1000000,
        strict_mode: bool = False,
        debug_mode: bool = False,
        replacement: bool = True,
        allow_overflow: bool = True,
        allow_empty_data: bool = False,
        allow_deduplicated_ds_name: bool = False,
        # long_seq_cut: int = 25600,           # A single instance longer than this will be truncated
        long_seq_threshold: int = 6144,      # Instance longer than this will be individually processed
        max_instance_per_batch: int = 4,     # max num of instance per device
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.data_rank = data_rank
        self.data_world_size = data_world_size
        self.datasets = datasets
        self.num_images_expected = num_images_expected
        self.max_buffer_size = max_buffer_size
        self.log_freq = log_freq
        self.strict_mode = strict_mode
        self.debug_mode = debug_mode
        self.replacement = replacement
        self.allow_overflow = allow_overflow
        self.allow_empty_data = allow_empty_data

        self.max_packed_tokens = max_packed_tokens
        # self.long_seq_cut = long_seq_cut
        self.long_seq_threshold = long_seq_threshold
        self.max_instance_per_batch = max_instance_per_batch

        self.img_start_token_id = self.tokenizer.convert_tokens_to_ids(VISION_START_TOKEN)
        self.img_token_id = self.tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_TOKEN)
        self.img_end_token_id = self.tokenizer.convert_tokens_to_ids(VISION_END_TOKEN)
        self.lvr_token_id = self.tokenizer.convert_tokens_to_ids(LVR_TOKEN)

        assert self.img_start_token_id != self.tokenizer.unk_token_id
        assert self.img_token_id != self.tokenizer.unk_token_id
        assert self.img_end_token_id != self.tokenizer.unk_token_id

        if dataset_weight is None:
            dataset_weight = [len(d) for d in datasets]

        self.datasets_orig = datasets
        self.dataset_weight_orig = [w / sum(dataset_weight) for w in dataset_weight]


        self.datasets = [ds for ds in self.datasets_orig]
        self.dataset_weight = [w for w in self.dataset_weight_orig]

        # lazy init
        self.worker_id = None
        self.worker_state_key = None
        self.dataset_iter_list = None
        self._state_dict = {
            'sample_info': {d.ds_name:0 for d in self.datasets},
        }

        self.worker_custom_infos = None

        ds_name_list = [d.ds_name for d in self.datasets]
        if not allow_deduplicated_ds_name:
            assert len(ds_name_list) == len(set(ds_name_list)), f'deduplicated ds_name: {ds_name_list}'

        for ds in self.datasets:
            self._state_dict[ds.ds_name] = {}

        if get_rank() == 0:
            logger.info(
                f'Loaded dataset to pack: {ds_name_list}, '
                f'{self.num_images_expected=}, {self.max_packed_tokens=}, '
                f'{self.replacement=}, {self.allow_overflow=}',
            )

            temp = []
            for ds, ds_w in zip(self.datasets, self.dataset_weight):
                temp.append(f'{ds.ds_name:<25}: {ds_w*100:.2f}%')
            temp = '\n'.join(temp)
            logger.info(
                f'Sampling prob for each dataset:\n{temp}'
            )

        if self.allow_empty_data:
            logger.warning('allow_empty_data is enabled, note that empty data may be generated!')

    def load_state_dict(self, state_dict, custom_infos=None):

        self.worker_custom_infos = custom_infos

        self._state_dict.update(state_dict)
        for ds in self.datasets:
            if ds.ds_name in self._state_dict:
                ds.load_state_dict(self._state_dict[ds.ds_name])
                logger.info(f'{ds.ds_name=} is resumed.')
            else:
                logger.warning(f'{ds.ds_name=} is not resumed.')

    def _should_log(self):
        worker_id = 0 if get_worker_info() is None else get_worker_info().id
        num_workers = 1 if get_worker_info() is None else get_worker_info().num_workers

        worker_id = num_workers * get_rank() + worker_id
        num_workers = num_workers * get_world_size()

        return worker_id == 0

    def next_data(self, current_dataset_idx):
        while True:
            try:
                current_sample = next(self.dataset_iter_list[current_dataset_idx])
                
                # CRITICAL: Check for empty lvr_tokens before adding to buffer
                # This prevents NaN loss from propagating through packing
                if 'lvr_tokens' in current_sample:
                    lvr_tokens = current_sample['lvr_tokens']
                    if isinstance(lvr_tokens, list):
                        empty_found = False
                        for token_group in lvr_tokens:
                            if isinstance(token_group, torch.Tensor):
                                if token_group.numel() == 0:
                                    empty_found = True
                                    break
                            elif isinstance(token_group, list):
                                if len(token_group) == 0:
                                    empty_found = True
                                    break
                        
                        # TEMPORARILY DISABLED: Skip logic disabled for testing
                        if empty_found:
                            # Always log but don't skip, not just when _should_log()
                            logger.error(
                                f"[PackedDataset] ❌ TEMPORARILY DISABLED: Empty lvr_tokens detected in sample from {self.datasets[current_dataset_idx].ds_name}! "
                                f"worker_id={self.worker_id}, data_rank={self.data_rank}, "
                                f"lvr_tokens numel: {[t.numel() if isinstance(t, torch.Tensor) else len(t) for t in lvr_tokens]}. "
                                f"Would skip but continuing anyway - may cause NaN loss."
                            )
                            print(f"[SKIP_SAMPLE_PACKED_DISABLED] worker_id={self.worker_id}, data_rank={self.data_rank}, "
                                  f"ds_name={self.datasets[current_dataset_idx].ds_name}, "
                                  f"lvr_tokens_empty=True", flush=True)
                            # continue  # TEMPORARILY DISABLED - Skip this sample and get next one
                            # Replace empty tensors with dummy token [0] to avoid downstream errors
                            for idx, token_group in enumerate(lvr_tokens):
                                if isinstance(token_group, torch.Tensor) and token_group.numel() == 0:
                                    lvr_tokens[idx] = torch.tensor([0], dtype=torch.int)
                                elif isinstance(token_group, list) and len(token_group) == 0:
                                    lvr_tokens[idx] = torch.tensor([0], dtype=torch.int)
                
                break  # Exit loop if successful
            except StopIteration:
                if self.replacement:
                    # logger.info(f'[Worker id {self.worker_id}] Dataset {self.datasets[current_dataset_idx].ds_name} is exhausted, restart it.')
                    try:
                        self.dataset_iter_list[current_dataset_idx] = iter(self.datasets[current_dataset_idx])
                        current_sample = next(self.dataset_iter_list[current_dataset_idx])
                        break
                    except:
                        # logger.error(f'{self.worker_id=} Fail to get any data from {self.datasets[current_dataset_idx].ds_name}! length={len(self.datasets)}')
                        self.datasets.pop(current_dataset_idx)
                        self.dataset_iter_list.pop(current_dataset_idx)
                        self.dataset_weight.pop(current_dataset_idx)

                        if len(self.datasets) == 0:
                            raise StopIteration
                        current_dataset_idx = np.random.choice(len(self.datasets))
                else:
                    # logger.error(f'{self.worker_id=} Fail to get any data from {self.datasets[current_dataset_idx].ds_name}! length={len(self.datasets)}')
                    self.datasets.pop(current_dataset_idx)
                    self.dataset_iter_list.pop(current_dataset_idx)
                    self.dataset_weight.pop(current_dataset_idx)

                    if len(self.datasets) == 0:
                        raise StopIteration
                    current_dataset_idx = np.random.choice(len(self.datasets))
            except Exception as e:
                import traceback
                import sys
                error_msg = f'worker_id{self.worker_id} data_rank={self.data_rank} data_world_size={self.data_world_size} Unexpected error: {type(e).__name__}: {str(e)}'
                traceback_msg = f'Traceback: {traceback.format_exc()}'
                # Use both print and logger to ensure error is visible
                print(f'ERROR: {error_msg}', file=sys.stderr, flush=True)
                print(f'ERROR: {traceback_msg}', file=sys.stderr, flush=True)
                sys.stderr.flush()
                logger.error(error_msg)
                logger.error(traceback_msg)
                if len(self.datasets) == 0:
                    raise StopIteration
                current_dataset_idx = np.random.choice(len(self.datasets))

        current_ds_name = self.datasets[current_dataset_idx].ds_name

        if self.worker_state_key not in self._state_dict[current_ds_name]:
            self._state_dict[current_ds_name][self.worker_state_key] = {}

        meta_info = current_sample.pop('meta_info', {})
        self._state_dict[current_ds_name][self.worker_state_key].update(**meta_info)
        self._state_dict['sample_info'][self.datasets[current_dataset_idx].ds_name] += 1
        return current_sample

    def find_buffer(self, buffer_list, new_sample):
        # NOTE: use `bisect` to search might be faster
        #  deleted the condition on # of images

        find = False
        find_idx = -1

        # if we see a new sample > LST, we need it to be in the buffer list,
        # instead of concatenating it to any existing buffer
        if new_sample['input_ids'].size(0) >= self.long_seq_threshold:
            return None

        for buffer_idx, buffer in enumerate(buffer_list):
            num_merged_tokens = new_sample['input_ids'].size(0) + buffer['input_ids'].size(0)
            num_instance_buffer = buffer['input_lengths'].size(0)
            if num_instance_buffer + 1 <= self.max_instance_per_batch:
                if num_merged_tokens <= self.max_packed_tokens:
                    find = True
                    find_idx = buffer_idx
                    break

                if self.allow_overflow and len(buffer_list) >= self.max_buffer_size // 2:
                    find = True
                    find_idx = buffer_idx

        if find:
            return buffer_list.pop(find_idx)
        else:
            return None

    def update_buffer(self, buffer, new_sample):
        if buffer is None:
            new_sample['data_index'] = torch.zeros_like(new_sample['input_ids'])
            return new_sample

        new_sample['data_index'] = torch.ones_like(new_sample['input_ids']) + buffer['data_index'][-1].item()

        # Handle debug keys separately - they should be lists
        buffer_keys = set(k for k in buffer.keys() if not k.startswith('_debug_'))
        sample_keys = set(k for k in new_sample.keys() if not k.startswith('_debug_'))
        assert buffer_keys == sample_keys, f"Key mismatch: buffer={buffer_keys}, sample={sample_keys}"
        
        for k in buffer:
            if k.startswith('_debug_'):
                # For debug keys, append to list
                if not isinstance(buffer[k], list):
                    # Convert to list if not already
                    buffer[k] = [buffer[k]]
                if k in new_sample:
                    buffer[k].append(new_sample[k])
            elif k == 'lvr_tokens':
                # CRITICAL: Check for empty lvr_tokens before adding to buffer
                if isinstance(new_sample[k], list):
                    for idx, lvr_token_group in enumerate(new_sample[k]):
                        if isinstance(lvr_token_group, torch.Tensor) and lvr_token_group.numel() == 0:
                            logger.error(
                                f"[PackedDataset] ❌ CRITICAL: Empty lvr_tokens[{idx}] detected in new_sample before update_buffer! "
                                f"worker_id={self.worker_id}, data_rank={self.data_rank}. "
                                f"Using fallback token 0."
                            )
                            print(f"[UPDATE_BUFFER_EMPTY_LVR] idx={idx}, worker_id={self.worker_id}, "
                                  f"data_rank={self.data_rank}, using_fallback_token_0=True", flush=True)
                            # Replace empty tensor with token 0
                            new_sample[k][idx] = torch.tensor([0], dtype=torch.int)
                buffer[k] = buffer[k] + new_sample[k]
            else:
                buffer[k] = torch.cat([buffer[k], new_sample[k]])
        
        # Handle debug keys that are in new_sample but not in buffer
        for k in new_sample:
            if k.startswith('_debug_') and k not in buffer:
                buffer[k] = [new_sample[k]]
        
        return buffer

    @staticmethod
    def split_buffer(buffer, max_tokens, img_start_token_id, img_token_id, img_end_token_id,lvr_token_id,long_seq_threshold,max_instance_per_batch):
        
        if not long_seq_threshold:
            long_seq_threshold = max_tokens // 2

        def _image_is_splitted(input_ids, cut_idx):
            if cut_idx >= input_ids.size(0):
                return False
            else:
                is_image_start = input_ids[cut_idx].item() == img_start_token_id
                is_image_token = input_ids[cut_idx].item() == img_token_id
                is_image_end = input_ids[cut_idx].item() == img_end_token_id
                return is_image_start or is_image_token or is_image_end
        
        def _find_lvr_aware_cut_id(input_ids, default_cut_id, lvr_token_id, max_tokens):
            """
            Improved truncation strategy: try to preserve <lvr> tokens when possible.
            
            Strategy:
            1. Find all <lvr> token positions in the sequence
            2. If first <lvr> token is before default_cut_id, use default (all <lvr> tokens preserved)
            3. If first <lvr> token is after default_cut_id:
               - Try to extend cut_id to include at least the first group of <lvr> tokens
               - But don't extend beyond max_tokens * 1.2 to avoid memory issues
               - Also ensure we don't cut in the middle of image tokens
            4. Return the adjusted cut_id
            """
            if lvr_token_id is None:
                return default_cut_id
            
            # Find all <lvr> token positions
            lvr_positions = (input_ids == lvr_token_id).nonzero(as_tuple=True)[0]
            
            if len(lvr_positions) == 0:
                # No <lvr> tokens in sequence, use default
                return default_cut_id
            
            first_lvr_pos = lvr_positions[0].item()
            last_lvr_pos = lvr_positions[-1].item()
            
            # If first <lvr> token is already within cut range, use default
            if first_lvr_pos < default_cut_id:
                return default_cut_id
            
            # First <lvr> token is outside cut range, try to extend
            # We want to include all <lvr> tokens (typically 8 per bbox)
            # Add some buffer (+10) after last <lvr> token for potential following text
            desired_cut_id = last_lvr_pos + 10
            
            # Don't extend too much - cap at 1.2x max_tokens to avoid memory explosion
            max_extended_cut_id = int(max_tokens * 1.2)
            
            if desired_cut_id <= max_extended_cut_id and desired_cut_id <= input_ids.size(0):
                # Check if this extended cut would split an image
                if not _image_is_splitted(input_ids, desired_cut_id):
                    logger.info(
                        f"[PackedDataset] 📐 Extended cut_id from {default_cut_id} to {desired_cut_id} "
                        f"to preserve <lvr> tokens (first_lvr_pos={first_lvr_pos}, last_lvr_pos={last_lvr_pos})"
                    )
                    return desired_cut_id
                else:
                    # Extended cut would split image, try to find a safe cut point after last <lvr>
                    for safe_cut in range(last_lvr_pos + 1, min(desired_cut_id + 50, input_ids.size(0))):
                        if not _image_is_splitted(input_ids, safe_cut):
                            logger.info(
                                f"[PackedDataset] 📐 Found safe extended cut_id={safe_cut} "
                                f"to preserve <lvr> tokens (avoiding image split)"
                            )
                            return safe_cut
            
            # Cannot extend safely, use default (will trigger lvr_tokens clearing)
            logger.warning(
                f"[PackedDataset] ⚠️ Cannot extend cut_id to preserve <lvr> tokens: "
                f"first_lvr_pos={first_lvr_pos} > default_cut_id={default_cut_id}, "
                f"desired_cut_id={desired_cut_id} > max_extended={max_extended_cut_id}"
            )
            return default_cut_id
        
        '''
            Handles long single-/multi- instance buffer differently
        '''
        # condition 1: single instance
        if buffer['data_index'][-1].item() == 0:
            # condition 1.1: single long instance
            if buffer['input_ids'].size(0) >= long_seq_threshold:

                '''cut_id is the idx of the first token to be dropped'''
                default_cut_id = min(max_tokens, buffer['input_ids'].size(0))
                
                # Try to find a better cut_id that preserves <lvr> tokens
                cut_id = _find_lvr_aware_cut_id(
                    buffer['input_ids'], default_cut_id, lvr_token_id, max_tokens
                )

                if not _image_is_splitted(buffer['input_ids'], cut_id):
                    # count discarded lvr tokens before slicing

                    if lvr_token_id is not None and 'lvr_tokens' in buffer and len(buffer['lvr_tokens']) > 0:
                        num_discarded_lvr_tokens = (buffer['input_ids'][cut_id:] == lvr_token_id).sum().item()
                        lvr_tokens_size = buffer['lvr_tokens'][0].size(0)
                        cut_id_lvr = max(0, min(lvr_tokens_size, lvr_tokens_size - num_discarded_lvr_tokens))
                        
                        # CRITICAL: Ensure cut_id_lvr is valid and doesn't create empty tensor
                        if cut_id_lvr <= 0:
                            logger.error(
                                f"[PackedDataset] ❌ CRITICAL: Invalid cut_id_lvr={cut_id_lvr} for lvr_tokens! "
                                f"lvr_tokens_size={lvr_tokens_size}, num_discarded_lvr_tokens={num_discarded_lvr_tokens}, "
                                f"cut_id={cut_id}. This would create empty lvr_tokens! Using full lvr_tokens instead."
                            )
                            cut_id_lvr = lvr_tokens_size  # Use full lvr_tokens to avoid empty tensor
                    else:
                        # If lvr_token_id is None or lvr_tokens is empty, keep all lvr_tokens
                        cut_id_lvr = None
                    
                    # Track if lvr_tokens were cleared, which means we should discard this buffer
                    should_discard_buffer = False
                    
                    for k in buffer:
                        if k in ['input_ids', 'labels', 'attention_mask', 'position_ids', 'data_index']:
                            buffer[k] = buffer[k][:cut_id]
                        elif k in ['pixel_values', 'image_flags','image_grid_thw', 'cropped_bbox_images']:
                            buffer[k] = buffer[k]
                        elif k in ['lvr_tokens']:
                            # CRITICAL: After truncating input_ids, check if any <lvr> tokens remain
                            # If not, clear lvr_tokens to avoid mismatch
                            lvr_tokens_cleared = False
                            if 'input_ids' in buffer and lvr_token_id is not None:
                                remaining_lvr_tokens = (buffer['input_ids'] == lvr_token_id).sum().item()
                                if remaining_lvr_tokens == 0 and len(buffer[k]) > 0 and buffer[k][0].numel() > 0:
                                    logger.warning(
                                        f"[PackedDataset] ⚠️  After truncation, no <lvr> tokens found in input_ids "
                                        f"but lvr_tokens has {buffer[k][0].numel()} tokens. Discarding this buffer to avoid NaN loss."
                                    )
                                    # Mark buffer for discard instead of clearing lvr_tokens
                                    should_discard_buffer = True
                                    lvr_tokens_cleared = True
                            
                            # Only process lvr_tokens if they weren't cleared and cut_id_lvr is set
                            if not lvr_tokens_cleared and cut_id_lvr is not None and len(buffer[k]) > 0:
                                # CRITICAL: Log before slicing for debugging
                                original_size = buffer[k][0].size(0)
                                # CRITICAL: Double-check before slicing to prevent empty tensor
                                if cut_id_lvr > 0 and cut_id_lvr <= buffer[k][0].size(0):
                                    buffer[k][0] = buffer[k][0][:cut_id_lvr]
                                elif cut_id_lvr == 0:
                                    # If cut_id_lvr is 0, keep at least one token to avoid empty tensor
                                    logger.warning(
                                        f"[PackedDataset] ⚠️  cut_id_lvr=0 would create empty lvr_tokens, "
                                        f"keeping at least 1 token. Original size: {buffer[k][0].size(0)}"
                                    )
                                    buffer[k][0] = buffer[k][0][:1] if buffer[k][0].size(0) > 0 else buffer[k][0]
                                else:
                                    # If cut_id_lvr > buffer[k][0].size(0), keep full tensor
                                    pass
                                
                                # CRITICAL: Final check - ensure tensor is not empty
                                if buffer[k][0].numel() == 0:
                                    logger.error(
                                        f"[PackedDataset] ❌ CRITICAL: Empty lvr_tokens[0] after split_buffer! "
                                        f"cut_id_lvr={cut_id_lvr}, original_size={original_size}, "
                                        f"cut_id={cut_id}. Discarding buffer to avoid NaN loss."
                                    )
                                    should_discard_buffer = True
                            # If cut_id_lvr is None, keep lvr_tokens as is
                        elif k in ['input_lengths']:
                            pass
                        elif k.startswith('_debug_'):
                            # Skip debug keys, keep them as is
                            pass
                        else:
                            raise NotImplementedError(f'find unsupported keys: {k} from {buffer.keys()}')
                    
                    # If lvr_tokens were cleared or became empty after truncation, clear them and continue
                    # Forward function will handle empty lvr_tokens by returning zero loss for resampler
                    # This avoids NCCL timeout caused by discarding buffer in distributed training
                    if should_discard_buffer:
                        logger.warning(
                            f"[PackedDataset] ⚠️ Empty lvr_tokens after truncation. "
                            f"Clearing lvr_tokens and continuing (forward will return zero resampler loss)."
                        )
                        # Clear lvr_tokens instead of discarding the entire buffer
                        buffer['lvr_tokens'] = []
                    
                    # re-assign lengths and return buffer
                    buffer['input_lengths'][0] = buffer['input_ids'].size(0)
                    buffer_ready = [buffer]
                    buffer_unready = []
                else:   # if image is getting cut, discard the overlong instance
                    buffer_ready = []
                    buffer_unready = []
            
            # condition 1.2: single short instance
            else:
                buffer_ready = []
                buffer_unready = [buffer]

        # condition 2: multi instance
        else:
            # condition 2.1: < maxToken AND < max_instance_per_batch
            if (buffer['input_ids'].size(0) < max_tokens) and (buffer['input_lengths'].size(0) < max_instance_per_batch):
                buffer_ready = []
                buffer_unready = [buffer]
            # condition 2.2: < maxToken AND == max_instance_per_batch
            elif (buffer['input_ids'].size(0) < max_tokens) and (buffer['input_lengths'].size(0) == max_instance_per_batch):
                # CRITICAL: Check if lvr_tokens are empty before adding to buffer_ready
                should_discard_buffer_multi = False
                if 'lvr_tokens' in buffer and lvr_token_id is not None:
                    # Check if input_ids has any <lvr> tokens
                    remaining_lvr_tokens = (buffer['input_ids'] == lvr_token_id).sum().item()
                    # Check if lvr_tokens list has any empty tensors
                    if len(buffer['lvr_tokens']) > 0:
                        for idx, lvr_token_group in enumerate(buffer['lvr_tokens']):
                            if isinstance(lvr_token_group, torch.Tensor) and lvr_token_group.numel() == 0:
                                logger.error(
                                    f"[PackedDataset] ❌ CRITICAL: Empty lvr_tokens[{idx}] detected in condition 2.2! "
                                    f"Discarding buffer to avoid NaN loss."
                                )
                                should_discard_buffer_multi = True
                                break
                    # Check if input_ids has no <lvr> tokens but lvr_tokens has values
                    if remaining_lvr_tokens == 0 and len(buffer['lvr_tokens']) > 0:
                        has_non_empty_lvr = any(
                            isinstance(t, torch.Tensor) and t.numel() > 0 
                            for t in buffer['lvr_tokens']
                        )
                        if has_non_empty_lvr:
                            logger.error(
                                f"[PackedDataset] ❌ CRITICAL: No <lvr> tokens in input_ids but lvr_tokens has values in condition 2.2! "
                                f"Discarding buffer to avoid NaN loss."
                            )
                            should_discard_buffer_multi = True
                
                # If should_discard_buffer_multi, clear lvr_tokens instead of discarding buffer
                # This avoids NCCL timeout in distributed training
                if should_discard_buffer_multi:
                    logger.warning(
                        f"[PackedDataset] ⚠️ Empty/invalid lvr_tokens in condition 2.2. "
                        f"Clearing lvr_tokens and continuing (forward will return zero resampler loss)."
                    )
                    buffer['lvr_tokens'] = []
                
                buffer_ready = [buffer]
                buffer_unready = []
            # condition 2.3: otherwise
            else:
                buffer_ready = []
                buffer_unready = []
                while buffer['input_ids'].size(0) >= max_tokens:
                    buffer_right = {}
                    cut_idx_right_size = buffer['input_lengths'][-1].item()     # number of tokens to be cut from right side
                    image_cut_idx_right_size = buffer['image_grid_thw'][-1].prod()  # number of pixels to be cut from right side
                    for k in buffer:
                        if k in ['input_ids', 'labels', 'attention_mask', 'position_ids', 'data_index']:
                            buffer_right[k] = buffer[k][-cut_idx_right_size:]
                            buffer[k] = buffer[k][:-cut_idx_right_size]
                        elif k in ['pixel_values', 'image_flags']:
                            buffer_right[k] = buffer[k][-image_cut_idx_right_size:]
                            buffer[k] = buffer[k][:-image_cut_idx_right_size]
                        elif k in ['lvr_tokens','image_grid_thw','input_lengths']:
                            # CRITICAL: Ensure lvr_tokens are not empty after splitting
                            if k == 'lvr_tokens' and len(buffer[k]) > 0:
                                # Check if the last lvr_tokens element is empty before splitting
                                last_lvr_tokens = buffer[k][-1]
                                if isinstance(last_lvr_tokens, torch.Tensor) and last_lvr_tokens.numel() == 0:
                                    logger.error(
                                        f"[PackedDataset] ❌ CRITICAL: Last lvr_tokens element is empty before split! "
                                        f"This should not happen!"
                                    )
                            buffer_right[k] = buffer[k][-1:]
                            buffer[k] = buffer[k][:-1]
                            
                            # CRITICAL: Verify lvr_tokens are not empty after splitting
                            if k == 'lvr_tokens':
                                if len(buffer[k]) > 0:
                                    for idx, lvr_token_group in enumerate(buffer[k]):
                                        if isinstance(lvr_token_group, torch.Tensor) and lvr_token_group.numel() == 0:
                                            logger.error(
                                                f"[PackedDataset] ❌ CRITICAL: Empty lvr_tokens[{idx}] detected after split! "
                                                f"Using fallback token 0."
                                            )
                                            # Use token 0 as fallback
                                            buffer[k][idx] = torch.tensor([0], dtype=torch.int)
                                if len(buffer_right[k]) > 0:
                                    for idx, lvr_token_group in enumerate(buffer_right[k]):
                                        if isinstance(lvr_token_group, torch.Tensor) and lvr_token_group.numel() == 0:
                                            logger.error(
                                                f"[PackedDataset] ❌ CRITICAL: Empty lvr_tokens[{idx}] in buffer_right after split! "
                                                f"Using fallback token 0."
                                            )
                                            # Use token 0 as fallback
                                            buffer_right[k][idx] = torch.tensor([0], dtype=torch.int)
                        elif k.startswith('_debug_'):
                            # For debug keys, keep them in the right buffer (most recent instance)
                            if k not in buffer_right:
                                buffer_right[k] = []
                            if isinstance(buffer[k], list):
                                buffer_right[k] = buffer[k][-1:] if len(buffer[k]) > 0 else []
                                buffer[k] = buffer[k][:-1] if len(buffer[k]) > 0 else []
                            else:
                                # For non-list debug keys, keep in right buffer
                                buffer_right[k] = buffer[k]
                        else:
                            raise NotImplementedError(f'find unsupported keys: {k} from {buffer.keys()}')
                    # Check left buffer - if invalid lvr_tokens, clear them instead of discarding
                    # This avoids NCCL timeout in distributed training
                    if 'lvr_tokens' in buffer and lvr_token_id is not None:
                        remaining_lvr_tokens_left = (buffer['input_ids'] == lvr_token_id).sum().item()
                        left_buffer_invalid = False
                        if len(buffer['lvr_tokens']) > 0:
                            for idx, lvr_token_group in enumerate(buffer['lvr_tokens']):
                                if isinstance(lvr_token_group, torch.Tensor) and lvr_token_group.numel() == 0:
                                    left_buffer_invalid = True
                                    break
                        if remaining_lvr_tokens_left == 0 and len(buffer['lvr_tokens']) > 0:
                            has_non_empty_lvr = any(
                                isinstance(t, torch.Tensor) and t.numel() > 0 
                                for t in buffer['lvr_tokens']
                            )
                            if has_non_empty_lvr:
                                left_buffer_invalid = True
                        
                        if left_buffer_invalid:
                            logger.warning(
                                f"[PackedDataset] ⚠️ Invalid lvr_tokens in left buffer (condition 2.3). "
                                f"Clearing lvr_tokens (forward will return zero resampler loss)."
                            )
                            buffer['lvr_tokens'] = []
                    
                    # Check right buffer - if invalid lvr_tokens, clear them instead of discarding
                    if 'lvr_tokens' in buffer_right and lvr_token_id is not None:
                        remaining_lvr_tokens_right = (buffer_right['input_ids'] == lvr_token_id).sum().item()
                        right_buffer_invalid = False
                        if len(buffer_right['lvr_tokens']) > 0:
                            for idx, lvr_token_group in enumerate(buffer_right['lvr_tokens']):
                                if isinstance(lvr_token_group, torch.Tensor) and lvr_token_group.numel() == 0:
                                    right_buffer_invalid = True
                                    break
                        if remaining_lvr_tokens_right == 0 and len(buffer_right['lvr_tokens']) > 0:
                            has_non_empty_lvr = any(
                                isinstance(t, torch.Tensor) and t.numel() > 0 
                                for t in buffer_right['lvr_tokens']
                            )
                            if has_non_empty_lvr:
                                right_buffer_invalid = True
                        
                        if right_buffer_invalid:
                            logger.warning(
                                f"[PackedDataset] ⚠️ Invalid lvr_tokens in right buffer (condition 2.3). "
                                f"Clearing lvr_tokens (forward will return zero resampler loss)."
                            )
                            buffer_right['lvr_tokens'] = []
                    
                    # Always append buffers (lvr_tokens already cleared if invalid)
                    if buffer['input_ids'].size(0) >= buffer_right['input_ids'].size(0):
                        buffer_ready.append(buffer)
                        buffer = buffer_right
                    else:   # buffer_right is longer than the accumulated left
                        buffer_ready.append(buffer_right)

                # Check final buffer - if invalid lvr_tokens, clear them instead of discarding
                # This avoids NCCL timeout in distributed training
                if 'lvr_tokens' in buffer and lvr_token_id is not None:
                    remaining_lvr_tokens_final = (buffer['input_ids'] == lvr_token_id).sum().item()
                    final_buffer_invalid = False
                    if len(buffer['lvr_tokens']) > 0:
                        for idx, lvr_token_group in enumerate(buffer['lvr_tokens']):
                            if isinstance(lvr_token_group, torch.Tensor) and lvr_token_group.numel() == 0:
                                final_buffer_invalid = True
                                break
                    if remaining_lvr_tokens_final == 0 and len(buffer['lvr_tokens']) > 0:
                        has_non_empty_lvr = any(
                            isinstance(t, torch.Tensor) and t.numel() > 0 
                            for t in buffer['lvr_tokens']
                        )
                        if has_non_empty_lvr:
                            final_buffer_invalid = True
                    
                    if final_buffer_invalid:
                        logger.warning(
                            f"[PackedDataset] ⚠️ Invalid lvr_tokens in final buffer (condition 2.3). "
                            f"Clearing lvr_tokens (forward will return zero resampler loss)."
                        )
                        buffer['lvr_tokens'] = []
                
                # Always append buffer (lvr_tokens already cleared if invalid)
                buffer_unready.append(buffer)

        return buffer_ready, buffer_unready

    def update_buffer_list(self, buffer_list, buffer_max_len_list, buffer):
        # NOTE: in-place operation

        buffer_ready, buffer_unready = PackedDataset.split_buffer(
            buffer=buffer,
            max_tokens=self.max_packed_tokens,
            img_start_token_id=self.img_start_token_id,
            img_token_id=self.img_token_id,
            img_end_token_id=self.img_end_token_id,
            lvr_token_id=self.lvr_token_id,
            long_seq_threshold=self.long_seq_threshold,
            max_instance_per_batch=self.max_instance_per_batch
        )

        for each_buffer in buffer_ready:
            buffer_max_len_list.append(each_buffer)

        for each_buffer in buffer_unready:
            find_idx = len(buffer_list)
            num_tokens_new_sample = each_buffer['input_ids'].size(0)
            for buffer_idx in range(len(buffer_list)):
                if buffer_list[buffer_idx]['input_ids'].size(0) < num_tokens_new_sample:
                    find_idx = buffer_idx
                    break
            buffer_list.insert(find_idx, each_buffer)

        return buffer_list, buffer_max_len_list

    def print_log(self, iter_idx, buffer_list):
        if iter_idx % self.log_freq != 0:
            return

        if self._should_log():
            logger.info(
                f"{iter_idx=}, {len(buffer_list)=}, {self._state_dict['sample_info']}"
            )

    def __iter__(self):
        iter_idx = 0
        buffer_list = []
        buffer_max_len_list = []

        if self._should_log():
            logger.info(f'Begin to iter, {len(buffer_list)=}')

        worker_id = 0 if get_worker_info() is None else get_worker_info().id
        num_workers = 1 if get_worker_info() is None else get_worker_info().num_workers

        worker_id = num_workers * self.data_rank + worker_id
        num_workers = num_workers * self.data_world_size

        rng = np.random.default_rng(seed=worker_id)

        # reset states of each dataset
        self.worker_id = worker_id
        self.worker_state_key = f'work_state_{self.worker_id}'
        self.datasets = [d for d in self.datasets_orig]

        self.dataset_weight = [w for w in self.dataset_weight_orig]
        self.dataset_iter_list = [iter(d) for d in self.datasets]

        for ds in self.datasets:
            # if not isinstance(ds, (ImageTextPairDataset, InterleavedDataset)):
            ds.worker_id = worker_id
            ds.worker_state_key = f'work_state_{self.worker_id}'
            ds.num_workers = num_workers
            if self._should_log() and worker_id == 0:
                logger.info(f'set worker_id and num_workers of {ds.__class__.__name__} {ds.ds_name}')

        if self.worker_custom_infos is not None and self.worker_state_key in self.worker_custom_infos:
            custom_infos = self.worker_custom_infos[self.worker_state_key]
            # buffer list
            if 'buffer_list' in custom_infos and isinstance(custom_infos['buffer_list'], list):
                buffer_list = custom_infos['buffer_list']
                if self._should_log() and worker_id == 0:
                    logger.info(f'[{self.worker_state_key}] load buffer list --> {len(buffer_list)=}')
            # other infos

            # reset
            self.worker_custom_infos = None

        logger.debug(
            f'{self.__class__.__name__} Rank {self.data_rank} '
            f'Worker {worker_id} begin to load data'
        )

        while True:
            self.dataset_weight = [w / sum(self.dataset_weight) for w in self.dataset_weight]
            current_dataset_idx = rng.choice(len(self.dataset_iter_list), p=self.dataset_weight)

            try:
                current_sample = self.next_data(current_dataset_idx)
            except:
                logger.info(f'All datasets are exhausted, begin to empty the buffer_list ({len(buffer_list)=})')
                while len(buffer_list) > 0:
                    yield buffer_list.pop(0)
                logger.info(f'buffer_list is empty! ({len(buffer_list)=})')
                return

            # it is guaranteed in self.find_buffer() that if current_sample is >= max_tokens,
            # it will not get concatenated to any existing buffer
            buffer = self.find_buffer(buffer_list, current_sample)
            buffer = self.update_buffer(buffer, current_sample)
            '''
                A greedy method to balance effifiency and memory safety:
                1. if buffer stacks up tp max_packed_tokens, it is poped
                2. if buffer has >= max_instance_per_batch, it is poped
                3. if a single sample is >= long_seq_thresh, it is poped as a single buffer

                This is intended to avoid a long seq paired with multiple short seqs since
                padding them will explode the memory
            '''
            buffer_list, buffer_max_len_list = self.update_buffer_list(buffer_list, buffer_max_len_list, buffer)

            while len(buffer_max_len_list) > 0:
                buffer_to_yield = buffer_max_len_list.pop(0)
                # Check for extremely long sequences that may cause NCCL timeout in DeepSpeed ZeRO-3
                seq_len = buffer_to_yield['input_ids'].size(0)
                if seq_len > 3000:  # Threshold for very long sequences
                    if self._should_log():
                        logger.warning(
                            f"[WARNING] Very long sequence detected: {seq_len} tokens. "
                            f"This may cause NCCL timeout in DeepSpeed ZeRO-3. "
                            f"Consider reducing max_packed_tokens or filtering long sequences. "
                            f"Current max_packed_tokens={self.max_packed_tokens}, long_seq_threshold={self.long_seq_threshold}"
                        )
                yield buffer_to_yield

            while len(buffer_list) > self.max_buffer_size:
                buffer_to_yield = buffer_list.pop(0)
                # Check for extremely long sequences that may cause NCCL timeout in DeepSpeed ZeRO-3
                seq_len = buffer_to_yield['input_ids'].size(0)
                if seq_len > 3000:  # Threshold for very long sequences
                    if self._should_log():
                        logger.warning(
                            f"[WARNING] Very long sequence detected: {seq_len} tokens. "
                            f"This may cause NCCL timeout in DeepSpeed ZeRO-3. "
                            f"Consider reducing max_packed_tokens or filtering long sequences. "
                            f"Current max_packed_tokens={self.max_packed_tokens}, long_seq_threshold={self.long_seq_threshold}"
                        )
                yield buffer_to_yield

            self.print_log(iter_idx=iter_idx, buffer_list=buffer_list)
            iter_idx += 1


WARNING_CNT = defaultdict(int)

class PackedDataCollatorForSupervisedDatasetLVR(object):

    def __init__(self,pad_token_id):
        self.pad_token_id = pad_token_id 
    
    def __call__(self, features):
        # features is supposed to be a list of packed items
        # We will set batch_per_device to 1

        if not isinstance(features, list):
            features = [features]

        #  Unpack all sequences
        all_input_ids = []
        all_attention_masks = []
        all_labels = []

        all_lvr_tokens = []
        all_pixel_values = []
        all_image_grid_thw = []
        all_cropped_bbox_images = []
        
        # Collect debug information
        all_debug_questions = []
        all_debug_answers = []
        all_debug_image_paths = []
        all_debug_bboxes = []
        all_debug_data_indices = []

        for feature in features:
            # each feature is a packed mini-batch
            all_input_ids.extend(torch.split(feature["input_ids"], feature["input_lengths"].tolist()))
            all_attention_masks.extend(torch.split(feature["attention_mask"], feature["input_lengths"].tolist()))
            all_labels.extend(torch.split(feature["labels"], feature["input_lengths"].tolist()))

            all_lvr_tokens.extend(feature['lvr_tokens'])
            all_pixel_values.append(feature['pixel_values'])
            all_image_grid_thw.append(feature['image_grid_thw'])
            if 'cropped_bbox_images' in feature and feature['cropped_bbox_images'] is not None:
                all_cropped_bbox_images.append(feature['cropped_bbox_images'])
            
            # Extract debug information if available
            if '_debug_question' in feature:
                debug_list = feature['_debug_question'] if isinstance(feature['_debug_question'], list) else [feature['_debug_question']]
                all_debug_questions.extend(debug_list)
            if '_debug_answer' in feature:
                debug_list = feature['_debug_answer'] if isinstance(feature['_debug_answer'], list) else [feature['_debug_answer']]
                all_debug_answers.extend(debug_list)
            if '_debug_image_paths' in feature:
                debug_list = feature['_debug_image_paths'] if isinstance(feature['_debug_image_paths'], list) else [feature['_debug_image_paths']]
                all_debug_image_paths.extend(debug_list)
            if '_debug_bboxes' in feature:
                debug_list = feature['_debug_bboxes'] if isinstance(feature['_debug_bboxes'], list) else [feature['_debug_bboxes']]
                all_debug_bboxes.extend(debug_list)
            if '_debug_data_idx' in feature:
                debug_list = feature['_debug_data_idx'] if isinstance(feature['_debug_data_idx'], list) else [feature['_debug_data_idx']]
                all_debug_data_indices.extend(debug_list)
        
        # pad all sequences
        max_len = max(len(seq) for seq in all_input_ids)
        padded_input_ids = []
        padded_attention_masks = []
        padded_labels = []
        for i in range(len(all_input_ids)):
            seq_len = len(all_input_ids[i])
            padding_needed = max_len - seq_len

            # Pad on the right side.
            padded_input_ids.append(torch.nn.functional.pad(
                all_input_ids[i], (0, padding_needed), value=self.pad_token_id))
            
            padded_attention_masks.append(torch.nn.functional.pad(
                all_attention_masks[i], (0, padding_needed), value=0))
            
            padded_labels.append(torch.nn.functional.pad(
                all_labels[i], (0, padding_needed), value=IGNORE_INDEX))

        data_dict = {
            "input_ids": torch.stack(padded_input_ids),
            "attention_mask": torch.stack(padded_attention_masks),
            "labels": torch.stack(padded_labels),

            "lvr_tokens": all_lvr_tokens,
            "pixel_values": torch.cat(all_pixel_values),
            "image_grid_thw": torch.cat(all_image_grid_thw),
        }
        if all_cropped_bbox_images:
            data_dict["cropped_bbox_images"] = torch.cat(all_cropped_bbox_images, dim=0)
        
        # Add debug information if available
        # CRITICAL: Convert to simple strings to prevent RecursionError in pin_memory
        # pin_memory recursively processes nested lists, deep nesting causes stack overflow
        if all_debug_questions:
            data_dict['_debug_question'] = str(all_debug_questions)[:2000]  # Truncate to avoid memory issues
        if all_debug_answers:
            data_dict['_debug_answer'] = str(all_debug_answers)[:2000]
        if all_debug_image_paths:
            data_dict['_debug_image_paths'] = str(all_debug_image_paths)[:2000]
        if all_debug_bboxes:
            data_dict['_debug_bboxes'] = str(all_debug_bboxes)[:2000]
        if all_debug_data_indices:
            data_dict['_debug_data_idx'] = str(all_debug_data_indices)[:500]
        
        return data_dict