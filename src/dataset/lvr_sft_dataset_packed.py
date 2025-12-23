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
    LVR_TOKEN
)
from transformers import TrainingArguments

from .data_utils import get_image_info, llava_to_openai_lvr, pad_sequence, map_image_path
import numpy as np
from PIL import Image
from typing import List, Tuple
import math

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

  
def make_packed_supervised_data_module_lvr(model_id, processor, data_args, training_args: TrainingArguments,latent_end_token=False):

    """Make dataset and collator for supervised fine-tuning."""

    data_rank = dist.get_rank()
    data_world_size = dist.get_world_size()

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
            latent_end_token=latent_end_token
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

    def __len__(self):
        return len(self.raw_data)

    def bbox_to_token_idxs(self, bboxes, image_grid_thw):
        """
            This function is intended for Qwen-VL series only.
            The target visual tokens is computed based on image_grid_thw,
            which is the best estimation

            image_grid_thw is a 2D tensor with a single item

        """
        _, h, w = image_grid_thw[0].tolist()
        token_idxs = []
        H2, W2 = h // 2, w // 2
        
        # CRITICAL: Ensure H2 and W2 are valid (at least 1)
        if H2 <= 0 or W2 <= 0:
            worker_id_str = getattr(self, 'worker_id', 'unknown')
            logger.error(
                f"[{self.ds_name}] ❌ CRITICAL: Invalid H2={H2} or W2={W2} (h={h}, w={w}), "
                f"worker_id={worker_id_str}. Cannot generate tokens. Using token 0 for all bboxes."
            )
            print(f"[BBOX_INVALID_GRID] h={h}, w={w}, H2={H2}, W2={W2}, worker_id={worker_id_str}, "
                  f"using_token_0_for_all", flush=True)
            # Return token 0 for all bboxes
            for bbox_idx, bbox in enumerate(bboxes):
                token_idxs.append([0])
            return token_idxs
        
        # Log non-standard grid sizes for debugging
        if h != 14 or w != 14:
            if hasattr(self, 'worker_id') and (self.worker_id == 0 or self.worker_id is None):
                logger.debug(
                    f"[{self.ds_name}] Non-standard image_grid_thw: h={h}, w={w}, H2={H2}, W2={W2} "
                    f"(expected h=14, w=14 for standard Qwen2-VL)"
                )
        
        for bbox_idx, bbox in enumerate(bboxes): 
            x0, y0, x1, y1 = bbox

            # Validate bbox - but don't return empty list, use fallback instead
            if x1 <= x0 or y1 <= y0:
                worker_id_str = getattr(self, 'worker_id', 'unknown')
                logger.warning(
                    f"[{self.ds_name}] Invalid bbox[{bbox_idx}]: {bbox} (x1<=x0 or y1<=y0), "
                    f"h={h}, w={w}, H2={H2}, W2={W2}, worker_id={worker_id_str}. "
                    f"Using image center token as fallback."
                )
                print(f"[BBOX_INVALID_FALLBACK] bbox={bbox}, h={h}, w={w}, H2={H2}, W2={W2}, "
                      f"worker_id={worker_id_str}, reason=invalid_bbox", flush=True)
                # Use image center as fallback for invalid bbox
                center_x_token = W2 // 2
                center_y_token = H2 // 2
                center_token_idx = int(center_y_token * W2 + center_x_token)
                valid_idxs = [center_token_idx]
                token_idxs.append(valid_idxs)
                continue

            # Scale to grid (typically 14x14)
            x0_grid = max(0, min(int(np.floor(x0 * w)), w-1))
            x1_grid = max(0, min(int(np.ceil (x1 * w)), w))
            y0_grid = max(0, min(int(np.floor(y0 * h)), h-1))
            y1_grid = max(0, min(int(np.ceil (y1 * h)), h))

            # Handle edge case: if bbox maps to same grid cell, ensure at least one token
            # This happens when bbox is very small and x0_grid == x1_grid or y0_grid == y1_grid
            if x0_grid == x1_grid:
                # Expand x1_grid to ensure at least one grid cell
                if x1_grid < w:
                    x1_grid = x1_grid + 1
                elif x0_grid > 0:
                    x0_grid = x0_grid - 1
                else:
                    # Can't expand, will result in empty tokens
                    pass
            
            if y0_grid == y1_grid:
                # Expand y1_grid to ensure at least one grid cell
                if y1_grid < h:
                    y1_grid = y1_grid + 1
                elif y0_grid > 0:
                    y0_grid = y0_grid - 1
                else:
                    # Can't expand, will result in empty tokens
                    pass

            # Map to token grid (H2 x W2, typically 7x7)
            x0_token = x0_grid // 2
            x1_token = (x1_grid + 1) // 2
            y0_token = y0_grid // 2
            y1_token = (y1_grid + 1) // 2

            # Ensure at least one token in each dimension
            if x0_token >= x1_token:
                x1_token = x0_token + 1
            if y0_token >= y1_token:
                y1_token = y0_token + 1

            # Final bounds check to ensure valid ranges
            x0_token = max(0, min(x0_token, W2 - 1))
            x1_token = max(x0_token + 1, min(x1_token, W2))
            y0_token = max(0, min(y0_token, H2 - 1))
            y1_token = max(y0_token + 1, min(y1_token, H2))
            
            # Check if ranges are valid (should not happen after above fixes, but keep for safety)
            if x0_token >= x1_token or y0_token >= y1_token:
                logger.warning(
                    f"[{self.ds_name}] Empty token range after all fixes for bbox[{bbox_idx}]: {bbox}, "
                    f"h={h}, w={w}, H2={H2}, W2={W2}, "
                    f"x0_grid={x0_grid}, x1_grid={x1_grid}, y0_grid={y0_grid}, y1_grid={y1_grid}, "
                    f"x0_token={x0_token}, x1_token={x1_token}, y0_token={y0_token}, y1_token={y1_token}. "
                    f"Will use fallback strategy in token generation."
                )
                # Don't append empty list - let fallback strategy handle it below
                # Set to use first token as minimum fallback
                x0_token, x1_token = 0, min(1, W2)
                y0_token, y1_token = 0, min(1, H2)

            idxs = [
                int(yy * W2 + xx)
                for yy in range(y0_token, y1_token)
                for xx in range(x0_token, x1_token)
            ]
            
            # Filter out invalid indices first
            valid_idxs = [idx for idx in idxs if 0 <= idx < H2 * W2]
            
            # CRITICAL: If no valid tokens generated, use fallback strategy
            if len(valid_idxs) == 0:
                # Always log, not just worker_id == 0, to catch all cases
                worker_id_str = getattr(self, 'worker_id', 'unknown')
                logger.error(
                    f"[{self.ds_name}] ❌ CRITICAL: Generated empty or invalid idxs for bbox[{bbox_idx}]: {bbox}, "
                    f"h={h}, w={w}, H2={H2}, W2={W2}, worker_id={worker_id_str}, "
                    f"x0_grid={x0_grid}, x1_grid={x1_grid}, y0_grid={y0_grid}, y1_grid={y1_grid}, "
                    f"x0_token={x0_token}, x1_token={x1_token}, y0_token={y0_token}, y1_token={y1_token}, "
                    f"x_range={list(range(x0_token, x1_token))}, y_range={list(range(y0_token, y1_token))}, "
                    f"original_idxs={idxs[:10]}. Using fallback: bbox center token."
                )
                print(f"[BBOX_FALLBACK_TRIGGERED] bbox={bbox}, h={h}, w={w}, H2={H2}, W2={W2}, "
                      f"worker_id={worker_id_str}, x0_token={x0_token}, x1_token={x1_token}, "
                      f"y0_token={y0_token}, y1_token={y1_token}, empty_idxs=True", flush=True)
                
                # Fallback strategy 1: Use bbox center
                x_center = (x0 + x1) / 2.0
                y_center = (y0 + y1) / 2.0
                
                # Map center to grid
                x_center_grid = max(0, min(int(np.round(x_center * w)), w-1))
                y_center_grid = max(0, min(int(np.round(y_center * h)), h-1))
                
                # Map to token grid
                x_center_token = max(0, min(x_center_grid // 2, W2 - 1))
                y_center_token = max(0, min(y_center_grid // 2, H2 - 1))
                
                center_token_idx = int(y_center_token * W2 + x_center_token)
                
                if 0 <= center_token_idx < H2 * W2:
                    valid_idxs = [center_token_idx]
                    worker_id_str = getattr(self, 'worker_id', 'unknown')
                    logger.info(
                        f"[{self.ds_name}] ✓ Fallback successful: using center token {center_token_idx} "
                        f"for bbox[{bbox_idx}] center=({x_center:.4f}, {y_center:.4f}), worker_id={worker_id_str}"
                    )
                    print(f"[BBOX_FALLBACK_SUCCESS] bbox={bbox}, center_token={center_token_idx}, "
                          f"center=({x_center:.4f}, {y_center:.4f}), worker_id={worker_id_str}", flush=True)
                    print(f"[BBOX_FALLBACK_SUCCESS] bbox={bbox}, center_token={center_token_idx}, "
                          f"center=({x_center:.4f}, {y_center:.4f}), worker_id={worker_id_str}", flush=True)
                else:
                    # Fallback strategy 2: Use image center
                    center_x_token = W2 // 2
                    center_y_token = H2 // 2
                    center_token_idx = int(center_y_token * W2 + center_x_token)
                    valid_idxs = [center_token_idx]
                    logger.warning(
                        f"[{self.ds_name}] Bbox center fallback failed, using image center token {center_token_idx} "
                        f"for bbox[{bbox_idx}]"
                    )
            
            # Check for invalid indices (should not happen after filtering, but keep for safety)
            invalid_indices = [idx for idx in valid_idxs if idx < 0 or idx >= H2 * W2]
            if invalid_indices:
                logger.error(
                    f"[{self.ds_name}] CRITICAL: Still have invalid token indices after filtering! "
                    f"bbox[{bbox_idx}]: {bbox}, invalid_indices={invalid_indices}, "
                    f"valid range: [0, {H2 * W2}). This should not happen!"
                )
                # Remove invalid indices
                valid_idxs = [idx for idx in valid_idxs if 0 <= idx < H2 * W2]
                
                # If still empty after filtering, use image center as last resort
                if len(valid_idxs) == 0:
                    center_x_token = W2 // 2
                    center_y_token = H2 // 2
                    center_token_idx = int(center_y_token * W2 + center_x_token)
                    valid_idxs = [center_token_idx]
                    logger.error(
                        f"[{self.ds_name}] CRITICAL: All tokens invalid, using image center token {center_token_idx} "
                        f"as last resort for bbox[{bbox_idx}]"
                    )
            
            # FINAL GUARANTEE: Ensure valid_idxs is never empty
            if len(valid_idxs) == 0:
                # Ultimate fallback: use first token (0, 0)
                center_token_idx = 0
                valid_idxs = [center_token_idx]
                worker_id_str = getattr(self, 'worker_id', 'unknown')
                logger.error(
                    f"[{self.ds_name}] ❌ CRITICAL: All fallback strategies failed for bbox[{bbox_idx}]: {bbox}, "
                    f"h={h}, w={w}, H2={H2}, W2={W2}, worker_id={worker_id_str}. Using token 0 as ultimate fallback."
                )
                print(f"[BBOX_ULTIMATE_FALLBACK] bbox={bbox}, h={h}, w={w}, H2={H2}, W2={W2}, "
                      f"worker_id={worker_id_str}, using_token_0=True", flush=True)

            token_idxs.append(valid_idxs)

        # FINAL VERIFICATION: Ensure no empty lists in token_idxs
        for bbox_idx, idx_list in enumerate(token_idxs):
            if len(idx_list) == 0:
                worker_id_str = getattr(self, 'worker_id', 'unknown')
                logger.error(
                    f"[{self.ds_name}] ❌ CRITICAL: Found empty token list at bbox_idx={bbox_idx} "
                    f"after all processing! h={h}, w={w}, H2={H2}, W2={W2}, worker_id={worker_id_str}. "
                    f"Using token 0 as emergency fallback."
                )
                print(f"[BBOX_EMERGENCY_FALLBACK] bbox_idx={bbox_idx}, h={h}, w={w}, H2={H2}, W2={W2}, "
                      f"worker_id={worker_id_str}, using_token_0=True", flush=True)
                token_idxs[bbox_idx] = [0]  # Emergency fallback: use token 0
        
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

            if isinstance(image_files, str):
                image_files = [image_files]

            images = []
            skipped_images = []

            # Get dataset name from sources if available
            dataset_name = sources.get('dataset', None)
            
            for image_file in image_files:
                # Map image path using dataset-specific mapping
                mapped_path = map_image_path(image_file, image_folder, dataset_name)
                image_info = get_image_info(mapped_path, self.image_min_pixel, self.image_max_pixel, self.image_resized_w, self.image_resized_h)
                # Skip if image file not found
                if image_info is None:
                    display_path = mapped_path if mapped_path is not None else image_file
                    skipped_images.append(display_path)
                    logger.warning(f"[{self.ds_name}] Skipping missing image - original: {image_file}, searched path: {display_path}")
                    continue
                images.append(image_info)
            
            # Skip this sample if no valid images found
            if len(images) == 0:
                logger.warning(f"[{self.ds_name}] Skipping sample {i} - all images missing: {skipped_images}")
                continue

            # Extract LVR tokens
            image_grid_thw = processor(text=[""], images=images, videos=videos, padding=False, do_resize=False, return_tensors='pt')['image_grid_thw']
            lvr_token_idxs_list = self.bbox_to_token_idxs(sources['bboxes'], image_grid_thw)
            
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
            if len(lvr_token_idxs_list) == 0:
                if self.worker_id == 0:
                    logger.warning(
                        f"[{self.ds_name}] Empty lvr_token_idxs_list for data_idx={i}, "
                        f"bboxes={sources.get('bboxes', [])}, "
                        f"image_grid_thw={image_grid_thw[0].tolist()}. "
                        f"Skipping this sample."
                    )
                continue

            sources = copy.deepcopy(llava_to_openai_lvr(sources['conversations'], is_video=is_video,lvr_token_idxs_list=lvr_token_idxs_list,latent_end_token=self.latent_end_token))

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
            print(f"[BEFORE_CREATE_LVR_TOKENS] data_idx={i}, worker_id={worker_id_str}, "
                  f"lvr_token_idxs_list_length={len(lvr_token_idxs_list)}, "
                  f"lvr_token_idxs_list={[[len(g) for g in lvr_token_idxs_list]]}, "
                  f"image_grid_thw={image_grid_thw[0].tolist()}", flush=True)
            
            for group_idx, group in enumerate(lvr_token_idxs_list):
                if len(group) == 0:
                    empty_lvr_tokens_found = True
                    # Always log, not just worker_id == 0, to catch all cases
                    logger.error(
                        f"[{self.ds_name}] ❌ CRITICAL: Empty lvr_token_idxs_list[{group_idx}] for bbox {sources.get('bboxes', [])[group_idx] if group_idx < len(sources.get('bboxes', [])) else 'unknown'}, "
                        f"image_grid_thw={image_grid_thw[0].tolist()}, "
                        f"data_idx={i}, worker_id={self.worker_id}. This will cause NaN loss!"
                    )
                    print(f"[EMPTY_GROUP_DETECTED] group_idx={group_idx}, data_idx={i}, worker_id={worker_id_str}, "
                          f"bbox={sources.get('bboxes', [])[group_idx] if group_idx < len(sources.get('bboxes', [])) else 'unknown'}, "
                          f"image_grid_thw={image_grid_thw[0].tolist()}", flush=True)
                
                token_tensor = torch.tensor(group, dtype=torch.int)
                print(f"[CREATE_LVR_TOKEN] group_idx={group_idx}, group_length={len(group)}, "
                      f"tensor_shape={token_tensor.shape}, tensor_numel={token_tensor.numel()}, "
                      f"empty={token_tensor.numel() == 0}, data_idx={i}, worker_id={worker_id_str}", flush=True)
                lvr_tokens.append(token_tensor)
            
            # CRITICAL: Double-check after creating tensors - skip if ANY lvr_tokens are empty
            # This prevents NaN loss from empty lvr_tokens
            for token_idx, token_tensor in enumerate(lvr_tokens):
                if token_tensor.numel() == 0:
                    empty_lvr_tokens_found = True
                    break
            
            if empty_lvr_tokens_found:
                # Always log and skip, regardless of worker_id
                logger.error(
                    f"[{self.ds_name}] ❌ CRITICAL: Empty lvr_tokens detected for data_idx={i}, worker_id={self.worker_id}! "
                    f"Bboxes: {sources.get('bboxes', [])}, "
                    f"image_grid_thw={image_grid_thw[0].tolist()}, "
                    f"lvr_token_idxs_list lengths: {[len(g) for g in lvr_token_idxs_list]}, "
                    f"lvr_tokens numel: {[t.numel() for t in lvr_tokens]}. "
                    f"Skipping this sample to prevent NaN loss."
                )
                print(f"[SKIP_SAMPLE] data_idx={i}, worker_id={self.worker_id}, bboxes={sources.get('bboxes', [])}, "
                      f"image_grid_thw={image_grid_thw[0].tolist()}, "
                      f"lvr_tokens_empty=True", flush=True)
                continue  # Skip this sample immediately

            data_dict = dict(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                lvr_tokens=lvr_tokens,
            )

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

        print("#"*42)
        print(f"Training with datasets and their weights:")
        for d,w in zip(datasets,self.dataset_weight):
            print(f"{d.ds_name}:\t{w}")
        print("#"*42)

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
                        
                        if empty_found:
                            # Always log and skip, not just when _should_log()
                            logger.error(
                                f"[PackedDataset] ❌ CRITICAL: Empty lvr_tokens detected in sample from {self.datasets[current_dataset_idx].ds_name}! "
                                f"worker_id={self.worker_id}, data_rank={self.data_rank}, "
                                f"lvr_tokens numel: {[t.numel() if isinstance(t, torch.Tensor) else len(t) for t in lvr_tokens]}. "
                                f"Skipping this sample to prevent NaN loss."
                            )
                            print(f"[SKIP_SAMPLE_PACKED] worker_id={self.worker_id}, data_rank={self.data_rank}, "
                                  f"ds_name={self.datasets[current_dataset_idx].ds_name}, "
                                  f"lvr_tokens_empty=True", flush=True)
                            continue  # Skip this sample and get next one
                
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
        
        '''
            Handles long single-/multi- instance buffer differently
        '''
        # condition 1: single instance
        if buffer['data_index'][-1].item() == 0:
            # condition 1.1: single long instance
            if buffer['input_ids'].size(0) >= long_seq_threshold:

                '''cut_id is the idx of the first token to be dropped'''
                cut_id = min(max_tokens, buffer['input_ids'].size(0))

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
                            print(f"[SPLIT_BUFFER_INVALID_CUT] cut_id_lvr={cut_id_lvr}, lvr_tokens_size={lvr_tokens_size}, "
                                  f"num_discarded_lvr_tokens={num_discarded_lvr_tokens}, cut_id={cut_id}, "
                                  f"using_full_lvr_tokens=True", flush=True)
                            cut_id_lvr = lvr_tokens_size  # Use full lvr_tokens to avoid empty tensor
                    else:
                        # If lvr_token_id is None or lvr_tokens is empty, keep all lvr_tokens
                        cut_id_lvr = None
                    
                    # Track if lvr_tokens were cleared, which means we should discard this buffer
                    should_discard_buffer = False
                    
                    for k in buffer:
                        if k in ['input_ids', 'labels', 'attention_mask', 'position_ids', 'data_index']:
                            buffer[k] = buffer[k][:cut_id]
                        elif k in ['pixel_values', 'image_flags','image_grid_thw']:
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
                                    print(f"[SPLIT_BUFFER_DISCARD_BUFFER] cut_id={cut_id}, remaining_lvr_tokens=0, "
                                          f"lvr_tokens_size={buffer[k][0].numel()}, discarding_buffer=True", flush=True)
                                    # Mark buffer for discard instead of clearing lvr_tokens
                                    should_discard_buffer = True
                                    lvr_tokens_cleared = True
                            
                            # Only process lvr_tokens if they weren't cleared and cut_id_lvr is set
                            if not lvr_tokens_cleared and cut_id_lvr is not None and len(buffer[k]) > 0:
                                # CRITICAL: Log before slicing for debugging
                                original_size = buffer[k][0].size(0)
                                print(f"[SPLIT_BUFFER_BEFORE_SLICE] cut_id_lvr={cut_id_lvr}, original_size={original_size}, "
                                      f"cut_id={cut_id}, num_discarded_lvr_tokens={num_discarded_lvr_tokens if 'num_discarded_lvr_tokens' in locals() else 'N/A'}", flush=True)
                                
                                # CRITICAL: Double-check before slicing to prevent empty tensor
                                if cut_id_lvr > 0 and cut_id_lvr <= buffer[k][0].size(0):
                                    buffer[k][0] = buffer[k][0][:cut_id_lvr]
                                    print(f"[SPLIT_BUFFER_AFTER_SLICE] cut_id_lvr={cut_id_lvr}, new_size={buffer[k][0].size(0)}, "
                                          f"empty={buffer[k][0].numel() == 0}", flush=True)
                                elif cut_id_lvr == 0:
                                    # If cut_id_lvr is 0, keep at least one token to avoid empty tensor
                                    logger.warning(
                                        f"[PackedDataset] ⚠️  cut_id_lvr=0 would create empty lvr_tokens, "
                                        f"keeping at least 1 token. Original size: {buffer[k][0].size(0)}"
                                    )
                                    print(f"[SPLIT_BUFFER_KEEP_ONE_TOKEN] cut_id_lvr=0, keeping 1 token instead", flush=True)
                                    buffer[k][0] = buffer[k][0][:1] if buffer[k][0].size(0) > 0 else buffer[k][0]
                                    print(f"[SPLIT_BUFFER_AFTER_KEEP_ONE] new_size={buffer[k][0].size(0)}, "
                                          f"empty={buffer[k][0].numel() == 0}", flush=True)
                                else:
                                    # If cut_id_lvr > buffer[k][0].size(0), keep full tensor
                                    print(f"[SPLIT_BUFFER_KEEP_FULL] cut_id_lvr={cut_id_lvr} > original_size={original_size}, "
                                          f"keeping full tensor", flush=True)
                                
                                # CRITICAL: Final check - ensure tensor is not empty
                                if buffer[k][0].numel() == 0:
                                    logger.error(
                                        f"[PackedDataset] ❌ CRITICAL: Empty lvr_tokens[0] after split_buffer! "
                                        f"cut_id_lvr={cut_id_lvr}, original_size={original_size}, "
                                        f"cut_id={cut_id}. Discarding buffer to avoid NaN loss."
                                    )
                                    print(f"[SPLIT_BUFFER_EMPTY_AFTER_SPLICE] cut_id_lvr={cut_id_lvr}, original_size={original_size}, "
                                          f"discarding_buffer=True", flush=True)
                                    should_discard_buffer = True
                            # If cut_id_lvr is None, keep lvr_tokens as is
                        elif k in ['input_lengths']:
                            pass
                        elif k.startswith('_debug_'):
                            # Skip debug keys, keep them as is
                            pass
                        else:
                            raise NotImplementedError(f'find unsupported keys: {k} from {buffer.keys()}')
                    
                    # CRITICAL: If lvr_tokens were cleared or became empty, discard this buffer
                    # This prevents empty lvr_tokens from reaching the forward function
                    if should_discard_buffer:
                        logger.error(
                            f"[PackedDataset] ❌ CRITICAL: Discarding buffer due to empty lvr_tokens after truncation. "
                            f"This prevents NaN loss. Buffer will be discarded and not added to buffer_ready."
                        )
                        print(f"[SPLIT_BUFFER_DISCARD_FINAL] discarding_buffer=True, buffer_not_added_to_ready=True", flush=True)
                        buffer_ready = []
                        buffer_unready = []
                    else:
                        # re-assign lengths
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
                                print(f"[SPLIT_BUFFER_COND2_2_EMPTY_LVR] idx={idx}, discarding_buffer=True", flush=True)
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
                            print(f"[SPLIT_BUFFER_COND2_2_MISMATCH] remaining_lvr_tokens=0, "
                                  f"lvr_tokens_has_values=True, discarding_buffer=True", flush=True)
                            should_discard_buffer_multi = True
                
                if should_discard_buffer_multi:
                    buffer_ready = []
                    buffer_unready = []
                else:
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
                                    print(f"[SPLIT_BUFFER_EMPTY_LVR] last_lvr_tokens_empty=True", flush=True)
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
                                            print(f"[SPLIT_BUFFER_EMPTY_AFTER_SPLIT] idx={idx}, using_fallback_token_0=True", flush=True)
                                            # Use token 0 as fallback
                                            buffer[k][idx] = torch.tensor([0], dtype=torch.int)
                                if len(buffer_right[k]) > 0:
                                    for idx, lvr_token_group in enumerate(buffer_right[k]):
                                        if isinstance(lvr_token_group, torch.Tensor) and lvr_token_group.numel() == 0:
                                            logger.error(
                                                f"[PackedDataset] ❌ CRITICAL: Empty lvr_tokens[{idx}] in buffer_right after split! "
                                                f"Using fallback token 0."
                                            )
                                            print(f"[SPLIT_BUFFER_RIGHT_EMPTY] idx={idx}, using_fallback_token_0=True", flush=True)
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
                    # CRITICAL: Check if lvr_tokens are empty before adding to buffer_ready
                    # Check left buffer
                    left_buffer_valid = True
                    if 'lvr_tokens' in buffer and lvr_token_id is not None:
                        remaining_lvr_tokens_left = (buffer['input_ids'] == lvr_token_id).sum().item()
                        if len(buffer['lvr_tokens']) > 0:
                            for idx, lvr_token_group in enumerate(buffer['lvr_tokens']):
                                if isinstance(lvr_token_group, torch.Tensor) and lvr_token_group.numel() == 0:
                                    left_buffer_valid = False
                                    logger.error(
                                        f"[PackedDataset] ❌ CRITICAL: Empty lvr_tokens[{idx}] in left buffer after split in condition 2.3! "
                                        f"Discarding buffer."
                                    )
                                    print(f"[SPLIT_BUFFER_COND2_3_LEFT_EMPTY] idx={idx}, discarding_left_buffer=True", flush=True)
                                    break
                        if remaining_lvr_tokens_left == 0 and len(buffer['lvr_tokens']) > 0:
                            has_non_empty_lvr = any(
                                isinstance(t, torch.Tensor) and t.numel() > 0 
                                for t in buffer['lvr_tokens']
                            )
                            if has_non_empty_lvr:
                                left_buffer_valid = False
                                logger.error(
                                    f"[PackedDataset] ❌ CRITICAL: No <lvr> tokens in left buffer input_ids but lvr_tokens has values! "
                                    f"Discarding buffer."
                                )
                                print(f"[SPLIT_BUFFER_COND2_3_LEFT_MISMATCH] discarding_left_buffer=True", flush=True)
                    
                    # Check right buffer
                    right_buffer_valid = True
                    if 'lvr_tokens' in buffer_right and lvr_token_id is not None:
                        remaining_lvr_tokens_right = (buffer_right['input_ids'] == lvr_token_id).sum().item()
                        if len(buffer_right['lvr_tokens']) > 0:
                            for idx, lvr_token_group in enumerate(buffer_right['lvr_tokens']):
                                if isinstance(lvr_token_group, torch.Tensor) and lvr_token_group.numel() == 0:
                                    right_buffer_valid = False
                                    logger.error(
                                        f"[PackedDataset] ❌ CRITICAL: Empty lvr_tokens[{idx}] in right buffer after split in condition 2.3! "
                                        f"Discarding buffer."
                                    )
                                    print(f"[SPLIT_BUFFER_COND2_3_RIGHT_EMPTY] idx={idx}, discarding_right_buffer=True", flush=True)
                                    break
                        if remaining_lvr_tokens_right == 0 and len(buffer_right['lvr_tokens']) > 0:
                            has_non_empty_lvr = any(
                                isinstance(t, torch.Tensor) and t.numel() > 0 
                                for t in buffer_right['lvr_tokens']
                            )
                            if has_non_empty_lvr:
                                right_buffer_valid = False
                                logger.error(
                                    f"[PackedDataset] ❌ CRITICAL: No <lvr> tokens in right buffer input_ids but lvr_tokens has values! "
                                    f"Discarding buffer."
                                )
                                print(f"[SPLIT_BUFFER_COND2_3_RIGHT_MISMATCH] discarding_right_buffer=True", flush=True)
                    
                    # if left buffer is longer
                    if buffer['input_ids'].size(0) >= buffer_right['input_ids'].size(0):
                        if left_buffer_valid:
                            buffer_ready.append(buffer)
                        buffer = buffer_right
                    else:   # buffer_right is longer than the accumulated left
                        if right_buffer_valid:
                            buffer_ready.append(buffer_right)

                # CRITICAL: Check final buffer before adding to buffer_unready
                final_buffer_valid = True
                if 'lvr_tokens' in buffer and lvr_token_id is not None:
                    remaining_lvr_tokens_final = (buffer['input_ids'] == lvr_token_id).sum().item()
                    if len(buffer['lvr_tokens']) > 0:
                        for idx, lvr_token_group in enumerate(buffer['lvr_tokens']):
                            if isinstance(lvr_token_group, torch.Tensor) and lvr_token_group.numel() == 0:
                                final_buffer_valid = False
                                logger.error(
                                    f"[PackedDataset] ❌ CRITICAL: Empty lvr_tokens[{idx}] in final buffer in condition 2.3! "
                                    f"Discarding buffer."
                                )
                                print(f"[SPLIT_BUFFER_COND2_3_FINAL_EMPTY] idx={idx}, discarding_final_buffer=True", flush=True)
                                break
                    if remaining_lvr_tokens_final == 0 and len(buffer['lvr_tokens']) > 0:
                        has_non_empty_lvr = any(
                            isinstance(t, torch.Tensor) and t.numel() > 0 
                            for t in buffer['lvr_tokens']
                        )
                        if has_non_empty_lvr:
                            final_buffer_valid = False
                            logger.error(
                                f"[PackedDataset] ❌ CRITICAL: No <lvr> tokens in final buffer input_ids but lvr_tokens has values! "
                                f"Discarding buffer."
                            )
                            print(f"[SPLIT_BUFFER_COND2_3_FINAL_MISMATCH] discarding_final_buffer=True", flush=True)
                
                # if buffer['input_ids'].size(0) <= max_tokens and PackedDataset.check_valid(buffer):
                if final_buffer_valid:
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
        
        # Add debug information if available
        if all_debug_questions:
            data_dict['_debug_question'] = all_debug_questions
        if all_debug_answers:
            data_dict['_debug_answer'] = all_debug_answers
        if all_debug_image_paths:
            data_dict['_debug_image_paths'] = all_debug_image_paths
        if all_debug_bboxes:
            data_dict['_debug_bboxes'] = all_debug_bboxes
        if all_debug_data_indices:
            data_dict['_debug_data_idx'] = all_debug_data_indices
        
        return data_dict