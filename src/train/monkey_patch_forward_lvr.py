import os
import warnings

import torch
from typing import Optional, List, Union, Tuple
from torch.nn import CrossEntropyLoss, MSELoss, L1Loss
import numpy as np
import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl
from transformers.utils import is_torchdynamo_compiling
# TransformersKwargs is not available in transformers 4.51.3, using Any as fallback
from typing import Any
TransformersKwargs = Any  # Type alias for kwargs compatibility
from transformers.processing_utils import Unpack
from src.constants import IGNORE_INDEX
import torch.distributed as dist

import torch.nn.functional as F


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


def replace_qwen2_5_with_mixed_modality_forward_lvr(inference_mode=False,
                                                    coconut=True,
                                                    lvr_head=True,
                                                    mode_switch_loss=False,
                                                    latent_end_token=False,
                                                    use_box_feature_resampler=False,
                                                    use_stage2_distillation=False,
                                                    rl=False):
    
    if inference_mode:
        if lvr_head:
            transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLForConditionalGeneration.forward = qwen2_5_mixed_modality_forward_lvr_with_head_inference
        elif use_box_feature_resampler or use_stage2_distillation:
            transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLForConditionalGeneration.forward = qwen2_5_mixed_modality_forward_lvr_with_resampler_inference
        else:
            transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLForConditionalGeneration.forward = qwen2_5_mixed_modality_forward_lvr_inference
    elif rl:
        transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLForConditionalGeneration.forward = qwen2_5_mixed_modality_forward_lvr_grpo
    else:
        if latent_end_token and lvr_head:
            transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLForConditionalGeneration.forward = qwen2_5_mixed_modality_forward_lvr_with_head_with_latentEndToken
        elif (latent_end_token or use_box_feature_resampler or use_stage2_distillation) and not lvr_head:
            transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLForConditionalGeneration.forward = qwen2_5_mixed_modality_forward_lvr_with_latentEndToken
        elif mode_switch_loss:
            transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLForConditionalGeneration.forward = qwen2_5_mixed_modality_forward_lvr_with_head_with_modeSwitchLoss
        elif lvr_head:
            transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLForConditionalGeneration.forward = qwen2_5_mixed_modality_forward_lvr_with_head
        else:
            transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLForConditionalGeneration.forward = qwen2_5_mixed_modality_forward_lvr


def set_lvr_loss_fct(loss_lvr_fct: str):
    """
        Set the loss function for LVR.
        Args:
            loss_lvr_fct (str): The type of loss function to use for LVR.
        Returns:
            A loss function object.
    """
    if loss_lvr_fct == 'mse':
        return MSELoss()
    elif loss_lvr_fct == 'mae':
        return L1Loss()
    elif loss_lvr_fct == 'cosine':
        # Returns a loss function: 1 - cosine similarity
        def cosine_loss(x, y):
            return 1 - F.cosine_similarity(x, y, dim=-1).mean()
        return cosine_loss
    else:
        raise ValueError(f"Unsupported lvr_loss: {loss_lvr_fct}")


def safe_assign_lvr_embeds(inputs_embeds, batch_indices, seq_positions, selected_lvr_embeds, location_hint=""):
    """
    Safely assign LVR embeddings to inputs_embeds with shape validation.
    
    This prevents RuntimeError: shape mismatch when lvr_tokens count doesn't match
    the number of <lvr> token positions in input_ids (corrupted data).
    
    Args:
        inputs_embeds: The input embeddings tensor to modify
        batch_indices: Batch indices for assignment
        seq_positions: Sequence positions for assignment
        selected_lvr_embeds: The LVR embeddings to assign
        location_hint: String to help identify which forward function called this
    
    Returns:
        True if assignment was successful, False if skipped due to shape mismatch
    """
    expected_positions = seq_positions.shape[0]
    actual_embeds = selected_lvr_embeds.shape[0]
    
    if expected_positions != actual_embeds:
        warnings.warn(
            f"[SKIP LVR {location_hint}] Shape mismatch detected: "
            f"seq_positions has {expected_positions} positions but "
            f"selected_lvr_embeds has {actual_embeds} embeddings. "
            f"This indicates corrupted data. Skipping LVR replacement for this batch."
        )
        return False
    
    inputs_embeds[batch_indices, seq_positions] = selected_lvr_embeds
    return True


'''
    Coconut mode
    No LVR Head
'''
def qwen2_5_mixed_modality_forward_lvr(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    rope_deltas: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
    lvr_tokens: Optional[torch.Tensor] = None,      # This is for TRAINING: Where should the lvr img tokens be
    lvr_tokens_thw: Optional[List[torch.Tensor]] = None,      # This is for TRAINING: Where should the lvr img tokens be
    lvr_mode_switch: Optional[torch.Tensor] = None, # This is for INFERENCE: Which instance in the batch is in lvr mode
    last_position_hidden_state: Optional[torch.FloatTensor] = None, # This is for INFERENCE: last hidden state of the last position
) -> Union[Tuple, "Qwen2_5_VLCausalLMOutputWithPast"]:
    
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if inputs_embeds is None:
        inputs_embeds = self.model.get_input_embeddings()(input_ids)

    if lvr_mode_switch:
        # only happen during inference
        # in fact, each instance's seq_len will be 1 in inference
        inputs_embeds[lvr_mode_switch,-1,:] = last_position_hidden_state[lvr_mode_switch]

    ''' 
        only happen during inference 
        inputs_embeds in shape (bs, seq_len, hidden)
    '''
    if last_position_hidden_state is not None:
        # in fact, each instance's seq_len will be 1 in inference
        inputs_embeds[lvr_mode_switch,-1,:] = last_position_hidden_state[lvr_mode_switch]

    '''Only necessary in training'''
    # Pass dummy image and dummy grid to the visual model to avoid deepspeed error.
    if not lvr_mode_switch and (pixel_values is None and pixel_values_videos is None):
        # Create dummy pixel_values and grid_thw for avoiding deepspeed error.
        dummy_pixel = torch.zeros(784, 1176).to(self.model.visual.device)
        dummy_grid = torch.tensor([[1, 28, 28]]).to(self.model.visual.device)
        
        dummy_pixel = dummy_pixel.type(self.model.visual.dtype)
        image_embeds = self.model.visual(dummy_pixel, grid_thw=dummy_grid)
        # Operates as maksed_scatter for the image tokens
        # However the values are all zeros so it dosen't affect the embeddings.
        # This could avoid deepspeed error when some batch only has texts.
        inputs_embeds += image_embeds.mean() * 0
            
    if pixel_values is not None:
            
        # FIX: Use self.get_image_features() instead of self.model.get_image_features()
        # self = Qwen2_5_VLForConditionalGeneration (has get_image_features)
        # self.model = Qwen2_5_VLModel (does NOT have get_image_features)
        image_embeds = self.get_image_features(pixel_values, image_grid_thw)
        image_embeds = torch.cat(image_embeds, dim=0)
    
        n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
        n_image_features = image_embeds.shape[0]
        if n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )

        if input_ids is None:
            image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            image_mask = image_mask.all(-1)
        else:
            image_mask = input_ids == self.config.image_token_id


        n_image_tokens = (image_mask).sum()
        image_mask_unsqueeze = image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        n_image_features = image_embeds.shape[0]
        if not is_torchdynamo_compiling() and n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )
        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask_unsqueeze, image_embeds)

        # IN TRAINING should we fill the lvr token positions with selected img tokrnd
        if lvr_tokens is not None:
            '''
                Filling the lvr tokens with image embeddings.
                Applicable when each image input has multiple bboxes
            '''
            total_tokens = torch.sum(image_mask, dim=1)   # 1d tensor([216, 234, 234, 234]) for #vis_tokens in each instance in batch
            batch_size = input_ids.size(0) 
            # lvr mask for lvr token locations in the batch, [bs, seq_length]
            # in each instance, lvr tokens are True, others are False
            lvr_mask = input_ids == self.config.lvr_id  
            # Total length = number of <lvr> tokens in the batch
            # seq_positions: flattend LOCAL positions of lvr tokens in the inputs_ids
            batch_indices, seq_positions = torch.nonzero(lvr_mask, as_tuple=True)  

            if isinstance(lvr_tokens,list):
                '''Exrtacting tokens from original image'''
                #  GLOBAL starting index in `image_embeds` of each image in the batch
                image_token_offsets = torch.cumsum(
                    F.pad(total_tokens, (1, 0)), dim=0
                )[:-1]  # shape [B], offset into image_embeds for each batch element

                global_lvr_token_indices = []
                num_latent_tokens = getattr(self.config, 'num_latent_tokens', 8)
                num_bboxes = len(batch_indices) // num_latent_tokens if len(batch_indices) > 0 else 0

                if len(lvr_tokens) == batch_size:
                    # One entry per batch item: lvr_tokens[b] -> batch b
                    for b, lvr_ids in enumerate(lvr_tokens):
                        offset = image_token_offsets[b].item()
                        global_lvr_token_indices.append(lvr_ids + offset)
                elif len(lvr_tokens) == num_bboxes and num_bboxes > 0:
                    # One entry per bbox: map bbox index to batch via batch_indices (same as _prepare_bbox_region_features)
                    for i, lvr_ids in enumerate(lvr_tokens):
                        batch_idx = batch_indices[i * num_latent_tokens].item()
                        offset = image_token_offsets[batch_idx].item()
                        global_lvr_token_indices.append(lvr_ids + offset)
                else:
                    # Fallback: avoid IndexError by only iterating valid indices
                    n = min(len(lvr_tokens), image_token_offsets.shape[0])
                    for b in range(n):
                        offset = image_token_offsets[b].item()
                        global_lvr_token_indices.append(lvr_tokens[b] + offset)
                    if len(lvr_tokens) != image_token_offsets.shape[0]:
                        warnings.warn(
                            f"lvr_tokens length ({len(lvr_tokens)}) != batch_size ({batch_size}) and != num_bboxes ({num_bboxes}); "
                            f"processed first {n} items only."
                        )
                global_lvr_token_indices = torch.cat(global_lvr_token_indices, dim=0)  # [L_total]

                # Step 3: Gather the selected visual embeddings
                selected_lvr_embeds = image_embeds[global_lvr_token_indices]  # [L_total, H]

                # Step 4: Replace in input_embeds at the right batch and position
                # SAFETY CHECK: Validate shapes match before assignment to prevent crash
                expected_positions = seq_positions.shape[0]
                actual_embeds = selected_lvr_embeds.shape[0]
                if expected_positions != actual_embeds:
                    warnings.warn(
                        f"[SKIP LVR] Shape mismatch detected: seq_positions has {expected_positions} positions "
                        f"but selected_lvr_embeds has {actual_embeds} embeddings. "
                        f"This indicates corrupted data. Skipping LVR replacement for this batch."
                    )
                else:
                    safe_assign_lvr_embeds(inputs_embeds, batch_indices, seq_positions, selected_lvr_embeds, "forward")
            else:
                '''re-encode target area'''
                # Now lvr_tokens is pixel_values of the cropped targets
                # FIX: Use self.get_image_features() instead of self.model.get_image_features()
                selected_lvr_embeds = self.get_image_features(lvr_tokens, lvr_tokens_thw)
                selected_lvr_embeds = torch.cat(selected_lvr_embeds, dim=0)
                # SAFETY CHECK: Validate shapes match before assignment to prevent crash
                expected_positions = seq_positions.shape[0]
                actual_embeds = selected_lvr_embeds.shape[0]
                if expected_positions != actual_embeds:
                    warnings.warn(
                        f"[SKIP LVR] Shape mismatch detected: seq_positions has {expected_positions} positions "
                        f"but selected_lvr_embeds has {actual_embeds} embeddings. "
                        f"This indicates corrupted data. Skipping LVR replacement for this batch."
                    )
                else:
                    safe_assign_lvr_embeds(inputs_embeds, batch_indices, seq_positions, selected_lvr_embeds, "forward")

    if attention_mask is not None:
        attention_mask = attention_mask.to(inputs_embeds.device)

    if position_ids is None:
            # Calculate RoPE index once per generation in the pre-fill stage only.
            # When compiling, we can't check tensor values thus we check only input length
            # It is safe to assume that `length!=1` means we're in pre-fill because compiled
            # models currently cannot do asssisted decoding
            prefill_compiled_stage = is_torchdynamo_compiling() and (
                (input_ids is not None and input_ids.shape[1] != 1)
                or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
            )
            prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
                (cache_position is not None and cache_position[0] == 0)
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            )
            if (prefill_compiled_stage or prefill_noncompiled_stage) or self.model.rope_deltas is None:
                # get_rope_index is on Qwen2_5_VLModel (self.model), not on ForConditionalGeneration
                position_ids, rope_deltas = self.model.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    second_per_grid_ts=second_per_grid_ts,
                    attention_mask=attention_mask,
                )
                self.model.rope_deltas = rope_deltas
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, 1, -1).expand(3, batch_size, -1)
                if cache_position is not None:
                    delta = (cache_position[0] + self.model.rope_deltas).to(inputs_embeds.device)
                else:
                    delta = torch.zeros((batch_size, seq_length), device=inputs_embeds.device)
                delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=1)
                position_ids += delta.to(position_ids.device)

    # FIX: Use self.model() instead of self.model.language_model()
    # self.model = Qwen2_5_VLModel (IS the language model itself)
    # self.model.language_model does NOT exist
    outputs = self.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

    hidden_states = outputs[0]
    last_position_hidden_state = outputs.last_hidden_state[:,-1,:]
    logits = self.lm_head(hidden_states)

    lvr_loss_fct = set_lvr_loss_fct(self.config.loss_lvr_fct)

    loss = None
    loss_ce = None
    loss_lvr = None
    if labels is not None:
        # Upcast to float if we need to compute the loss to avoid potential precision issues
        logits = logits.float()
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Don't want CE loss for <lvr> token
        shift_labels = shift_labels.masked_fill(shift_labels == self.config.lvr_id, IGNORE_INDEX)

        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss_ce = loss_fct(shift_logits, shift_labels)

        # lvr loss
        # Get last hidden states for <lvr> token positions
        seq_positions_start = seq_positions - 1  # Now points to lvr_start
        ''' We need to convert to fp32 to avoid overflow by mse'''
        selected_hidden_states = hidden_states[batch_indices, seq_positions_start].to(torch.float32)  # [L_total, H]
        selected_lvr_embeds = selected_lvr_embeds.to(torch.float32)
        # Compute LVR loss between predicted and inserted lvr embeddings
        loss_lvr = lvr_loss_fct(selected_hidden_states, selected_lvr_embeds)

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return Qwen2_5_VLCausalLMOutputWithPast(
        # loss=loss,
        loss_ce=loss_ce,
        loss_lvr=loss_lvr,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=self.model.rope_deltas,
        last_position_hidden_state =last_position_hidden_state
    )


from transformers.modeling_outputs import ModelOutput
from dataclasses import dataclass

def _get_lvr_head_dtype(lvr_head) -> torch.dtype:
    """
    Get the dtype of LVR head parameters efficiently.
    Caches the result to avoid repeated parameter iteration.
    """
    if not hasattr(_get_lvr_head_dtype, '_cache'):
        _get_lvr_head_dtype._cache = {}
    
    # Use id() as cache key to avoid issues with object identity
    cache_key = id(lvr_head)
    if cache_key not in _get_lvr_head_dtype._cache:
        try:
            # Try to get dtype from first parameter (most efficient)
            dtype = next(lvr_head.parameters()).dtype
        except StopIteration:
            # No parameters, use default
            dtype = torch.bfloat16
        _get_lvr_head_dtype._cache[cache_key] = dtype
    
    return _get_lvr_head_dtype._cache[cache_key]

def _build_global_lvr_token_indices(lvr_tokens, image_token_offsets, batch_indices=None, num_latent_tokens=8):
    """
    Build global LVR token indices efficiently without list append + cat.
    
    Args:
        lvr_tokens: List of tensors, each containing local token indices (per batch item or per bbox)
        image_token_offsets: (batch_size,) tensor of offsets into image_embeds
        batch_indices: Optional (L,) tensor from lvr_mask.nonzero(); used when lvr_tokens is per-bbox
        num_latent_tokens: Tokens per bbox (default 8); used with batch_indices for per-bbox mapping
    
    Returns:
        global_lvr_token_indices: (L_total,) tensor of global indices
    """
    # Pre-allocate tensor instead of list append + cat (more memory efficient)
    total_lvr_tokens = sum(len(lvr_ids) for lvr_ids in lvr_tokens)
    if total_lvr_tokens == 0:
        return torch.empty(0, dtype=torch.long, device=image_token_offsets.device)
    batch_size = image_token_offsets.shape[0]
    num_bboxes = (len(batch_indices) // num_latent_tokens) if batch_indices is not None and len(batch_indices) > 0 else 0
    indices_list = []
    if len(lvr_tokens) == batch_size:
        # One entry per batch item
        for b, lvr_ids in enumerate(lvr_tokens):
            if len(lvr_ids) > 0:
                offset = image_token_offsets[b].item()
                indices_list.append(lvr_ids + offset)
    elif len(lvr_tokens) == num_bboxes and num_bboxes > 0 and batch_indices is not None:
        # One entry per bbox: map bbox index to batch via batch_indices
        for i, lvr_ids in enumerate(lvr_tokens):
            if len(lvr_ids) > 0:
                batch_idx = batch_indices[i * num_latent_tokens].item()
                offset = image_token_offsets[batch_idx].item()
                indices_list.append(lvr_ids + offset)
    else:
        # Fallback: only iterate valid indices to avoid IndexError
        n = min(len(lvr_tokens), batch_size)
        for b in range(n):
            lvr_ids = lvr_tokens[b]
            if len(lvr_ids) > 0:
                offset = image_token_offsets[b].item()
                indices_list.append(lvr_ids + offset)
    if len(indices_list) > 0:
        return torch.cat(indices_list, dim=0)
    else:
        return torch.empty(0, dtype=torch.long, device=image_token_offsets.device)

def _prepare_batched_image_embeds(
    image_embeds: torch.Tensor,
    total_tokens: torch.Tensor,
    batch_indices: torch.Tensor,
    target_dtype: Optional[torch.dtype] = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Prepare batched image embeddings for batch processing (MLP-mask or other image-based LVR heads).
    
    This function pads image embeddings of different lengths to enable efficient batch processing.
    
    Args:
        image_embeds: (total_num_tokens, hidden_size) - All image token embeddings concatenated
        total_tokens: (batch_size,) - Number of image tokens for each batch item
        batch_indices: (L_total,) - Batch indices of items to process (may contain duplicates)
        target_dtype: Optional target dtype for the output tensor. If None, uses image_embeds.dtype
    
    Returns:
        batched_image_embeds: (L_total, max_num_tokens, hidden_size) - Padded image embeddings
        image_attention_mask: (L_total, max_num_tokens) - True for valid tokens, False for padding
    """
    import os
    # Calculate offsets for each batch item in the original batch
    image_token_offsets = torch.cumsum(F.pad(total_tokens, (1, 0)), dim=0)[:-1]  # (batch_size,)
    
    # Get unique batch indices and their counts
    unique_batch_indices, inverse_indices = torch.unique(batch_indices, return_inverse=True)
    
    # Calculate number of image tokens for each unique batch item
    num_tokens_per_item = total_tokens[unique_batch_indices]  # (num_unique,)
    
    # Find maximum number of tokens
    max_num_tokens = num_tokens_per_item.max().item()
    
    # Get hidden size
    hidden_size = image_embeds.shape[1]
    device = image_embeds.device
    # Use target_dtype if provided, otherwise use image_embeds dtype
    dtype = target_dtype if target_dtype is not None else image_embeds.dtype
    
    # Initialize
    L_total = len(batch_indices)
    
    # Pre-compute dtype conversion flag to avoid checking in loop
    needs_dtype_conversion = target_dtype is not None and image_embeds.dtype != target_dtype
    
    # Vectorized approach: compute offsets and lengths for all items at once
    batch_indices_tensor = torch.tensor(batch_indices, device=device, dtype=torch.long)
    offsets = image_token_offsets[batch_indices_tensor]  # (L_total,)
    num_tokens_per_item = total_tokens[batch_indices_tensor]  # (L_total,)
    
    # Fill batched embeddings using vectorized operations where possible
    # CRITICAL: Use .detach() to avoid gradient computation on image_embeds
    # LVR head only needs gradients for hidden_state, not image_embeds
    # This prevents huge gradient communication in DeepSpeed ZeRO-3
    # IMPORTANT: Avoid inplace operations to prevent gradient computation errors
    batch_list = []
    mask_list = []
    for i in range(L_total):
        offset = offsets[i].item()
        num_tokens = num_tokens_per_item[i].item()
        
        if num_tokens > 0:
            # Extract and convert image embeddings for this batch item directly to target dtype
            if needs_dtype_conversion:
                # Convert slice directly to target dtype and copy, detach to avoid gradients
                slice_data = image_embeds[offset:offset+num_tokens].detach().to(target_dtype)
            else:
                # No conversion needed, but still detach to avoid gradients
                slice_data = image_embeds[offset:offset+num_tokens].detach()
        else:
            slice_data = torch.empty(0, hidden_size, device=device, dtype=dtype)
        
        # Pad to max_num_tokens
        if slice_data.shape[0] < max_num_tokens:
            padding = torch.zeros(max_num_tokens - slice_data.shape[0], hidden_size, device=device, dtype=dtype)
            padded_slice = torch.cat([slice_data, padding], dim=0)
        else:
            padded_slice = slice_data
        
        batch_list.append(padded_slice)
        
        # Create attention mask without inplace operations
        if num_tokens > 0:
            valid_mask = torch.ones(num_tokens, dtype=torch.bool, device=device)
            padding_mask = torch.zeros(max_num_tokens - num_tokens, dtype=torch.bool, device=device)
            mask = torch.cat([valid_mask, padding_mask], dim=0)
        else:
            mask = torch.zeros(max_num_tokens, dtype=torch.bool, device=device)
        mask_list.append(mask)
    
    # Stack all items without inplace operations
    batched_image_embeds = torch.stack(batch_list, dim=0)  # (L_total, max_num_tokens, hidden_size)
    image_attention_mask = torch.stack(mask_list, dim=0)  # (L_total, max_num_tokens)
    
    return batched_image_embeds, image_attention_mask


@dataclass
class Qwen2_5_VLCausalLMOutputWithPast(ModelOutput):
    """
        please refer to the original Qwen2_5_VLCausalLMOutputWithPast in transformers.models.qwen2_5_vl.modeling_qwen2_5_vl
    """

    loss: Optional[torch.FloatTensor] = None
    loss_lvr: Optional[torch.FloatTensor] = None
    loss_lvr_resampler: Optional[torch.FloatTensor] = None
    loss_ortho: Optional[torch.FloatTensor] = None
    loss_attn_div: Optional[torch.FloatTensor] = None
    loss_attn_guidance: Optional[torch.FloatTensor] = None
    loss_attn_transfer: Optional[torch.FloatTensor] = None
    loss_ce: Optional[torch.FloatTensor] = None
    loss_mode_switch: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    rope_deltas: Optional[torch.LongTensor] = None
    
    last_position_hidden_state: Optional[Tuple[torch.FloatTensor]] = None
    # next_pos_lvr:Optional[bool] = False


def  set_lvr_loss_fct(loss_lvr_fct: str):
    """
        Set the loss function for LVR.
        Args:
            loss_lvr_fct (str): The type of loss function to use for LVR.
        Returns:
            A loss function object.
    """
    if loss_lvr_fct == 'mse':
        return MSELoss()
    elif loss_lvr_fct == 'mae':
        return L1Loss()
    elif loss_lvr_fct == 'cosine':
        # Returns a loss function: 1 - cosine similarity. eps avoids NaN when norm is 0.
        def cosine_loss(x, y):
            return 1 - F.cosine_similarity(x, y, dim=-1, eps=1e-8).mean()
        return cosine_loss
    else:
        raise ValueError(f"Unsupported lvr_loss: {loss_lvr_fct}")


def _prepare_bbox_region_features(
    image_embeds: torch.Tensor,
    total_tokens: torch.Tensor,
    batch_indices: torch.Tensor,
    lvr_tokens: list,
    num_latent_tokens: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Build (num_bboxes, max_N, D) bbox region features and key_padding_mask for BoxFeatureResampler.
    batch_indices from lvr_mask has length 8*num_bboxes; lvr_tokens is list of length num_bboxes (one tensor per bbox).
    Returns bbox_region_features (L, max_N, D), key_padding_mask (L, max_N) with True = padding.
    """
    num_bboxes = len(batch_indices) // num_latent_tokens
    if num_bboxes == 0 or len(lvr_tokens) != num_bboxes:
        return torch.empty(0, 0, image_embeds.shape[1], device=device, dtype=dtype), None
    image_token_offsets = torch.cumsum(F.pad(total_tokens, (1, 0)), dim=0)[:-1]
    hidden_size = image_embeds.shape[1]
    feats_list = []
    mask_list = []
    for i in range(num_bboxes):
        batch_idx = batch_indices[i * num_latent_tokens].item()
        local_idx = lvr_tokens[i]
        if isinstance(local_idx, (list, tuple)):
            local_idx = torch.tensor(local_idx, device=device, dtype=torch.long)
        else:
            local_idx = local_idx.to(device=device)
        offset = image_token_offsets[batch_idx].item()
        global_idx = offset + local_idx
        feats = image_embeds[global_idx].detach()
        feats_list.append(feats)
        mask_list.append(torch.zeros(feats.shape[0], dtype=torch.bool, device=device))
    max_N = max(f.shape[0] for f in feats_list)
    batch_list = []
    mask_out_list = []
    for feats, valid_mask in zip(feats_list, mask_list):
        n = feats.shape[0]
        if n < max_N:
            pad = torch.zeros(max_N - n, hidden_size, device=device, dtype=feats.dtype)
            batch_list.append(torch.cat([feats, pad], dim=0))
            mask_out_list.append(torch.cat([valid_mask, torch.ones(max_N - n, dtype=torch.bool, device=device)], dim=0))
        else:
            batch_list.append(feats)
            mask_out_list.append(valid_mask)
    bbox_region_features = torch.stack(batch_list, dim=0).to(dtype=dtype)
    key_padding_mask = torch.stack(mask_out_list, dim=0)
    return bbox_region_features, key_padding_mask


def create_spatial_mask_from_lvr_tokens(
    lvr_tokens: list,
    batch_indices_per_bbox: torch.Tensor,
    image_grid_thw: torch.Tensor,
    max_seq_len: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Create spatial mask from lvr_tokens (bbox token indices).
    lvr_tokens[i] corresponds to the image at batch_indices_per_bbox[i].
    image_grid_thw: [batch_size, 3] as [T, H, W]; tokens per image = H*W//4.
    Returns: spatial_mask [num_bboxes, max_seq_len], 1.0=inside bbox, 0.0=background.
    """
    num_bboxes = len(lvr_tokens)
    if num_bboxes == 0:
        return torch.zeros(0, max_seq_len, device=device, dtype=torch.float32)
    spatial_masks = torch.zeros((num_bboxes, max_seq_len), device=device, dtype=torch.float32)
    for i in range(num_bboxes):
        batch_idx = batch_indices_per_bbox[i].item()
        seq_len = image_grid_thw[batch_idx, 1].item() * image_grid_thw[batch_idx, 2].item() // 4
        local_idx = lvr_tokens[i]
        if isinstance(local_idx, (list, tuple)):
            local_idx = torch.tensor(local_idx, device=device, dtype=torch.long)
        else:
            local_idx = local_idx.to(device=device)
        for idx in local_idx.tolist():
            if 0 <= idx < seq_len and idx < max_seq_len:
                spatial_masks[i, idx] = 1.0
    return spatial_masks


def get_aligned_teacher_attn_from_lvr_tokens(
    teacher_attn: torch.Tensor,
    lvr_tokens: list,
    key_padding_mask: torch.Tensor,
    max_seq_len: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Map teacher's local bbox attention [B, 8, max_N] to full-image [B, 8, max_seq_len].
    teacher_attn[i,:,j] corresponds to lvr_tokens[i][j]; key_padding_mask marks padding.
    """
    B, num_queries, _ = teacher_attn.shape
    aligned_attn = torch.zeros((B, num_queries, max_seq_len), device=device, dtype=teacher_attn.dtype)
    for i in range(B):
        num_valid = (~key_padding_mask[i]).sum().item()
        if num_valid == 0:
            continue
        local_idx = lvr_tokens[i]
        if isinstance(local_idx, (list, tuple)):
            local_idx = torch.tensor(local_idx, device=device, dtype=torch.long)
        else:
            local_idx = local_idx.to(device=device)
        valid_indices = local_idx[:num_valid]
        mask = valid_indices < max_seq_len
        if mask.any():
            aligned_attn[i, :, valid_indices[mask]] = teacher_attn[i, :, :num_valid][:, mask]
    return aligned_attn


def qwen2_5_mixed_modality_forward_lvr_inference(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    rope_deltas: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
    lvr_tokens: Optional[torch.Tensor] = None,      # This is for TRAINING: Where should the lvr img tokens be
    lvr_tokens_thw: Optional[List[torch.Tensor]] = None,      # This is for TRAINING: Where should the lvr img tokens be
    lvr_mode_switch: Optional[torch.Tensor] = None, # This is for INFERENCE: Which instance in the batch is in lvr mode
    last_position_hidden_state: Optional[torch.FloatTensor] = None, # This is for INFERENCE: last hidden state of the last position
    **kwargs: Unpack[TransformersKwargs],
) -> Union[Tuple, Qwen2_5_VLCausalLMOutputWithPast]:
    
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if inputs_embeds is None:
        inputs_embeds = self.model.get_input_embeddings()(input_ids)

    if last_position_hidden_state is not None:
        # only happen during inference
        # in fact, each instance's seq_len will be 1 in inference
        inputs_embeds[lvr_mode_switch,-1,:] = last_position_hidden_state[lvr_mode_switch]

    ''' 
        only happen during inference 
        inputs_embeds in shape (bs, seq_len, hidden)
    '''
    if last_position_hidden_state is not None:
        # in fact, each instance's seq_len will be 1 in inference
        inputs_embeds[lvr_mode_switch,-1,:] = last_position_hidden_state[lvr_mode_switch]

    '''Only necessary in training'''
    # Pass dummy image and dummy grid to the visual model to avoid deepspeed error.
    # if not lvr_mode_switch and (pixel_values is None and pixel_values_videos is None):
    #     # Create dummy pixel_values and grid_thw for avoiding deepspeed error.
    #     dummy_pixel = torch.zeros(784, 1176).to(self.model.visual.device)
    #     dummy_grid = torch.tensor([[1, 28, 28]]).to(self.model.visual.device)
        
    #     dummy_pixel = dummy_pixel.type(self.model.visual.dtype)
    #     image_embeds = self.model.visual(dummy_pixel, grid_thw=dummy_grid)
    #     # Operates as maksed_scatter for the image tokens
    #     # However the values are all zeros so it dosen't affect the embeddings.
    #     # This could avoid deepspeed error when some batch only has texts.
    #     inputs_embeds += image_embeds.mean() * 0
            
    if pixel_values is not None:
            
        image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
        # Handle both list and tensor returns from visual module
        if isinstance(image_embeds, (list, tuple)):
            image_embeds = torch.cat(image_embeds, dim=0)
        # If already a tensor, use it directly
    
        n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
        n_image_features = image_embeds.shape[0]
        if n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )

        if input_ids is None:
            image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            image_mask = image_mask.all(-1)
        else:
            image_mask = input_ids == self.config.image_token_id


        n_image_tokens = (image_mask).sum()
        image_mask_unsqueeze = image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        n_image_features = image_embeds.shape[0]
        if not is_torchdynamo_compiling() and n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )
        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask_unsqueeze, image_embeds)

        # IN TRAINING should we fill the lvr token positions with selected img tokrnd
        if lvr_tokens is not None:
            '''
                Filling the lvr tokens with image embeddings.
                Applicable when each image input has multiple bboxes
            '''
            total_tokens = torch.sum(image_mask, dim=1)   # 1d tensor([216, 234, 234, 234]) for #vis_tokens in each instance in batch
            batch_size = input_ids.size(0) 
            # lvr mask for lvr token locations in the batch, [bs, seq_length]
            # in each instance, lvr tokens are True, others are False
            lvr_mask = input_ids == self.config.lvr_id  
            # Total length = number of <lvr> tokens in the batch
            # seq_positions: flattend LOCAL positions of lvr tokens in the inputs_ids
            batch_indices, seq_positions = torch.nonzero(lvr_mask, as_tuple=True)  

            if isinstance(lvr_tokens,list):
                '''Fill LVR slots: either use_resampler_fill (GT box -> resampler -> 8 tokens) or raw image embeddings.'''
                num_latent_tokens = getattr(self.config, 'num_latent_tokens', 8)
                num_bboxes = len(batch_indices) // num_latent_tokens
                use_resampler_fill = (
                    getattr(self.config, 'use_box_feature_resampler', False)
                    and hasattr(self, 'box_feature_resampler')
                    and num_bboxes > 0
                    and num_latent_tokens * num_bboxes == len(batch_indices)
                    and len(lvr_tokens) == num_bboxes
                )
                if use_resampler_fill:
                    # GT box -> BoxFeatureResampler -> 8 tokens -> fill 8 latent slots (same as training)
                    bbox_feats, key_pad_mask = _prepare_bbox_region_features(
                        image_embeds, total_tokens, batch_indices, lvr_tokens, num_latent_tokens,
                        device=inputs_embeds.device, dtype=next(self.box_feature_resampler.parameters()).dtype,
                    )
                    if bbox_feats.shape[0] > 0 and bbox_feats.shape[1] > 0:
                        fill_embeds = self.box_feature_resampler(bbox_feats, key_padding_mask=key_pad_mask).detach()
                        fill_embeds = fill_embeds.to(inputs_embeds.dtype).view(-1, fill_embeds.shape[-1])
                        n_fill = min(len(batch_indices), fill_embeds.shape[0])
                        if n_fill != len(batch_indices) or n_fill != fill_embeds.shape[0]:
                            warnings.warn(
                                f"LVR resampler fill shape mismatch: batch_indices len={len(batch_indices)}, "
                                f"fill_embeds shape[0]={fill_embeds.shape[0]}; using n_fill={n_fill} to avoid crash."
                            )
                        if inputs_embeds.requires_grad:
                            inputs_embeds = inputs_embeds.clone()
                        if n_fill > 0:
                            inputs_embeds[batch_indices[:n_fill], seq_positions[:n_fill]] = fill_embeds[:n_fill]
                else:
                    '''Extracting tokens from original image'''
                    image_token_offsets = torch.cumsum(
                        F.pad(total_tokens, (1, 0)), dim=0
                    )[:-1]  # shape [B], offset into image_embeds for each batch element

                    num_latent_tokens = getattr(self.config, 'num_latent_tokens', 8)
                    global_lvr_token_indices = _build_global_lvr_token_indices(
                        lvr_tokens, image_token_offsets,
                        batch_indices=batch_indices, num_latent_tokens=num_latent_tokens,
                    )

                    selected_lvr_embeds = image_embeds[global_lvr_token_indices]  # [L_total, H]

                    if inputs_embeds.requires_grad:
                        inputs_embeds = inputs_embeds.clone()
                    n_fill = min(len(batch_indices), selected_lvr_embeds.shape[0])
                    if n_fill > 0:
                        inputs_embeds[batch_indices[:n_fill], seq_positions[:n_fill]] = selected_lvr_embeds[:n_fill]
            else:
                '''re-encode target area'''
                # Now lvr_tokens is pixel_values of the cropped targets
                # FIX: Use self.get_image_features() instead of self.model.get_image_features()
                selected_lvr_embeds = self.get_image_features(lvr_tokens, lvr_tokens_thw)
                selected_lvr_embeds = torch.cat(selected_lvr_embeds, dim=0)
                if inputs_embeds.requires_grad:
                    inputs_embeds = inputs_embeds.clone()
                n_fill = min(len(batch_indices), selected_lvr_embeds.shape[0])
                if n_fill > 0:
                    inputs_embeds[batch_indices[:n_fill], seq_positions[:n_fill]] = selected_lvr_embeds[:n_fill]

    if attention_mask is not None:
        attention_mask = attention_mask.to(inputs_embeds.device)

    if position_ids is None:
            # Calculate RoPE index once per generation in the pre-fill stage only.
            # When compiling, we can't check tensor values thus we check only input length
            # It is safe to assume that `length!=1` means we're in pre-fill because compiled
            # models currently cannot do asssisted decoding
            prefill_compiled_stage = is_torchdynamo_compiling() and (
                (input_ids is not None and input_ids.shape[1] != 1)
                or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
            )
            prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
                (cache_position is not None and cache_position[0] == 0)
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            )
            if (prefill_compiled_stage or prefill_noncompiled_stage) or self.model.rope_deltas is None:
                position_ids, rope_deltas = self.model.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    second_per_grid_ts=second_per_grid_ts,
                    attention_mask=attention_mask,
                )
                self.model.rope_deltas = rope_deltas
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, 1, -1).expand(3, batch_size, -1)
                if cache_position is not None:
                    delta = (cache_position[0] + self.model.rope_deltas).to(inputs_embeds.device)
                else:
                    delta = torch.zeros((batch_size, seq_length), device=inputs_embeds.device)
                delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=1)
                position_ids = position_ids.clone() + delta.to(position_ids.device)

    outputs = self.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs
        )

    hidden_states = outputs[0]
    last_position_hidden_state = outputs.last_hidden_state[:,-1,:]
    logits = self.lm_head(hidden_states)

    lvr_loss_fct = set_lvr_loss_fct(self.config.loss_lvr_fct)

    loss = None
    loss_ce = None
    loss_lvr = None
    if labels is not None:
        # Upcast to float if we need to compute the loss to avoid potential precision issues
        logits = logits.float()
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Don't want CE loss for <lvr> token
        shift_labels = shift_labels.masked_fill(shift_labels == self.config.lvr_id, IGNORE_INDEX)

        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss_ce = loss_fct(shift_logits, shift_labels)

        # lvr loss
        # Get last hidden states for <lvr> token positions
        seq_positions_start = seq_positions - 1  # Now points to lvr_start
        
        if hasattr(self.config, 'lvr_head_type') and (self.config.lvr_head_type == 'attention-mask'):
            # LVR loss: compute reconstruction loss between V_focal and grounding box image tokens
            # Use selected_lvr_embeds as ground truth (same as original LVR behavior)
            ''' We need to convert to fp32 to avoid overflow by mse'''
            # Batch processing for V_focal computation
            if len(batch_indices) > 0:
                # Get model parameter dtype first to avoid creating unnecessary float32 tensors
                model_dtype = _get_lvr_head_dtype(self.lvr_head)
                
                # Prepare batched hidden states directly in model dtype
                batched_hidden_states = hidden_states[batch_indices, seq_positions_start].to(model_dtype)  # (L_total, hidden_size)
                
                # Prepare batched image embeddings directly in model dtype to save memory
                batched_image_embeds, image_attention_mask = _prepare_batched_image_embeds(
                    image_embeds, total_tokens, batch_indices, target_dtype=model_dtype
                )
                
                # Batch compute V_focal in model dtype, then convert output to float32 for loss computation
                v_focal_tensor = self.lvr_head(batched_hidden_states, batched_image_embeds, image_attention_mask)  # (L_total, hidden_size)
                
                # Free memory: delete large intermediate tensors immediately (before dtype conversion)
                del batched_hidden_states, batched_image_embeds, image_attention_mask
                
                # Convert to float32 for loss computation (only convert what's needed)
                v_focal_tensor = v_focal_tensor.to(torch.float32)
                selected_lvr_embeds_fp32 = selected_lvr_embeds.to(torch.float32)  # (L_total, hidden_size)
                
                # Compute reconstruction loss (MSE/L1) between V_focal and selected_lvr_embeds
                loss_lvr = lvr_loss_fct(v_focal_tensor, selected_lvr_embeds_fp32)
                
                # Free memory: delete temporary tensors after loss computation
                del v_focal_tensor, selected_lvr_embeds_fp32
            else:
                loss_lvr = None
        else:
            ''' We need to convert to fp32 to avoid overflow by mse'''
            selected_hidden_states = hidden_states[batch_indices, seq_positions_start].to(torch.float32)  # [L_total, H]
            selected_lvr_embeds = selected_lvr_embeds.to(torch.float32)
            # Compute LVR loss between predicted and inserted lvr embeddings
            loss_lvr = lvr_loss_fct(selected_hidden_states, selected_lvr_embeds)

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return Qwen2_5_VLCausalLMOutputWithPast(
        # loss=loss,
        loss_ce=loss_ce,
        loss_lvr=loss_lvr,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=self.model.rope_deltas,
        last_position_hidden_state =last_position_hidden_state
    )


'''
    Inference mode with BoxFeatureResampler (no LVR head).
    This forward function is used for inferencing models trained with use_box_feature_resampler=True.
    
    Key differences from qwen2_5_mixed_modality_forward_lvr_inference:
    - In LVR mode, uses box_feature_resampler to process hidden states
    - Saves image_embeds during prefill for use during LVR thinking steps
'''
def qwen2_5_mixed_modality_forward_lvr_with_resampler_inference(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    rope_deltas: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
    lvr_tokens: Optional[torch.Tensor] = None,
    lvr_tokens_thw: Optional[List[torch.Tensor]] = None,
    lvr_mode_switch: Optional[torch.Tensor] = None,
    last_position_hidden_state: Optional[torch.FloatTensor] = None,
    **kwargs,
) -> Union[Tuple, Qwen2_5_VLCausalLMOutputWithPast]:
    """
    Inference forward for BoxFeatureResampler models.
    In LVR mode, uses box_feature_resampler to enhance hidden states if available.
    """
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if inputs_embeds is None:
        inputs_embeds = self.model.get_input_embeddings()(input_ids)

    # In LVR mode, replace current input with last position hidden state
    if last_position_hidden_state is not None and lvr_mode_switch is not None:
        inputs_embeds[lvr_mode_switch, -1, :] = last_position_hidden_state[lvr_mode_switch]

    # Track image embeddings for use in LVR mode
    image_embeds = None
    total_tokens = None
    
    if pixel_values is not None:
        image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
        if isinstance(image_embeds, (list, tuple)):
            image_embeds = torch.cat(image_embeds, dim=0)
        
        # Save image_embeds for LVR processing
        self._cached_image_embeds = image_embeds
        
        n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
        n_image_features = image_embeds.shape[0]
        if n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )

        if input_ids is None:
            image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            image_mask = image_mask.all(-1)
        else:
            image_mask = input_ids == self.config.image_token_id

        n_image_tokens = (image_mask).sum()
        image_mask_unsqueeze = image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        n_image_features = image_embeds.shape[0]
        if not is_torchdynamo_compiling() and n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )
        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask_unsqueeze, image_embeds)
        total_tokens = torch.sum(image_mask, dim=1)
        
        # Save total_tokens for LVR processing
        self._cached_total_tokens = total_tokens

    if attention_mask is not None:
        attention_mask = attention_mask.to(inputs_embeds.device)

    if position_ids is None:
        prefill_compiled_stage = is_torchdynamo_compiling() and (
            (input_ids is not None and input_ids.shape[1] != 1)
            or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
        )
        prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
            (cache_position is not None and cache_position[0] == 0)
            or (past_key_values is None or past_key_values.get_seq_length() == 0)
        )
        if (prefill_compiled_stage or prefill_noncompiled_stage) or self.model.rope_deltas is None:
            position_ids, rope_deltas = self.model.get_rope_index(
                input_ids,
                image_grid_thw,
                video_grid_thw,
                second_per_grid_ts=second_per_grid_ts,
                attention_mask=attention_mask,
            )
            self.model.rope_deltas = rope_deltas
        else:
            batch_size, seq_length, _ = inputs_embeds.shape
            position_ids = torch.arange(seq_length, device=inputs_embeds.device)
            position_ids = position_ids.view(1, 1, -1).expand(3, batch_size, -1)
            if cache_position is not None:
                delta = (cache_position[0] + self.model.rope_deltas).to(inputs_embeds.device)
            else:
                delta = torch.zeros((batch_size, seq_length), device=inputs_embeds.device)
            delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=1)
            position_ids = position_ids.clone() + delta.to(position_ids.device)

    outputs = self.model(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
        **kwargs
    )

    hidden_states = outputs[0]
    last_position_hidden_state = outputs.last_hidden_state[:, -1, :]
    
    # NOTE: Do NOT apply box_feature_resampler enhancement to hidden states here.
    # Training behavior: resampler output is used to FILL INPUT EMBEDDINGS at LVR positions,
    # NOT to modify OUTPUT hidden states.
    # Inference: we use last_position_hidden_state as input for next step (Coconut style).

    logits = self.lm_head(hidden_states)

    loss = None
    loss_ce = None
    loss_lvr = None

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return Qwen2_5_VLCausalLMOutputWithPast(
        loss_ce=loss_ce,
        loss_lvr=loss_lvr,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=self.model.rope_deltas,
        last_position_hidden_state=last_position_hidden_state
    )


'''
    Coconut mode;
    LVR head;
    Note that this forward function is used for inferencing all the LVR models with a LVR head
'''
def qwen2_5_mixed_modality_forward_lvr_with_head(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    rope_deltas: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
    lvr_tokens: Optional[torch.Tensor] = None,      # This is for TRAINING: Where should the lvr img tokens be
    lvr_tokens_thw: Optional[List[torch.Tensor]] = None,      # This is for TRAINING: Where should the lvr img tokens be
    lvr_mode_switch: Optional[torch.Tensor] = None, # This is for INFERENCE: Which instance in the batch is in lvr mode
    last_position_hidden_state: Optional[torch.FloatTensor] = None, # This is for INFERENCE: last hidden state of the last position
) -> Union[Tuple, Qwen2_5_VLCausalLMOutputWithPast]:
    
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    

    if inputs_embeds is None:
        inputs_embeds = self.model.get_input_embeddings()(input_ids)

    ''' 
        only happen during inference 
        inputs_embeds in shape (bs, seq_len, hidden)
    '''
    if last_position_hidden_state is not None:
        # in fact, each instance's seq_len will be 1 in inference
        inputs_embeds[lvr_mode_switch,-1,:] = last_position_hidden_state[lvr_mode_switch]
    
    ''' Only necessary in training '''
    # Pass dummy image and dummy grid to the visual model to avoid deepspeed error.
    if not lvr_mode_switch and (pixel_values is None and pixel_values_videos is None):
        # Create dummy pixel_values and grid_thw for avoiding deepspeed error.
        dummy_pixel = torch.zeros(784, 1176).to(self.visual.device)
        dummy_grid = torch.tensor([[1, 28, 28]]).to(self.visual.device)
        
        dummy_pixel = dummy_pixel.type(self.visual.dtype)
        image_embeds = self.visual(dummy_pixel, grid_thw=dummy_grid)
        # Operates as maksed_scatter for the image tokens
        # However the values are all zeros so it dosen't affect the embeddings.
        # This could avoid deepspeed error when some batch only has texts.
        inputs_embeds += image_embeds.mean() * 0
            
    if pixel_values is not None:

        # with torch.autocast(device_type='cuda', enabled=True, dtype=torch.float32):
        #     # Ensure vision tower inputs are float32
        #     pixel_values = pixel_values.to(torch.float32) 
        image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
        # Handle both list and tensor returns from visual module
        if isinstance(image_embeds, (list, tuple)):
            image_embeds = torch.cat(image_embeds, dim=0)
        # If already a tensor, use it directly


        n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
        n_image_features = image_embeds.shape[0]
        if n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )

        if input_ids is None:
            image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            image_mask = image_mask.all(-1)
        else:
            image_mask = input_ids == self.config.image_token_id


        n_image_tokens = (image_mask).sum()
        image_mask_unsqueeze = image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        n_image_features = image_embeds.shape[0]
        if not is_torchdynamo_compiling() and n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )
        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask_unsqueeze, image_embeds)

        # IN TRAINING should we fill the lvr token positions with selected img tokrnd
        if lvr_tokens is not None:
            '''
                Filling the lvr tokens with image embeddings.
                Applicable when each image input has multiple bboxes
            '''
            total_tokens = torch.sum(image_mask, dim=1)   # 1d tensor([216, 234, 234, 234]) for #vis_tokens in each instance in batch
            batch_size = input_ids.size(0) 
            # lvr mask for lvr token locations in the batch, [bs, seq_length]
            # in each instance, lvr tokens are True, others are False
            lvr_mask = input_ids == self.config.lvr_id  
            # Total length = number of <lvr> tokens in the batch
            # seq_positions: flattend LOCAL positions of lvr tokens in the inputs_ids
            batch_indices, seq_positions = torch.nonzero(lvr_mask, as_tuple=True)  
            if isinstance(lvr_tokens,list):
                '''Exrtacting tokens from original image'''
                #  GLOBAL starting index in `image_embeds` of each image in the batch
                image_token_offsets = torch.cumsum(
                    F.pad(total_tokens, (1, 0)), dim=0
                )[:-1]  # shape [B], offset into image_embeds for each batch element

                # Use helper function to handle different lvr_tokens vs batch_size scenarios
                num_latent_tokens = getattr(self.config, 'num_latent_tokens', 8)
                global_lvr_token_indices = _build_global_lvr_token_indices(
                    lvr_tokens, image_token_offsets, batch_indices, num_latent_tokens
                )

                # Step 3: Gather the selected visual embeddings
                if len(global_lvr_token_indices) > 0:
                    selected_lvr_embeds = image_embeds[global_lvr_token_indices]  # [L_total, H]
                    # Step 4: Replace in input_embeds at the right batch and position
                    safe_assign_lvr_embeds(inputs_embeds, batch_indices, seq_positions, selected_lvr_embeds, "forward")
            
            else:
                '''re-encode target area'''
                # Now lvr_tokens is pixel_values of the cropped targets
                # FIX: Use self.get_image_features() instead of self.model.get_image_features()
                selected_lvr_embeds = self.get_image_features(lvr_tokens, lvr_tokens_thw)
                selected_lvr_embeds = torch.cat(selected_lvr_embeds, dim=0)
                safe_assign_lvr_embeds(inputs_embeds, batch_indices, seq_positions, selected_lvr_embeds, "forward_reencode")

            

    if attention_mask is not None:
        attention_mask = attention_mask.to(inputs_embeds.device)

    if position_ids is None:
        # Calculate RoPE index once per generation in the pre-fill stage only.
        # When compiling, we can't check tensor values thus we check only input length
        # It is safe to assume that `length!=1` means we're in pre-fill because compiled
        # models currently cannot do asssisted decoding
        prefill_compiled_stage = is_torchdynamo_compiling() and (
            (input_ids is not None and input_ids.shape[1] != 1)
            or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
        )
        prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
            (cache_position is not None and cache_position[0] == 0)
            or (past_key_values is None or past_key_values.get_seq_length() == 0)
        )
        if (prefill_compiled_stage or prefill_noncompiled_stage) or self.model.rope_deltas is None:
            position_ids, rope_deltas = self.model.get_rope_index(
                input_ids,
                image_grid_thw,
                video_grid_thw,
                second_per_grid_ts=second_per_grid_ts,
                attention_mask=attention_mask,
            )
            self.model.rope_deltas = rope_deltas
        else:
            batch_size, seq_length, _ = inputs_embeds.shape
            position_ids = torch.arange(seq_length, device=inputs_embeds.device)
            position_ids = position_ids.view(1, 1, -1).expand(3, batch_size, -1)
            if cache_position is not None:
                delta = (cache_position[0] + self.model.rope_deltas).to(inputs_embeds.device)
            else:
                delta = torch.zeros((batch_size, seq_length), device=inputs_embeds.device)
            delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=1)
            position_ids = position_ids.clone() + delta.to(position_ids.device)

    outputs = self.model(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
    )

    '''apply lvr_head in training mode'''
    if lvr_tokens is not None and lvr_mask.any():
        # batch_indices, seq_positions = torch.nonzero(lvr_mask, as_tuple=True)
        if len(batch_indices) > 0:
            # Get last hidden states for <lvr> token positions, starting <lvr_start>
            seq_positions_start = seq_positions - 1  # shift left by 1 pos, now points to lvr_start
            
            if hasattr(self.config, 'lvr_head_type') and (self.config.lvr_head_type == 'attention-mask' or self.config.lvr_head_type == 'ivr' or self.config.lvr_head_type == 'gated-focus' or self.config.lvr_head_type == 'gfr' or self.config.lvr_head_type == 'intrinsic-similarity' or self.config.lvr_head_type == 'isg'):
                # MLP-mask and Intrinsic-Similarity LVR heads need image embeddings - batch processing
                # Get model dtype to avoid unnecessary dtype conversions
                model_dtype = _get_lvr_head_dtype(self.lvr_head)
                
                # Prepare batched hidden states in model dtype
                batched_hidden_states = outputs.last_hidden_state[batch_indices, seq_positions_start].to(model_dtype)  # (L_total, hidden_size)
                
                # Prepare batched image embeddings directly in model dtype to save memory
                batched_image_embeds, image_attention_mask = _prepare_batched_image_embeds(
                    image_embeds, total_tokens, batch_indices, target_dtype=model_dtype
                )
                
                # Batch call LVR head (works for attention-mask and other batch-processing heads)
                v_focal_batch = self.lvr_head(batched_hidden_states, batched_image_embeds, image_attention_mask)  # (L_total, hidden_size)
                
                # Batch replace hidden states (convert back to original dtype if needed)
                # Avoid inplace operation by cloning and reconstructing
                new_hidden_state = outputs.last_hidden_state.clone()
                new_hidden_state[batch_indices, seq_positions_start] = v_focal_batch.to(outputs.last_hidden_state.dtype)
                outputs.last_hidden_state = new_hidden_state
                
                # Free memory: delete all intermediate tensors
                del batched_hidden_states, batched_image_embeds, image_attention_mask, v_focal_batch
            else:
                # Original LVR head behavior (simple or glu)
                # Extract once to avoid double indexing
                hidden_states_lvr = outputs.last_hidden_state[batch_indices, seq_positions_start]
                # Avoid inplace operation by cloning and reconstructing
                new_hidden_state = outputs.last_hidden_state.clone()
                new_hidden_state[batch_indices, seq_positions_start] = self.lvr_head(hidden_states_lvr)
                outputs.last_hidden_state = new_hidden_state
                del hidden_states_lvr

    '''apply lvr_head in _inference mode'''
    if lvr_mode_switch:
        if hasattr(self.config, 'lvr_head_type') and (self.config.lvr_head_type == 'attention-mask' or self.config.lvr_head_type == 'ivr' or self.config.lvr_head_type == 'gated-focus' or self.config.lvr_head_type == 'gfr' or self.config.lvr_head_type == 'intrinsic-similarity' or self.config.lvr_head_type == 'isg'):
            # MLP-mask and Intrinsic-Similarity LVR heads in inference mode - batch processing
            if pixel_values is not None:
                # Get image embeddings (already computed above)
                # For inference, we need to map batch indices to image embeddings
                batch_size = outputs.last_hidden_state.shape[0]
                lvr_batch_indices = torch.nonzero(lvr_mode_switch, as_tuple=True)[0]  # Get indices where LVR mode is active
                
                if len(lvr_batch_indices) > 0 and 'total_tokens' in locals() and 'image_embeds' in locals():
                    # Batch processing for multiple LVR mode items
                    # Get hidden states for LVR mode items
                    batched_hidden_states = outputs.last_hidden_state[lvr_batch_indices, -1]  # (num_lvr_items, hidden_size)
                    
                    # Prepare batched image embeddings and attention mask
                    batched_image_embeds, image_attention_mask = _prepare_batched_image_embeds(
                        image_embeds, total_tokens, lvr_batch_indices
                    )
                    
                    # Batch call LVR head (works for attention-mask and other batch-processing heads)
                    v_focal_batch = self.lvr_head(batched_hidden_states, batched_image_embeds, image_attention_mask)  # (num_lvr_items, hidden_size)
                    
                    # Batch replace hidden states
                    outputs.last_hidden_state[lvr_batch_indices, -1] = v_focal_batch
                    
                    # Free memory
                    del batched_hidden_states, batched_image_embeds, image_attention_mask, v_focal_batch
                elif len(lvr_batch_indices) > 0:
                    # Fallback: use original behavior if image info not available
                    for b_idx in lvr_batch_indices:
                        hidden_state = outputs.last_hidden_state[b_idx, -1]
                        outputs.last_hidden_state[b_idx, -1] = self.lvr_head(hidden_state.unsqueeze(0)).squeeze(0)
            else:
                # No images available, use original behavior
                outputs.last_hidden_state[lvr_mode_switch,:,:] = self.lvr_head(outputs.last_hidden_state[lvr_mode_switch,:,:])
        else:
            # Original LVR head behavior (simple or glu)
            outputs.last_hidden_state[lvr_mode_switch,:,:] = self.lvr_head(outputs.last_hidden_state[lvr_mode_switch,:,:])

    hidden_states = outputs[0]
    last_position_hidden_state = outputs.last_hidden_state[:,-1,:]
    logits = self.lm_head(hidden_states)

    lvr_loss_fct = set_lvr_loss_fct(self.config.loss_lvr_fct)

    loss = None
    loss_ce = None
    loss_lvr = None
    if labels is not None:
        # Upcast to float if we need to compute the loss to avoid potential precision issues
        logits = logits.float()
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Don't want CE loss for <lvr> token
        shift_labels = shift_labels.masked_fill(shift_labels == self.config.lvr_id, IGNORE_INDEX)

        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss_ce = loss_fct(shift_logits, shift_labels)

        # lvr loss
        # Get last hidden states for <lvr> token positions
        seq_positions_start = seq_positions - 1  # Now points to lvr_start
        
        if hasattr(self.config, 'lvr_head_type') and (self.config.lvr_head_type == 'attention-mask' or self.config.lvr_head_type == 'ivr' or self.config.lvr_head_type == 'gated-focus' or self.config.lvr_head_type == 'gfr'):
            # MLP-mask LVR loss: compute reconstruction loss between V_focal and grounding box image tokens
            # Use selected_lvr_embeds as ground truth (same as original LVR behavior)
            ''' We need to convert to fp32 to avoid overflow by mse'''
            # Batch processing for V_focal computation
            if len(batch_indices) > 0:
                # Get model parameter dtype first to avoid creating unnecessary float32 tensors
                model_dtype = _get_lvr_head_dtype(self.lvr_head)
                
                # Prepare batched hidden states directly in model dtype
                batched_hidden_states = hidden_states[batch_indices, seq_positions_start].to(model_dtype)  # (L_total, hidden_size)
                
                # Prepare batched image embeddings directly in model dtype to save memory
                batched_image_embeds, image_attention_mask = _prepare_batched_image_embeds(
                    image_embeds, total_tokens, batch_indices, target_dtype=model_dtype
                )
                
                # Batch compute V_focal in model dtype, then convert output to float32 for loss computation
                v_focal_tensor = self.lvr_head(batched_hidden_states, batched_image_embeds, image_attention_mask)  # (L_total, hidden_size)
                
                # Free memory: delete large intermediate tensors immediately (before dtype conversion)
                del batched_hidden_states, batched_image_embeds, image_attention_mask
                
                # Convert to float32 for loss computation (only convert what's needed)
                v_focal_tensor = v_focal_tensor.to(torch.float32)
                selected_lvr_embeds_fp32 = selected_lvr_embeds.to(torch.float32)  # (L_total, hidden_size)
                
                # Compute reconstruction loss (MSE/L1) between V_focal and selected_lvr_embeds
                loss_lvr = lvr_loss_fct(v_focal_tensor, selected_lvr_embeds_fp32)
                
                # Free memory: delete temporary tensors after loss computation
                del v_focal_tensor, selected_lvr_embeds_fp32
            else:
                loss_lvr = None
        else:
            ''' We need to convert to fp32 to avoid overflow by mse'''
            selected_hidden_states = hidden_states[batch_indices, seq_positions_start].to(torch.float32)  # [L_total, H]
            selected_lvr_embeds = selected_lvr_embeds.to(torch.float32)
            # Compute LVR loss between predicted and inserted lvr embeddings
            loss_lvr = lvr_loss_fct(selected_hidden_states, selected_lvr_embeds)


    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return Qwen2_5_VLCausalLMOutputWithPast(
        # loss=loss,
        loss_ce=loss_ce,
        loss_lvr=loss_lvr,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=self.model.rope_deltas,
        last_position_hidden_state =last_position_hidden_state
    )





def qwen2_5_mixed_modality_forward_lvr_with_head_inference(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    rope_deltas: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
    lvr_tokens: Optional[torch.Tensor] = None,      # This is for TRAINING: Where should the lvr img tokens be
    lvr_tokens_thw: Optional[List[torch.Tensor]] = None,      # This is for TRAINING: Where should the lvr img tokens be
    lvr_mode_switch: Optional[torch.Tensor] = None, # This is for INFERENCE: Which instance in the batch is in lvr mode
    last_position_hidden_state: Optional[torch.FloatTensor] = None, # This is for INFERENCE: last hidden state of the last position
    **kwargs: Unpack[TransformersKwargs],
) -> Union[Tuple, Qwen2_5_VLCausalLMOutputWithPast]:
    
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    

    if inputs_embeds is None:
        inputs_embeds = self.model.get_input_embeddings()(input_ids)

    ''' 
        only happen during inference 
        inputs_embeds in shape (bs, seq_len, hidden)
    '''
    if last_position_hidden_state is not None:
        # in fact, each instance's seq_len will be 1 in inference
        inputs_embeds[lvr_mode_switch,-1,:] = last_position_hidden_state[lvr_mode_switch]
    
    ''' Only necessary in training '''
    # Pass dummy image and dummy grid to the visual model to avoid deepspeed error.
    if not lvr_mode_switch and (pixel_values is None and pixel_values_videos is None):
        # Create dummy pixel_values and grid_thw for avoiding deepspeed error.
        dummy_pixel = torch.zeros(784, 1176).to(self.visual.device)
        dummy_grid = torch.tensor([[1, 28, 28]]).to(self.visual.device)
        
        dummy_pixel = dummy_pixel.type(self.visual.dtype)
        image_embeds = self.visual(dummy_pixel, grid_thw=dummy_grid)
        # Operates as maksed_scatter for the image tokens
        # However the values are all zeros so it dosen't affect the embeddings.
        # This could avoid deepspeed error when some batch only has texts.
        inputs_embeds += image_embeds.mean() * 0
            
    if pixel_values is not None:

        # with torch.autocast(device_type='cuda', enabled=True, dtype=torch.float32):
        #     # Ensure vision tower inputs are float32
        #     pixel_values = pixel_values.to(torch.float32) 
        image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
        # Handle both list and tensor returns from visual module
        if isinstance(image_embeds, (list, tuple)):
            image_embeds = torch.cat(image_embeds, dim=0)
        # If already a tensor, use it directly


        n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
        n_image_features = image_embeds.shape[0]
        if n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )

        if input_ids is None:
            image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            image_mask = image_mask.all(-1)
        else:
            image_mask = input_ids == self.config.image_token_id


        n_image_tokens = (image_mask).sum()
        image_mask_unsqueeze = image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        n_image_features = image_embeds.shape[0]
        if not is_torchdynamo_compiling() and n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )
        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask_unsqueeze, image_embeds)

        # IN TRAINING should we fill the lvr token positions with selected img tokrnd
        if lvr_tokens is not None:
            '''
                Filling the lvr tokens with image embeddings.
                Applicable when each image input has multiple bboxes
            '''
            total_tokens = torch.sum(image_mask, dim=1)   # 1d tensor([216, 234, 234, 234]) for #vis_tokens in each instance in batch
            batch_size = input_ids.size(0) 
            # lvr mask for lvr token locations in the batch, [bs, seq_length]
            # in each instance, lvr tokens are True, others are False
            lvr_mask = input_ids == self.config.lvr_id  
            # Total length = number of <lvr> tokens in the batch
            # seq_positions: flattend LOCAL positions of lvr tokens in the inputs_ids
            batch_indices, seq_positions = torch.nonzero(lvr_mask, as_tuple=True)  
            if isinstance(lvr_tokens,list):
                '''Exrtacting tokens from original image'''
                #  GLOBAL starting index in `image_embeds` of each image in the batch
                image_token_offsets = torch.cumsum(
                    F.pad(total_tokens, (1, 0)), dim=0
                )[:-1]  # shape [B], offset into image_embeds for each batch element

                # Use helper function to handle different lvr_tokens vs batch_size scenarios
                num_latent_tokens = getattr(self.config, 'num_latent_tokens', 8)
                global_lvr_token_indices = _build_global_lvr_token_indices(
                    lvr_tokens, image_token_offsets, batch_indices, num_latent_tokens
                )

                # Step 3: Gather the selected visual embeddings
                if len(global_lvr_token_indices) > 0:
                    selected_lvr_embeds = image_embeds[global_lvr_token_indices]  # [L_total, H]
                    # Step 4: Replace in input_embeds at the right batch and position
                    safe_assign_lvr_embeds(inputs_embeds, batch_indices, seq_positions, selected_lvr_embeds, "forward")
            
            else:
                '''re-encode target area'''
                # Now lvr_tokens is pixel_values of the cropped targets
                # FIX: Use self.get_image_features() instead of self.model.get_image_features()
                selected_lvr_embeds = self.get_image_features(lvr_tokens, lvr_tokens_thw)
                selected_lvr_embeds = torch.cat(selected_lvr_embeds, dim=0)
                safe_assign_lvr_embeds(inputs_embeds, batch_indices, seq_positions, selected_lvr_embeds, "forward_reencode")

            

    if attention_mask is not None:
        attention_mask = attention_mask.to(inputs_embeds.device)

    if position_ids is None:
        # Calculate RoPE index once per generation in the pre-fill stage only.
        # When compiling, we can't check tensor values thus we check only input length
        # It is safe to assume that `length!=1` means we're in pre-fill because compiled
        # models currently cannot do asssisted decoding
        prefill_compiled_stage = is_torchdynamo_compiling() and (
            (input_ids is not None and input_ids.shape[1] != 1)
            or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
        )
        prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
            (cache_position is not None and cache_position[0] == 0)
            or (past_key_values is None or past_key_values.get_seq_length() == 0)
        )
        if (prefill_compiled_stage or prefill_noncompiled_stage) or self.model.rope_deltas is None:
            position_ids, rope_deltas = self.model.get_rope_index(
                input_ids,
                image_grid_thw,
                video_grid_thw,
                second_per_grid_ts=second_per_grid_ts,
                attention_mask=attention_mask,
            )
            self.model.rope_deltas = rope_deltas
        else:
            batch_size, seq_length, _ = inputs_embeds.shape
            position_ids = torch.arange(seq_length, device=inputs_embeds.device)
            position_ids = position_ids.view(1, 1, -1).expand(3, batch_size, -1)
            if cache_position is not None:
                delta = (cache_position[0] + self.model.rope_deltas).to(inputs_embeds.device)
            else:
                delta = torch.zeros((batch_size, seq_length), device=inputs_embeds.device)
            delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=1)
            position_ids = position_ids.clone() + delta.to(position_ids.device)

    outputs = self.model(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
        **kwargs
    )

    '''apply lvr_head in training mode'''
    if lvr_tokens is not None and lvr_mask.any():
        # batch_indices, seq_positions = torch.nonzero(lvr_mask, as_tuple=True)
        if len(batch_indices) > 0:
            # Get last hidden states for <lvr> token positions, starting <lvr_start>
            seq_positions_start = seq_positions - 1  # shift left by 1 pos, now points to lvr_start
            
            if hasattr(self.config, 'lvr_head_type') and (self.config.lvr_head_type == 'attention-mask' or self.config.lvr_head_type == 'ivr' or self.config.lvr_head_type == 'gated-focus' or self.config.lvr_head_type == 'gfr' or self.config.lvr_head_type == 'intrinsic-similarity' or self.config.lvr_head_type == 'isg'):
                # MLP-mask LVR head needs image embeddings - batch processing
                # Get model dtype to avoid unnecessary dtype conversions
                model_dtype = _get_lvr_head_dtype(self.lvr_head)
                batched_hidden_states = outputs.last_hidden_state[batch_indices, seq_positions_start].to(model_dtype)  # (L_total, hidden_size)
                batched_image_embeds, image_attention_mask = _prepare_batched_image_embeds(
                    image_embeds, total_tokens, batch_indices, target_dtype=model_dtype
                )
                v_focal_batch = self.lvr_head(batched_hidden_states, batched_image_embeds, image_attention_mask)  # (L_total, hidden_size)
                outputs.last_hidden_state[batch_indices, seq_positions_start] = v_focal_batch
                # Free memory
                del batched_hidden_states, batched_image_embeds, image_attention_mask, v_focal_batch
            else:
                outputs.last_hidden_state[batch_indices, seq_positions_start] = self.lvr_head(outputs.last_hidden_state[batch_indices, seq_positions_start])

    '''apply lvr_head in _inference mode'''
    if lvr_mode_switch:
        if hasattr(self.config, 'lvr_head_type') and (self.config.lvr_head_type == 'attention-mask' or self.config.lvr_head_type == 'ivr' or self.config.lvr_head_type == 'gated-focus' or self.config.lvr_head_type == 'gfr' or self.config.lvr_head_type == 'intrinsic-similarity' or self.config.lvr_head_type == 'isg'):
            # MLP-mask LVR head in inference mode - batch processing
            # For intrinsic-similarity/isg head type, image_embeds is required
            requires_image_embeds = (self.config.lvr_head_type == 'intrinsic-similarity' or self.config.lvr_head_type == 'isg')
            
            if pixel_values is not None:
                batch_size = outputs.last_hidden_state.shape[0]
                lvr_batch_indices = torch.nonzero(lvr_mode_switch, as_tuple=True)[0]
                
                # Check if image_embeds is available
                # image_embeds is only created when pixel_values is not None (line 1279)
                # So if pixel_values is None here, image_embeds doesn't exist
                # We need to check if it exists in the current scope
                local_vars = locals()
                has_image_embeds = 'image_embeds' in local_vars and local_vars.get('image_embeds') is not None
                
                # Compute total_tokens from image_mask if not available
                if 'total_tokens' not in locals() or total_tokens is None:
                    if 'image_mask' in locals() and image_mask is not None:
                        total_tokens = torch.sum(image_mask, dim=1)  # Compute from image_mask
                        has_total_tokens = True
                    else:
                        has_total_tokens = False
                else:
                    has_total_tokens = True
                
                if len(lvr_batch_indices) > 0 and has_image_embeds and has_total_tokens:
                    # Get model dtype to avoid unnecessary dtype conversions
                    model_dtype = next(self.lvr_head.parameters()).dtype if len(list(self.lvr_head.parameters())) > 0 else torch.bfloat16
                    
                    batched_hidden_states = outputs.last_hidden_state[lvr_batch_indices, -1].to(model_dtype)  # (num_lvr_items, hidden_size)
                    # Safely access image_embeds - it should exist if has_image_embeds is True
                    # Get image_embeds from local scope safely
                    local_vars_check = locals()
                    if 'image_embeds' in local_vars_check and local_vars_check['image_embeds'] is not None:
                        current_image_embeds = local_vars_check['image_embeds']
                        batched_image_embeds, image_attention_mask = _prepare_batched_image_embeds(
                            current_image_embeds, total_tokens, lvr_batch_indices, target_dtype=model_dtype
                        )
                    else:
                        # image_embeds not available, fall through to fallback handling
                        has_image_embeds = False
                        batched_image_embeds = None
                        image_attention_mask = None
                    
                    if batched_image_embeds is not None:
                        v_focal_batch = self.lvr_head(batched_hidden_states, batched_image_embeds, image_attention_mask)  # (num_lvr_items, hidden_size)
                        outputs.last_hidden_state[lvr_batch_indices, -1] = v_focal_batch.to(outputs.last_hidden_state.dtype)
                        
                        # Free memory
                        del batched_hidden_states, batched_image_embeds, image_attention_mask, v_focal_batch
                    else:
                        # image_embeds not available, fall through to fallback handling below
                        has_image_embeds = False
                elif len(lvr_batch_indices) > 0:
                    # Fallback: if image info not available
                    if requires_image_embeds:
                        # For intrinsic-similarity head, skip LVR processing if image_embeds not available
                        # This can happen in inference when image info is not properly passed
                        pass  # Skip LVR head processing - use original hidden states
                    else:
                        # For other head types, use original behavior
                        for b_idx in lvr_batch_indices:
                            hidden_state = outputs.last_hidden_state[b_idx, -1]
                            outputs.last_hidden_state[b_idx, -1] = self.lvr_head(hidden_state.unsqueeze(0)).squeeze(0)
            else:
                # No images available (pixel_values is None)
                if requires_image_embeds:
                    # For intrinsic-similarity head, skip LVR processing when no images
                    pass  # Skip LVR head processing - use original hidden states
                else:
                    # For other head types, use original behavior
                    outputs.last_hidden_state[lvr_mode_switch,:,:] = self.lvr_head(outputs.last_hidden_state[lvr_mode_switch,:,:])
        else:
            # Fallback for head types that don't require image_embeds (like 'simple', 'glu')
            # But check if it's actually a head type that requires image_embeds
            if hasattr(self.config, 'lvr_head_type') and (self.config.lvr_head_type == 'intrinsic-similarity' or self.config.lvr_head_type == 'isg'):
                # Skip LVR processing for intrinsic-similarity when not in the main condition block
                # This can happen if image_embeds is not available
                pass
            else:
                outputs.last_hidden_state[lvr_mode_switch,:,:] = self.lvr_head(outputs.last_hidden_state[lvr_mode_switch,:,:])

    hidden_states = outputs[0]
    last_position_hidden_state = outputs.last_hidden_state[:,-1,:]
    logits = self.lm_head(hidden_states)

    lvr_loss_fct = set_lvr_loss_fct(self.config.loss_lvr_fct)

    loss = None
    loss_ce = None
    loss_lvr = None
    if labels is not None:
        # Upcast to float if we need to compute the loss to avoid potential precision issues
        logits = logits.float()
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Don't want CE loss for <lvr> token
        shift_labels = shift_labels.masked_fill(shift_labels == self.config.lvr_id, IGNORE_INDEX)

        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss_ce = loss_fct(shift_logits, shift_labels)

        # lvr loss
        # Get last hidden states for <lvr> token positions
        seq_positions_start = seq_positions - 1  # Now points to lvr_start
        
        if hasattr(self.config, 'lvr_head_type') and (self.config.lvr_head_type == 'attention-mask' or self.config.lvr_head_type == 'ivr' or self.config.lvr_head_type == 'gated-focus' or self.config.lvr_head_type == 'gfr'):
            # MLP-mask LVR loss: compute reconstruction loss between V_focal and grounding box image tokens
            # Use selected_lvr_embeds as ground truth (same as original LVR behavior)
            ''' We need to convert to fp32 to avoid overflow by mse'''
            # Batch processing for V_focal computation
            if len(batch_indices) > 0:
                # Get model parameter dtype first to avoid creating unnecessary float32 tensors
                model_dtype = _get_lvr_head_dtype(self.lvr_head)
                
                # Prepare batched hidden states directly in model dtype
                batched_hidden_states = hidden_states[batch_indices, seq_positions_start].to(model_dtype)  # (L_total, hidden_size)
                
                # Prepare batched image embeddings directly in model dtype to save memory
                batched_image_embeds, image_attention_mask = _prepare_batched_image_embeds(
                    image_embeds, total_tokens, batch_indices, target_dtype=model_dtype
                )
                
                # Batch compute V_focal in model dtype, then convert output to float32 for loss computation
                v_focal_tensor = self.lvr_head(batched_hidden_states, batched_image_embeds, image_attention_mask)  # (L_total, hidden_size)
                
                # Free memory: delete large intermediate tensors immediately (before dtype conversion)
                del batched_hidden_states, batched_image_embeds, image_attention_mask
                
                # Convert to float32 for loss computation (only convert what's needed)
                v_focal_tensor = v_focal_tensor.to(torch.float32)
                selected_lvr_embeds_fp32 = selected_lvr_embeds.to(torch.float32)  # (L_total, hidden_size)
                
                # Compute reconstruction loss (MSE/L1) between V_focal and selected_lvr_embeds
                loss_lvr = lvr_loss_fct(v_focal_tensor, selected_lvr_embeds_fp32)
                
                # Free memory: delete temporary tensors after loss computation
                del v_focal_tensor, selected_lvr_embeds_fp32
            else:
                loss_lvr = None
        else:
            ''' We need to convert to fp32 to avoid overflow by mse'''
            selected_hidden_states = hidden_states[batch_indices, seq_positions_start].to(torch.float32)  # [L_total, H]
            selected_lvr_embeds = selected_lvr_embeds.to(torch.float32)
            # Compute LVR loss between predicted and inserted lvr embeddings
            loss_lvr = lvr_loss_fct(selected_hidden_states, selected_lvr_embeds)


    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return Qwen2_5_VLCausalLMOutputWithPast(
        # loss=loss,
        loss_ce=loss_ce,
        loss_lvr=loss_lvr,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=self.model.rope_deltas,
        last_position_hidden_state =last_position_hidden_state
    )


'''
    Coconut mode
    LVR head
'''
def qwen2_5_mixed_modality_forward_lvr_with_head_with_modeSwitchLoss(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    rope_deltas: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
    lvr_tokens: Optional[torch.Tensor] = None,      # This is for TRAINING: Where should the lvr img tokens be
    lvr_mode_switch: Optional[torch.Tensor] = None, # This is for INFERENCE: Which instance in the batch is in lvr mode
    last_position_hidden_state: Optional[torch.FloatTensor] = None, # This is for INFERENCE: last hidden state of the last position
) -> Union[Tuple, Qwen2_5_VLCausalLMOutputWithPast]:
    
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    

    if inputs_embeds is None:
        inputs_embeds = self.model.get_input_embeddings()(input_ids)

    ''' 
        only happen during inference 
        inputs_embeds in shape (bs, seq_len, hidden)
    '''
    if lvr_mode_switch:
        # in fact, each instance's seq_len will be 1 in inference
        inputs_embeds[lvr_mode_switch,-1,:] = last_position_hidden_state[lvr_mode_switch]
    
    ''' Only necessary in training '''
    # Pass dummy image and dummy grid to the visual model to avoid deepspeed error.
    if not lvr_mode_switch and (pixel_values is None and pixel_values_videos is None):
        # Create dummy pixel_values and grid_thw for avoiding deepspeed error.
        dummy_pixel = torch.zeros(784, 1176).to(self.visual.device)
        dummy_grid = torch.tensor([[1, 28, 28]]).to(self.visual.device)
        
        dummy_pixel = dummy_pixel.type(self.visual.dtype)
        image_embeds = self.visual(dummy_pixel, grid_thw=dummy_grid)
        # Operates as maksed_scatter for the image tokens
        # However the values are all zeros so it dosen't affect the embeddings.
        # This could avoid deepspeed error when some batch only has texts.
        inputs_embeds += image_embeds.mean() * 0
            
    if pixel_values is not None:

        # with torch.autocast(device_type='cuda', enabled=True, dtype=torch.float32):
        #     # Ensure vision tower inputs are float32
        #     pixel_values = pixel_values.to(torch.float32) 
        image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
        # Handle both list and tensor returns from visual module
        if isinstance(image_embeds, (list, tuple)):
            image_embeds = torch.cat(image_embeds, dim=0)
        # If already a tensor, use it directly


        n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
        n_image_features = image_embeds.shape[0]
        if n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )

        if input_ids is None:
            image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            image_mask = image_mask.all(-1)
        else:
            image_mask = input_ids == self.config.image_token_id


        n_image_tokens = (image_mask).sum()
        image_mask_unsqueeze = image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        n_image_features = image_embeds.shape[0]
        if not is_torchdynamo_compiling() and n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )
        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask_unsqueeze, image_embeds)

        # IN TRAINING should we fill the lvr token positions with selected img tokrnd
        if lvr_tokens:
            '''
                Filling the lvr tokens with image embeddings.
                Applicable when each image input has multiple bboxes
            '''
            total_tokens = torch.sum(image_mask, dim=1)   # 1d tensor([216, 234, 234, 234]) for #vis_tokens in each instance in batch
            batch_size = input_ids.size(0) 
            # lvr mask for lvr token locations in the batch, [bs, seq_length]
            # in each instance, lvr tokens are True, others are False
            lvr_mask = input_ids == self.config.lvr_id  
            # Total length = number of <lvr> tokens in the batch
            # seq_positions: flattend LOCAL positions of lvr tokens in the inputs_ids
            batch_indices, seq_positions = torch.nonzero(lvr_mask, as_tuple=True)  

            #  GLOBAL starting index in `image_embeds` of each image in the batch
            image_token_offsets = torch.cumsum(
                F.pad(total_tokens, (1, 0)), dim=0
            )[:-1]  # shape [B], offset into image_embeds for each batch element

            # Use helper function to handle different lvr_tokens vs batch_size scenarios
            num_latent_tokens = getattr(self.config, 'num_latent_tokens', 8)
            global_lvr_token_indices = _build_global_lvr_token_indices(
                lvr_tokens, image_token_offsets, batch_indices, num_latent_tokens
            )

            # Step 3: Gather the selected visual embeddings
            if len(global_lvr_token_indices) > 0:
                selected_lvr_embeds = image_embeds[global_lvr_token_indices]  # [L_total, H]
                # Step 4: Replace in input_embeds at the right batch and position
                safe_assign_lvr_embeds(inputs_embeds, batch_indices, seq_positions, selected_lvr_embeds, "forward_reencode")
            

    if attention_mask is not None:
        attention_mask = attention_mask.to(inputs_embeds.device)

    if position_ids is None:
        # Calculate RoPE index once per generation in the pre-fill stage only.
        # When compiling, we can't check tensor values thus we check only input length
        # It is safe to assume that `length!=1` means we're in pre-fill because compiled
        # models currently cannot do asssisted decoding
        prefill_compiled_stage = is_torchdynamo_compiling() and (
            (input_ids is not None and input_ids.shape[1] != 1)
            or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
        )
        prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
            (cache_position is not None and cache_position[0] == 0)
            or (past_key_values is None or past_key_values.get_seq_length() == 0)
        )
        if (prefill_compiled_stage or prefill_noncompiled_stage) or self.model.rope_deltas is None:
            position_ids, rope_deltas = self.model.get_rope_index(
                input_ids,
                image_grid_thw,
                video_grid_thw,
                second_per_grid_ts=second_per_grid_ts,
                attention_mask=attention_mask,
            )
            self.model.rope_deltas = rope_deltas
        else:
            batch_size, seq_length, _ = inputs_embeds.shape
            position_ids = torch.arange(seq_length, device=inputs_embeds.device)
            position_ids = position_ids.view(1, 1, -1).expand(3, batch_size, -1)
            if cache_position is not None:
                delta = (cache_position[0] + self.model.rope_deltas).to(inputs_embeds.device)
            else:
                delta = torch.zeros((batch_size, seq_length), device=inputs_embeds.device)
            delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=1)
            position_ids = position_ids.clone() + delta.to(position_ids.device)

    outputs = self.model(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
    )

    '''apply lvr_head in training mode'''
    if lvr_tokens and lvr_mask.any():
        # batch_indices, seq_positions = torch.nonzero(lvr_mask, as_tuple=True)
        if len(batch_indices) > 0:
            # Get last hidden states for <lvr> token positions, starting <lvr_start>
            seq_positions_start = seq_positions - 1  # shift left by 1 pos, now points to lvr_start
            
            if hasattr(self.config, 'lvr_head_type') and (self.config.lvr_head_type == 'attention-mask' or self.config.lvr_head_type == 'ivr' or self.config.lvr_head_type == 'gated-focus' or self.config.lvr_head_type == 'gfr'):
                # MLP-mask LVR head needs image embeddings - batch processing
                if 'image_embeds' in locals() and 'total_tokens' in locals():
                    # Get model dtype to avoid unnecessary dtype conversions
                    model_dtype = next(self.lvr_head.parameters()).dtype if len(list(self.lvr_head.parameters())) > 0 else torch.bfloat16
                    
                    batched_hidden_states = outputs.last_hidden_state[batch_indices, seq_positions_start].to(model_dtype)  # (L_total, hidden_size)
                    batched_image_embeds, image_attention_mask = _prepare_batched_image_embeds(
                        image_embeds, total_tokens, batch_indices, target_dtype=model_dtype
                    )
                    v_focal_batch = self.lvr_head(batched_hidden_states, batched_image_embeds, image_attention_mask)  # (L_total, hidden_size)
                    outputs.last_hidden_state[batch_indices, seq_positions_start] = v_focal_batch.to(outputs.last_hidden_state.dtype)
                    
                    # Free memory
                    del batched_hidden_states, batched_image_embeds, image_attention_mask, v_focal_batch
                else:
                    # Fallback: use original behavior if image info not available
                    for i, (b_idx, seq_pos) in enumerate(zip(batch_indices, seq_positions_start)):
                        hidden_state = outputs.last_hidden_state[b_idx, seq_pos]
                        outputs.last_hidden_state[b_idx, seq_pos] = self.lvr_head(hidden_state.unsqueeze(0)).squeeze(0)
            else:
                outputs.last_hidden_state[batch_indices, seq_positions_start] = self.lvr_head(outputs.last_hidden_state[batch_indices, seq_positions_start])

    '''apply lvr_head in _inference mode'''
    if lvr_mode_switch:
        if hasattr(self.config, 'lvr_head_type') and (self.config.lvr_head_type == 'attention-mask' or self.config.lvr_head_type == 'ivr' or self.config.lvr_head_type == 'gated-focus' or self.config.lvr_head_type == 'gfr'):
            # MLP-mask LVR head in inference mode - batch processing
            if pixel_values is not None:
                batch_size = outputs.last_hidden_state.shape[0]
                lvr_batch_indices = torch.nonzero(lvr_mode_switch, as_tuple=True)[0]
                if len(lvr_batch_indices) > 0 and 'total_tokens' in locals() and 'image_embeds' in locals():
                    # Get model dtype to avoid unnecessary dtype conversions
                    model_dtype = next(self.lvr_head.parameters()).dtype if len(list(self.lvr_head.parameters())) > 0 else torch.bfloat16
                    
                    batched_hidden_states = outputs.last_hidden_state[lvr_batch_indices, -1].to(model_dtype)  # (num_lvr_items, hidden_size)
                    batched_image_embeds, image_attention_mask = _prepare_batched_image_embeds(
                        image_embeds, total_tokens, lvr_batch_indices, target_dtype=model_dtype
                    )
                    v_focal_batch = self.lvr_head(batched_hidden_states, batched_image_embeds, image_attention_mask)  # (num_lvr_items, hidden_size)
                    outputs.last_hidden_state[lvr_batch_indices, -1] = v_focal_batch.to(outputs.last_hidden_state.dtype)
                    
                    # Free memory
                    del batched_hidden_states, batched_image_embeds, image_attention_mask, v_focal_batch
                elif len(lvr_batch_indices) > 0:
                    # Fallback: use original behavior if image info not available
                    for b_idx in lvr_batch_indices:
                        hidden_state = outputs.last_hidden_state[b_idx, -1]
                        outputs.last_hidden_state[b_idx, -1] = self.lvr_head(hidden_state.unsqueeze(0)).squeeze(0)
            else:
                # No images available, use original behavior
                outputs.last_hidden_state[lvr_mode_switch,:,:] = self.lvr_head(outputs.last_hidden_state[lvr_mode_switch,:,:])
        else:
            outputs.last_hidden_state[lvr_mode_switch,:,:] = self.lvr_head(outputs.last_hidden_state[lvr_mode_switch,:,:])

    hidden_states = outputs[0]
    last_position_hidden_state = outputs.last_hidden_state[:,-1,:]
    logits = self.lm_head(hidden_states)

    lvr_loss_fct = set_lvr_loss_fct(self.config.loss_lvr_fct)

    loss = None
    loss_ce = None
    loss_lvr = None
    if labels is not None:
        # Upcast to float if we need to compute the loss to avoid potential precision issues
        logits = logits.float()
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Don't want CE loss for <lvr> token
        shift_labels = shift_labels.masked_fill(shift_labels == self.config.lvr_id, IGNORE_INDEX)

        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss_ce = loss_fct(shift_logits, shift_labels)

        # lvr loss
        # Get last hidden states for <lvr> token positions
        seq_positions_start = seq_positions - 1  # Now points to lvr_start
        
        if hasattr(self.config, 'lvr_head_type') and (self.config.lvr_head_type == 'attention-mask'):
            # MLP-mask LVR loss: compute reconstruction loss between V_focal and grounding box image tokens
            # Use selected_lvr_embeds as ground truth (same as original LVR behavior)
            ''' We need to convert to fp32 to avoid overflow by mse'''
            # Batch processing for V_focal computation
            if len(batch_indices) > 0:
                # Get model parameter dtype first to avoid creating unnecessary float32 tensors
                model_dtype = _get_lvr_head_dtype(self.lvr_head)
                
                # Prepare batched hidden states directly in model dtype
                batched_hidden_states = hidden_states[batch_indices, seq_positions_start].to(model_dtype)  # (L_total, hidden_size)
                
                # Prepare batched image embeddings directly in model dtype to save memory
                batched_image_embeds, image_attention_mask = _prepare_batched_image_embeds(
                    image_embeds, total_tokens, batch_indices, target_dtype=model_dtype
                )
                
                # Batch compute V_focal in model dtype, then convert output to float32 for loss computation
                v_focal_tensor = self.lvr_head(batched_hidden_states, batched_image_embeds, image_attention_mask)  # (L_total, hidden_size)
                
                # Free memory: delete large intermediate tensors immediately (before dtype conversion)
                del batched_hidden_states, batched_image_embeds, image_attention_mask
                
                # Convert to float32 for loss computation (only convert what's needed)
                v_focal_tensor = v_focal_tensor.to(torch.float32)
                selected_lvr_embeds_fp32 = selected_lvr_embeds.to(torch.float32)  # (L_total, hidden_size)
                
                # Compute reconstruction loss (MSE/L1) between V_focal and selected_lvr_embeds
                loss_lvr = lvr_loss_fct(v_focal_tensor, selected_lvr_embeds_fp32)
                
                # Free memory: delete temporary tensors after loss computation
                del v_focal_tensor, selected_lvr_embeds_fp32
            else:
                loss_lvr = None
        else:
            ''' We need to convert to fp32 to avoid overflow by mse'''
            selected_hidden_states = hidden_states[batch_indices, seq_positions_start].to(torch.float32)  # [L_total, H]
            selected_lvr_embeds = selected_lvr_embeds.to(torch.float32)
            # Compute LVR loss between predicted and inserted lvr embeddings
            loss_lvr = lvr_loss_fct(selected_hidden_states, selected_lvr_embeds)

            # mode switch loss

            lvr_or_lvrstart_mask = (input_ids == self.config.lvr_start_id) | (input_ids == self.config.lvr_id)

            # Find the next tokens of each position
            shifted_input_ids = torch.roll(input_ids, shifts=-1, dims=1)
            # the lvr token that is right before lvr_end token
            is_last_lvr = lvr_or_lvrstart_mask & (shifted_input_ids == self.config.lvr_end_id)
            # 1 if it's the last <lvr> before <lvr_end>, else 0
            targets = is_last_lvr.float()  # [batch_size, seq_len]

            lvr_end_logits = logits[..., self.config.lvr_end_id]  # [batch_size, seq_len]

            # Apply mask to focus only on <lvr_start>,<lvr> token positions
            masked_logits = lvr_end_logits[lvr_or_lvrstart_mask]  # [num_lvr_tokens]
            masked_targets = targets[lvr_or_lvrstart_mask]        # [num_lvr_tokens]

            loss_mode_switch = F.binary_cross_entropy_with_logits(masked_logits, masked_targets)


    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return Qwen2_5_VLCausalLMOutputWithPast(
        # loss=loss,
        loss_ce=loss_ce,
        loss_lvr=loss_lvr,
        loss_mode_switch=loss_mode_switch,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=self.model.rope_deltas,
        last_position_hidden_state =last_position_hidden_state
    )


'''
    Coconut mode
    LVR Head
    Padded <LVR_end> latent token as the mode switching signal
'''
def qwen2_5_mixed_modality_forward_lvr_with_head_with_latentEndToken(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    rope_deltas: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
    lvr_tokens: Optional[torch.Tensor] = None,      # This is for TRAINING: Where should the lvr img tokens be
    lvr_mode_switch: Optional[torch.Tensor] = None, # This is for INFERENCE: Which instance in the batch is in lvr mode
    last_position_hidden_state: Optional[torch.FloatTensor] = None, # This is for INFERENCE: last hidden state of the last position
) -> Union[Tuple, Qwen2_5_VLCausalLMOutputWithPast]:
    
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    

    if inputs_embeds is None:
        inputs_embeds = self.model.get_input_embeddings()(input_ids)

    ''' 
        only happen during inference 
        inputs_embeds in shape (bs, seq_len, hidden)
    '''
    if lvr_mode_switch:
        # in fact, each instance's seq_len will be 1 in inference
        inputs_embeds[lvr_mode_switch,-1,:] = last_position_hidden_state[lvr_mode_switch]
    
    ''' Only necessary in training '''
    # Pass dummy image and dummy grid to the visual model to avoid deepspeed error.
    if not lvr_mode_switch and (pixel_values is None and pixel_values_videos is None):
        # Create dummy pixel_values and grid_thw for avoiding deepspeed error.
        dummy_pixel = torch.zeros(784, 1176).to(self.visual.device)
        dummy_grid = torch.tensor([[1, 28, 28]]).to(self.visual.device)
        
        dummy_pixel = dummy_pixel.type(self.visual.dtype)
        image_embeds = self.visual(dummy_pixel, grid_thw=dummy_grid)
        # Operates as maksed_scatter for the image tokens
        # However the values are all zeros so it dosen't affect the embeddings.
        # This could avoid deepspeed error when some batch only has texts.
        inputs_embeds += image_embeds.mean() * 0
            
    if pixel_values is not None:

        # with torch.autocast(device_type='cuda', enabled=True, dtype=torch.float32):
        #     # Ensure vision tower inputs are float32
        #     pixel_values = pixel_values.to(torch.float32) 
        image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
        # Handle both list and tensor returns from visual module
        if isinstance(image_embeds, (list, tuple)):
            image_embeds = torch.cat(image_embeds, dim=0)
        # If already a tensor, use it directly


        n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
        n_image_features = image_embeds.shape[0]
        if n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )

        if input_ids is None:
            image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            image_mask = image_mask.all(-1)
        else:
            image_mask = input_ids == self.config.image_token_id


        n_image_tokens = (image_mask).sum()
        image_mask_unsqueeze = image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        n_image_features = image_embeds.shape[0]
        if not is_torchdynamo_compiling() and n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )
        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask_unsqueeze, image_embeds)

        # IN TRAINING should we fill the lvr token positions with selected img tokrnd
        if lvr_tokens:
            '''
                Filling the lvr tokens with image embeddings.
                Applicable when each image input has multiple bboxes
            '''
            total_tokens = torch.sum(image_mask, dim=1)   # 1d tensor([216, 234, 234, 234]) for #vis_tokens in each instance in batch
            batch_size = input_ids.size(0) 
            # lvr mask for lvr token locations in the batch, [bs, seq_length]
            # in each instance, lvr tokens are True, others are False
            lvr_mask = input_ids == self.config.lvr_id  
            # Total length = number of <lvr> tokens in the batch
            # seq_positions: flattend LOCAL positions of lvr tokens in the inputs_ids
            batch_indices, seq_positions = torch.nonzero(lvr_mask, as_tuple=True)  

            #  GLOBAL starting index in `image_embeds` of each image in the batch
            image_token_offsets = torch.cumsum(
                F.pad(total_tokens, (1, 0)), dim=0
            )[:-1]  # shape [B], offset into image_embeds for each batch element

            # Use helper function to handle different lvr_tokens vs batch_size scenarios
            num_latent_tokens = getattr(self.config, 'num_latent_tokens', 8)
            global_lvr_token_indices = _build_global_lvr_token_indices(
                lvr_tokens, image_token_offsets, batch_indices, num_latent_tokens
            )

            # Step 3: Gather the selected visual embeddings
            if len(global_lvr_token_indices) > 0:
                selected_lvr_embeds = image_embeds[global_lvr_token_indices]  # [L_total, H]
                # Step 4: Replace in input_embeds at the right batch and position
                safe_assign_lvr_embeds(inputs_embeds, batch_indices, seq_positions, selected_lvr_embeds, "forward_reencode")

            '''Apply lvr_latent_end_token (only when latent_end_token is enabled and model has lvr_latent_end_emb)'''
            lvr_latent_end_mask = (input_ids == self.config.lvr_latent_end_id)
            batch_indices_latentend, seq_positions_latentend = torch.nonzero(lvr_latent_end_mask, as_tuple=True)
            if lvr_latent_end_mask.any() and hasattr(self, 'lvr_latent_end_emb'):
                inputs_embeds[lvr_latent_end_mask] = self.lvr_latent_end_emb.to(inputs_embeds.device)
            

    if attention_mask is not None:
        attention_mask = attention_mask.to(inputs_embeds.device)

    if position_ids is None:
        # Calculate RoPE index once per generation in the pre-fill stage only.
        # When compiling, we can't check tensor values thus we check only input length
        # It is safe to assume that `length!=1` means we're in pre-fill because compiled
        # models currently cannot do asssisted decoding
        prefill_compiled_stage = is_torchdynamo_compiling() and (
            (input_ids is not None and input_ids.shape[1] != 1)
            or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
        )
        prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
            (cache_position is not None and cache_position[0] == 0)
            or (past_key_values is None or past_key_values.get_seq_length() == 0)
        )
        if (prefill_compiled_stage or prefill_noncompiled_stage) or self.model.rope_deltas is None:
            position_ids, rope_deltas = self.model.get_rope_index(
                input_ids,
                image_grid_thw,
                video_grid_thw,
                second_per_grid_ts=second_per_grid_ts,
                attention_mask=attention_mask,
            )
            self.model.rope_deltas = rope_deltas
        else:
            batch_size, seq_length, _ = inputs_embeds.shape
            position_ids = torch.arange(seq_length, device=inputs_embeds.device)
            position_ids = position_ids.view(1, 1, -1).expand(3, batch_size, -1)
            if cache_position is not None:
                delta = (cache_position[0] + self.model.rope_deltas).to(inputs_embeds.device)
            else:
                delta = torch.zeros((batch_size, seq_length), device=inputs_embeds.device)
            delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=1)
            position_ids = position_ids.clone() + delta.to(position_ids.device)

    outputs = self.model(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
    )

    '''apply lvr_head in training mode'''
    if lvr_tokens and lvr_mask.any():
        # batch_indices, seq_positions = torch.nonzero(lvr_mask, as_tuple=True)
        if len(batch_indices) > 0:
            # Get last hidden states for <lvr> token positions, starting <lvr_start>
            seq_positions_start = seq_positions - 1  # shift left by 1 pos, now points to lvr_start
            
            if hasattr(self.config, 'lvr_head_type') and (self.config.lvr_head_type == 'attention-mask' or self.config.lvr_head_type == 'ivr' or self.config.lvr_head_type == 'gated-focus' or self.config.lvr_head_type == 'gfr'):
                # MLP-mask LVR head needs image embeddings - batch processing
                if 'image_embeds' in locals() and 'total_tokens' in locals():
                    # Get model dtype to avoid unnecessary dtype conversions
                    model_dtype = next(self.lvr_head.parameters()).dtype if len(list(self.lvr_head.parameters())) > 0 else torch.bfloat16
                    
                    batched_hidden_states = outputs.last_hidden_state[batch_indices, seq_positions_start].to(model_dtype)  # (L_total, hidden_size)
                    batched_image_embeds, image_attention_mask = _prepare_batched_image_embeds(
                        image_embeds, total_tokens, batch_indices, target_dtype=model_dtype
                    )
                    v_focal_batch = self.lvr_head(batched_hidden_states, batched_image_embeds, image_attention_mask)  # (L_total, hidden_size)
                    outputs.last_hidden_state[batch_indices, seq_positions_start] = v_focal_batch.to(outputs.last_hidden_state.dtype)
                    
                    # Free memory
                    del batched_hidden_states, batched_image_embeds, image_attention_mask, v_focal_batch
                else:
                    # Fallback: use original behavior if image info not available
                    for i, (b_idx, seq_pos) in enumerate(zip(batch_indices, seq_positions_start)):
                        hidden_state = outputs.last_hidden_state[b_idx, seq_pos]
                        outputs.last_hidden_state[b_idx, seq_pos] = self.lvr_head(hidden_state.unsqueeze(0)).squeeze(0)
            else:
                outputs.last_hidden_state[batch_indices, seq_positions_start] = self.lvr_head(outputs.last_hidden_state[batch_indices, seq_positions_start])

            '''In this mode, <|lvr_latent_end|> is also a latent token'''
            seq_positions_start_latentend = seq_positions_latentend - 1
            if hasattr(self.config, 'lvr_head_type') and (self.config.lvr_head_type == 'attention-mask' or self.config.lvr_head_type == 'ivr' or self.config.lvr_head_type == 'gated-focus' or self.config.lvr_head_type == 'gfr'):
                # MLP-mask LVR head for latent_end tokens - batch processing
                if len(batch_indices_latentend) > 0:
                    if 'image_embeds' in locals() and 'total_tokens' in locals():
                        # Get model dtype to avoid unnecessary dtype conversions
                        model_dtype = next(self.lvr_head.parameters()).dtype if len(list(self.lvr_head.parameters())) > 0 else torch.bfloat16
                        
                        batched_hidden_states_latentend = outputs.last_hidden_state[batch_indices_latentend, seq_positions_start_latentend].to(model_dtype)  # (L_latentend, hidden_size)
                        batched_image_embeds_latentend, image_attention_mask_latentend = _prepare_batched_image_embeds(
                            image_embeds, total_tokens, batch_indices_latentend, target_dtype=model_dtype
                        )
                        v_focal_batch_latentend = self.lvr_head(batched_hidden_states_latentend, batched_image_embeds_latentend, image_attention_mask_latentend)  # (L_latentend, hidden_size)
                        outputs.last_hidden_state[batch_indices_latentend, seq_positions_start_latentend] = v_focal_batch_latentend.to(outputs.last_hidden_state.dtype)
                        
                        # Free memory
                        del batched_hidden_states_latentend, batched_image_embeds_latentend, image_attention_mask_latentend, v_focal_batch_latentend
                    else:
                        # Fallback: use original behavior if image info not available
                        for i, (b_idx, seq_pos) in enumerate(zip(batch_indices_latentend, seq_positions_start_latentend)):
                            hidden_state = outputs.last_hidden_state[b_idx, seq_pos]
                            outputs.last_hidden_state[b_idx, seq_pos] = self.lvr_head(hidden_state.unsqueeze(0)).squeeze(0)
            else:
                outputs.last_hidden_state[batch_indices_latentend, seq_positions_start_latentend] = self.lvr_head(outputs.last_hidden_state[batch_indices_latentend, seq_positions_start_latentend])


    '''apply lvr_head in _inference mode'''
    if lvr_mode_switch:
        if hasattr(self.config, 'lvr_head_type') and (self.config.lvr_head_type == 'attention-mask'):
            # MLP-mask LVR head in inference mode - batch processing
            if pixel_values is not None:
                batch_size = outputs.last_hidden_state.shape[0]
                lvr_batch_indices = torch.nonzero(lvr_mode_switch, as_tuple=True)[0]
                if len(lvr_batch_indices) > 0 and 'total_tokens' in locals() and 'image_embeds' in locals():
                    # Get model dtype to avoid unnecessary dtype conversions
                    model_dtype = next(self.lvr_head.parameters()).dtype if len(list(self.lvr_head.parameters())) > 0 else torch.bfloat16
                    
                    batched_hidden_states = outputs.last_hidden_state[lvr_batch_indices, -1].to(model_dtype)  # (num_lvr_items, hidden_size)
                    batched_image_embeds, image_attention_mask = _prepare_batched_image_embeds(
                        image_embeds, total_tokens, lvr_batch_indices, target_dtype=model_dtype
                    )
                    v_focal_batch = self.lvr_head(batched_hidden_states, batched_image_embeds, image_attention_mask)  # (num_lvr_items, hidden_size)
                    outputs.last_hidden_state[lvr_batch_indices, -1] = v_focal_batch.to(outputs.last_hidden_state.dtype)
                    
                    # Free memory
                    del batched_hidden_states, batched_image_embeds, image_attention_mask, v_focal_batch
                elif len(lvr_batch_indices) > 0:
                    # Fallback: use original behavior if image info not available
                    for b_idx in lvr_batch_indices:
                        hidden_state = outputs.last_hidden_state[b_idx, -1]
                        outputs.last_hidden_state[b_idx, -1] = self.lvr_head(hidden_state.unsqueeze(0)).squeeze(0)
            else:
                # No images available, use original behavior
                outputs.last_hidden_state[lvr_mode_switch,:,:] = self.lvr_head(outputs.last_hidden_state[lvr_mode_switch,:,:])
        else:
            outputs.last_hidden_state[lvr_mode_switch,:,:] = self.lvr_head(outputs.last_hidden_state[lvr_mode_switch,:,:])

    hidden_states = outputs[0]
    last_position_hidden_state = outputs.last_hidden_state[:,-1,:]
    logits = self.lm_head(hidden_states)

    lvr_loss_fct = set_lvr_loss_fct(self.config.loss_lvr_fct)
    mode_switch_loss_fct = set_lvr_loss_fct(getattr(self.config, 'loss_mode_switch_fct', 'mse'))

    loss = None
    loss_ce = None
    loss_lvr = None
    loss_lvr_resampler = None
    loss_ortho = None
    loss_attn_div = None
    loss_attn_guidance = None
    loss_attn_transfer = None
    if labels is not None:
        # Upcast to float if we need to compute the loss to avoid potential precision issues
        logits = logits.float()
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Stage 1: keep CE loss for <lvr> so model learns to predict correct token sequence (fixes lvr_end collapse).
        # Stage 2: mask both lvr and lvr_latent_end to avoid double-counting with Student MSE.
        # Stage 3 E2E: only CE on text tokens (mask lvr and lvr_latent_end, same as Stage 2)
        if getattr(self.config, 'use_stage3_e2e', False) or getattr(self.config, 'use_stage2_distillation', False):
            shift_labels = shift_labels.masked_fill((shift_labels == self.config.lvr_id)|(shift_labels == self.config.lvr_latent_end_id), IGNORE_INDEX)
        else:
            shift_labels = shift_labels.masked_fill(shift_labels == self.config.lvr_latent_end_id, IGNORE_INDEX)

        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss_ce = loss_fct(shift_logits, shift_labels)

        # lvr loss
        # Get last hidden states for <lvr> token positions
        seq_positions_start = seq_positions - 1
        selected_hidden_states = hidden_states[batch_indices, seq_positions_start].to(torch.float32)  # [L_total, H]

        if selected_lvr_embeds is None:
            loss_lvr = torch.tensor(0.0, device=selected_hidden_states.device, dtype=torch.float32, requires_grad=True)
        else:
            ''' We need to convert to fp32 to avoid overflow by mse'''
            selected_lvr_embeds = selected_lvr_embeds.to(torch.float32)
            # Truncate to same length so all ranks have same loss tensor shape (avoids NCCL hang when else_fill mismatch)
            n_align = min(selected_hidden_states.shape[0], selected_lvr_embeds.shape[0])
            if n_align == 0:
                loss_lvr = torch.tensor(0.0, device=selected_hidden_states.device, dtype=torch.float32, requires_grad=True)
            else:
                loss_lvr = lvr_loss_fct(selected_hidden_states[:n_align], selected_lvr_embeds[:n_align])

        # loss_mode_switch: only when latent_end_token is used (model has lvr_latent_end_emb) and batch has <lvr_latent_end> positions.
        # Uses hidden state at position (seq_positions_latentend - 1), i.e. the *last* of the 8 <lvr> slots (same as llm_latent_8[:, 7, :] in loss_lvr_resampler).
        # So when both are computed, the 8th latent slot is penalized twice; if resampler's 8th target ≈ lvr_latent_end_emb and per-slot MSE is similar, the two losses can be almost identical.
        loss_mode_switch = None
        if batch_indices_latentend.numel() > 0 and hasattr(self, 'lvr_latent_end_emb'):
            seq_positions_start_latentend = seq_positions_latentend - 1
            selected_hidden_states_latentend = hidden_states[batch_indices_latentend, seq_positions_start_latentend].to(torch.float32)
            selected_lvr_embeds_latentend = self.lvr_latent_end_emb.unsqueeze(0).expand_as(selected_hidden_states_latentend).to(torch.float32)
            selected_lvr_embeds_latentend = selected_lvr_embeds_latentend.to(selected_hidden_states_latentend.device)
            loss_mode_switch = mode_switch_loss_fct(selected_hidden_states_latentend, selected_lvr_embeds_latentend)

        # BoxFeatureResampler / Stage 2 distillation: fixed N latent MSE. Stage 2: Teacher frozen, Student (DynamicAutoregressiveResampler) aligns to Target.
        # When use_box_feature_resampler or use_stage2_distillation is enabled, always return a valid loss tensor (0.0 if cannot compute)
        # This ensures consistent gradient sync across all ranks in distributed training
        _resampler_output_for_dit = None  # Save resampler output for DiT condition (50/50 GT mechanism)
        use_stage2 = getattr(self.config, 'use_stage2_distillation', False)
        if (getattr(self.config, 'use_box_feature_resampler', False) or (use_stage2 and hasattr(self, 'student_resampler'))) and hasattr(self, 'box_feature_resampler'):
            num_latent_tokens = getattr(self.config, 'num_latent_tokens', 8)
            
            # Check if we have valid data to compute resampler loss
            can_compute_resampler_loss = (
                len(batch_indices) > 0 and
                isinstance(lvr_tokens, list) and 
                len(lvr_tokens) > 0
            )
            
            if can_compute_resampler_loss:
                num_bboxes = len(batch_indices) // num_latent_tokens
                if num_bboxes > 0 and num_latent_tokens * num_bboxes == len(batch_indices) and len(lvr_tokens) == num_bboxes:
                    bbox_feats, key_pad_mask = _prepare_bbox_region_features(
                        image_embeds, total_tokens, batch_indices, lvr_tokens, num_latent_tokens,
                        device=hidden_states.device, dtype=next(self.box_feature_resampler.parameters()).dtype,
                    )
                    if bbox_feats.shape[0] > 0 and bbox_feats.shape[1] > 0:
                        if use_stage2 and hasattr(self, 'student_resampler'):
                            # Stage 2: Teacher (frozen) -> target; Student (LLM hidden states as Q, full image as KV) -> predicted; MSE loss
                            # Use hidden_states.dtype to match model compute dtype (bf16/fp16) - ZeRO-3 param dtype may be unreliable
                            compute_dtype = hidden_states.dtype
                            save_vis = os.environ.get("STAGE2_VIS_ATTENTION", "0") == "1" or getattr(self, "_vis_stage2_attention", False)
                            need_teacher_attn = (
                                getattr(self.config, 'loss_attn_transfer_lambda', 0.0) > 0 or save_vis
                            )
                            teacher_attn = None
                            with torch.no_grad():
                                if need_teacher_attn:
                                    target_latent_tokens, teacher_attn = self.box_feature_resampler(
                                        bbox_feats, key_padding_mask=key_pad_mask, return_attention=True
                                    )
                                else:
                                    target_latent_tokens = self.box_feature_resampler(bbox_feats, key_padding_mask=key_pad_mask)
                            _resampler_output_for_dit = target_latent_tokens  # For DiT 50/50 GT condition
                            # LLM hidden states at 8 <lvr> positions (autoregressive outputs) - keep in model dtype for student_resampler
                            lvr_hidden_states = hidden_states[batch_indices, seq_positions].to(compute_dtype)
                            lvr_hidden_states = lvr_hidden_states.view(num_bboxes, num_latent_tokens, -1)
                            # Full image features per bbox (batch_indices[0::8] gives one index per bbox)
                            batch_indices_per_bbox = batch_indices[0::num_latent_tokens]
                            full_image_features, image_attention_mask = _prepare_batched_image_embeds(
                                image_embeds, total_tokens, batch_indices_per_bbox,
                                target_dtype=compute_dtype,
                            )
                            key_padding_mask = ~image_attention_mask if image_attention_mask is not None else None
                            predicted_latent_tokens, student_attn = self.student_resampler(
                                lvr_hidden_states=lvr_hidden_states,
                                full_image_features=full_image_features,
                                key_padding_mask=key_padding_mask,
                                return_attention=True,
                            )
                            if save_vis and teacher_attn is not None:
                                self._stage2_vis_buffer = {
                                    "teacher_attn": teacher_attn.detach().cpu().float(),
                                    "student_attn": student_attn.detach().cpu().float(),
                                    "lvr_tokens": [t.cpu() if isinstance(t, torch.Tensor) else t for t in lvr_tokens],
                                    "batch_indices": batch_indices.cpu(),
                                    "batch_indices_per_bbox": batch_indices_per_bbox.cpu(),
                                    "image_grid_thw": image_grid_thw.cpu() if image_grid_thw is not None else None,
                                    "total_tokens": total_tokens.cpu(),
                                    "key_pad_mask": key_pad_mask.cpu() if key_pad_mask is not None else None,
                                    "image_attention_mask": image_attention_mask.cpu() if image_attention_mask is not None else None,
                                    "num_bboxes": num_bboxes,
                                    "num_latent_tokens": num_latent_tokens,
                                }
                            predicted_latent_tokens = predicted_latent_tokens.to(torch.float32)
                            target_latent_tokens = target_latent_tokens.to(torch.float32)
                            if torch.isnan(predicted_latent_tokens).any() or torch.isinf(predicted_latent_tokens).any() or torch.isnan(target_latent_tokens).any() or torch.isinf(target_latent_tokens).any():
                                loss_lvr_resampler = torch.tensor(0.0, device=predicted_latent_tokens.device, dtype=torch.float32, requires_grad=True)
                                loss_attn_transfer = None
                            else:
                                loss_lvr_resampler = F.mse_loss(predicted_latent_tokens, target_latent_tokens.detach())
                                # Pixel-level Attention Distillation: KL divergence to align Student with Teacher attention
                                loss_attn_transfer = None
                                loss_attn_transfer_lambda = getattr(self.config, 'loss_attn_transfer_lambda', 0.0)
                                if loss_attn_transfer_lambda > 0 and teacher_attn is not None and student_attn is not None and key_pad_mask is not None:
                                    seq_len_full = full_image_features.shape[1]
                                    target_attn = get_aligned_teacher_attn_from_lvr_tokens(
                                        teacher_attn.float(), lvr_tokens, key_pad_mask,
                                        seq_len_full, hidden_states.device,
                                    )
                                    eps = 1e-9
                                    student_attn_log = torch.log(student_attn.float() + eps)
                                    loss_attn_transfer = F.kl_div(student_attn_log, target_attn, reduction='batchmean')
                                    if not (torch.isnan(loss_attn_transfer) or torch.isinf(loss_attn_transfer)):
                                        loss_lvr_resampler = loss_lvr_resampler + loss_attn_transfer_lambda * loss_attn_transfer
                                    else:
                                        loss_attn_transfer = None
                            if loss_lvr_resampler is not None and (torch.isnan(loss_lvr_resampler) or torch.isinf(loss_lvr_resampler)):
                                loss_lvr_resampler = predicted_latent_tokens.sum() * 0.0
                            elif loss_lvr_resampler is not None and os.environ.get("LVR_DEBUG", "0") == "1":
                                print(f"[LVR.forward] rank={get_rank()} loss_lvr_resampler (Stage2)={loss_lvr_resampler.item():.6f} "
                                      f"predicted shape={predicted_latent_tokens.shape} target shape={target_latent_tokens.shape}", flush=True)
                        else:
                            # Stage 1: bidirectional symmetric loss only (docs/BoxFeatureResampler.md)
                            resampler_output = self.box_feature_resampler(
                                bbox_feats, key_padding_mask=key_pad_mask
                            )
                            _resampler_output_for_dit = resampler_output  # Save for DiT 50/50 GT condition
                            seq_positions_start = seq_positions - 1
                            llm_latent_8 = hidden_states[batch_indices, seq_positions_start].to(torch.float32)
                            llm_latent_8 = llm_latent_8.view(num_bboxes, num_latent_tokens, -1)
                            resampler_output = resampler_output.to(torch.float32)
                            # Avoid NaN: skip loss if inputs contain NaN/Inf; clamp to reduce overflow
                            if torch.isnan(llm_latent_8).any() or torch.isinf(llm_latent_8).any() or torch.isnan(resampler_output).any() or torch.isinf(resampler_output).any():
                                loss_lvr_resampler = torch.tensor(0.0, device=llm_latent_8.device, dtype=torch.float32, requires_grad=True)
                                loss_attn_div = None
                            else:
                                llm_latent_8_clamped = torch.clamp(llm_latent_8, min=-1e4, max=1e4)
                                resampler_output_clamped = torch.clamp(resampler_output, min=-1e4, max=1e4)
                                # Bidirectional symmetric loss: each side only receives its own gradient (original design)
                                loss_resampler = lvr_loss_fct(llm_latent_8_clamped.detach(), resampler_output_clamped)  # train resampler
                                loss_llm = lvr_loss_fct(llm_latent_8_clamped, resampler_output_clamped.detach())        # train LLM
                                loss_lvr_resampler = (loss_resampler + loss_llm) / 2
                                loss_ortho = None
                                loss_attn_div = None
                            if loss_lvr_resampler is not None and (torch.isnan(loss_lvr_resampler) or torch.isinf(loss_lvr_resampler)):
                                loss_lvr_resampler = llm_latent_8.sum() * 0.0
                            elif loss_lvr_resampler is not None and os.environ.get("LVR_DEBUG", "0") == "1":
                                print(f"[LVR.forward] rank={get_rank()} loss_lvr_resampler={loss_lvr_resampler.item():.6f} "
                                      f"llm_latent_8 shape={llm_latent_8.shape} resampler_output shape={resampler_output.shape}", flush=True)
                    else:
                        # bbox_feats is empty, return zero loss
                        loss_lvr_resampler = torch.tensor(0.0, device=hidden_states.device, dtype=torch.float32, requires_grad=True)
                        loss_ortho = None
                        loss_attn_div = None
                else:
                    # num_bboxes mismatch, return zero loss
                    loss_lvr_resampler = torch.tensor(0.0, device=hidden_states.device, dtype=torch.float32, requires_grad=True)
                    loss_ortho = None
                    loss_attn_div = None
            else:
                # Empty lvr_tokens or batch_indices, return zero loss
                # This happens when lvr_tokens was cleared due to truncation in packed dataset
                loss_lvr_resampler = torch.tensor(0.0, device=hidden_states.device, dtype=torch.float32, requires_grad=True)
                loss_ortho = None
                loss_attn_div = None
                if os.environ.get("LVR_DEBUG", "0") == "1":
                    print(f"[LVR.forward] rank={get_rank()} loss_lvr_resampler=0.0 (empty lvr_tokens or batch_indices)", flush=True)

            # loss_ortho for logging only when not already baked into loss_lvr_resampler
            if loss_ortho is None and not use_stage2 and hasattr(self, 'box_feature_resampler'):
                loss_ortho = self.box_feature_resampler.get_orthogonality_loss()

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return Qwen2_5_VLCausalLMOutputWithPast(
        # loss=loss,
        loss_ce=loss_ce,
        loss_lvr=loss_lvr,
        loss_lvr_resampler=loss_lvr_resampler,
        loss_ortho=loss_ortho,
        loss_attn_div=loss_attn_div,
        loss_attn_guidance=loss_attn_guidance,
        loss_attn_transfer=loss_attn_transfer,
        loss_mode_switch=loss_mode_switch,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=self.model.rope_deltas,
        last_position_hidden_state =last_position_hidden_state
    )

'''
    Coconut mode
    LVR Head
    Padded <LVR_end> latent token as the mode switching signal
'''
def qwen2_5_mixed_modality_forward_lvr_with_latentEndToken(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    rope_deltas: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
    lvr_tokens: Optional[torch.Tensor] = None,      # This is for TRAINING: Where should the lvr img tokens be
    lvr_mode_switch: Optional[torch.Tensor] = None, # This is for INFERENCE: Which instance in the batch is in lvr mode
    last_position_hidden_state: Optional[torch.FloatTensor] = None, # This is for INFERENCE: last hidden state of the last position
) -> Union[Tuple, Qwen2_5_VLCausalLMOutputWithPast]:
    
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    

    if inputs_embeds is None:
        inputs_embeds = self.model.get_input_embeddings()(input_ids)

    ''' 
        only happen during inference 
        inputs_embeds in shape (bs, seq_len, hidden)
    '''
    if lvr_mode_switch:
        # in fact, each instance's seq_len will be 1 in inference
        inputs_embeds[lvr_mode_switch,-1,:] = last_position_hidden_state[lvr_mode_switch]
    
    ''' Only necessary in training '''
    # Pass dummy image and dummy grid to the visual model to avoid deepspeed error.
    if not lvr_mode_switch and (pixel_values is None and pixel_values_videos is None):
        # Create dummy pixel_values and grid_thw for avoiding deepspeed error.
        dummy_pixel = torch.zeros(784, 1176).to(self.visual.device)
        dummy_grid = torch.tensor([[1, 28, 28]]).to(self.visual.device)
        
        dummy_pixel = dummy_pixel.type(self.visual.dtype)
        image_embeds = self.visual(dummy_pixel, grid_thw=dummy_grid)
        # Operates as maksed_scatter for the image tokens
        # However the values are all zeros so it dosen't affect the embeddings.
        # This could avoid deepspeed error when some batch only has texts.
        inputs_embeds += image_embeds.mean() * 0
            
    if pixel_values is not None:

        # with torch.autocast(device_type='cuda', enabled=True, dtype=torch.float32):
        #     # Ensure vision tower inputs are float32
        #     pixel_values = pixel_values.to(torch.float32) 
        image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
        # Handle both list and tensor returns from visual module
        if isinstance(image_embeds, (list, tuple)):
            image_embeds = torch.cat(image_embeds, dim=0)
        # If already a tensor, use it directly


        n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
        n_image_features = image_embeds.shape[0]
        if n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )

        if input_ids is None:
            image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            image_mask = image_mask.all(-1)
        else:
            image_mask = input_ids == self.config.image_token_id


        n_image_tokens = (image_mask).sum()
        image_mask_unsqueeze = image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        n_image_features = image_embeds.shape[0]
        if not is_torchdynamo_compiling() and n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )
        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask_unsqueeze, image_embeds)

        # IN TRAINING should we fill the lvr token positions with selected img tokrnd
        if lvr_tokens:
            '''
                Filling the lvr tokens with image embeddings.
                Applicable when each image input has multiple bboxes
            '''
            total_tokens = torch.sum(image_mask, dim=1)   # 1d tensor([216, 234, 234, 234]) for #vis_tokens in each instance in batch
            batch_size = input_ids.size(0) 
            # lvr mask for lvr token locations in the batch, [bs, seq_length]
            # in each instance, lvr tokens are True, others are False
            lvr_mask = input_ids == self.config.lvr_id  
            # Total length = number of <lvr> tokens in the batch
            # seq_positions: flattend LOCAL positions of lvr tokens in the inputs_ids
            batch_indices, seq_positions = torch.nonzero(lvr_mask, as_tuple=True)  

        #  GLOBAL starting index in `image_embeds` of each image in the batch
            image_token_offsets = torch.cumsum(
                F.pad(total_tokens, (1, 0)), dim=0
            )[:-1]  # shape [B], offset into image_embeds for each batch element

            num_latent_tokens = getattr(self.config, 'num_latent_tokens', 8)
            num_bboxes = len(batch_indices) // num_latent_tokens
            use_resampler_fill = (
                getattr(self.config, 'use_box_feature_resampler', False)
                and hasattr(self, 'box_feature_resampler')
                and num_bboxes > 0
                and num_latent_tokens * num_bboxes == len(batch_indices)
                and isinstance(lvr_tokens, list)
                and len(lvr_tokens) == num_bboxes
                and not getattr(self.config, 'use_stage3_e2e', False)  # Stage 3: use GT fill, no BCM
            )
            if os.environ.get("LVR_DEBUG", "0") == "1":
                print(f"[LVR.forward] rank={get_rank()} lvr_fill: len(batch_indices)={len(batch_indices)} "
                      f"num_bboxes={num_bboxes} num_latent_tokens={num_latent_tokens} use_resampler_fill={use_resampler_fill} "
                      f"len(lvr_tokens)={len(lvr_tokens) if isinstance(lvr_tokens, list) else 'N/A'}", flush=True)
            if isinstance(lvr_tokens, list):
                '''lvr_tokens is a list of token indices for each bbox'''
                if use_resampler_fill:
                    # Fixed 8: GT bbox tokens -> BoxFeatureResampler -> 8 latent; fill those 8 positions per bbox with resampler output
                    bbox_feats, key_pad_mask = _prepare_bbox_region_features(
                        image_embeds, total_tokens, batch_indices, lvr_tokens, num_latent_tokens,
                        device=inputs_embeds.device, dtype=next(self.box_feature_resampler.parameters()).dtype,
                    )
                    if bbox_feats.shape[0] > 0 and bbox_feats.shape[1] > 0:
                        fill_embeds = self.box_feature_resampler(bbox_feats, key_padding_mask=key_pad_mask).detach()
                        fill_embeds = fill_embeds.to(inputs_embeds.dtype).view(-1, fill_embeds.shape[-1])
                        n_fill = min(len(batch_indices), fill_embeds.shape[0])
                        if n_fill != len(batch_indices) or n_fill != fill_embeds.shape[0]:
                            warnings.warn(
                                f"LVR resampler fill shape mismatch: batch_indices len={len(batch_indices)}, "
                                f"fill_embeds shape[0]={fill_embeds.shape[0]}; using n_fill={n_fill} to avoid crash."
                            )
                        if n_fill > 0:
                            for k in range(n_fill):
                                inputs_embeds[batch_indices[k], seq_positions[k]] = fill_embeds[k]
                    selected_lvr_embeds = None
                else:
                    '''Extracting tokens from original image'''
                    global_lvr_token_indices = []
                    for i in range(len(lvr_tokens)):
                        batch_idx = batch_indices[i * num_latent_tokens].item() if num_latent_tokens > 0 and i * num_latent_tokens < len(batch_indices) else i
                        lvr_ids = lvr_tokens[i]
                        offset = image_token_offsets[batch_idx].item()
                        if isinstance(lvr_ids, torch.Tensor):
                            global_lvr_token_indices.append(lvr_ids.to(image_token_offsets.device) + offset)
                        else:
                            global_lvr_token_indices.append(torch.tensor(lvr_ids, device=image_token_offsets.device, dtype=torch.long) + offset)
                    if global_lvr_token_indices:
                        global_lvr_token_indices = torch.cat(global_lvr_token_indices, dim=0)
                        selected_lvr_embeds = image_embeds[global_lvr_token_indices]
                        n_fill = min(len(batch_indices), selected_lvr_embeds.shape[0])
                        if n_fill > 0:
                            # Assign one row at a time so that when len(batch_indices)=1 but selected_lvr_embeds
                            # has more rows (e.g. 1 LVR token in sequence vs 3 source embeds from lvr_tokens),
                            # we never assign [3,H] to [1,H] and avoid broadcast error.
                            for k in range(n_fill):
                                inputs_embeds[batch_indices[k], seq_positions[k]] = selected_lvr_embeds[k]
                    else:
                        selected_lvr_embeds = None
            else:
                '''re-encode target area - lvr_tokens is pixel_values of the cropped targets'''
                # Note: This function does not have lvr_tokens_thw parameter, so we cannot use get_image_features here.
                # If lvr_tokens is a tensor (pixel_values), we need to handle it appropriately.
                # For now, log a warning and skip to avoid crash.
                warnings.warn(
                    f"[LVR.forward] lvr_tokens is a tensor (not list) in qwen2_5_mixed_modality_forward_lvr_with_latentEndToken. "
                    f"This path requires lvr_tokens_thw which is not available. Skipping LVR fill."
                )
                selected_lvr_embeds = None

            '''Apply lvr_latent_end_token (only when latent_end_token is enabled and model has lvr_latent_end_emb)'''
            lvr_latent_end_mask = (input_ids == self.config.lvr_latent_end_id)
            batch_indices_latentend, seq_positions_latentend = torch.nonzero(lvr_latent_end_mask, as_tuple=True)
            if lvr_latent_end_mask.any() and hasattr(self, 'lvr_latent_end_emb'):
                inputs_embeds[lvr_latent_end_mask] = self.lvr_latent_end_emb.to(inputs_embeds.device)
            

    if attention_mask is not None:
        attention_mask = attention_mask.to(inputs_embeds.device)

    if position_ids is None:
        # Calculate RoPE index once per generation in the pre-fill stage only.
        # When compiling, we can't check tensor values thus we check only input length
        # It is safe to assume that `length!=1` means we're in pre-fill because compiled
        # models currently cannot do asssisted decoding
        prefill_compiled_stage = is_torchdynamo_compiling() and (
            (input_ids is not None and input_ids.shape[1] != 1)
            or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
        )
        prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
            (cache_position is not None and cache_position[0] == 0)
            or (past_key_values is None or past_key_values.get_seq_length() == 0)
        )
        if (prefill_compiled_stage or prefill_noncompiled_stage) or self.model.rope_deltas is None:
            position_ids, rope_deltas = self.model.get_rope_index(
                input_ids,
                image_grid_thw,
                video_grid_thw,
                second_per_grid_ts=second_per_grid_ts,
                attention_mask=attention_mask,
            )
            self.model.rope_deltas = rope_deltas
        else:
            batch_size, seq_length, _ = inputs_embeds.shape
            position_ids = torch.arange(seq_length, device=inputs_embeds.device)
            position_ids = position_ids.view(1, 1, -1).expand(3, batch_size, -1)
            if cache_position is not None:
                delta = (cache_position[0] + self.model.rope_deltas).to(inputs_embeds.device)
            else:
                delta = torch.zeros((batch_size, seq_length), device=inputs_embeds.device)
            delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=1)
            position_ids = position_ids.clone() + delta.to(position_ids.device)

    outputs = self.model(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
    )

    hidden_states = outputs[0]
    # Stage 3 E2E: Replace hidden_states at <lvr> positions with DAR output so CE loss flows to DAR
    use_stage3 = getattr(self.config, 'use_stage3_e2e', False)
    if use_stage3 and labels is not None and lvr_tokens and len(batch_indices) > 0 and hasattr(self, 'student_resampler') and self.student_resampler is not None:
        num_latent_tokens = getattr(self.config, 'num_latent_tokens', 8)
        num_bboxes = len(batch_indices) // num_latent_tokens
        if num_bboxes > 0 and num_latent_tokens * num_bboxes == len(batch_indices) and isinstance(lvr_tokens, list) and len(lvr_tokens) == num_bboxes:
            compute_dtype = hidden_states.dtype
            lvr_hidden_states = hidden_states[batch_indices, seq_positions].to(compute_dtype)
            lvr_hidden_states = lvr_hidden_states.view(num_bboxes, num_latent_tokens, -1)
            batch_indices_per_bbox = batch_indices[0::num_latent_tokens]
            full_image_features, image_attention_mask = _prepare_batched_image_embeds(
                image_embeds, total_tokens, batch_indices_per_bbox,
                target_dtype=compute_dtype,
            )
            key_padding_mask = ~image_attention_mask if image_attention_mask is not None else None
            predicted_latent_tokens = self.student_resampler(
                lvr_hidden_states=lvr_hidden_states,
                full_image_features=full_image_features,
                key_padding_mask=key_padding_mask,
                return_attention=False,
            )
            predicted_latent_tokens = predicted_latent_tokens.view(-1, predicted_latent_tokens.shape[-1])
            hidden_states[batch_indices, seq_positions] = predicted_latent_tokens.to(hidden_states.dtype)
    last_position_hidden_state = outputs.last_hidden_state[:,-1,:]
    logits = self.lm_head(hidden_states)

    lvr_loss_fct = set_lvr_loss_fct(self.config.loss_lvr_fct)
    mode_switch_loss_fct = set_lvr_loss_fct(getattr(self.config, 'loss_mode_switch_fct', 'mse'))

    loss = None
    loss_ce = None
    loss_lvr = None
    loss_lvr_resampler = None
    loss_ortho = None
    loss_attn_div = None
    loss_attn_guidance = None
    loss_attn_transfer = None
    if labels is not None:
        # Upcast to float if we need to compute the loss to avoid potential precision issues
        logits = logits.float()
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Stage 3 E2E: only CE on text tokens (mask lvr and lvr_latent_end). Stage 2: same.
        if getattr(self.config, 'use_stage3_e2e', False) or getattr(self.config, 'use_stage2_distillation', False):
            shift_labels = shift_labels.masked_fill((shift_labels == self.config.lvr_id)|(shift_labels == self.config.lvr_latent_end_id), IGNORE_INDEX)
        else:
            shift_labels = shift_labels.masked_fill(shift_labels == self.config.lvr_latent_end_id, IGNORE_INDEX)

        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss_ce = loss_fct(shift_logits, shift_labels)

        # Stage 3 E2E: skip all auxiliary losses (loss_lvr, loss_mode_switch, loss_lvr_resampler, etc.)
        loss_mode_switch = None  # Default; set below when not Stage 3
        if not use_stage3:
            # lvr loss
            seq_positions_start = seq_positions - 1
            selected_hidden_states = hidden_states[batch_indices, seq_positions_start].to(torch.float32)  # [L_total, H]
            if selected_lvr_embeds is not None:
                selected_lvr_embeds = selected_lvr_embeds.to(torch.float32)
                n_lvr = min(selected_hidden_states.shape[0], selected_lvr_embeds.shape[0])
                if n_lvr > 0:
                    loss_lvr = lvr_loss_fct(selected_hidden_states[:n_lvr], selected_lvr_embeds[:n_lvr])
                else:
                    loss_lvr = None
            else:
                loss_lvr = None

        # loss_mode_switch: only when latent_end_token is used (model has lvr_latent_end_emb) and batch has <lvr_latent_end> positions.
        # Stage 3 E2E: skip (no auxiliary losses)
        if not use_stage3:
            if batch_indices_latentend.numel() > 0 and hasattr(self, 'lvr_latent_end_emb'):
                seq_positions_start_latentend = seq_positions_latentend - 1
                selected_hidden_states_latentend = hidden_states[batch_indices_latentend, seq_positions_start_latentend].to(torch.float32)
                selected_lvr_embeds_latentend = self.lvr_latent_end_emb.unsqueeze(0).expand_as(selected_hidden_states_latentend).to(torch.float32)
                selected_lvr_embeds_latentend = selected_lvr_embeds_latentend.to(selected_hidden_states_latentend.device)
                loss_mode_switch = mode_switch_loss_fct(selected_hidden_states_latentend, selected_lvr_embeds_latentend)

        # BoxFeatureResampler / Stage 2 distillation: fixed N latent MSE. Stage 2: Teacher frozen, Student aligns to Target.
        # Stage 3 E2E: skip entire block (no BCM forward, no distillation losses)
        _resampler_output_for_dit = None  # Save resampler output for DiT condition (50/50 GT mechanism)
        use_stage2 = getattr(self.config, 'use_stage2_distillation', False)
        if not use_stage3 and (getattr(self.config, 'use_box_feature_resampler', False) or (use_stage2 and hasattr(self, 'student_resampler'))) and hasattr(self, 'box_feature_resampler'):
            num_latent_tokens = getattr(self.config, 'num_latent_tokens', 8)
            
            # Check if we have valid data to compute resampler loss
            can_compute_resampler_loss = (
                len(batch_indices) > 0 and
                isinstance(lvr_tokens, list) and 
                len(lvr_tokens) > 0
            )
            
            if can_compute_resampler_loss:
                num_bboxes = len(batch_indices) // num_latent_tokens
                if num_bboxes > 0 and num_latent_tokens * num_bboxes == len(batch_indices) and len(lvr_tokens) == num_bboxes:
                    bbox_feats, key_pad_mask = _prepare_bbox_region_features(
                        image_embeds, total_tokens, batch_indices, lvr_tokens, num_latent_tokens,
                        device=hidden_states.device, dtype=next(self.box_feature_resampler.parameters()).dtype,
                    )
                    if bbox_feats.shape[0] > 0 and bbox_feats.shape[1] > 0:
                        if use_stage2 and hasattr(self, 'student_resampler'):
                            # Stage 2: Teacher (frozen) -> target; Student (LLM hidden states as Q, full image as KV) -> predicted; MSE loss
                            # Use hidden_states.dtype to match model compute dtype (bf16/fp16) - ZeRO-3 param dtype may be unreliable
                            compute_dtype = hidden_states.dtype
                            save_vis = os.environ.get("STAGE2_VIS_ATTENTION", "0") == "1" or getattr(self, "_vis_stage2_attention", False)
                            need_teacher_attn = (
                                getattr(self.config, 'loss_attn_transfer_lambda', 0.0) > 0 or save_vis
                            )
                            teacher_attn = None
                            with torch.no_grad():
                                if need_teacher_attn:
                                    target_latent_tokens, teacher_attn = self.box_feature_resampler(
                                        bbox_feats, key_padding_mask=key_pad_mask, return_attention=True
                                    )
                                else:
                                    target_latent_tokens = self.box_feature_resampler(bbox_feats, key_padding_mask=key_pad_mask)
                            _resampler_output_for_dit = target_latent_tokens  # For DiT 50/50 GT condition
                            # LLM hidden states at 8 <lvr> positions (autoregressive outputs) - keep in model dtype for student_resampler
                            lvr_hidden_states = hidden_states[batch_indices, seq_positions].to(compute_dtype)
                            lvr_hidden_states = lvr_hidden_states.view(num_bboxes, num_latent_tokens, -1)
                            # Full image features per bbox
                            batch_indices_per_bbox = batch_indices[0::num_latent_tokens]
                            full_image_features, image_attention_mask = _prepare_batched_image_embeds(
                                image_embeds, total_tokens, batch_indices_per_bbox,
                                target_dtype=compute_dtype,
                            )
                            key_padding_mask = ~image_attention_mask if image_attention_mask is not None else None
                            predicted_latent_tokens, student_attn = self.student_resampler(
                                lvr_hidden_states=lvr_hidden_states,
                                full_image_features=full_image_features,
                                key_padding_mask=key_padding_mask,
                                return_attention=True,
                            )
                            if save_vis and teacher_attn is not None:
                                self._stage2_vis_buffer = {
                                    "teacher_attn": teacher_attn.detach().cpu().float(),
                                    "student_attn": student_attn.detach().cpu().float(),
                                    "lvr_tokens": [t.cpu() if isinstance(t, torch.Tensor) else t for t in lvr_tokens],
                                    "batch_indices": batch_indices.cpu(),
                                    "batch_indices_per_bbox": batch_indices_per_bbox.cpu(),
                                    "image_grid_thw": image_grid_thw.cpu() if image_grid_thw is not None else None,
                                    "total_tokens": total_tokens.cpu(),
                                    "key_pad_mask": key_pad_mask.cpu() if key_pad_mask is not None else None,
                                    "image_attention_mask": image_attention_mask.cpu() if image_attention_mask is not None else None,
                                    "num_bboxes": num_bboxes,
                                    "num_latent_tokens": num_latent_tokens,
                                }
                            predicted_latent_tokens = predicted_latent_tokens.to(torch.float32)
                            target_latent_tokens = target_latent_tokens.to(torch.float32)
                            if torch.isnan(predicted_latent_tokens).any() or torch.isinf(predicted_latent_tokens).any() or torch.isnan(target_latent_tokens).any() or torch.isinf(target_latent_tokens).any():
                                loss_lvr_resampler = torch.tensor(0.0, device=predicted_latent_tokens.device, dtype=torch.float32, requires_grad=True)
                                loss_attn_transfer = None
                            else:
                                loss_lvr_resampler = F.mse_loss(predicted_latent_tokens, target_latent_tokens.detach())
                                # Pixel-level Attention Distillation: KL divergence to align Student with Teacher attention
                                loss_attn_transfer = None
                                loss_attn_transfer_lambda = getattr(self.config, 'loss_attn_transfer_lambda', 0.0)
                                if loss_attn_transfer_lambda > 0 and teacher_attn is not None and student_attn is not None and key_pad_mask is not None:
                                    seq_len_full = full_image_features.shape[1]
                                    target_attn = get_aligned_teacher_attn_from_lvr_tokens(
                                        teacher_attn.float(), lvr_tokens, key_pad_mask,
                                        seq_len_full, hidden_states.device,
                                    )
                                    eps = 1e-9
                                    student_attn_log = torch.log(student_attn.float() + eps)
                                    loss_attn_transfer = F.kl_div(student_attn_log, target_attn, reduction='batchmean')
                                    if not (torch.isnan(loss_attn_transfer) or torch.isinf(loss_attn_transfer)):
                                        loss_lvr_resampler = loss_lvr_resampler + loss_attn_transfer_lambda * loss_attn_transfer
                                    else:
                                        loss_attn_transfer = None
                            if loss_lvr_resampler is not None and (torch.isnan(loss_lvr_resampler) or torch.isinf(loss_lvr_resampler)):
                                loss_lvr_resampler = predicted_latent_tokens.sum() * 0.0
                            elif loss_lvr_resampler is not None and os.environ.get("LVR_DEBUG", "0") == "1":
                                print(f"[LVR.forward_latentEnd] rank={get_rank()} loss_lvr_resampler (Stage2)={loss_lvr_resampler.item():.6f} "
                                      f"predicted shape={predicted_latent_tokens.shape} target shape={target_latent_tokens.shape}", flush=True)
                        else:
                            # Stage 1: bidirectional symmetric loss only (docs/BoxFeatureResampler.md)
                            resampler_output = self.box_feature_resampler(
                                bbox_feats, key_padding_mask=key_pad_mask
                            )
                            _resampler_output_for_dit = resampler_output  # Save for DiT 50/50 GT condition
                            llm_latent_8 = hidden_states[batch_indices, seq_positions_start].to(torch.float32)
                            llm_latent_8 = llm_latent_8.view(num_bboxes, num_latent_tokens, -1)
                            resampler_output = resampler_output.to(torch.float32)
                            # Avoid NaN: skip loss if inputs contain NaN/Inf; clamp to reduce overflow
                            if torch.isnan(llm_latent_8).any() or torch.isinf(llm_latent_8).any() or torch.isnan(resampler_output).any() or torch.isinf(resampler_output).any():
                                loss_lvr_resampler = torch.tensor(0.0, device=llm_latent_8.device, dtype=torch.float32, requires_grad=True)
                                loss_attn_div = None
                            else:
                                llm_latent_8_clamped = torch.clamp(llm_latent_8, min=-1e4, max=1e4)
                                resampler_output_clamped = torch.clamp(resampler_output, min=-1e4, max=1e4)
                                # Bidirectional symmetric loss: each side only receives its own gradient (original design)
                                loss_resampler = lvr_loss_fct(llm_latent_8_clamped.detach(), resampler_output_clamped)  # train resampler
                                loss_llm = lvr_loss_fct(llm_latent_8_clamped, resampler_output_clamped.detach())        # train LLM
                                loss_lvr_resampler = (loss_resampler + loss_llm) / 2
                                loss_ortho = None
                                loss_attn_div = None
                            if loss_lvr_resampler is not None and (torch.isnan(loss_lvr_resampler) or torch.isinf(loss_lvr_resampler)):
                                loss_lvr_resampler = llm_latent_8.sum() * 0.0
                            elif loss_lvr_resampler is not None and os.environ.get("LVR_DEBUG", "0") == "1":
                                print(f"[LVR.forward_latentEnd] rank={get_rank()} loss_lvr_resampler={loss_lvr_resampler.item():.6f} "
                                      f"llm_latent_8 shape={llm_latent_8.shape} resampler_output shape={resampler_output.shape}", flush=True)
                    else:
                        # bbox_feats is empty, return zero loss
                        loss_lvr_resampler = torch.tensor(0.0, device=hidden_states.device, dtype=torch.float32, requires_grad=True)
                        loss_ortho = None
                        loss_attn_div = None
                else:
                    # num_bboxes mismatch, return zero loss
                    loss_lvr_resampler = torch.tensor(0.0, device=hidden_states.device, dtype=torch.float32, requires_grad=True)
                    loss_ortho = None
                    loss_attn_div = None
            else:
                # Empty lvr_tokens or batch_indices, return zero loss
                loss_lvr_resampler = torch.tensor(0.0, device=hidden_states.device, dtype=torch.float32, requires_grad=True)
                loss_ortho = None
                loss_attn_div = None
                if os.environ.get("LVR_DEBUG", "0") == "1":
                    print(f"[LVR.forward_latentEnd] rank={get_rank()} loss_lvr_resampler=0.0 (empty lvr_tokens or batch_indices)", flush=True)

            # loss_ortho for logging only when not already baked into loss_lvr_resampler
            if loss_ortho is None and not use_stage2 and hasattr(self, 'box_feature_resampler'):
                loss_ortho = self.box_feature_resampler.get_orthogonality_loss()

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return Qwen2_5_VLCausalLMOutputWithPast(
        # loss=loss,
        loss_ce=loss_ce,
        loss_lvr=loss_lvr,
        loss_lvr_resampler=loss_lvr_resampler,
        loss_ortho=loss_ortho,
        loss_attn_div=loss_attn_div,
        loss_attn_guidance=loss_attn_guidance,
        loss_attn_transfer=loss_attn_transfer,
        loss_mode_switch=loss_mode_switch,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=self.model.rope_deltas,
        last_position_hidden_state =last_position_hidden_state
    )


"""
    Forward function for stage 2 RL
    Kinda messy since in this stage, the transofmers will be 4.51.3 < 4.54 in stage I
    Will fix this inconsistency in final release
"""
def qwen2_5_mixed_modality_forward_lvr_rl(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    rope_deltas: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
    lvr_mode_switch: Optional[torch.Tensor] = None, # This is for INFERENCE: Which instance in the batch is in lvr mode
    last_position_hidden_state: Optional[torch.FloatTensor] = None, # This is for INFERENCE: last hidden state of the last position
) -> Union[Tuple, Qwen2_5_VLCausalLMOutputWithPast]:
    '''In this mode, no lvr_tokens'''
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    

    if inputs_embeds is None:
        inputs_embeds = self.model.get_input_embeddings()(input_ids)

    ''' 
        only happen during inference 
        inputs_embeds in shape (bs, seq_len, hidden)
    '''
    if last_position_hidden_state is not None:
        # in fact, each instance's seq_len will be 1 in inference
        inputs_embeds[lvr_mode_switch,-1,:] = last_position_hidden_state[lvr_mode_switch]
    
    ''' Only necessary in training '''
    # Pass dummy image and dummy grid to the visual model to avoid deepspeed error.
    if not lvr_mode_switch and (pixel_values is None and pixel_values_videos is None):
        # Create dummy pixel_values and grid_thw for avoiding deepspeed error.
        dummy_pixel = torch.zeros(784, 1176).to(self.visual.device)
        dummy_grid = torch.tensor([[1, 28, 28]]).to(self.visual.device)
        
        dummy_pixel = dummy_pixel.type(self.visual.dtype)
        image_embeds = self.visual(dummy_pixel, grid_thw=dummy_grid)
        # Operates as maksed_scatter for the image tokens
        # However the values are all zeros so it dosen't affect the embeddings.
        # This could avoid deepspeed error when some batch only has texts.
        inputs_embeds += image_embeds.mean() * 0
            
    if pixel_values is not None:

        # with torch.autocast(device_type='cuda', enabled=True, dtype=torch.float32):
        #     # Ensure vision tower inputs are float32
        #     pixel_values = pixel_values.to(torch.float32) 
        image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
        # Handle both list and tensor returns from visual module
        if isinstance(image_embeds, (list, tuple)):
            image_embeds = torch.cat(image_embeds, dim=0)
        # If already a tensor, use it directly


        n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
        n_image_features = image_embeds.shape[0]
        if n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )

        if input_ids is None:
            image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            image_mask = image_mask.all(-1)
        else:
            image_mask = input_ids == self.config.image_token_id


        n_image_tokens = (image_mask).sum()
        image_mask_unsqueeze = image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        n_image_features = image_embeds.shape[0]
        if not is_torchdynamo_compiling() and n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )
        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask_unsqueeze, image_embeds)
        
        # Calculate total_tokens for each batch item (needed for attention-mask LVR head in inference)
        total_tokens = torch.sum(image_mask, dim=1)  # (batch_size,)
            

    if attention_mask is not None:
        attention_mask = attention_mask.to(inputs_embeds.device)

    if position_ids is None:
        # Calculate RoPE index once per generation in the pre-fill stage only.
        # When compiling, we can't check tensor values thus we check only input length
        # It is safe to assume that `length!=1` means we're in pre-fill because compiled
        # models currently cannot do asssisted decoding
        prefill_compiled_stage = is_torchdynamo_compiling() and (
            (input_ids is not None and input_ids.shape[1] != 1)
            or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
        )
        prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
            (cache_position is not None and cache_position[0] == 0)
            or (past_key_values is None or past_key_values.get_seq_length() == 0)
        )
        if (prefill_compiled_stage or prefill_noncompiled_stage) or self.model.rope_deltas is None:
            position_ids, rope_deltas = self.model.get_rope_index(
                input_ids,
                image_grid_thw,
                video_grid_thw,
                second_per_grid_ts=second_per_grid_ts,
                attention_mask=attention_mask,
            )
            self.model.rope_deltas = rope_deltas
        else:
            batch_size, seq_length, _ = inputs_embeds.shape
            position_ids = torch.arange(seq_length, device=inputs_embeds.device)
            position_ids = position_ids.view(1, 1, -1).expand(3, batch_size, -1)
            if cache_position is not None:
                delta = (cache_position[0] + self.model.rope_deltas).to(inputs_embeds.device)
            else:
                delta = torch.zeros((batch_size, seq_length), device=inputs_embeds.device)
            delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=1)
            position_ids = position_ids.clone() + delta.to(position_ids.device)

    outputs = self.model(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
    )

    # check if there is lvr_head
    if self.config.lvr_head:
        '''apply lvr_head in _inference mode'''
        if lvr_mode_switch is not None:
            if hasattr(self.config, 'lvr_head_type') and (self.config.lvr_head_type == 'attention-mask' or self.config.lvr_head_type == 'ivr' or self.config.lvr_head_type == 'gated-focus' or self.config.lvr_head_type == 'gfr'):
                # MLP-mask LVR head in inference mode - batch processing
                if pixel_values is not None:
                    batch_size = outputs.last_hidden_state.shape[0]
                    lvr_batch_indices = torch.nonzero(lvr_mode_switch, as_tuple=True)[0]
                    if len(lvr_batch_indices) > 0 and pixel_values is not None:
                        # Get model dtype to avoid unnecessary dtype conversions
                        model_dtype = next(self.lvr_head.parameters()).dtype if len(list(self.lvr_head.parameters())) > 0 else torch.bfloat16
                        
                        batched_hidden_states = outputs.last_hidden_state[lvr_batch_indices, -1].to(model_dtype)  # (num_lvr_items, hidden_size)
                        batched_image_embeds, image_attention_mask = _prepare_batched_image_embeds(
                            image_embeds, total_tokens, lvr_batch_indices, target_dtype=model_dtype
                        )
                        v_focal_batch = self.lvr_head(batched_hidden_states, batched_image_embeds, image_attention_mask)  # (num_lvr_items, hidden_size)
                        outputs.last_hidden_state[lvr_batch_indices, -1] = v_focal_batch.to(outputs.last_hidden_state.dtype)
                        
                        # Free memory
                        del batched_hidden_states, batched_image_embeds, image_attention_mask, v_focal_batch
                    elif len(lvr_batch_indices) > 0:
                        # Fallback: use original behavior if image info not available
                        for b_idx in lvr_batch_indices:
                            hidden_state = outputs.last_hidden_state[b_idx, -1]
                            outputs.last_hidden_state[b_idx, -1] = self.lvr_head(hidden_state.unsqueeze(0)).squeeze(0)
                else:
                    # Fallback: use original behavior if image info not available
                    outputs.last_hidden_state[lvr_mode_switch,:,:] = self.lvr_head(outputs.last_hidden_state[lvr_mode_switch,:,:])
            else:
                outputs.last_hidden_state[lvr_mode_switch,:,:] = self.lvr_head(outputs.last_hidden_state[lvr_mode_switch,:,:])

    hidden_states = outputs[0]
    last_position_hidden_state = outputs.last_hidden_state[:,-1,:]
    logits = self.lm_head(hidden_states)

    loss = None
    loss_ce = None
    loss_lvr = None
    if labels is not None:
        # Upcast to float if we need to compute the loss to avoid potential precision issues
        logits = logits.float()
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Don't want CE loss for <lvr> token
        shift_labels = shift_labels.masked_fill(shift_labels == self.config.lvr_id, IGNORE_INDEX)

        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss_ce = loss_fct(shift_logits, shift_labels)

        # No lvr loss in this mode
        loss_lvr = None


    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return Qwen2_5_VLCausalLMOutputWithPast(
        # loss=loss,
        loss_ce=loss_ce,
        loss_lvr=loss_lvr,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=self.model.rope_deltas,
        last_position_hidden_state =last_position_hidden_state
    )
