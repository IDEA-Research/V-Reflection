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
                                                    rl = False):
    
    print("#"*42)
    if inference_mode:
        if lvr_head:
            print("Inference mode with Lvr_head!!!")
            transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLForConditionalGeneration.forward = qwen2_5_mixed_modality_forward_lvr_with_head_inference
        else:
            print("Inference mode without Lvr_head!!!")
            transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLForConditionalGeneration.forward = qwen2_5_mixed_modality_forward_lvr_inference
    elif rl:
        print("Activated stage 2 training!!!")
        transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLForConditionalGeneration.forward = qwen2_5_mixed_modality_forward_lvr_grpo
    else:
        if latent_end_token and lvr_head:
            print("Activated latent end token mode with LVR_Head!!!")
            transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLForConditionalGeneration.forward = qwen2_5_mixed_modality_forward_lvr_with_head_with_latentEndToken
        elif latent_end_token and not lvr_head:
            print("Activated latent end token mode without LVR_Head!!!")
            transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLForConditionalGeneration.forward = qwen2_5_mixed_modality_forward_lvr_with_latentEndToken
        elif mode_switch_loss:
            print("Activated BCE mode swtich loss!!!")
            transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLForConditionalGeneration.forward = qwen2_5_mixed_modality_forward_lvr_with_head_with_modeSwitchLoss
        elif lvr_head:
            print("Activated naive LVR with head mode!!!")
            transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLForConditionalGeneration.forward = qwen2_5_mixed_modality_forward_lvr_with_head
        else:
            print("Activated naive LVR without head mode!!!")
            transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLForConditionalGeneration.forward = qwen2_5_mixed_modality_forward_lvr
    
    print("#"*42)


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

def _build_global_lvr_token_indices(lvr_tokens, image_token_offsets):
    """
    Build global LVR token indices efficiently without list append + cat.
    
    Args:
        lvr_tokens: List of tensors, each containing local token indices for a batch item
        image_token_offsets: (batch_size,) tensor of offsets into image_embeds
    
    Returns:
        global_lvr_token_indices: (L_total,) tensor of global indices
    """
    # Pre-allocate tensor instead of list append + cat (more memory efficient)
    total_lvr_tokens = sum(len(lvr_ids) for lvr_ids in lvr_tokens)
    if total_lvr_tokens == 0:
        return torch.empty(0, dtype=torch.long, device=image_token_offsets.device)
    
    # Build tensor using list and cat to avoid inplace operations
    indices_list = []
    for b, lvr_ids in enumerate(lvr_tokens):
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
    import time
    debug_enabled = os.getenv("LVR_DEBUG", "0") == "1"
    if debug_enabled:
        prep_start = time.time()
        print(f"[_prepare_batched_image_embeds] START: image_embeds.shape={image_embeds.shape}, "
              f"total_tokens.shape={total_tokens.shape}, len(batch_indices)={len(batch_indices)}")
    
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
    import os
    debug_enabled = os.getenv("LVR_DEBUG", "0") == "1"
    if debug_enabled:
        import time
        fill_start = time.time()
        print(f"[_prepare_batched_image_embeds] Filling {L_total} items, max_num_tokens={max_num_tokens}")
    
    # Collect all slices and build the batched tensor without inplace operations
    batch_list = []
    mask_list = []
    for i in range(L_total):
        if debug_enabled and i % 100 == 0:
            print(f"[_prepare_batched_image_embeds] Processing item {i}/{L_total}, elapsed={time.time()-fill_start:.3f}s")
        
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
    
    if debug_enabled:
        print(f"[_prepare_batched_image_embeds] COMPLETE: total elapsed={time.time()-fill_start:.3f}s")
    
    return batched_image_embeds, image_attention_mask

@dataclass
class Qwen2_5_VLCausalLMOutputWithPast(ModelOutput):
    """
        please refer to the original Qwen2_5_VLCausalLMOutputWithPast in transformers.models.qwen2_5_vl.modeling_qwen2_5_vl
    """

    loss: Optional[torch.FloatTensor] = None
    loss_lvr: Optional[torch.FloatTensor] = None
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
        # Returns a loss function: 1 - cosine similarity
        def cosine_loss(x, y):
            return 1 - F.cosine_similarity(x, y, dim=-1).mean()
        return cosine_loss
    else:
        raise ValueError(f"Unsupported lvr_loss: {loss_lvr_fct}")

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
) -> Union[Tuple, Qwen2_5_VLCausalLMOutputWithPast]:
    
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
        # Avoid inplace operation by cloning and reconstructing
        inputs_embeds = inputs_embeds.clone()
        inputs_embeds[lvr_mode_switch,-1,:] = last_position_hidden_state[lvr_mode_switch]

    ''' 
        only happen during inference 
        inputs_embeds in shape (bs, seq_len, hidden)
    '''
    if last_position_hidden_state is not None:
        # in fact, each instance's seq_len will be 1 in inference
        # Avoid inplace operation by cloning and reconstructing
        if not inputs_embeds.is_leaf or inputs_embeds.requires_grad:
            inputs_embeds = inputs_embeds.clone()
        inputs_embeds[lvr_mode_switch,-1,:] = last_position_hidden_state[lvr_mode_switch]

    '''Only necessary in training'''
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

                # Build global indices efficiently (avoids list append + cat)
                global_lvr_token_indices = _build_global_lvr_token_indices(lvr_tokens, image_token_offsets)

                # Step 3: Gather the selected visual embeddings
                selected_lvr_embeds = image_embeds[global_lvr_token_indices]  # [L_total, H]

                # Step 4: Replace in input_embeds at the right batch and position
                # Avoid inplace operation by cloning if needed
                if inputs_embeds.requires_grad:
                    inputs_embeds = inputs_embeds.clone()
                inputs_embeds[batch_indices, seq_positions] = selected_lvr_embeds
            else:
                '''re-encode target area'''
                # Now lvr_tokens is pixel_values of the cropped targets
                selected_lvr_embeds = self.model.get_image_features(lvr_tokens, lvr_tokens_thw)
                selected_lvr_embeds = torch.cat(selected_lvr_embeds, dim=0)
                # Avoid inplace operation by cloning if needed
                if inputs_embeds.requires_grad:
                    inputs_embeds = inputs_embeds.clone()
                inputs_embeds[batch_indices, seq_positions] = selected_lvr_embeds

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
                position_ids, rope_deltas = self.get_rope_index(
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
    No LVR Head
'''
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
                '''Exrtacting tokens from original image'''
                #  GLOBAL starting index in `image_embeds` of each image in the batch
                image_token_offsets = torch.cumsum(
                    F.pad(total_tokens, (1, 0)), dim=0
                )[:-1]  # shape [B], offset into image_embeds for each batch element

                # Build global indices efficiently (avoids list append + cat)
                global_lvr_token_indices = _build_global_lvr_token_indices(lvr_tokens, image_token_offsets)

                # Step 3: Gather the selected visual embeddings
                selected_lvr_embeds = image_embeds[global_lvr_token_indices]  # [L_total, H]

                # Step 4: Replace in input_embeds at the right batch and position
                # Avoid inplace operation by cloning if needed
                if inputs_embeds.requires_grad:
                    inputs_embeds = inputs_embeds.clone()
                inputs_embeds[batch_indices, seq_positions] = selected_lvr_embeds
            else:
                '''re-encode target area'''
                # Now lvr_tokens is pixel_values of the cropped targets
                selected_lvr_embeds = self.model.get_image_features(lvr_tokens, lvr_tokens_thw)
                selected_lvr_embeds = torch.cat(selected_lvr_embeds, dim=0)
                # Avoid inplace operation by cloning if needed
                if inputs_embeds.requires_grad:
                    inputs_embeds = inputs_embeds.clone()
                inputs_embeds[batch_indices, seq_positions] = selected_lvr_embeds

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
                position_ids, rope_deltas = self.get_rope_index(
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

                global_lvr_token_indices = []
                for b, lvr_ids in enumerate(lvr_tokens):
                    # Convert local to global index
                    offset = image_token_offsets[b].item()
                    global_lvr_token_indices.append(lvr_ids + offset)
                global_lvr_token_indices = torch.cat(global_lvr_token_indices, dim=0)  # [L_total]

                # Step 3: Gather the selected visual embeddings
                selected_lvr_embeds = image_embeds[global_lvr_token_indices]  # [L_total, H]

                # Step 4: Replace in input_embeds at the right batch and position
                inputs_embeds[batch_indices, seq_positions] = selected_lvr_embeds
            
            else:
                '''re-encode target area'''
                # Now lvr_tokens is pixel_values of the cropped targets
                selected_lvr_embeds = self.model.get_image_features(lvr_tokens, lvr_tokens_thw)
                selected_lvr_embeds = torch.cat(selected_lvr_embeds, dim=0)
                inputs_embeds[batch_indices, seq_positions] = selected_lvr_embeds

            

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
            position_ids, rope_deltas = self.get_rope_index(
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
            
            if hasattr(self.config, 'lvr_head_type') and (self.config.lvr_head_type == 'attention-mask' or self.config.lvr_head_type == 'slot-attention' or self.config.lvr_head_type == 'ivr' or self.config.lvr_head_type == 'implicit-visual-routing' or self.config.lvr_head_type == 'gated-focus' or self.config.lvr_head_type == 'gfr' or self.config.lvr_head_type == 'intrinsic-similarity' or self.config.lvr_head_type == 'isg'):
                # MLP-mask, Slot-Attention, and Intrinsic-Similarity LVR heads need image embeddings - batch processing
                # Get model dtype to avoid unnecessary dtype conversions
                model_dtype = _get_lvr_head_dtype(self.lvr_head)
                
                # Prepare batched hidden states in model dtype
                batched_hidden_states = outputs.last_hidden_state[batch_indices, seq_positions_start].to(model_dtype)  # (L_total, hidden_size)
                
                # Prepare batched image embeddings directly in model dtype to save memory
                batched_image_embeds, image_attention_mask = _prepare_batched_image_embeds(
                    image_embeds, total_tokens, batch_indices, target_dtype=model_dtype
                )
                
                # Batch call LVR head (works for both attention-mask and slot-attention)
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
        if hasattr(self.config, 'lvr_head_type') and (self.config.lvr_head_type == 'attention-mask' or self.config.lvr_head_type == 'slot-attention' or self.config.lvr_head_type == 'ivr' or self.config.lvr_head_type == 'implicit-visual-routing' or self.config.lvr_head_type == 'gated-focus' or self.config.lvr_head_type == 'gfr' or self.config.lvr_head_type == 'intrinsic-similarity' or self.config.lvr_head_type == 'isg'):
            # MLP-mask, Slot-Attention, and Intrinsic-Similarity LVR heads in inference mode - batch processing
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
                    
                    # Batch call LVR head (works for both attention-mask and slot-attention)
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
        
        if hasattr(self.config, 'lvr_head_type') and (self.config.lvr_head_type == 'attention-mask' or self.config.lvr_head_type == 'ivr' or self.config.lvr_head_type == 'implicit-visual-routing' or self.config.lvr_head_type == 'gated-focus' or self.config.lvr_head_type == 'gfr'):
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

                global_lvr_token_indices = []
                for b, lvr_ids in enumerate(lvr_tokens):
                    # Convert local to global index
                    offset = image_token_offsets[b].item()
                    global_lvr_token_indices.append(lvr_ids + offset)
                global_lvr_token_indices = torch.cat(global_lvr_token_indices, dim=0)  # [L_total]

                # Step 3: Gather the selected visual embeddings
                selected_lvr_embeds = image_embeds[global_lvr_token_indices]  # [L_total, H]

                # Step 4: Replace in input_embeds at the right batch and position
                inputs_embeds[batch_indices, seq_positions] = selected_lvr_embeds
            
            else:
                '''re-encode target area'''
                # Now lvr_tokens is pixel_values of the cropped targets
                selected_lvr_embeds = self.model.get_image_features(lvr_tokens, lvr_tokens_thw)
                selected_lvr_embeds = torch.cat(selected_lvr_embeds, dim=0)
                inputs_embeds[batch_indices, seq_positions] = selected_lvr_embeds

            

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
            position_ids, rope_deltas = self.get_rope_index(
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
            
            if hasattr(self.config, 'lvr_head_type') and (self.config.lvr_head_type == 'attention-mask' or self.config.lvr_head_type == 'ivr' or self.config.lvr_head_type == 'implicit-visual-routing' or self.config.lvr_head_type == 'gated-focus' or self.config.lvr_head_type == 'gfr' or self.config.lvr_head_type == 'intrinsic-similarity' or self.config.lvr_head_type == 'isg'):
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
        if hasattr(self.config, 'lvr_head_type') and (self.config.lvr_head_type == 'attention-mask' or self.config.lvr_head_type == 'ivr' or self.config.lvr_head_type == 'implicit-visual-routing' or self.config.lvr_head_type == 'gated-focus' or self.config.lvr_head_type == 'gfr' or self.config.lvr_head_type == 'intrinsic-similarity' or self.config.lvr_head_type == 'isg'):
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
        
        if hasattr(self.config, 'lvr_head_type') and (self.config.lvr_head_type == 'attention-mask' or self.config.lvr_head_type == 'ivr' or self.config.lvr_head_type == 'implicit-visual-routing' or self.config.lvr_head_type == 'gated-focus' or self.config.lvr_head_type == 'gfr'):
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

            global_lvr_token_indices = []

            for b, lvr_ids in enumerate(lvr_tokens):
                # Convert local to global index
                offset = image_token_offsets[b].item()
                global_lvr_token_indices.append(lvr_ids + offset)
            global_lvr_token_indices = torch.cat(global_lvr_token_indices, dim=0)  # [L_total]

            # Step 3: Gather the selected visual embeddings
            selected_lvr_embeds = image_embeds[global_lvr_token_indices]  # [L_total, H]

            # Step 4: Replace in input_embeds at the right batch and position
            inputs_embeds[batch_indices, seq_positions] = selected_lvr_embeds
            

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
            position_ids, rope_deltas = self.get_rope_index(
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
            
            if hasattr(self.config, 'lvr_head_type') and (self.config.lvr_head_type == 'attention-mask' or self.config.lvr_head_type == 'ivr' or self.config.lvr_head_type == 'implicit-visual-routing' or self.config.lvr_head_type == 'gated-focus' or self.config.lvr_head_type == 'gfr'):
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
        if hasattr(self.config, 'lvr_head_type') and (self.config.lvr_head_type == 'attention-mask' or self.config.lvr_head_type == 'ivr' or self.config.lvr_head_type == 'implicit-visual-routing' or self.config.lvr_head_type == 'gated-focus' or self.config.lvr_head_type == 'gfr'):
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

            global_lvr_token_indices = []

            for b, lvr_ids in enumerate(lvr_tokens):
                # Convert local to global index
                offset = image_token_offsets[b].item()
                global_lvr_token_indices.append(lvr_ids + offset)
            global_lvr_token_indices = torch.cat(global_lvr_token_indices, dim=0)  # [L_total]

            # Step 3: Gather the selected visual embeddings
            selected_lvr_embeds = image_embeds[global_lvr_token_indices]  # [L_total, H]

            # Step 4: Replace in input_embeds at the right batch and position
            inputs_embeds[batch_indices, seq_positions] = selected_lvr_embeds

            '''Apply lvr_latent_end_token'''
            lvr_latent_end_mask = (input_ids == self.config.lvr_latent_end_id)
            batch_indices_latentend, seq_positions_latentend = torch.nonzero(lvr_latent_end_mask, as_tuple=True)
            if lvr_latent_end_mask.any():
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
            position_ids, rope_deltas = self.get_rope_index(
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
            
            if hasattr(self.config, 'lvr_head_type') and (self.config.lvr_head_type == 'attention-mask' or self.config.lvr_head_type == 'ivr' or self.config.lvr_head_type == 'implicit-visual-routing' or self.config.lvr_head_type == 'gated-focus' or self.config.lvr_head_type == 'gfr'):
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
            if hasattr(self.config, 'lvr_head_type') and (self.config.lvr_head_type == 'attention-mask' or self.config.lvr_head_type == 'ivr' or self.config.lvr_head_type == 'implicit-visual-routing' or self.config.lvr_head_type == 'gated-focus' or self.config.lvr_head_type == 'gfr'):
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
    mode_switch_loss_fct = set_lvr_loss_fct(self.config.loss_mode_switch_fct)

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
        shift_labels = shift_labels.masked_fill((shift_labels == self.config.lvr_id)|(shift_labels == self.config.lvr_latent_end_id), IGNORE_INDEX)

        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss_ce = loss_fct(shift_logits, shift_labels)

        # lvr loss
        # Get last hidden states for <lvr> token positions
        seq_positions_start = seq_positions - 1
        selected_hidden_states = hidden_states[batch_indices, seq_positions_start].to(torch.float32)  # [L_total, H]
        # Get last hidden states for <lvr_latent_end> token positions
        seq_positions_start_latentend = seq_positions_latentend - 1
        selected_hidden_states_latentend = hidden_states[batch_indices_latentend, seq_positions_start_latentend].to(torch.float32)  # [L_total, H]

        ''' We need to convert to fp32 to avoid overflow by mse'''
        selected_lvr_embeds = selected_lvr_embeds.to(torch.float32)
        selected_lvr_embeds_latentend = self.lvr_latent_end_emb.unsqueeze(0).expand_as(selected_hidden_states_latentend).to(torch.float32)
        selected_lvr_embeds_latentend = selected_lvr_embeds_latentend.to(selected_hidden_states_latentend.device)
        # Compute LVR loss between predicted and inserted lvr embeddings
        loss_lvr = lvr_loss_fct(selected_hidden_states, selected_lvr_embeds) 
        loss_mode_switch = mode_switch_loss_fct(selected_hidden_states_latentend, selected_lvr_embeds_latentend)


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

            global_lvr_token_indices = []

            for b, lvr_ids in enumerate(lvr_tokens):
                # Convert local to global index
                offset = image_token_offsets[b].item()
                global_lvr_token_indices.append(lvr_ids + offset)
            global_lvr_token_indices = torch.cat(global_lvr_token_indices, dim=0)  # [L_total]

            # Step 3: Gather the selected visual embeddings
            selected_lvr_embeds = image_embeds[global_lvr_token_indices]  # [L_total, H]

            # Step 4: Replace in input_embeds at the right batch and position
            inputs_embeds[batch_indices, seq_positions] = selected_lvr_embeds

            '''Apply lvr_latent_end_token'''
            lvr_latent_end_mask = (input_ids == self.config.lvr_latent_end_id)
            batch_indices_latentend, seq_positions_latentend = torch.nonzero(lvr_latent_end_mask, as_tuple=True)
            if lvr_latent_end_mask.any():
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
            position_ids, rope_deltas = self.get_rope_index(
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
    last_position_hidden_state = outputs.last_hidden_state[:,-1,:]
    logits = self.lm_head(hidden_states)

    lvr_loss_fct = set_lvr_loss_fct(self.config.loss_lvr_fct)
    mode_switch_loss_fct = set_lvr_loss_fct(self.config.loss_mode_switch_fct)

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
        shift_labels = shift_labels.masked_fill((shift_labels == self.config.lvr_id)|(shift_labels == self.config.lvr_latent_end_id), IGNORE_INDEX)

        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss_ce = loss_fct(shift_logits, shift_labels)

        # lvr loss
        # Get last hidden states for <lvr> token positions
        seq_positions_start = seq_positions - 1
        selected_hidden_states = hidden_states[batch_indices, seq_positions_start].to(torch.float32)  # [L_total, H]
        # Get last hidden states for <lvr_latent_end> token positions
        seq_positions_start_latentend = seq_positions_latentend - 1
        selected_hidden_states_latentend = hidden_states[batch_indices_latentend, seq_positions_start_latentend].to(torch.float32)  # [L_total, H]

        ''' We need to convert to fp32 to avoid overflow by mse'''
        selected_lvr_embeds = selected_lvr_embeds.to(torch.float32)
        selected_lvr_embeds_latentend = self.lvr_latent_end_emb.unsqueeze(0).expand_as(selected_hidden_states_latentend).to(torch.float32)
        selected_lvr_embeds_latentend = selected_lvr_embeds_latentend.to(selected_hidden_states_latentend.device)
        # Compute LVR loss between predicted and inserted lvr embeddings
        loss_lvr = lvr_loss_fct(selected_hidden_states, selected_lvr_embeds) 
        loss_mode_switch = mode_switch_loss_fct(selected_hidden_states_latentend, selected_lvr_embeds_latentend)


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
            position_ids, rope_deltas = self.get_rope_index(
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
            if hasattr(self.config, 'lvr_head_type') and (self.config.lvr_head_type == 'attention-mask' or self.config.lvr_head_type == 'ivr' or self.config.lvr_head_type == 'implicit-visual-routing' or self.config.lvr_head_type == 'gated-focus' or self.config.lvr_head_type == 'gfr'):
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
