import os
import torch
import torch.nn as nn
import torch.distributed as dist
import wandb
from transformers import Trainer
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    TRAINER_STATE_NAME,
    PREFIX_CHECKPOINT_DIR,
    logger,
    ExportableState,
    SaveStrategy
)

from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS

from src.constants import IGNORE_INDEX

def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE and not ignore_status:
            pass  # Param not available, will gather
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param

class QwenLVRSFTTrainer(Trainer):

    def create_optimizer(self):
        """
        Setup the optimizer.
        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            lr_mapper = {}
            visual_parameters = []
            merger_parameters = []
            lvr_head_parameters =[]

            if self.args.vision_lr is not None:
                lr_mapper["visual"] = self.args.vision_lr
                visual_parameters = [name for name, _ in opt_model.named_parameters() if "visual" in name and "merger" not in name]
            if self.args.merger_lr is not None:
                lr_mapper["merger"] = self.args.merger_lr
                merger_parameters = [name for name, _ in opt_model.named_parameters() if "merger" in name]
            if self.args.lvr_head_lr is not None:
                lr_mapper["lvr_head"] = self.args.lvr_head_lr
                lvr_head_parameters = [name for name, _ in opt_model.named_parameters() if "lvr_head" in name]

            if len(lr_mapper) > 0:
                special_lr_parameters = merger_parameters + visual_parameters + lvr_head_parameters
                
                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in special_lr_parameters and p.requires_grad)],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in special_lr_parameters and p.requires_grad)],
                        "weight_decay": 0.0,
                    },
                ]
                
                if visual_parameters: 
                    optimizer_grouped_parameters.extend(
                        [
                            {
                                "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in visual_parameters and p.requires_grad)],
                                "weight_decay": self.args.weight_decay,
                                "lr": self.args.vision_lr,
                            },
                            {
                                "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in visual_parameters and p.requires_grad)],
                                "weight_decay": 0.0,
                                "lr": self.args.vision_lr,
                            },
                        ]
                    )
                
                if merger_parameters: 
                    optimizer_grouped_parameters.extend(
                        [
                            {
                                "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in merger_parameters and p.requires_grad)],
                                "weight_decay": self.args.weight_decay,
                                "lr": self.args.merger_lr,
                            },
                            {
                                "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in merger_parameters and p.requires_grad)],
                                "weight_decay": 0.0,
                                "lr": self.args.merger_lr,
                            },
                        ]
                    )
                
                if lvr_head_parameters: 
                    optimizer_grouped_parameters.extend(
                        [
                            {
                                "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in lvr_head_parameters and p.requires_grad)],
                                "weight_decay": self.args.weight_decay,
                                "lr": self.args.lvr_head_lr,
                            },
                            {
                                "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in lvr_head_parameters and p.requires_grad)],
                                "weight_decay": 0.0,
                                "lr": self.args.lvr_head_lr,
                            },
                        ]
                    )
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)],
                        "weight_decay": 0.0,
                    },
                ]
            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        return self.optimizer
    
    def _save_checkpoint(self, model, trial):
        # In all cases, including ddp/dp/deepspeed, self.model is always a reference to the model we
        # want to save except FullyShardedDDP.
        # assert unwrap_model(model) is self.model, "internal model should be a reference to self.model"

        # Save model checkpoint
        # modified to support online checkpointing
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

        if self.hp_search_backend is None and trial is None:
            self.store_flos()

        run_dir = self._get_output_dir(trial=trial)
        # output_dir is the local path forcheckpoint
        output_dir = os.path.join(run_dir, checkpoint_folder)
        
        # Clean up Tensor objects from config before saving to avoid JSON serialization errors
        # Store original values temporarily and restore after saving
        config_tensor_backup = {}
        if hasattr(self.model, 'config'):
            config = self.model.config
            # Check all attributes in config for Tensor objects
            # Use __dict__ to get actual attributes, avoiding methods and properties
            if hasattr(config, '__dict__'):
                for attr_name in list(config.__dict__.keys()):
                    try:
                        attr_value = getattr(config, attr_name, None)
                        if isinstance(attr_value, torch.Tensor):
                            config_tensor_backup[attr_name] = attr_value
                            # Remove Tensor from config to allow JSON serialization
                            delattr(config, attr_name)
                    except (AttributeError, TypeError, RuntimeError):
                        pass
        
        try:
            self.save_model(output_dir, _internal_call=True)
        finally:
            # Restore Tensor attributes after saving
            if hasattr(self.model, 'config'):
                config = self.model.config
                for attr_name, attr_value in config_tensor_backup.items():
                    setattr(config, attr_name, attr_value)

        if self.args.save_strategy in [SaveStrategy.STEPS, SaveStrategy.EPOCH] and self.state.best_global_step:
            best_checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.best_global_step}"
            best_checkpoint_dir = os.path.join(run_dir, best_checkpoint_folder)

            if os.path.exists(best_checkpoint_dir):
                self.state.best_model_checkpoint = best_checkpoint_dir

        if not self.args.save_only_model:
            # Save optimizer and scheduler
            self._save_optimizer_and_scheduler(output_dir)
            self._save_scaler(output_dir)
            # Save RNG state
            self._save_rng_state(output_dir)

        # Save the Trainer state
        if self.args.should_save:
            # Update `ExportableState` callbacks and `TrainerControl` state to where we are currently
            for cb in [
                cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)
            ]:
                cb_name = cb.__class__.__name__
                cb_state = cb.state()
                if isinstance(self.state.stateful_callbacks[cb_name], list):
                    self.state.stateful_callbacks[cb_name].append(cb_state)
                else:
                    self.state.stateful_callbacks[cb_name] = cb_state
            self.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))

        if self.args.push_to_hub:
            self._push_from_checkpoint(output_dir)

        # Maybe delete some older checkpoints.
        if self.args.should_save:
            # Solely rely on numerical checkpoint id for rotation.
            # mtime is not reliable especially on some fuse fs in cloud environments.
            self._rotate_checkpoints(use_mtime=False, output_dir=run_dir)

    def compute_loss(self, model, inputs,num_items_in_batch=None, return_outputs=False):

        if self.args.enable_data_packing:
            batch_size = inputs['input_ids'].size(0)
            total_tokens = inputs['input_ids'].size(0) * inputs['input_ids'].size(1)
            self.log({
            "batch_size": batch_size,
            "tokens_per_device": total_tokens,})

        # Removed shape logging - focus on gradients only

        # Store model reference for fallback loss creation if needed
        model_ref = model
        self._current_model = model  # Store for debug logging
        
        # Filter out debug keys before passing to model
        model_inputs = {k: v for k, v in inputs.items() if not k.startswith('_debug_')}
        
        # Helper: current rank for debug prints (so we know which rank had the issue)
        def _rank():
            try:
                return dist.get_rank() if dist.is_initialized() else int(os.environ.get("RANK", 0))
            except Exception:
                return int(os.environ.get("RANK", 0))

        # CRITICAL: Assertion check for NaN/Inf in inputs BEFORE forward pass
        # This prevents NaN from propagating through the model and causing NCCL hangs
        # If detected, skip this sample and return safe zero loss
        nan_in_inputs = False
        nan_location = []
        nan_details = {}
        
        # Check input_ids
        if 'input_ids' in model_inputs:
            nan_mask = torch.isnan(model_inputs['input_ids'])
            inf_mask = torch.isinf(model_inputs['input_ids'].float())
            if nan_mask.any() or inf_mask.any():
                nan_in_inputs = True
                nan_location.append('input_ids')
                nan_details['input_ids'] = {
                    'nan_count': nan_mask.sum().item(),
                    'inf_count': inf_mask.sum().item(),
                    'total': model_inputs['input_ids'].numel()
                }
        
        # Check pixel_values (if present)
        if 'pixel_values' in model_inputs:
            nan_mask = torch.isnan(model_inputs['pixel_values'])
            inf_mask = torch.isinf(model_inputs['pixel_values'])
            if nan_mask.any() or inf_mask.any():
                nan_in_inputs = True
                nan_location.append('pixel_values')
                nan_details['pixel_values'] = {
                    'nan_count': nan_mask.sum().item(),
                    'inf_count': inf_mask.sum().item(),
                    'total': model_inputs['pixel_values'].numel()
                }
        
        # Check attention_mask (if present)
        if 'attention_mask' in model_inputs:
            nan_mask = torch.isnan(model_inputs['attention_mask'])
            inf_mask = torch.isinf(model_inputs['attention_mask'].float())
            if nan_mask.any() or inf_mask.any():
                nan_in_inputs = True
                nan_location.append('attention_mask')
                nan_details['attention_mask'] = {
                    'nan_count': nan_mask.sum().item(),
                    'inf_count': inf_mask.sum().item(),
                    'total': model_inputs['attention_mask'].numel()
                }
        
        # Check labels (if present)
        if 'labels' in model_inputs:
            # Labels may contain IGNORE_INDEX (-100), so check only valid labels
            valid_mask = model_inputs['labels'] != IGNORE_INDEX
            if valid_mask.any():
                valid_labels = model_inputs['labels'][valid_mask].float()
                nan_mask = torch.isnan(valid_labels)
                inf_mask = torch.isinf(valid_labels)
                if nan_mask.any() or inf_mask.any():
                    nan_in_inputs = True
                    nan_location.append('labels')
                    nan_details['labels'] = {
                        'nan_count': nan_mask.sum().item(),
                        'inf_count': inf_mask.sum().item(),
                        'valid_labels': valid_mask.sum().item()
                    }
        
        # Assertion: If NaN detected in inputs, skip this sample and return safe zero loss
        if nan_in_inputs:
            rank = _rank()
            self._log_debug_info(inputs, "input_nan_detected")
            print(f"[TRAIN.compute_loss] ASSERTION FAILED | rank={rank} step={self.state.global_step} | "
                  f"NaN/Inf detected in inputs before forward pass!", flush=True)
            print(f"[TRAIN.compute_loss] rank={rank} Location: {nan_location}", flush=True)
            for key, details in nan_details.items():
                print(f"[TRAIN.compute_loss] rank={rank}   {key}: NaN={details.get('nan_count', 0)}, "
                      f"Inf={details.get('inf_count', 0)}, Total={details.get('total', details.get('valid_labels', 'N/A'))}", flush=True)
            print(f"[TRAIN.compute_loss] rank={rank} Skipping this sample and returning zero loss to prevent NaN propagation.", flush=True)
            
            # Return a safe zero loss tensor on the correct device
            # Use input_ids device as reference
            device = model_inputs['input_ids'].device
            dtype = torch.float32
            
            # Create a zero loss tensor with requires_grad=True
            # Zero loss with gradient is safe - gradient will be zero, so it won't affect model parameters
            # This allows training to continue while effectively skipping this problematic sample
            zero_loss = torch.tensor(0.0, device=device, dtype=dtype, requires_grad=True)
            
            # Create dummy outputs with zero losses (with gradient for compatibility)
            class DummyOutputs:
                def __init__(self, device, dtype):
                    self.loss_ce = torch.tensor(0.0, device=device, dtype=dtype, requires_grad=True)
                    self.loss_lvr = torch.tensor(0.0, device=device, dtype=dtype, requires_grad=True)
                    self.loss_lvr_resampler = torch.tensor(0.0, device=device, dtype=dtype, requires_grad=True)
                    self.loss_ortho = torch.tensor(0.0, device=device, dtype=dtype, requires_grad=True)
                    self.loss_attn_div = torch.tensor(0.0, device=device, dtype=dtype, requires_grad=True)
                    self.loss_mode_switch = torch.tensor(0.0, device=device, dtype=dtype, requires_grad=True)
                    self.loss = zero_loss
            
            dummy_outputs = DummyOutputs(device, dtype)
            
            if return_outputs:
                return (zero_loss, dummy_outputs)
            else:
                return zero_loss
        
        outputs = model(**model_inputs)
        # loss = outputs.loss  # total loss
        loss_ce = outputs.loss_ce
        loss_lvr = outputs.loss_lvr
        loss_lvr_resampler = getattr(outputs, 'loss_lvr_resampler', None)
        loss_ortho = getattr(outputs, 'loss_ortho', None)
        loss_attn_div = getattr(outputs, 'loss_attn_div', None)
        loss_attn_guidance = getattr(outputs, 'loss_attn_guidance', None)
        loss_attn_transfer = getattr(outputs, 'loss_attn_transfer', None)
        loss_mode_switch = getattr(outputs, 'loss_mode_switch', None)

        # NaN detection and protection for individual loss components
        # Clamp loss values to prevent extreme values that could lead to NaN
        MAX_LOSS_VALUE = 100.0  # Reasonable upper bound for loss values
        
        if loss_ce is not None:
            # Check for NaN or Inf in loss_ce
            if torch.isnan(loss_ce) or torch.isinf(loss_ce):
                rank = _rank()
                self._log_debug_info(inputs, "loss_ce")
                self._log_detailed_debug_info(model_inputs, "loss_ce")
                print(f"[TRAIN.compute_loss] WARNING | rank={rank} step={self.state.global_step} | "
                      f"loss_ce is NaN/Inf: {loss_ce.item()}, replacing with 0.0", flush=True)
                # Use nan_to_num to preserve computation graph connection
                loss_ce = torch.nan_to_num(loss_ce, nan=0.0, posinf=0.0, neginf=0.0)
            else:
                # Clamp to prevent extreme values
                loss_ce = torch.clamp(loss_ce, min=0.0, max=MAX_LOSS_VALUE)
        
        if loss_lvr is not None:
            # Check for NaN or Inf in loss_lvr
            if torch.isnan(loss_lvr) or torch.isinf(loss_lvr):
                rank = _rank()
                self._log_debug_info(inputs, "loss_lvr")
                self._log_detailed_debug_info(model_inputs, "loss_lvr")
                print(f"[TRAIN.compute_loss] WARNING | rank={rank} step={self.state.global_step} | "
                      f"loss_lvr is NaN/Inf: {loss_lvr.item()}, replacing with 0.0", flush=True)
                # Use nan_to_num to preserve computation graph connection
                loss_lvr = torch.nan_to_num(loss_lvr, nan=0.0, posinf=0.0, neginf=0.0)
            else:
                # Clamp to prevent extreme values
                loss_lvr = torch.clamp(loss_lvr, min=0.0, max=MAX_LOSS_VALUE)
        
        if loss_lvr_resampler is not None:
            if torch.isnan(loss_lvr_resampler) or torch.isinf(loss_lvr_resampler):
                rank = _rank()
                self._log_debug_info(inputs, "loss_lvr_resampler")
                self._log_detailed_debug_info(model_inputs, "loss_lvr_resampler")
                print(f"[TRAIN.compute_loss] WARNING | rank={rank} step={self.state.global_step} | "
                      f"loss_lvr_resampler is NaN/Inf: {loss_lvr_resampler.item()}, replacing with 0.0", flush=True)
                print(f"[TRAIN.compute_loss] rank={rank} loss_ce={loss_ce.item() if loss_ce is not None else 'None'}, "
                      f"loss_lvr={loss_lvr.item() if loss_lvr is not None else 'None'}", flush=True)
                loss_lvr_resampler = torch.nan_to_num(loss_lvr_resampler, nan=0.0, posinf=0.0, neginf=0.0)
            else:
                loss_lvr_resampler = torch.clamp(loss_lvr_resampler, min=0.0, max=MAX_LOSS_VALUE)

        if loss_ortho is not None:
            if torch.isnan(loss_ortho) or torch.isinf(loss_ortho):
                rank = _rank()
                self._log_debug_info(inputs, "loss_ortho")
                print(f"[TRAIN.compute_loss] WARNING | rank={rank} step={self.state.global_step} | "
                      f"loss_ortho is NaN/Inf: {loss_ortho.item()}, replacing with 0.0", flush=True)
                loss_ortho = torch.nan_to_num(loss_ortho, nan=0.0, posinf=0.0, neginf=0.0)
            else:
                loss_ortho = torch.clamp(loss_ortho, min=0.0, max=MAX_LOSS_VALUE)

        if loss_attn_div is not None:
            if torch.isnan(loss_attn_div) or torch.isinf(loss_attn_div):
                rank = _rank()
                self._log_debug_info(inputs, "loss_attn_div")
                print(f"[TRAIN.compute_loss] WARNING | rank={rank} step={self.state.global_step} | "
                      f"loss_attn_div is NaN/Inf: {loss_attn_div.item()}, replacing with 0.0", flush=True)
                loss_attn_div = torch.nan_to_num(loss_attn_div, nan=0.0, posinf=0.0, neginf=0.0)
            else:
                loss_attn_div = torch.clamp(loss_attn_div, min=0.0, max=MAX_LOSS_VALUE)

        # Build combined loss (two-stage: no mode_switch_loss)
        loss = loss_ce
        if self.args.use_mse_loss and loss_lvr is not None and self.args.loss_lvr_lambda > 0:
            loss = loss + self.args.loss_lvr_lambda * loss_lvr
        if loss_lvr_resampler is not None and getattr(self.args, 'loss_lvr_resampler_lambda', 0.0) > 0:
            loss = loss + self.args.loss_lvr_resampler_lambda * loss_lvr_resampler
        # Final NaN check and protection for combined loss
        if torch.isnan(loss) or torch.isinf(loss):
            rank = _rank()
            self._log_debug_info(inputs, "combined_loss")
            self._log_detailed_debug_info(model_inputs, "combined_loss")
            print(f"[TRAIN.compute_loss] ERROR | rank={rank} step={self.state.global_step} | "
                  f"Combined loss is NaN/Inf: {loss.item()}, replacing with loss_ce only", flush=True)
            print(f"[TRAIN.compute_loss] rank={rank} components: loss_ce={loss_ce.item() if loss_ce is not None else 'None'}, "
                  f"loss_lvr={loss_lvr.item() if loss_lvr is not None else 'None'}, "
                  f"loss_lvr_resampler={loss_lvr_resampler.item() if loss_lvr_resampler is not None else 'None'}, "
                  f"loss_ortho={loss_ortho.item() if loss_ortho is not None else 'None'}, "
                  f"loss_attn_div={loss_attn_div.item() if loss_attn_div is not None else 'None'}", flush=True)
            # Fallback to loss_ce only if combined loss is NaN
            loss = loss_ce if loss_ce is not None else torch.tensor(0.0, device=loss.device, dtype=loss.dtype, requires_grad=True)
        
        # Clamp final loss to prevent extreme values
        loss = torch.clamp(loss, min=0.0, max=MAX_LOSS_VALUE)

        # Log each component (use safe item() extraction)
        try:
            loss_total_val = loss.detach().item()
            loss_ce_val = loss_ce.detach().item() if loss_ce is not None else 0.0
            loss_lvr_val = loss_lvr.detach().item() if loss_lvr is not None else 0.0
            loss_lvr_resampler_val = loss_lvr_resampler.detach().item() if loss_lvr_resampler is not None else 0.0
            loss_ortho_val = loss_ortho.detach().item() if loss_ortho is not None else 0.0
            loss_attn_div_val = loss_attn_div.detach().item() if loss_attn_div is not None else 0.0
            # When lambda=0, log 0 to indicate the term is disabled (not used in combined loss)
            if getattr(self.args, 'loss_ortho_lambda', 0.1) == 0:
                loss_ortho_val = 0.0
            if getattr(self.args, 'loss_attn_div_lambda', 0.0) == 0:
                loss_attn_div_val = 0.0
            loss_attn_guidance_val = loss_attn_guidance.detach().item() if loss_attn_guidance is not None else 0.0
            loss_attn_transfer_val = loss_attn_transfer.detach().item() if loss_attn_transfer is not None else 0.0
            loss_mode_switch_val = 0.0  # Not used in two-stage pipeline
            
            # Check for NaN before logging
            if torch.isnan(loss) or torch.isnan(torch.tensor(loss_total_val)):
                rank = _rank()
                print(f"[TRAIN.compute_loss] CRITICAL | rank={rank} step={self.state.global_step} | "
                      f"Loss is still NaN after protection! Using fallback loss.", flush=True)
                # Use loss_ce if available and valid, otherwise create a valid loss with requires_grad
                if loss_ce is not None and not torch.isnan(loss_ce) and not torch.isinf(loss_ce):
                    loss = loss_ce
                    loss_total_val = loss_ce_val
                    print(f"[TRAIN.compute_loss] FALLBACK | rank={rank} Using loss_ce={loss_ce_val:.6f}", flush=True)
                else:
                    # Create a valid loss tensor connected to the computation graph
                    # Use a small value from model output to ensure requires_grad=True
                    try:
                        # Get a parameter from the model to create a connected loss
                        model_param = next(model_ref.parameters())
                        # Create zero loss but connected to computation graph via a zero multiplication
                        # This ensures requires_grad=True and proper gradient flow
                        loss = torch.tensor(0.0, device=model_param.device, dtype=model_param.dtype) * model_param.sum() * 0.0
                        loss_total_val = 0.0
                        print(f"[TRAIN.compute_loss] FALLBACK | rank={rank} Created zero loss connected to computation graph", flush=True)
                    except Exception as fallback_error:
                        print(f"[TRAIN.compute_loss] ERROR | rank={rank} Fallback creation failed: {fallback_error}", flush=True)
                        # Last resort: use loss_ce even if NaN (will be handled by training_step)
                        # But ensure it has requires_grad if it's a tensor
                        if loss_ce is not None and isinstance(loss_ce, torch.Tensor):
                            loss = loss_ce
                        else:
                            # Create a basic tensor with requires_grad
                            loss = torch.tensor(0.0, device='cuda', dtype=torch.float32, requires_grad=True)
                        loss_total_val = 0.0
        except Exception as e:
            rank = _rank()
            print(f"[TRAIN.compute_loss] ERROR | rank={rank} step={self.state.global_step} | "
                  f"Error extracting loss values: {e}, using fallback", flush=True)
            loss_total_val = 0.0
            loss_ce_val = 0.0
            loss_lvr_val = 0.0
            loss_lvr_resampler_val = 0.0
            loss_ortho_val = 0.0
            loss_attn_div_val = 0.0
            loss_attn_guidance_val = 0.0
            loss_attn_transfer_val = 0.0
            loss_mode_switch_val = 0.0
            # Try to use loss_ce if available
            try:
                if loss_ce is not None and not torch.isnan(loss_ce) and not torch.isinf(loss_ce):
                    loss = loss_ce
                    loss_total_val = loss_ce_val
                else:
                    # Create a valid loss tensor with requires_grad connected to model
                    model_param = next(model_ref.parameters())
                    loss = torch.tensor(0.0, device=model_param.device, dtype=model_param.dtype) * model_param.sum() * 0.0
            except:
                # Ultimate fallback: use loss_ce or create basic tensor
                loss = loss_ce if loss_ce is not None else torch.tensor(0.0, device='cuda', dtype=torch.float32, requires_grad=True)

        self.log({
            "loss_total": loss_total_val,
            "loss_ce": loss_ce_val,
            "loss_lvr": loss_lvr_val,
            "loss_lvr_resampler": loss_lvr_resampler_val,
            "loss_ortho": loss_ortho_val,
            "loss_attn_div": loss_attn_div_val,
            "loss_attn_guidance": loss_attn_guidance_val,
            "loss_attn_transfer": loss_attn_transfer_val,
            "loss_mode_switch": loss_mode_switch_val,
        })

        return (loss, outputs) if return_outputs else loss
    
    def _log_debug_info(self, inputs, loss_type):
        """Log debug information about the data when NaN/Inf is detected"""
        try:
            rank = dist.get_rank() if dist.is_initialized() else int(os.environ.get("RANK", 0))
            if '_debug_question' in inputs:
                questions = inputs['_debug_question']
                if isinstance(questions, list) and len(questions) > 0:
                    print(f"[DEBUG] rank={rank} {loss_type} NaN detected - Question: {questions[0][:200]}", flush=True)
            if '_debug_answer' in inputs:
                answers = inputs['_debug_answer']
                if isinstance(answers, list) and len(answers) > 0:
                    print(f"[DEBUG] rank={rank} {loss_type} NaN detected - Answer: {answers[0][:200]}", flush=True)
            if '_debug_image_paths' in inputs:
                image_paths = inputs['_debug_image_paths']
                if isinstance(image_paths, list) and len(image_paths) > 0:
                    first_path = image_paths[0]
                    if isinstance(first_path, list):
                        paths_str = ', '.join(str(p)[:100] for p in first_path)
                    else:
                        paths_str = str(first_path)[:100]
                    print(f"[DEBUG] rank={rank} {loss_type} NaN detected - Image paths: {paths_str}", flush=True)
            if '_debug_bboxes' in inputs:
                bboxes = inputs['_debug_bboxes']
                if isinstance(bboxes, list) and len(bboxes) > 0:
                    bboxes_str = str(bboxes[0])[:200]
                    print(f"[DEBUG] rank={rank} {loss_type} NaN detected - Bboxes: {bboxes_str}", flush=True)
            if '_debug_data_idx' in inputs:
                data_indices = inputs['_debug_data_idx']
                if isinstance(data_indices, list) and len(data_indices) > 0:
                    print(f"[DEBUG] rank={rank} {loss_type} NaN detected - Data index: {data_indices[0]}", flush=True)
        except Exception as e:
            rank = dist.get_rank() if dist.is_initialized() else int(os.environ.get("RANK", 0))
            print(f"[DEBUG] rank={rank} Error logging debug info: {e}", flush=True)
    
    def _log_detailed_debug_info(self, model_inputs, loss_type):
        """Log detailed information about model inputs when NaN/Inf is detected"""
        try:
            rank = dist.get_rank() if dist.is_initialized() else int(os.environ.get("RANK", 0))
            # Check input shapes
            if 'input_ids' in model_inputs:
                input_ids = model_inputs['input_ids']
                print(f"[DEBUG] rank={rank} {loss_type} - input_ids shape: {input_ids.shape}, dtype: {input_ids.dtype}", flush=True)
                # Check for lvr token in input_ids
                from src.constants import LVR_TOKEN
                # Try to get lvr_token_id from model config if available
                lvr_token_id = None
                try:
                    model_to_check = getattr(self, '_current_model', None) or getattr(self, 'model', None)
                    if model_to_check is not None and hasattr(model_to_check, 'config') and hasattr(model_to_check.config, 'lvr_id'):
                        lvr_token_id = model_to_check.config.lvr_id
                except:
                    pass
                if lvr_token_id is not None:
                    lvr_positions = (input_ids == lvr_token_id).nonzero(as_tuple=True)
                    if len(lvr_positions[0]) > 0:
                        print(f"[DEBUG] rank={rank} {loss_type} - Found {len(lvr_positions[0])} LVR tokens in input_ids at positions: {lvr_positions[1][:10].tolist()}", flush=True)
                    else:
                        print(f"[DEBUG] rank={rank} {loss_type} - WARNING: No LVR tokens found in input_ids!", flush=True)
            
            if 'labels' in model_inputs:
                labels = model_inputs['labels']
                print(f"[DEBUG] rank={rank} {loss_type} - labels shape: {labels.shape}, dtype: {labels.dtype}", flush=True)
                # Check for valid labels (not IGNORE_INDEX)
                valid_labels = labels[labels != IGNORE_INDEX]
                if len(valid_labels) > 0:
                    print(f"[DEBUG] rank={rank} {loss_type} - Valid labels count: {len(valid_labels)}, min: {valid_labels.min().item()}, max: {valid_labels.max().item()}", flush=True)
                    # Check for NaN/Inf in valid labels
                    if torch.isnan(valid_labels.float()).any() or torch.isinf(valid_labels.float()).any():
                        print(f"[DEBUG] rank={rank} {loss_type} - WARNING: NaN/Inf found in valid labels!", flush=True)
            
            # Check lvr_tokens
            if 'lvr_tokens' in model_inputs:
                lvr_tokens = model_inputs['lvr_tokens']
                print(f"[DEBUG] rank={rank} {loss_type} - lvr_tokens type: {type(lvr_tokens)}, length: {len(lvr_tokens) if isinstance(lvr_tokens, list) else 'N/A'}", flush=True)
                if isinstance(lvr_tokens, list) and len(lvr_tokens) > 0:
                    for i, lvr_token_group in enumerate(lvr_tokens):
                        if isinstance(lvr_token_group, torch.Tensor):
                            is_empty = lvr_token_group.numel() == 0
                            print(f"[DEBUG] rank={rank} {loss_type} - lvr_tokens[{i}] shape: {lvr_token_group.shape}, dtype: {lvr_token_group.dtype}, "
                                  f"numel: {lvr_token_group.numel()}, empty: {is_empty}", flush=True)
                            if is_empty:
                                print(f"[DEBUG] rank={rank} {loss_type} - ❌ CRITICAL: Empty lvr_tokens[{i}]! This will cause NaN loss!", flush=True)
                            else:
                                print(f"[DEBUG] rank={rank} {loss_type} - lvr_tokens[{i}] min: {lvr_token_group.min().item()}, max: {lvr_token_group.max().item()}, "
                                      f"values: {lvr_token_group.tolist()[:20]}", flush=True)
                        elif isinstance(lvr_token_group, list):
                            print(f"[DEBUG] rank={rank} {loss_type} - lvr_tokens[{i}] type: list, length: {len(lvr_token_group)}, "
                                  f"empty: {len(lvr_token_group) == 0}, values: {lvr_token_group[:20]}", flush=True)
                            if len(lvr_token_group) == 0:
                                print(f"[DEBUG] rank={rank} {loss_type} - ❌ CRITICAL: Empty lvr_tokens[{i}]! This will cause NaN loss!", flush=True)
                        else:
                            print(f"[DEBUG] rank={rank} {loss_type} - lvr_tokens[{i}] type: {type(lvr_token_group)}, value: {lvr_token_group}", flush=True)
                else:
                    print(f"[DEBUG] rank={rank} {loss_type} - WARNING: lvr_tokens is empty or not a list!", flush=True)
            
            # Check pixel_values
            if 'pixel_values' in model_inputs:
                pixel_values = model_inputs['pixel_values']
                print(f"[DEBUG] rank={rank} {loss_type} - pixel_values shape: {pixel_values.shape}, dtype: {pixel_values.dtype}", flush=True)
                if torch.isnan(pixel_values).any() or torch.isinf(pixel_values).any():
                    nan_count = torch.isnan(pixel_values).sum().item()
                    inf_count = torch.isinf(pixel_values).sum().item()
                    print(f"[DEBUG] rank={rank} {loss_type} - WARNING: NaN/Inf in pixel_values: NaN={nan_count}, Inf={inf_count}", flush=True)
            
            # Check image_grid_thw
            if 'image_grid_thw' in model_inputs:
                image_grid_thw = model_inputs['image_grid_thw']
                print(f"[DEBUG] rank={rank} {loss_type} - image_grid_thw shape: {image_grid_thw.shape}, dtype: {image_grid_thw.dtype}", flush=True)
                print(f"[DEBUG] rank={rank} {loss_type} - image_grid_thw values: {image_grid_thw[:3] if len(image_grid_thw) >= 3 else image_grid_thw}", flush=True)
        except Exception as e:
            rank = dist.get_rank() if dist.is_initialized() else int(os.environ.get("RANK", 0))
            print(f"[DEBUG] rank={rank} Error logging detailed debug info: {e}", flush=True)
            import traceback
            print(f"[DEBUG] Traceback: {traceback.format_exc()}", flush=True)
    
    def training_step(self, model, inputs, num_items_in_batch=None):
        """
        Override training_step to add gradient logging for debugging NCCL timeout issues.
        """
        import torch
        import time
        
        step_start_time = time.time()
        step = self.state.global_step
        try:
            _rank = dist.get_rank() if dist.is_initialized() else int(os.environ.get("RANK", 0))
        except Exception:
            _rank = int(os.environ.get("RANK", 0))
        
        # Call parent training_step which handles forward, backward, and optimizer step
        try:
            loss = super().training_step(model, inputs, num_items_in_batch)
            
            # Check for NaN loss after training step
            if isinstance(loss, torch.Tensor):
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"[TRAIN.training_step] WARNING | rank={_rank} step={step} | "
                          f"Loss is NaN/Inf after training_step: {loss.item()}, will continue with next step", flush=True)
                    # Use nan_to_num to preserve computation graph if possible, otherwise create connected zero loss
                    try:
                        loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)
                    except:
                        # If nan_to_num fails, create a zero loss connected to model parameters
                        model_param = next(model.parameters())
                        loss = torch.tensor(0.0, device=model_param.device, dtype=model_param.dtype) * model_param.sum() * 0.0
            elif isinstance(loss, (float, int)):
                if not (isinstance(loss, (int, float)) and (loss == loss)):  # Check for NaN
                    print(f"[TRAIN.training_step] WARNING | rank={_rank} step={step} | "
                          f"Loss is NaN after training_step: {loss}, will continue with next step", flush=True)
                    loss = 0.0
        except RuntimeError as e:
            error_str = str(e).lower()
            rank = _rank
            if "nan" in error_str or "inf" in error_str or "nccl" in error_str:
                print(f"[TRAIN.training_step] WARNING | rank={_rank} step={step} | "
                      f"RuntimeError with NaN/Inf/NCCL: {e}, will continue with next step", flush=True)
                # Create a zero loss connected to model to ensure proper gradient flow
                try:
                    model_param = next(model.parameters())
                    loss = torch.tensor(0.0, device=model_param.device, dtype=model_param.dtype) * model_param.sum() * 0.0
                except:
                    loss = torch.tensor(0.0, device='cuda' if torch.cuda.is_available() else 'cpu', dtype=torch.float32)
            else:
                raise
        
        step_elapsed = time.time() - step_start_time
        
        # Check for NaN/Inf gradients after backward pass
        # Note: In DeepSpeed ZeRO-3, gradients are partitioned, so we check what's available
        if hasattr(model, 'module'):
            model_to_check = model.module
        else:
            model_to_check = model
        
        # Check all parameters for NaN/Inf gradients
        nan_grad_found = False
        if hasattr(model_to_check, 'lvr_head') and model_to_check.lvr_head is not None:
            lvr_params = list(model_to_check.lvr_head.parameters())
            for param in lvr_params:
                if param.grad is not None:
                    # Check for NaN/Inf gradients
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        self._log_debug_info(inputs, "gradient")
                        print(f"[TRAIN] WARNING | rank={_rank} step={step} | NaN/Inf detected in LVR head gradients (param shape: {param.shape}), zeroing out gradient", flush=True)
                        nan_grad_found = True
                        # Zero out the gradient to prevent propagation
                        param.grad.zero_()
        
        # If NaN gradient found, we've already zeroed it out, so we can continue
        # The loss is already computed and optimizer step has been taken, so we just log and continue
        if nan_grad_found:
            print(f"[TRAIN] INFO | rank={_rank} step={step} | NaN gradients detected and zeroed, continuing to next step", flush=True)
        
        # Log gradient statistics (if no NaN)
        grad_stats = {}
        if hasattr(model_to_check, 'lvr_head') and model_to_check.lvr_head is not None:
            lvr_params = list(model_to_check.lvr_head.parameters())
            grad_norms = []
            
            for param in lvr_params:
                if param.grad is not None:
                    try:
                        grad_norm = param.grad.norm().item() if param.grad.numel() > 0 else 0.0
                        if not (torch.isnan(torch.tensor(grad_norm)) or torch.isinf(torch.tensor(grad_norm))):
                            grad_norms.append(grad_norm)
                    except:
                        pass
            
            if grad_norms:
                grad_stats = {
                    'lvr_grad_norm_mean': sum(grad_norms) / len(grad_norms),
                    'lvr_grad_norm_max': max(grad_norms),
                }
        
        # Log gradient statistics (main focus)
        if grad_stats:
            print(f"[TRAIN] rank={_rank} step={step} | loss={loss.item() if isinstance(loss, torch.Tensor) else loss:.6f} | "
                  f"grad_norm_mean={grad_stats.get('lvr_grad_norm_mean', 0):.6f} | "
                  f"grad_norm_max={grad_stats.get('lvr_grad_norm_max', 0):.6f}", flush=True)
        
        return loss