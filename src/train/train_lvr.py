import sys
import os
import traceback
import signal

import os
import torch
from transformers import AutoProcessor, AutoConfig, HfArgumentParser
from transformers import AutoTokenizer, AutoModel

from src.model.qwen_lvr_model import QwenWithLVR
from src.trainer import QwenLVRSFTTrainer
from src.dataset import make_supervised_data_module_lvr, make_packed_supervised_data_module_lvr
from src.params import DataArguments, ModelArguments, TrainingArguments

from src.train.train_utils import safe_save_model_for_hf_trainer
from monkey_patch_forward_lvr import replace_qwen2_5_with_mixed_modality_forward_lvr

from src.s3_checkpoints_lvr import OCIFolderCheckpointHandler, create_temp_dir
from src.train.monkey_patch_patch_emb import replace_qwen_2_5_vl_patch_emb
from src.train.monkey_patch_dataloader import replace_train_dataloader

local_rank = None

# For debugging only Plese comment this during training
# torch.autograd.set_detect_anomaly(True)

def exception_handler(exc_type, exc_value, exc_traceback):
    """Global exception handler to catch all unhandled exceptions"""
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    
    rank0_print("\n" + "="*80)
    rank0_print("[FATAL ERROR] Unhandled exception occurred!")
    rank0_print(f"[FATAL ERROR] Exception type: {exc_type.__name__}")
    rank0_print(f"[FATAL ERROR] Exception message: {str(exc_value)}")
    rank0_print("[FATAL ERROR] Full traceback:")
    rank0_print("="*80)
    traceback.print_exception(exc_type, exc_value, exc_traceback)
    rank0_print("="*80 + "\n")

# Set global exception handler
sys.excepthook = exception_handler

def rank0_print(*args):
    if local_rank == 0 or local_rank == '0' or local_rank is None:
        print(*args)

def set_requires_grad(parameters, requires_grad):
    """Set requires_grad for parameters, skipping non-float types"""
    for p in parameters:
        if p.dtype.is_floating_point or p.dtype.is_complex:
            p.requires_grad = requires_grad

def configure_vision_tower(model, training_args, compute_dtype, device):
    vision_tower = model.visual
    vision_tower.to(dtype=compute_dtype, device=device)

    vision_model_params = model.visual.parameters()
    set_requires_grad(vision_model_params, not training_args.freeze_vision_tower)
    
    # Handle merger specifically
    merger_params = model.visual.merger.parameters()
    set_requires_grad(merger_params, not training_args.freeze_merger)

def configure_llm(model, training_args):
    lm_head = model.lm_head.parameters()
    set_requires_grad(lm_head, not training_args.freeze_llm)

    llm_params = model.model.parameters()
    set_requires_grad(llm_params, not training_args.freeze_llm)

def configure_lvr_head(model, training_args):
    """Configure LVR head parameters - always trainable (not frozen)"""
    if hasattr(model, 'lvr_head') and model.lvr_head is not None:
        lvr_head_params = list(model.lvr_head.parameters())
        set_requires_grad(lvr_head_params, True)  # LVR head is always trainable
        total_params = sum(p.numel() for p in lvr_head_params if p.requires_grad)
        rank0_print(f"[INFO] LVR head parameters set to trainable (total params: {total_params})")
    
    if hasattr(model, 'lvr_latent_end_emb') and model.lvr_latent_end_emb is not None:
        model.lvr_latent_end_emb.requires_grad = True
        rank0_print(f"[INFO] LVR latent end token set to trainable")


def train():
    global local_rank

    # Set NCCL timeout to detect communication issues quickly
    # Increased to 1800s (30min) to handle large gradient synchronization operations
    # NaN loss can cause very slow gradient synchronization, leading to timeout
    if 'NCCL_TIMEOUT' not in os.environ:
        os.environ['NCCL_TIMEOUT'] = '1800'  # 30 minutes timeout - handle NaN-induced slowdowns
        rank0_print(f"[INFO] Setting NCCL_TIMEOUT to 1800 seconds (30 minutes) to handle NaN-induced gradient sync delays")
    
    # Disable NCCL INFO logging for cleaner output (only show WARN/ERROR)
    if 'NCCL_DEBUG' not in os.environ:
        os.environ['NCCL_DEBUG'] = 'WARN'  # Only show warnings and errors, not INFO messages
    
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    '''
        set up oci checkpointing;
        set online_checkpoint to False if you dont need
    '''
    oci_handler = None
    temp_folder = None
    if training_args.online_checkpoint:
        # oci keys
        access_key_id = os.environ.get('ACCESS_KEY_ID')
        secret_access_key = os.environ.get('SECRET_ACCESS_KEY')
        endpoint_url = os.environ.get('ENDPOINT_URL')
        bucket_name = os.environ.get('BUCKET_NAME')
        region_name = os.environ.get('REGION_NAME', 'us-east-1')  # Default region if not set

        # Validate required OCI environment variables
        missing_vars = []
        if not access_key_id:
            missing_vars.append('ACCESS_KEY_ID')
        if not secret_access_key:
            missing_vars.append('SECRET_ACCESS_KEY')
        if not endpoint_url:
            missing_vars.append('ENDPOINT_URL')
        if not bucket_name:
            missing_vars.append('BUCKET_NAME')
        
        if missing_vars:
            error_msg = f"Error: online_checkpoint is enabled but required environment variables are missing: {', '.join(missing_vars)}"
            rank0_print(error_msg)
            raise ValueError(error_msg)

        model_name = model_args.model_id.split('/')[-1]     # "Qwen2.5-VL-7B-Instruct"
        # local cache dir and tempFile class
        cache_dir = os.getenv("CACHE_DIR")  #cache dir = "/dockerx/Local/users/bangzheng"
        # If CACHE_DIR is not set, use a default temporary directory
        if cache_dir is None:
            import tempfile
            cache_dir = os.path.join(tempfile.gettempdir(), "model_cache")
            os.makedirs(cache_dir, exist_ok=True)
            rank0_print(f"Warning: CACHE_DIR not set, using default: {cache_dir}")
        # temp_file class; "/dockerx/Local/users/bangzheng/model_name/run_name-[random]"
        local_model_name_or_path = create_temp_dir(base_path=os.path.join(cache_dir,model_name),prefix=training_args.run_name + '-')     
        temp_folder = local_model_name_or_path

        # remote dir
        remote_dir = training_args.output_dir  # output_dir is remote now; "/checkpoints"
        remote_dir = os.path.join(remote_dir,model_name,training_args.run_name)    # "/checkpoints/Qwen2.5-VL-7B-Instruct/run_name"
        training_args.remote_output_dir = remote_dir
        training_args.output_dir = local_model_name_or_path.name    # output_dir should always be local

        # oci handler
        oci_handler = OCIFolderCheckpointHandler(access_key_id, secret_access_key, endpoint_url, bucket_name, region_name)
    

    local_rank = training_args.local_rank

    '''
        Monkey patching model forward function with lvr
        Configure model
    '''
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
    
    # if we are starting from a checkpoint
    if training_args.checkpoint_name:
        if training_args.online_checkpoint and oci_handler is not None:
            # CHKPT_NAME="checkpoints_lvrHead_featureAlign/Qwen2.5-VL-7B-Instruct/BS256-LAMBDA1-LVR_HEAD_LR1e-5-MAXTOKEN{7680}/checkpoint-1578/"
            local_pth_to_download_chkpt = create_temp_dir(base_path=os.path.join(cache_dir,model_name),prefix=f"warmed_{model_args.lvr_head_type}" + '-')
            oci_handler.load_checkpoint(training_args.checkpoint_name, local_pth_to_download_chkpt,inference_mode=True)
            
            model_pth = local_pth_to_download_chkpt.name
        else:
            # Use local checkpoint path when not using online checkpointing
            model_pth = training_args.checkpoint_name
    # if its starting a new training
    else:
        model_pth = model_args.model_id
    
    # get the model config
    config = AutoConfig.from_pretrained(model_pth,trust_remote_code=True)
    config.latent_end_token = model_args.latent_end_token
    config.lvr_head = model_args.lvr_head
    config.lvr_head_type = model_args.lvr_head_type
    config.mlp_ratio = getattr(model_args, 'mlp_ratio', 1.0)
    rank0_print(f"[INFO] Config mlp_ratio set to: {config.mlp_ratio}")
    
    # Set IVR parameters if using IVR type
    if model_args.lvr_head_type == 'ivr':
        config.ivr_iterations = getattr(model_args, 'ivr_iterations', 3)
        config.ivr_chunk_size = getattr(model_args, 'ivr_chunk_size', None)
        config.ivr_use_output_norm = getattr(model_args, 'ivr_use_output_norm', True)
        config.ivr_temperature = getattr(model_args, 'ivr_temperature', 1.0)
        rank0_print(f"[INFO] IVR config: iterations={config.ivr_iterations}, "
                   f"chunk_size={config.ivr_chunk_size}, use_output_norm={config.ivr_use_output_norm}, "
                   f"temperature={config.ivr_temperature}")
    
    # Set GFR parameters if using GFR type
    if model_args.lvr_head_type == 'gated-focus' or model_args.lvr_head_type == 'gfr':
        config.gfr_visual_dim = getattr(model_args, 'gfr_visual_dim', None)
        config.gfr_chunk_size = getattr(model_args, 'gfr_chunk_size', None)
        config.gfr_use_output_norm = getattr(model_args, 'gfr_use_output_norm', True)
        rank0_print(f"[INFO] GFR config: visual_dim={config.gfr_visual_dim}, "
                   f"chunk_size={config.gfr_chunk_size}, use_output_norm={config.gfr_use_output_norm}")
    
    # Load model based on model type
    if "Qwen2.5" in model_args.model_id:
        # Patch the forward function
        replace_qwen2_5_with_mixed_modality_forward_lvr(coconut=model_args.coconut,
                                                        lvr_head=model_args.lvr_head,
                                                        mode_switch_loss=training_args.mode_switch_loss,
                                                        latent_end_token=model_args.latent_end_token)
        
        model = QwenWithLVR.from_pretrained(
            model_pth,
            config=config,
            torch_dtype=compute_dtype,
            attn_implementation="flash_attention_2" if not training_args.disable_flash_attn2 else "sdpa",
        )

        # init lvr_head (if not already initialized in __init__ or needs reinitialization)
        if model_args.lvr_head:
            mlp_ratio = getattr(model_args, 'mlp_ratio', 1.0)
            # Check if LVR head already exists and has correct mlp_ratio
            if hasattr(model, 'lvr_head') and model.lvr_head is not None:
                # Verify mlp_ratio matches (for attention-mask type)
                if model_args.lvr_head_type == 'attention-mask':
                    current_mlp_ratio = getattr(model.config, 'mlp_ratio', 1.0)
                    if abs(current_mlp_ratio - mlp_ratio) < 1e-6:
                        rank0_print(f"[INFO] LVR head already initialized with mlp_ratio={mlp_ratio}, skipping reinitialization")
                    else:
                        rank0_print(f"[INFO] Reinitializing LVR head: mlp_ratio {current_mlp_ratio} -> {mlp_ratio}")
                        model._init_lvr_head(lvr_head_type=model_args.lvr_head_type, 
                                            mlp_ratio=mlp_ratio)
                else:
                    rank0_print(f"[INFO] LVR head already initialized, skipping reinitialization")
            else:
                rank0_print(f"[INFO] Initializing LVR head with mlp_ratio={mlp_ratio}")
                model._init_lvr_head(lvr_head_type=model_args.lvr_head_type, 
                                    mlp_ratio=mlp_ratio)
        
        # init latent_end_token
        if model_args.latent_end_token:
            model._init_lvr_latent_end_emb()
            model.config.loss_mode_switch_fct = training_args.loss_mode_switch_fct

        
        ''' Patch the patch-emb with fp32; Avoid edge-case nermical stability issue '''
        replace_qwen_2_5_vl_patch_emb()

    else:
        raise("Unsupported model type. At this moment, we only support Qwen2.5LM-based Qwen2.5VL series and InternVL3 series.")

    model.config.use_cache = False
    model_to_configure = model
    configure_llm(model_to_configure, training_args)
    configure_vision_tower(model_to_configure, training_args, compute_dtype, training_args.device)
    
    # Configure LVR head
    if model_args.lvr_head:
        configure_lvr_head(model, training_args)  # Ensure LVR head is trainable

    ''' NaN sanitizer: Hook the patch-emb with torch.nan_to_num() '''
    # def output_nan_sanitizer_hook(module, input, output):
    #     if isinstance(output, torch.Tensor) and torch.isnan(output).any():
    #         print(f"[Sanitizer] {module.__class__.__name__}: NaN or Inf detected.")
    #         print(f"  Output stats - min: {output.min().item()}, max: {output.max().item()}, mean: {output.mean().item()}")
    #         return torch.nan_to_num(output, nan=0.0, posinf=1e4, neginf=-1e4)
    #     return output
    # model.model.visual.patch_embed.register_forward_hook(output_nan_sanitizer_hook)

    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}

    # configure processors and special tokens
    processor = AutoProcessor.from_pretrained(model_args.model_id,min_pixels=data_args.image_min_pixels,max_pixels=data_args.image_max_pixels)

    processor.tokenizer.add_tokens("<|lvr_start|>",special_tokens=True)
    processor.tokenizer.add_tokens("<|lvr|>",special_tokens=True)
    processor.tokenizer.add_tokens("<|lvr_latent_end|>",special_tokens=True)
    processor.tokenizer.add_tokens("<|lvr_end|>",special_tokens=True)

    lvr_id = processor.tokenizer.convert_tokens_to_ids("<|lvr|>")
    lvr_latent_end_id = processor.tokenizer.convert_tokens_to_ids("<|lvr_latent_end|>")
    lvr_start_id = processor.tokenizer.convert_tokens_to_ids("<|lvr_start|>")
    lvr_end_id = processor.tokenizer.convert_tokens_to_ids("<|lvr_end|>")

    model.config.lvr_id = lvr_id
    model.config.lvr_latent_end_id = lvr_latent_end_id
    model.config.lvr_start_id = lvr_start_id
    model.config.lvr_end_id = lvr_end_id


    # there are some dummy tokens in newer hf version
    if model.config.vocab_size < len(processor.tokenizer):
        model.resize_token_embeddings(len(processor.tokenizer))

    # configure lvr loss type
    model.config.loss_lvr_fct = training_args.loss_lvr_fct


    '''
        Data module configurations
        use data packing for faster training due to the random input lengths of LVR
    '''
    # model.config.tokenizer_model_max_length = processor.tokenizer.model_max_length
    if training_args.enable_data_packing:
        training_args.per_device_train_batch_size = 1
        if model_args.max_lvr_tokens is not None:
            data_module, total_data_len = make_packed_supervised_data_module_lvr_fixedToken(model_id=model_args.model_id,
                                                                                            processor=processor,
                                                                                            max_lvr_tokens=model_args.max_lvr_tokens,
                                                                                            data_args=data_args,
                                                                                            training_args=training_args,
                                                                                            latent_end_token=model_args.latent_end_token)
        else:
            data_module, total_data_len = make_packed_supervised_data_module_lvr(model_id=model_args.model_id,
                                                                                processor=processor,
                                                                                data_args=data_args,
                                                                                training_args=training_args,
                                                                                latent_end_token=model_args.latent_end_token)
        if not training_args.max_steps:
            training_args.max_steps = total_data_len // (training_args.gradient_accumulation_steps 
                                                         * training_args.world_size
                                                         * training_args.per_device_train_batch_size)
        # Very crucial or the packed data will get incorrectly sliced by the dataloader
        replace_train_dataloader()
    else:
        data_module = make_supervised_data_module_lvr(model_id=model_args.model_id,
                                              processor=processor,
                                              data_args=data_args)
    
    # tempFolder = temp_file class; "/dockerx/Local/users/bangzheng/model_name/run_name-[random]"
    trainer = QwenLVRSFTTrainer(
        model=model,
        processing_class=processor,
        args=training_args,
        temp_folder=temp_folder,
        oci_handler=oci_handler,
        **data_module
    )

    def signal_handler(signum, frame):
        """Handle signals like SIGTERM, SIGINT"""
        rank0_print(f"\n[ERROR] Received signal {signum}")
        rank0_print(f"[ERROR] Signal frame: {frame}")
        traceback.print_stack(frame)
        sys.exit(1)
    
    # Register signal handlers
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        trainer.train()
    except KeyboardInterrupt:
        rank0_print("\n[ERROR] Training interrupted by user")
        traceback.print_exc()
        raise
    except RuntimeError as e:
        rank0_print(f"\n[ERROR] RuntimeError occurred during training:")
        rank0_print(f"[ERROR] {str(e)}")
        rank0_print(f"[ERROR] Full traceback:")
        traceback.print_exc()
        raise
    except Exception as e:
        rank0_print(f"\n[ERROR] Unexpected error occurred during training:")
        rank0_print(f"[ERROR] Error type: {type(e).__name__}")
        rank0_print(f"[ERROR] Error message: {str(e)}")
        rank0_print(f"[ERROR] Full traceback:")
        traceback.print_exc()
        raise
    finally:
        rank0_print("[INFO] Training finished, saving state...")
        try:
            trainer.save_state()
            model.config.use_cache = True
            safe_save_model_for_hf_trainer(trainer, output_dir=training_args.output_dir)
        except Exception as e:
            rank0_print(f"[ERROR] Error during cleanup: {str(e)}")
            traceback.print_exc()



if __name__ == "__main__":
    try:
        train()
    except Exception as e:
        rank0_print(f"\n[FATAL ERROR] Training failed with error:")
        rank0_print(f"[FATAL ERROR] Error type: {type(e).__name__}")
        rank0_print(f"[FATAL ERROR] Error message: {str(e)}")
        rank0_print(f"[FATAL ERROR] Full traceback:")
        traceback.print_exc()
        sys.exit(1)
