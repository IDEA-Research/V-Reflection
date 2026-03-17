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

from src.train.monkey_patch_patch_emb import replace_qwen_2_5_vl_patch_emb
from src.train.monkey_patch_dataloader import replace_train_dataloader

local_rank = None

# Uncomment for debugging: sys.excepthook prints full traceback on unhandled exceptions
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

    freeze_vision = training_args.freeze_vision_tower
    freeze_merger = training_args.freeze_merger
    vision_model_params = model.visual.parameters()
    set_requires_grad(vision_model_params, not freeze_vision)
    merger_params = model.visual.merger.parameters()
    set_requires_grad(merger_params, not freeze_merger)

def configure_llm(model, training_args):
    freeze_llm = training_args.freeze_llm
    lm_head = model.lm_head.parameters()
    set_requires_grad(lm_head, not freeze_llm)
    llm_params = model.model.parameters()
    set_requires_grad(llm_params, not freeze_llm)

def configure_lvr_head(model, training_args):
    """Configure Box-Guided Compression and Dynamic Autoregressive Compression parameters"""
    # Configure Box-Guided Compression (trainable unless Stage 2, then frozen)
    if hasattr(model, 'box_feature_resampler') and model.box_feature_resampler is not None:
        if getattr(model.config, 'use_stage2_distillation', False):
            set_requires_grad(list(model.box_feature_resampler.parameters()), False)
            rank0_print(f"[INFO] Box-Guided Compression (BCM) frozen for Stage 2")
        else:
            resampler_params = list(model.box_feature_resampler.parameters())
            set_requires_grad(resampler_params, True)
            total_params = sum(p.numel() for p in resampler_params if p.requires_grad)
            rank0_print(f"[INFO] Box-Guided Compression parameters set to trainable (total params: {total_params})")

    # Configure Dynamic Autoregressive Compression (Stage 2 Student - always trainable for CE gradient flow)
    if hasattr(model, 'student_resampler') and model.student_resampler is not None:
        student_params = list(model.student_resampler.parameters())
        set_requires_grad(student_params, True)
        total_params = sum(p.numel() for p in student_params if p.requires_grad)
        rank0_print(f"[INFO] Dynamic Autoregressive Compression (DAC) parameters set to trainable (total params: {total_params})")

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

    local_rank = training_args.local_rank

    '''
        Monkey patching model forward function with lvr
        Configure model
    '''
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
    
    # if we are starting from a checkpoint
    if training_args.checkpoint_name:
        model_pth = training_args.checkpoint_name
    # if its starting a new training
    else:
        model_pth = model_args.model_id
    
    # get the model config
    config = AutoConfig.from_pretrained(model_pth,trust_remote_code=True)
    config.lvr_head = False
    config.latent_end_token = False
    # Box-Guided Compression: fixed num latent tokens per bbox for MSE target
    config.use_box_feature_resampler = getattr(model_args, 'use_box_feature_resampler', False)
    config.use_stage2_distillation = getattr(model_args, 'use_stage2_distillation', False)
    if config.use_stage2_distillation and not config.use_box_feature_resampler:
        config.use_box_feature_resampler = True  # Stage 2 requires Box-Guided Compression as Teacher
    config.num_latent_tokens = getattr(model_args, 'num_latent_tokens', 8)
    if config.use_box_feature_resampler:
        rank0_print(f"[INFO] Box-Guided Compression enabled with num_latent_tokens={config.num_latent_tokens}")
    if config.use_stage2_distillation:
        rank0_print(f"[INFO] Stage 2 distillation enabled: Dynamic Autoregressive Compression (Student) + frozen Box-Guided Compression (Teacher)")

    # Load model based on model type
    if "Qwen2.5" in model_args.model_id:
        # Patch the forward function (two-stage: resampler only, no lvr_head/latent_end/mode_switch)
        replace_qwen2_5_with_mixed_modality_forward_lvr(coconut=model_args.coconut,
                                                        use_box_feature_resampler=getattr(model_args, 'use_box_feature_resampler', False),
                                                        use_stage2_distillation=getattr(model_args, 'use_stage2_distillation', False))
        
        model = QwenWithLVR.from_pretrained(
            model_pth,
            config=config,
            torch_dtype=compute_dtype,
            attn_implementation="flash_attention_2" if not training_args.disable_flash_attn2 else "sdpa",
        )

        ''' Patch the patch-emb with fp32; Avoid edge-case nermical stability issue '''
        replace_qwen_2_5_vl_patch_emb()

    else:
        raise("Unsupported model type. At this moment, we only support Qwen2.5LM-based Qwen2.5VL series and InternVL3 series.")

    model.config.use_cache = False
    model_to_configure = model
    configure_llm(model_to_configure, training_args)
    configure_vision_tower(model_to_configure, training_args, compute_dtype, training_args.device)
    
    # Configure Box-Guided Compression / Dynamic Autoregressive Compression
    if getattr(model_args, 'use_box_feature_resampler', False) or getattr(model_args, 'use_stage2_distillation', False):
        configure_lvr_head(model, training_args)  # Ensure LVR head / resampler is trainable

    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}

    # configure processors and special tokens
    processor = AutoProcessor.from_pretrained(model_args.model_id,min_pixels=data_args.image_min_pixels,max_pixels=data_args.image_max_pixels)

    processor.tokenizer.add_tokens("<|lvr_start|>", special_tokens=True)
    processor.tokenizer.add_tokens("<|lvr|>", special_tokens=True)
    processor.tokenizer.add_tokens("<|lvr_latent_end|>", special_tokens=True)
    processor.tokenizer.add_tokens("<|lvr_end|>", special_tokens=True)

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
    # configure loss control flags
    model.config.use_mse_loss = training_args.use_mse_loss
    model.config.loss_ortho_lambda = training_args.loss_ortho_lambda
    model.config.loss_attn_lambda = training_args.loss_attn_lambda
    model.config.loss_attn_transfer_lambda = training_args.loss_attn_transfer_lambda
    model.config.loss_attn_div_lambda = training_args.loss_attn_div_lambda


    '''
        Data module configurations
        use data packing for faster training due to the random input lengths of LVR
    '''
    # model.config.tokenizer_model_max_length = processor.tokenizer.model_max_length
    if training_args.enable_data_packing:
        training_args.per_device_train_batch_size = 1
        fixed_num = (model_args.num_latent_tokens if getattr(model_args, 'use_box_feature_resampler', False) else getattr(data_args, 'fixed_num_of_lvr_tokens', None))
        if model_args.max_lvr_tokens is not None:
            fixed_num = model_args.max_lvr_tokens
        data_module, total_data_len = make_packed_supervised_data_module_lvr(
            model_id=model_args.model_id,
            processor=processor,
            data_args=data_args,
            training_args=training_args,
            latent_end_token=False,
            fixed_num_of_lvr_tokens=fixed_num,
        )
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
    
    trainer = QwenLVRSFTTrainer(
        model=model,
        processing_class=processor,
        args=training_args,
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
        # If resume_from_checkpoint is provided via TrainingArguments (e.g. from CLI),
        # explicitly pass it to the Trainer so that optimizer/scheduler/global_step are restored.
        # Stage 2: When loading from Stage 1 checkpoint, the DeepSpeed checkpoint has different
        # architecture (no student_resampler). Do NOT resume - only model was loaded via checkpoint_name.
        resume_ckpt = training_args.resume_from_checkpoint
        if getattr(model_args, 'use_stage2_distillation', False) and resume_ckpt:
            rank0_print(f"[INFO] Stage 2 distillation: skipping resume_from_checkpoint (Stage 1 ckpt has different arch). Model loaded from checkpoint_name.")
            resume_ckpt = None
        trainer.train(resume_from_checkpoint=resume_ckpt)
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
