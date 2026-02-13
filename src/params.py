from dataclasses import dataclass, field
from typing import Optional

from transformers import TrainingArguments as HFTrainingArguments
# from trl import DPOConfig as DPOConfigTRL
from trl import GRPOConfig as GRPOConfigTRL


@dataclass
class ModelArguments:
    model_id: Optional[str] = field(default="Qwen/Qwen2-VL-7B-Instruct")
    # set continuous reasoning mode
    coconut: bool = field(default=True)
    lvr_head: bool = field(default=False)
    lvr_head_type: str = field(default="simple")
    mlp_ratio: float = field(default=1.0, metadata={"help": "MLP ratio for attention-based LVR head. Controls projection dimension: 1.0=full, 0.5=half, 0.25=quarter"})
    # IVR (Implicit Visual Routing) parameters
    ivr_iterations: int = field(default=3, metadata={"help": "Number of routing iterations for IVR head"})
    ivr_chunk_size: Optional[int] = field(default=None, metadata={"help": "Chunk size for IVR head (None means auto-select based on sequence length)"})
    ivr_use_output_norm: bool = field(default=True, metadata={"help": "Whether to use output normalization for IVR head"})
    ivr_temperature: float = field(default=1.0, metadata={"help": "Temperature parameter for IVR routing"})
    # GFR (Gated Feature Reweighting) parameters
    gfr_visual_dim: Optional[int] = field(default=None, metadata={"help": "Visual dimension for GFR head (None means use hidden_size)"})
    gfr_chunk_size: Optional[int] = field(default=None, metadata={"help": "Chunk size for GFR head (None means auto-select, default 512)"})
    gfr_use_output_norm: bool = field(default=True, metadata={"help": "Whether to use output normalization for GFR head"})
    latent_end_token: bool = field(default=False)
    max_lvr_tokens: int = field(default=None)
    use_box_feature_resampler: bool = field(default=False, metadata={"help": "Use BoxFeatureResampler for fixed 8 latent tokens MSE target"})
    num_latent_tokens: int = field(default=8, metadata={"help": "Number of fixed latent tokens per bbox for resampler loss"})
    use_dit_reconstruction: bool = field(default=False, metadata={"help": "Use DiT-XL-2 pixel reconstruction head conditioned on LLM 8 tokens"})
    dit_pretrained_path: Optional[str] = field(default=None, metadata={"help": "Path to pretrained DiT-XL-2 weights (.pt file, e.g. DiT-XL-2-256x256.pt)"})
    dit_vae_repo: str = field(default="stabilityai/sd-vae-ft-mse", metadata={"help": "VAE repo for DiT reconstruction"})
    dit_hidden_size: int = field(default=1152, metadata={"help": "DiT hidden size (DiT-XL-2 uses 1152)"})
    dit_num_latent_tokens: int = field(default=8, metadata={"help": "Number of LLM tokens used as DiT condition per bbox"})
    # Attention isolation parameters


@dataclass
class TrainingArguments(HFTrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    adam_beta1: float = field(default=0.9)
    adam_beta2: float = field(default=0.999)
    adam_epsilon: float = field(default=1e-8)

    loss_lvr_fct: str = field(default="mse")
    loss_lvr_lambda: float = field(default=1e-1)
    loss_lvr_resampler_lambda: float = field(default=1e-1, metadata={"help": "Weight for MSE loss between LLM latent tokens and BoxFeatureResampler target"})
    loss_dit_recon_lambda: float = field(default=0.1, metadata={"help": "Weight for DiT pixel reconstruction loss"})
    dit_num_inference_steps: int = field(default=20, metadata={"help": "Number of denoising steps for DiT at inference"})
    dit_condition_gt_prob: float = field(default=0.5, metadata={"help": "Probability of using Resampler GT tokens (instead of LLM tokens) as DiT condition during training. 0.0=always LLM, 1.0=always GT, 0.5=50/50"})

    # Loss control flags - enable/disable specific losses
    use_mse_loss: bool = field(default=True, metadata={"help": "Whether to compute and use MSE/LVR reconstruction loss"})

    freeze_vision_tower: bool = field(default=False)
    freeze_llm: bool = field(default=False)
    freeze_merger: bool = field(default=False)
    disable_flash_attn2: bool = field(default=False)

    max_seq_length: int = field(
        default=32768, # This is the default value of the qwen2-vl model
        metadata={
            "help":
                "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )

    lora_enable: bool = False
    vision_lora: bool = False
    use_dora: bool = False
    lora_rank: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    vision_lr: Optional[float] = None
    merger_lr: Optional[float] = None
    lvr_head_lr: Optional[float] = None
    lora_namespan_exclude: str = field(default=None, metadata={"help": "List of namespan to exclude for LoRA"})
    num_lora_modules: int = -1
    # use_liger: bool = True
    run_name: Optional[str] = field(default="vscode debugger", metadata={"help": "Name of the run for logging purposes."})
    # True if serving the checkpoints and data on oci
    online_checkpoint: Optional[bool] = False
    checkpoint_name:Optional[str] = None
    # data packing-related params
    enable_data_packing: bool = False
    max_packed_tokens:Optional[int] = None
    # long_seq_cut:Optional[int] = field(default=25600, metadata={"help": "Max Len of long single data instnace allowed"})
    long_seq_threshold:Optional[int] = field(default=4096, metadata={"help": "Threshold to be a long single instance"})
    max_instance_per_batch:Optional[int] = 4
    max_steps:Optional[int] = 2500

    mode_switch_loss: Optional[bool] = False
    loss_mode_switch_fct: Optional[str] = field(default="mse")
    loss_mode_switch_lambda:Optional[float] = field(default=1e-1)
    


@dataclass
class GRPOArguments(GRPOConfigTRL):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    adam_beta1: float = field(default=0.9)
    adam_beta2: float = field(default=0.999)
    adam_epsilon: float = field(default=1e-8)

    freeze_vision_tower: bool = field(default=False)
    freeze_llm: bool = field(default=False)
    freeze_merger: bool = field(default=False)
    disable_flash_attn2: bool = field(default=False)
    lora_enable: bool = False
    vision_lora: bool = False
    use_dora: bool = False
    lora_rank: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    vision_lr: Optional[float] = None
    merger_lr: Optional[float] = None
    lora_namespan_exclude: str = field(default=None, metadata={"help": "List of namespan to exclude for LoRA"})
    num_lora_modules: int = -1
    beta: float = field(
        default=0.04,
        metadata={
            "help": "KL coefficient. If `0.0`, the reference model is not loaded, reducing memory usage and improving "
            "training speed, but may be numerically unstable for long training runs."
        },
    )
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 50
    min_p: Optional[float] = None
    repetition_penalty: float = 1.0
    max_completion_length: int = 256
    max_prompt_length: int = 512

    online_checkpoint: Optional[bool] = False
    checkpoint_name:Optional[str] = None
    decoding_strategy:str = "steps"
    lvr_steps: int = 16



@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    lazy_preprocess: bool = False
    image_folder: Optional[str] = field(default=None)
    image_min_pixels: Optional[int] = field(default=3136)
    image_max_pixels: Optional[int] = field(default=12845056)
    video_min_pixels: Optional[int] = field(default=100352)
    video_max_pixels: Optional[int] = field(default=602112)
    image_resized_width: int = field(default=None)
    image_resized_height: int = field(default=None)
    video_resized_width: int = field(default=None)
    video_resized_height: int = field(default=None)
    fps: float = 1.0
    random_seed: Optional[int] = field(default=None)
    fixed_num_of_lvr_tokens: Optional[int] = field(default=None, metadata={"help": "Fixed number of <lvr> tokens per bbox (e.g. 8 for BoxFeatureResampler). If set, template uses this many tokens + optional latent_end."})
    dit_crop_size: int = field(default=128, metadata={"help": "Bbox crop size for DiT reconstruction (128->16x16 latent, 256->32x32 latent). Smaller = faster training + less memory."})
