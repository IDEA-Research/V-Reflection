#!/bin/bash
#SBATCH --partition=cvr
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:hgx:8 # A800
#SBATCH --mem=640G
#SBATCH --qos=preemptive #specify preemptive Q0S
#SBATCH --output=/comp_robot/zhoujiazhou/projects/Active-Coconut/logs/GRPO_7b_%j.txt

source ~/miniconda3/etc/profile.d/conda.sh
conda activate train

cd /comp_robot/zhoujiazhou/projects/Active-Coconut
export PYTHONPATH="${PWD}:${PWD}/src/train:${PYTHONPATH:-}"
export LVR_TRACE=0
export LVR_DEBUG=0
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
NUM_DEVICES=$(echo "$GPU_IDS" | tr ',' '\n' | wc -l)
export PYTORCH_ALLOC_CONF=expandable_segments:True

# NCCL 优化配置
export NCCL_TIMEOUT=300  # 30 分钟超时（默认是 10 分钟）
export NCCL_IB_DISABLE=0  # 启用 InfiniBand（如果可用）
export NCCL_DEBUG=WARN  # 设置为 WARN 以减少日志输出（调试时可设为 INFO）

export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-Dynamic-Coconut-GRPO}"

# ============================================================================
# Model Configuration
# ============================================================================
MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"

# ============================================================================
# Stage-1 Checkpoint Configuration
# ============================================================================
# Stage-1 checkpoint path (required for GRPO training)
# Can be overridden via STAGE1_CHECKPOINT env var
STAGE1_STEPS="${STAGE1_STEPS:-1500}"
STAGE1_CHECKPOINT="${STAGE1_CHECKPOINT:-/comp_robot/zhoujiazhou/projects/Active-Coconut/result/intrinsic-similarity/SFT_steps2500_b1_mseLVR0.1_acc8_MSETrue_TripletFalse/checkpoint-${STAGE1_STEPS}/}"

# ============================================================================
# Data Configuration
# ============================================================================
# Data path: Can use either SFT format (meta_data with <image> tokens) or GRPO format
# - SFT format: meta_data file pointing to multiple data files (e.g., meta_data_lvr_sft_stage1.json)
# - GRPO format: Direct JSON array without <image> tokens (e.g., virl39k.json)
# Note: GRPO dataset will automatically remove <image> tokens from SFT format data
DATA_PATH="${DATA_PATH:-data/meta_data_lvr_sft_stage1.json}"
IMAGE_FOLDER="${IMAGE_FOLDER:-/comp_robot/zhoujiazhou/Datasets/Visual_cot/images}"

# ============================================================================
# GRPO Training Parameters
# ============================================================================
# Generation parameters (critical for GRPO)
NUM_GENERATIONS="${NUM_GENERATIONS:-8}"  # Number of completions per prompt (at least 2)
TEMPERATURE="${TEMPERATURE:-0.9}"  # Generation temperature (LVR is sensitive to this)
TOP_P="${TOP_P:-1.0}"  # Top-p sampling (default: 1.0)
TOP_K="${TOP_K:-50}"  # Top-k sampling (default: 50)
MIN_P="${MIN_P:-}"  # Min-p sampling (optional, default: None)
REPETITION_PENALTY="${REPETITION_PENALTY:-1.0}"  # Repetition penalty (default: 1.0)
DECODING_STRATEGY="${DECODING_STRATEGY:-steps}"  # Decoding strategy: "steps" or "max"
LVR_STEPS="${LVR_STEPS:-8}"  # Number of LVR steps when using "steps" strategy

# Completion and prompt length limits
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-512}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"

# Training parameters
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-2}"
BATCH_PER_DEVICE="${BATCH_PER_DEVICE:-1}"  # Smaller batch size for GRPO (generates multiple completions)
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-2}"
LR="${LR:-5e-6}"  # Lower learning rate for RL training

# KL divergence coefficient (0.0 means no ref model, saves memory but may be unstable)
BETA="${BETA:-0.04}"

# Image token limits
MAX_TOKEN="${MAX_TOKEN:-2560}"
MIN_TOKEN="${MIN_TOKEN:-128}"

# Model freezing (typically freeze vision and merger for GRPO)
FREEZE_VISION="${FREEZE_VISION:-True}"
FREEZE_MERGER="${FREEZE_MERGER:-True}"
FREEZE_LLM="${FREEZE_LLM:-False}"

# Online checkpointing (set to False if not using cloud storage)
ONLINE_CHECKPOINT="${ONLINE_CHECKPOINT:-False}"

# ============================================================================
# Build Run Name and Output Directory
# ============================================================================
RUN_NAME="GRPO_7B_steps${STAGE1_STEPS}_decoding${DECODING_STRATEGY}_lvrSteps${LVR_STEPS}_gen${NUM_GENERATIONS}_LR${LR}_TEMP${TEMPERATURE}_beta${BETA}"
OUTPUT_DIR="result/grpo/${RUN_NAME}/"

mkdir -p logs "$OUTPUT_DIR"
export WANDB_RUN_NAME="$RUN_NAME"
MASTER_PORT="${MASTER_PORT:-29500}"

# ============================================================================
# Print Configuration Summary
# ============================================================================
echo "=========================================="
echo "GRPO Training Configuration:"
echo "  Model: $MODEL_NAME"
echo "  Stage-1 Checkpoint: $STAGE1_CHECKPOINT"
echo "  Data Path: $DATA_PATH"
echo "  Image Folder: $IMAGE_FOLDER"
echo "  GPUs: ${GPU_IDS:-all available} ($NUM_DEVICES devices)"
echo "  Batch per device: $BATCH_PER_DEVICE"
echo "  Gradient accumulation steps: $GRAD_ACCUM_STEPS"
echo "  Effective batch size: $((BATCH_PER_DEVICE * NUM_DEVICES * GRAD_ACCUM_STEPS))"
echo "  Number of generations per prompt: $NUM_GENERATIONS"
echo "  Temperature: $TEMPERATURE"
echo "  Top-p: $TOP_P"
echo "  Top-k: $TOP_K"
echo "  Min-p: ${MIN_P:-None}"
echo "  Repetition penalty: $REPETITION_PENALTY"
echo "  Decoding strategy: $DECODING_STRATEGY"
echo "  LVR steps: $LVR_STEPS"
echo "  Max completion length: $MAX_COMPLETION_LENGTH"
echo "  Max prompt length: $MAX_PROMPT_LENGTH"
echo "  Learning rate: $LR"
echo "  Beta (KL coefficient): $BETA"
echo "  Freeze vision: $FREEZE_VISION"
echo "  Freeze merger: $FREEZE_MERGER"
echo "  Freeze LLM: $FREEZE_LLM"
echo "  Wandb Project: $WANDB_PROJECT"
echo "  Wandb Mode: $WANDB_MODE"
echo "  Output Dir: $OUTPUT_DIR"
echo "=========================================="

# ============================================================================
# Build DeepSpeed Command
# ============================================================================
DEEPSPEED_CMD="deepspeed --master_port=$MASTER_PORT src/train/train_grpo.py \
    --run_name \"$RUN_NAME\" \
    --deepspeed scripts/zero2.json \
    --checkpoint_name \"$STAGE1_CHECKPOINT\" \
    --model_id $MODEL_NAME \
    --data_path \"$DATA_PATH\" \
    --image_folder \"$IMAGE_FOLDER\" \
    --freeze_vision_tower $FREEZE_VISION \
    --freeze_merger $FREEZE_MERGER \
    --freeze_llm $FREEZE_LLM \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 False \
    --output_dir \"$OUTPUT_DIR\" \
    --temperature $TEMPERATURE \
    --top_p $TOP_P \
    --top_k $TOP_K \
    --repetition_penalty $REPETITION_PENALTY \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --num_generations $NUM_GENERATIONS \
    --per_device_train_batch_size $BATCH_PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --max_completion_length $MAX_COMPLETION_LENGTH \
    --max_prompt_length $MAX_PROMPT_LENGTH \
    --image_min_pixels $((MIN_TOKEN * 28 * 28)) \
    --image_max_pixels $((MAX_TOKEN * 28 * 28)) \
    --learning_rate $LR \
    --beta $BETA \
    --remove_unused_columns False \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type \"cosine\" \
    --logging_steps 1 \
    --tf32 False \
    --gradient_checkpointing True \
    --report_to wandb \
    --lazy_preprocess True \
    --save_strategy \"steps\" \
    --save_steps 100 \
    --save_total_limit 50 \
    --dataloader_num_workers 8 \
    --decoding_strategy $DECODING_STRATEGY \
    --lvr_steps $LVR_STEPS"

# Add min_p parameter if set
if [ -n "$MIN_P" ]; then
    DEEPSPEED_CMD="$DEEPSPEED_CMD --min_p $MIN_P"
fi

# Add online_checkpoint parameter if enabled
if [ "$ONLINE_CHECKPOINT" = "True" ]; then
    DEEPSPEED_CMD="$DEEPSPEED_CMD --online_checkpoint True"
fi

# ============================================================================
# Execute Training
# ============================================================================
eval $DEEPSPEED_CMD

if [ $? -ne 0 ]; then
    echo "Error: GRPO training failed"
    exit 1
fi

echo "GRPO training completed successfully!"
echo "Checkpoint saved to: $OUTPUT_DIR"

# ============================================================================
# Post-Training Evaluation
# ============================================================================
echo ""
echo "=========================================="
echo "Starting post-training evaluation..."
echo "=========================================="

BASE_CHECKPOINT_DIR="${PWD}/${OUTPUT_DIR}"

# Check if checkpoint directory exists
if [ -d "$BASE_CHECKPOINT_DIR" ]; then
    # Check if there are any checkpoints
    CHECKPOINT_COUNT=$(find "$BASE_CHECKPOINT_DIR" -type d -name "checkpoint-*" | wc -l)
    if [ "$CHECKPOINT_COUNT" -gt 0 ]; then
        echo "Found $CHECKPOINT_COUNT checkpoint(s) in $BASE_CHECKPOINT_DIR"
        echo "Starting batch evaluation for all checkpoints..."
        
        # Export environment variables for evaluation script
        export BASE_CHECKPOINT_DIR
        export DATASET_CONFIG="${DATASET_CONFIG:-default}"
        export EVAL_STEP_LIST="${EVAL_STEP_LIST:-4}"
        export LVR_SAVE_ACTIVATION_MAPS="${LVR_SAVE_ACTIVATION_MAPS:-0}"
        
        # Call the batch evaluation script (same as SFT)
        # Note: GRPO checkpoints should be compatible with SFT evaluation script
        bash "${PWD}/scripts/evaluation/evaluation_7b_SFT_all_ck.sh" || {
            echo "Warning: Batch evaluation failed"
            # Don't exit with error code - training succeeded, evaluation is optional
            exit 0
        }
        
        echo ""
        echo "=========================================="
        echo "Evaluation completed successfully!"
        echo "Results saved in: evaluation/results/"
        echo "=========================================="
    else
        echo "Warning: No checkpoint directories found in $BASE_CHECKPOINT_DIR. Skipping evaluation."
    fi
else
    echo "Warning: Checkpoint directory not found: $BASE_CHECKPOINT_DIR. Skipping evaluation."
fi

echo ""
echo "GRPO training and evaluation pipeline completed!"
