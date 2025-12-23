#!/bin/bash
#SBATCH --partition=cvr
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:hgx:8 # A800
#SBATCH --mem=640G
#SBATCH --qos=preemptive #specify preemptive Q0S
#SBATCH --output=/comp_robot/zhoujiazhou/projects/Active-Coconut/logs/lvr_stage1_3b_%j.txt

# ============================================================================
# Activate Conda Environment
# ============================================================================
source ~/miniconda3/etc/profile.d/conda.sh
conda activate train

# ============================================================================
# Set Working Directory
# ============================================================================
cd /comp_robot/zhoujiazhou/projects/Active-Coconut
echo "Working directory: $(pwd)"

# Add project root and src/train to PYTHONPATH so Python can find modules
# This allows both 'from src.xxx' and 'from train.xxx' imports to work
export PYTHONPATH="${PWD}:${PWD}/src/train:${PYTHONPATH:-}"
echo "PYTHONPATH: $PYTHONPATH"

# ============================================================================
# GPU Configuration
# ============================================================================
# Specify which GPUs to use (comma-separated, e.g., "0,1,2,3" or "0,1")
# If not set, will use all available GPUs
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"  # Default to GPU 0,1,2,3,4,5,6,7, can be overridden

# Auto-detect GPU count if GPU_IDS is set
if [ -n "$GPU_IDS" ]; then
    # Count number of GPUs specified
    NUM_DEVICES=$(echo "$GPU_IDS" | tr ',' '\n' | wc -l)
    export CUDA_VISIBLE_DEVICES="$GPU_IDS"
    USE_NUM_GPUS_FLAG=false  # Don't use --num_gpus when CUDA_VISIBLE_DEVICES is set
    echo "Using specified GPUs: $GPU_IDS (Total: $NUM_DEVICES GPUs)"
else
    # Use all available GPUs
    NUM_DEVICES=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
    if [ -z "$NUM_DEVICES" ] || [ "$NUM_DEVICES" -eq 0 ]; then
        echo "Warning: Cannot detect GPU count, defaulting to 4"
        NUM_DEVICES=4
    fi
    USE_NUM_GPUS_FLAG=true  # Use --num_gpus when using all GPUs
    echo "Using all available GPUs (Total: $NUM_DEVICES GPUs)"
fi

export PYTORCH_ALLOC_CONF=expandable_segments:True

# ============================================================================
# Wandb Configuration
# ============================================================================
# Set wandb mode: "online" (default) or "offline"
export WANDB_MODE="${WANDB_MODE:-online}"

# Set wandb project name
export WANDB_PROJECT="${WANDB_PROJECT:-Dynamic-Coconut}"

# ============================================================================
# Model Configs
# ============================================================================
MODEL_NAME="Qwen/Qwen2.5-VL-3B-Instruct"

# ============================================================================
# Data Config
# ============================================================================
DATA_PACKING=True

LST=4096
MAX_INSTANCE_PER_BATCH=6
MAX_PACKED_TOKENS=$((MAX_INSTANCE_PER_BATCH * LST))

RANDOM_SEED=42

# Dataset configuration
# Options: "default" (viscot_sroie_dude) or "viscot_full" (Visual_cot full dataset)
# - "default": Uses meta_data_lvr_sft_stage1.json (viscot_sroie_dude_lvr_formatted.json)
# - "viscot_full": Uses meta_data_lvr_sft_stage1_viscot_full.json (viscot_363k_lvr_formatted.json with coco/openimages)
DATASET_CONFIG="${DATASET_CONFIG:-default}"  # Default: default

# Set DATA_PATH based on DATASET_CONFIG
if [ "$DATASET_CONFIG" = "viscot_full" ]; then
    DATA_PATH="data/meta_data_lvr_sft_stage1_viscot_full.json"
    echo "Using Visual_cot full dataset configuration"
else
    DATA_PATH="data/meta_data_lvr_sft_stage1.json"
    echo "Using default dataset configuration (viscot_sroie_dude)"
fi

# ============================================================================
# General Training Params
# ============================================================================
MAX_STEPS=2500
BATCH_PER_DEVICE=1            # if use data packing, BS should always be 1
GRAD_ACCUM_STEPS=8

# LLM-related params
LR=1e-5
LVR_HEAD=False

# LVR-related params
LVR_LOSS_FCT=mse
LAMBDA_LVR=0.1

MAX_TOKEN=5120
MIN_TOKEN=128

RUN_NAME="bs${MAX_INSTANCE_PER_BATCH}_gradAcc${GRAD_ACCUM_STEPS}_${LVR_LOSS_FCT}LVR${LAMBDA_LVR}-MaxVisToken${MAX_TOKEN}-MinVisToken${MIN_TOKEN}"
OUTPUT_DIR="result/stage1_3b/${RUN_NAME}/"

# ============================================================================
# Create Logs Directory
# ============================================================================
mkdir -p logs
mkdir -p "$OUTPUT_DIR"

# ============================================================================
# Print Configuration Summary
# ============================================================================
echo "=========================================="
echo "Training Configuration:"
echo "  Model: $MODEL_NAME"
echo "  Dataset Config: $DATASET_CONFIG"
echo "  Data Path: $DATA_PATH"
echo "  GPUs: ${GPU_IDS:-all available} ($NUM_DEVICES devices)"
echo "  Batch per device: $BATCH_PER_DEVICE"
echo "  Gradient accumulation steps: $GRAD_ACCUM_STEPS"
echo "  Effective batch size: $((BATCH_PER_DEVICE * NUM_DEVICES * GRAD_ACCUM_STEPS))"
echo "  Max steps: $MAX_STEPS"
echo "  Learning rate: $LR"
echo "  Wandb Project: $WANDB_PROJECT"
echo "  Wandb Mode: $WANDB_MODE"
echo "  Wandb Group: ${WANDB_GROUP:-not set}"
echo "  Wandb Entity: ${WANDB_ENTITY:-not set}"
echo "  Wandb Tags: ${WANDB_TAGS:-not set}"
echo "=========================================="

# ============================================================================
# Launch Training
# ============================================================================
# Set master port (default: 29500, can be overridden via MASTER_PORT env var)
MASTER_PORT="${MASTER_PORT:-29500}"

# Build deepspeed command
if [ "$USE_NUM_GPUS_FLAG" = true ]; then
    # Use --num_gpus when CUDA_VISIBLE_DEVICES is not set
    DEEPSPEED_CMD="deepspeed --num_gpus=$NUM_DEVICES --master_port=$MASTER_PORT"
else
    # Don't use --num_gpus when CUDA_VISIBLE_DEVICES is set (deepspeed will auto-detect)
    DEEPSPEED_CMD="deepspeed --master_port=$MASTER_PORT"
fi

$DEEPSPEED_CMD src/train/train_lvr.py \
    --run_name "$RUN_NAME" \
    --coconut True \
    --loss_lvr_fct $LVR_LOSS_FCT \
    --deepspeed scripts/zero3_offload.json \
    --model_id $MODEL_NAME \
    --data_path "$DATA_PATH" \
    --remove_unused_columns False \
    --lvr_head $LVR_HEAD \
    --freeze_vision_tower True \
    --freeze_merger True \
    --freeze_llm False \
    --max_steps $MAX_STEPS \
    --learning_rate $LR \
    --loss_lvr_lambda $LAMBDA_LVR \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 False \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs 1 \
    --per_device_train_batch_size $BATCH_PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --image_min_pixels $((MIN_TOKEN * 28 * 28)) \
    --image_max_pixels $((MAX_TOKEN * 28 * 28)) \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --gradient_checkpointing True \
    --report_to wandb \
    --lazy_preprocess True \
    --save_strategy "steps" \
    --save_steps 300 \
    --save_total_limit 10 \
    --dataloader_num_workers 8 \
    --enable_data_packing $DATA_PACKING \
    --max_packed_tokens $MAX_PACKED_TOKENS \
    --random_seed $RANDOM_SEED \
    --long_seq_threshold $LST \
    --max_instance_per_batch $MAX_INSTANCE_PER_BATCH

echo "Training completed!"

