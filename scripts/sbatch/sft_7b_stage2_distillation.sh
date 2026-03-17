#!/bin/bash
# SFT Stage 2: Teacher-Student distillation with Dynamic Autoregressive Compression.
# Teacher (frozen): Box-Guided Compression from Stage 1 checkpoint.
# Student (trainable): Dynamic Autoregressive Compression - LLM 8 <lvr> hidden states as Q, full image as KV, MSE to Teacher target.
# Requires Stage 1 checkpoint with use_box_feature_resampler (e.g. sft_7b_box_resampler.sh output).
#SBATCH --partition=cvr
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:hgx:8
#SBATCH --mem=640G
#SBATCH --qos=preemptive
#SBATCH --output=logs/SFT_7b_stage2_distillation_resampler1.0_attn1.0_lr1e-6_v2_%j.txt

source ~/miniconda3/etc/profile.d/conda.sh
conda activate train

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="${PWD}:${PWD}/src/train:${PYTHONPATH:-}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export NCCL_IB_DISABLE=0
export NCCL_DEBUG=WARN

export WANDB_MODE="${WANDB_MODE:-online}"
# Set WANDB_API_KEY in your environment or .env for wandb logging (e.g. wandb login)
export WANDB_PROJECT="${WANDB_PROJECT:-V-Reflection}"

MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"
DATA_PACKING=True
LST=4096
MAX_INSTANCE_PER_BATCH=4
MAX_PACKED_TOKENS=$((MAX_INSTANCE_PER_BATCH * LST))
BATCH_PER_DEVICE=1
GRAD_ACCUM_STEPS=8
RANDOM_SEED=42
DATASET_CONFIG="${DATASET_CONFIG:-default}"
DATA_PATH=$([ "$DATASET_CONFIG" = "viscot_full" ] && \
    echo "data/meta_data_lvr_sft_stage1_viscot_full.json" || \
    echo "data/meta_data_lvr_sft_stage1.json")

MAX_STEPS=2500
LR=1e-6
LAMBDA_LVR="${LAMBDA_LVR:-0.1}"  # loss_lvr_lambda (unused when use_box_feature_resampler)

# Stage 2: use_box_feature_resampler (Teacher) + use_stage2_distillation (Student)
USE_BOX_FEATURE_RESAMPLER=True
USE_STAGE2_DISTILLATION=True
NUM_LATENT_TOKENS=8
LOSS_LVR_RESAMPLER_LAMBDA=1.0
LOSS_ATTN_TRANSFER_LAMBDA=1.0

# Loss control
USE_MSE_LOSS=True
LVR_LOSS_FCT=mse

MAX_TOKEN=5120
MIN_TOKEN=128

RUN_NAME="SFT_stage2_distillation_steps${MAX_STEPS}_b${MAX_INSTANCE_PER_BATCH}_LVR${LAMBDA_LVR}_resampler${LOSS_LVR_RESAMPLER_LAMBDA}_attnTransfer${LOSS_ATTN_TRANSFER_LAMBDA}_acc${GRAD_ACCUM_STEPS}_latent${NUM_LATENT_TOKENS}_lr${LR}_v2"
OUTPUT_DIR="result/stage2_distillation/${RUN_NAME}/"


CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"

mkdir -p logs "$OUTPUT_DIR"
export WANDB_RUN_NAME="$RUN_NAME"
MASTER_PORT="${MASTER_PORT:-29500}"
CHECKPOINT_ARGS=""
[ -n "$CHECKPOINT_PATH" ] && CHECKPOINT_ARGS="--checkpoint_name $CHECKPOINT_PATH --resume_from_checkpoint $CHECKPOINT_PATH"

DEEPSPEED_CMD="deepspeed --master_port=$MASTER_PORT src/train/train_lvr.py \
    --run_name \"$RUN_NAME\" \
    --coconut True \
    --loss_lvr_fct $LVR_LOSS_FCT \
    --deepspeed scripts/zero3_offload.json \
    --model_id $MODEL_NAME \
    --data_path \"$DATA_PATH\" \
    --remove_unused_columns False \
    --use_box_feature_resampler $USE_BOX_FEATURE_RESAMPLER \
    --use_stage2_distillation $USE_STAGE2_DISTILLATION \
    --num_latent_tokens $NUM_LATENT_TOKENS \
    --loss_lvr_resampler_lambda $LOSS_LVR_RESAMPLER_LAMBDA \
    --loss_attn_transfer_lambda $LOSS_ATTN_TRANSFER_LAMBDA \
    --freeze_vision_tower True \
    --freeze_merger True \
    --freeze_llm False \
    --max_steps $MAX_STEPS \
    --learning_rate $LR \
    --loss_lvr_lambda $LAMBDA_LVR \
    --use_mse_loss $USE_MSE_LOSS \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 False \
    --output_dir \"$OUTPUT_DIR\" \
    $CHECKPOINT_ARGS \
    --num_train_epochs 1 \
    --per_device_train_batch_size $BATCH_PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --image_min_pixels $((MIN_TOKEN * 28 * 28)) \
    --image_max_pixels $((MAX_TOKEN * 28 * 28)) \
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
    --save_total_limit 25 \
    --dataloader_num_workers 8 \
    --enable_data_packing $DATA_PACKING \
    --max_packed_tokens $MAX_PACKED_TOKENS \
    --random_seed $RANDOM_SEED \
    --long_seq_threshold $LST \
    --max_instance_per_batch $MAX_INSTANCE_PER_BATCH"

eval $DEEPSPEED_CMD

if [ $? -ne 0 ]; then
    echo "Error: Training failed"
    exit 1
fi

echo "Training completed. Starting evaluation of all checkpoints..."
export BASE_CHECKPOINT_DIR="${PWD}/${OUTPUT_DIR}"
export DATASET_CONFIG="${DATASET_CONFIG:-default}"
# Benchmarks: BLINK, MMVP, VSTAR, HRBench4K, HRBench8K, MME-RealWorld-Lite
export EVAL_BENCHMARKS="${EVAL_BENCHMARKS:-BLINK, MMVP, VSTAR, HRBench4K, HRBench8K, MME-RealWorld-Lite}"
bash "$PROJECT_ROOT/scripts/evaluation/evaluation_7b_sbatch.sh"
EVAL_EXIT=$?
if [ $EVAL_EXIT -ne 0 ]; then
    echo "Evaluation completed with errors (exit code $EVAL_EXIT)"
    exit $EVAL_EXIT
fi
echo "Training and evaluation completed successfully."
