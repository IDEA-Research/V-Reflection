#!/bin/bash
#SBATCH --partition=cvr
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:hgx:8 # A800
#SBATCH --mem=640G
#SBATCH --qos=preemptive #specify preemptive Q0S
#SBATCH --output=/comp_robot/zhoujiazhou/projects/Active-Coconut/logs/SFT_7b_b1_acc8_8GPU_ce+mse_lvr0.1_intrinsic-similarity_%j.txt

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
export NCCL_IB_DISABLE=0  # 启用 InfiniBand（如果可用）
export NCCL_DEBUG=WARN  # 设置为 WARN 以减少日志输出（调试时可设为 INFO）

export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-Dynamic-Coconut}"

MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"
DATA_PACKING=True
LST=4096 #4096
MAX_INSTANCE_PER_BATCH=1
MAX_PACKED_TOKENS=$((MAX_INSTANCE_PER_BATCH * LST))
BATCH_PER_DEVICE=1
GRAD_ACCUM_STEPS=8
RANDOM_SEED=42
DATASET_CONFIG="${DATASET_CONFIG:-default}"
DATA_PATH=$([ "$DATASET_CONFIG" = "viscot_full" ] && \
    echo "data/meta_data_lvr_sft_stage1_viscot_full.json" || \
    echo "data/meta_data_lvr_sft_stage1.json")

MAX_STEPS=2500
LR=1e-5
LVR_HEAD=True
LVR_HEAD_TYPE="intrinsic-similarity"  # Options: simple, glu, attention-mask, ivr, intrinsic-similarity, isg

# Loss control
USE_MSE_LOSS="${USE_MSE_LOSS:-True}"  # Enable MSE/reconstruction loss (default: True)
LVR_LOSS_FCT=mse
LAMBDA_LVR=0.1

MAX_TOKEN=5120
MIN_TOKEN=128

# Build run name and output dir (handle empty LVR_HEAD_TYPE)
if [ -z "$LVR_HEAD_TYPE" ]; then
    RUN_NAME="SFT_steps${MAX_STEPS}_b${MAX_INSTANCE_PER_BATCH}_${LVR_LOSS_FCT}LVR${LAMBDA_LVR}_acc${GRAD_ACCUM_STEPS}_MSE${USE_MSE_LOSS}"
    OUTPUT_DIR="result/no_head/${RUN_NAME}/"
else
    RUN_NAME="SFT_steps${MAX_STEPS}_b${MAX_INSTANCE_PER_BATCH}_${LVR_LOSS_FCT}LVR${LAMBDA_LVR}_acc${GRAD_ACCUM_STEPS}_MSE${USE_MSE_LOSS}"
    OUTPUT_DIR="result/${LVR_HEAD_TYPE}/${RUN_NAME}/"
fi

mkdir -p logs "$OUTPUT_DIR"
export WANDB_RUN_NAME="$RUN_NAME"
MASTER_PORT="${MASTER_PORT:-29500}"

# Build deepspeed command with IVR parameters
DEEPSPEED_CMD="deepspeed --master_port=$MASTER_PORT src/train/train_lvr.py \
    --run_name \"$RUN_NAME\" \
    --coconut True \
    --loss_lvr_fct $LVR_LOSS_FCT \
    --deepspeed scripts/zero3_offload.json \
    --model_id $MODEL_NAME \
    --data_path \"$DATA_PATH\" \
    --remove_unused_columns False \
    --lvr_head $LVR_HEAD \
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
    --save_steps 300 \
    --save_total_limit 10 \
    --dataloader_num_workers 8 \
    --enable_data_packing $DATA_PACKING \
    --max_packed_tokens $MAX_PACKED_TOKENS \
    --random_seed $RANDOM_SEED \
    --long_seq_threshold $LST \
    --max_instance_per_batch $MAX_INSTANCE_PER_BATCH"

# Add optional lvr_head_type parameter if it is set
[ -n "$LVR_HEAD_TYPE" ] && DEEPSPEED_CMD="$DEEPSPEED_CMD --lvr_head_type $LVR_HEAD_TYPE"

# Add IVR parameters only if LVR_HEAD_TYPE is "ivr" or if IVR variables are set
if [ "$LVR_HEAD_TYPE" = "ivr" ] || [ -n "$IVR_ITERATIONS" ] || [ -n "$IVR_USE_OUTPUT_NORM" ] || [ -n "$IVR_TEMPERATURE" ]; then
    [ -n "$IVR_ITERATIONS" ] && DEEPSPEED_CMD="$DEEPSPEED_CMD --ivr_iterations $IVR_ITERATIONS"
    [ -n "$IVR_USE_OUTPUT_NORM" ] && DEEPSPEED_CMD="$DEEPSPEED_CMD --ivr_use_output_norm $IVR_USE_OUTPUT_NORM"
    [ -n "$IVR_TEMPERATURE" ] && DEEPSPEED_CMD="$DEEPSPEED_CMD --ivr_temperature $IVR_TEMPERATURE"
    [ -n "$IVR_CHUNK_SIZE" ] && DEEPSPEED_CMD="$DEEPSPEED_CMD --ivr_chunk_size $IVR_CHUNK_SIZE"
fi

# Execute the command
eval $DEEPSPEED_CMD

if [ $? -ne 0 ]; then
    echo "Error: Training failed"
    exit 1
fi

echo "Training completed. Starting evaluation..."
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
        
        # Call the batch evaluation script
        bash "${PWD}/scripts/evaluation/evaluation_7b_SFT_all_ck.sh" || {
            echo "Warning: Batch evaluation failed"
            exit 1
        }
    else
        echo "Warning: No checkpoint directories found in $BASE_CHECKPOINT_DIR. Skipping evaluation."
    fi
else
    echo "Warning: Checkpoint directory not found: $BASE_CHECKPOINT_DIR. Skipping evaluation."
fi

