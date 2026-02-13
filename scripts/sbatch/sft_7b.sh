#!/bin/bash
#SBATCH --partition=cvr
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:hgx:8 # A800
#SBATCH --mem=640G
#SBATCH --qos=preemptive #specify preemptive Q0S
#SBATCH --output=/comp_robot/zhoujiazhou/projects/Active-Coconut/logs/baseline_SFT_7b_b4_ce_steps2500_%j.txt

source ~/miniconda3/etc/profile.d/conda.sh
conda activate train


cd /comp_robot/zhoujiazhou/projects/Active-Coconut
export PYTHONPATH="${PWD}:${PWD}/src/train:${PYTHONPATH:-}"

GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
NUM_DEVICES=$(echo "$GPU_IDS" | tr ',' '\n' | wc -l)
export PYTORCH_ALLOC_CONF=expandable_segments:True

export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_API_KEY=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
export WANDB_PROJECT="${WANDB_PROJECT:-Dynamic-Coconut}"

MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"
DATA_PACKING=True
LST=4096 #4096
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
LR=1e-5
LVR_HEAD=False

# Loss control
USE_MSE_LOSS="${USE_MSE_LOSS:-True}"  # Enable MSE/reconstruction loss (default: True for LVR baseline)
LVR_LOSS_FCT=mse
LAMBDA_LVR=0.1

MAX_TOKEN=5120
MIN_TOKEN=128
RUN_NAME="SFT_steps${MAX_STEPS}_b${MAX_INSTANCE_PER_BATCH}_${LVR_LOSS_FCT}LVR${LAMBDA_LVR}_acc${GRAD_ACCUM_STEPS}_MSE${USE_MSE_LOSS}_resample"
OUTPUT_DIR="result/no_head/${RUN_NAME}/"

# 如果需要从已有 checkpoint 继续训练，请在提交前设置 CHECKPOINT_PATH，例如：
# CHECKPOINT_PATH="/comp_robot/zhoujiazhou/projects/Active-Coconut/result/no_head/SFT_steps2500_b4_mseLVR0.1_acc8_MSETrue_resample/checkpoint-1800" sbatch scripts/sbatch/sft_7b.sh
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/comp_robot/zhoujiazhou/projects/Active-Coconut/result/no_head/SFT_steps2500_b4_mseLVR0.1_acc8_MSETrue_resample/checkpoint-1800}"

mkdir -p logs "$OUTPUT_DIR"
MASTER_PORT="${MASTER_PORT:-29500}"

CHECKPOINT_ARGS=""
[ -n "$CHECKPOINT_PATH" ] && CHECKPOINT_ARGS="--checkpoint_name $CHECKPOINT_PATH --resume_from_checkpoint $CHECKPOINT_PATH"

deepspeed --master_port=$MASTER_PORT src/train/train_lvr.py \
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
    --use_mse_loss $USE_MSE_LOSS \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 False \
    --output_dir "$OUTPUT_DIR" \
    $CHECKPOINT_ARGS \
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

if [ $? -ne 0 ]; then
    echo "Error: Training failed"
    exit 1
fi

echo "Training completed. Starting evaluation..."
# evaluation_7b_SFT_all_ck.sh discovers all checkpoints under BASE_CHECKPOINT_DIR (not CHECKPOINT_PATH)
BASE_CHECKPOINT_DIR="${PWD}/${OUTPUT_DIR}"
CHECKPOINT_PATH="${BASE_CHECKPOINT_DIR}/checkpoint-${MAX_STEPS}"

# Use latest checkpoint if final checkpoint doesn't exist (for existence check only)
[ ! -d "$CHECKPOINT_PATH" ] && \
    CHECKPOINT_PATH=$(find "$BASE_CHECKPOINT_DIR" -maxdepth 1 -type d -name "checkpoint-*" | sort -V | tail -n 1)

if [ -d "$CHECKPOINT_PATH" ]; then
    export BASE_CHECKPOINT_DIR CHECKPOINT_PATH DATASET_CONFIG
    export EVAL_STEP_LIST="${EVAL_STEP_LIST:-4}"
    bash "${PWD}/scripts/evaluation/evaluation_7b_SFT_all_ck.sh" || echo "Warning: Evaluation failed"
else
    echo "Warning: No checkpoint found. Skipping evaluation."
fi

