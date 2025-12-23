#!/bin/bash
#SBATCH --partition=cvr
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:hgx:8 # A800
#SBATCH --mem=640G
#SBATCH --qos=preemptive #specify preemptive Q0S
#SBATCH --output=/comp_robot/zhoujiazhou/projects/Active-Coconut/logs/SFT_7b_isg_b1_acc8_lvr0.1_%j.txt

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
LVR_HEAD_TYPE="isg"  # Options: simple, glu, attention-mask, slot-attention, ivr, implicit-visual-routing, intrinsic-similarity, isg

# ISG (Intrinsic Similarity Gating) 参数
ISG_CHUNK_SIZE="${ISG_CHUNK_SIZE:-}"  # Chunk大小，默认None（自动选择）
ISG_USE_OUTPUT_NORM="${ISG_USE_OUTPUT_NORM:-True}"  # 是否使用输出归一化，默认True

LVR_LOSS_FCT=mse
LAMBDA_LVR=0.1
MAX_TOKEN=5120
MIN_TOKEN=128
RUN_NAME="${LVR_HEAD_TYPE}_b${MAX_INSTANCE_PER_BATCH}_acc${GRAD_ACCUM_STEPS}_${LVR_LOSS_FCT}LVR${LAMBDA_LVR}"
OUTPUT_DIR="result/SFT_7b_8GPU_${LVR_HEAD_TYPE}/${RUN_NAME}/"

mkdir -p logs "$OUTPUT_DIR"
MASTER_PORT="${MASTER_PORT:-29500}"

# Build deepspeed command with ISG parameters
DEEPSPEED_CMD="deepspeed --master_port=$MASTER_PORT src/train/train_lvr.py \
    --run_name \"$RUN_NAME\" \
    --coconut True \
    --loss_lvr_fct $LVR_LOSS_FCT \
    --deepspeed scripts/zero3_offload_disk_optimized.json \
    --model_id $MODEL_NAME \
    --data_path \"$DATA_PATH\" \
    --remove_unused_columns False \
    --lvr_head $LVR_HEAD \
    --lvr_head_type $LVR_HEAD_TYPE \
    --isg_use_output_norm $ISG_USE_OUTPUT_NORM \
    --freeze_vision_tower True \
    --freeze_merger True \
    --freeze_llm False \
    --max_steps $MAX_STEPS \
    --learning_rate $LR \
    --loss_lvr_lambda $LAMBDA_LVR \
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

# Add optional ISG chunk_size parameter if it is set
[ -n "$ISG_CHUNK_SIZE" ] && DEEPSPEED_CMD="$DEEPSPEED_CMD --isg_chunk_size $ISG_CHUNK_SIZE"

# Execute the command
eval $DEEPSPEED_CMD

if [ $? -ne 0 ]; then
    echo "Error: Training failed"
    exit 1
fi