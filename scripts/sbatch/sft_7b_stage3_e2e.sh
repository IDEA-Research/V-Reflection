#!/bin/bash
# SFT Stage 3: End-to-end reasoning unlock. CE loss only, no distillation.
# Loads from Stage 2 checkpoint. DAR in forward path for CE gradient flow.
# LLM backbone, DAR, vision/merger unfrozen for joint optimization.
#SBATCH --partition=cvr
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:hgx:8
#SBATCH --mem=640G
#SBATCH --qos=preemptive
#SBATCH --output=/comp_robot/zhoujiazhou/projects/Active-Coconut/logs/SFT_7b_stage3_e2e_lr1e6_%j.txt

source ~/miniconda3/etc/profile.d/conda.sh
conda activate train

cd /comp_robot/zhoujiazhou/projects/Active-Coconut
export PYTHONPATH="${PWD}:${PWD}/src/train:${PYTHONPATH:-}"
export LVR_TRACE=0
export LVR_DEBUG=0
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export NCCL_IB_DISABLE=0
export NCCL_DEBUG=WARN

export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_API_KEY=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
export WANDB_PROJECT="${WANDB_PROJECT:-Dynamic-Coconut}"

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
LVR_HEAD=False

# Stage 3: use_stage3_e2e=True, CE loss only. Requires use_box_feature_resampler (from Stage 2)
USE_BOX_FEATURE_RESAMPLER=True
USE_STAGE2_DISTILLATION=True
USE_STAGE3_E2E=True
NUM_LATENT_TOKENS=8
LATENT_END_TOKEN=False

MAX_TOKEN=5120
MIN_TOKEN=128

RUN_NAME="SFT_stage3_e2e_steps${MAX_STEPS}_b${MAX_INSTANCE_PER_BATCH}_acc${GRAD_ACCUM_STEPS}_latent${NUM_LATENT_TOKENS}_lr${LR}"
OUTPUT_DIR="result/stage3_e2e/${RUN_NAME}/"

# Stage 2 checkpoint (required): model with student_resampler
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/comp_robot/zhoujiazhou/projects/Active-Coconut/result/stage2_distillation/SFT_stage2_distillation_steps2500_b4_LVR0.1_resampler0.5_attnTransfer1.0_acc8_latent8/checkpoint-2500}"
if [ -z "$CHECKPOINT_PATH" ]; then
    echo "[WARN] CHECKPOINT_PATH not set. Stage 3 requires Stage 2 checkpoint with student_resampler."
    echo "       Set env: CHECKPOINT_PATH=\"path/to/stage2_checkpoint\" sbatch scripts/sbatch/sft_7b_stage3_e2e.sh"
fi

mkdir -p logs "$OUTPUT_DIR"
export WANDB_RUN_NAME="$RUN_NAME"
MASTER_PORT="${MASTER_PORT:-29500}"
[ "$LVR_HEAD" = "True" ] && LVR_HEAD_TYPE_ARG="--lvr_head_type ${LVR_HEAD_TYPE:-simple}" || LVR_HEAD_TYPE_ARG=""

CHECKPOINT_ARGS=""
[ -n "$CHECKPOINT_PATH" ] && CHECKPOINT_ARGS="--checkpoint_name $CHECKPOINT_PATH --resume_from_checkpoint $CHECKPOINT_PATH"

DEEPSPEED_CMD="deepspeed --master_port=$MASTER_PORT src/train/train_lvr.py \
    --run_name \"$RUN_NAME\" \
    --coconut True \
    --deepspeed scripts/zero3_offload.json \
    --model_id $MODEL_NAME \
    --data_path \"$DATA_PATH\" \
    --remove_unused_columns False \
    --lvr_head $LVR_HEAD \
    --latent_end_token $LATENT_END_TOKEN \
    --use_box_feature_resampler $USE_BOX_FEATURE_RESAMPLER \
    --use_stage2_distillation $USE_STAGE2_DISTILLATION \
    --use_stage3_e2e $USE_STAGE3_E2E \
    --num_latent_tokens $NUM_LATENT_TOKENS \
    --freeze_vision_tower False \
    --freeze_merger False \
    --freeze_llm False \
    --max_steps $MAX_STEPS \
    --learning_rate $LR \
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
    --save_steps 300 \
    --save_total_limit 10 \
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
# 指定测试集: BLINK, MMVP, VSTAR, POPE
export EVAL_BENCHMARKS="${EVAL_BENCHMARKS:-BLINK, MMVP, VSTAR, POPE}"
bash /comp_robot/zhoujiazhou/projects/Active-Coconut/scripts/evaluation/evaluation_7b_SFT_all_ck.sh
EVAL_EXIT=$?
if [ $EVAL_EXIT -ne 0 ]; then
    echo "Evaluation completed with errors (exit code $EVAL_EXIT)"
    exit $EVAL_EXIT
fi
echo "Training and evaluation completed successfully."
