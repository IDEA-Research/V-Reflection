#!/bin/bash
# SFT with BoxFeatureResampler: fixed N latent tokens per bbox, MSE(LLM_N, target.detach()).
# BoxFeatureResampler is independent of LVR head: use LVR_HEAD=False (no intrinsic-similarity/other head).
# Uses fixed_num_of_lvr_tokens=8 (no latent_end_token / loss_mode_switch / lvr_latent_end_emb).
#SBATCH --partition=cvr
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:hgx:8
#SBATCH --mem=640G
#SBATCH --qos=preemptive
#SBATCH --output=/comp_robot/zhoujiazhou/projects/Active-Coconut/logs/SFT_7b_box_resampler_b4_latent8_lr1e6_%j.txt

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
# MODEL_NAME="Qwen/Qwen3.0-VL-8B-Instruct"
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
LR=1e-5
# LVR head disabled: only BoxFeatureResampler loss
LVR_HEAD=False
# LVR_HEAD_TYPE 仅在 LVR_HEAD=True 时需要设置，否则脚本不会传 --lvr_head_type

# BoxFeatureResampler: GT box tokens -> resampler -> 8 tokens; sequence has 8 latent slots per bbox.
USE_BOX_FEATURE_RESAMPLER=True
NUM_LATENT_TOKENS=8
LATENT_END_TOKEN=False
LOSS_LVR_RESAMPLER_LAMBDA=0.5

# Loss control (original design: Total Loss = loss_ce + lambda_resampler * loss_lvr_resampler)
USE_MSE_LOSS=True
LVR_LOSS_FCT=mse

MAX_TOKEN=5120
MIN_TOKEN=128

RUN_NAME="SFT_box_resampler_steps${MAX_STEPS}_b${MAX_INSTANCE_PER_BATCH}_resampler${LOSS_LVR_RESAMPLER_LAMBDA}_acc${GRAD_ACCUM_STEPS}_latent${NUM_LATENT_TOKENS}_lr${LR}"
OUTPUT_DIR="result/box_resampler/${RUN_NAME}/"

# 如果需要从已有 checkpoint 继续训练，请在提交前设置 CHECKPOINT_PATH，例如：
# CHECKPOINT_PATH=\"/comp_robot/zhoujiazhou/projects/Active-Coconut/result/box_resampler/SFT_box_resampler_steps2500_b4_resampler0.1_acc8_latent8/checkpoint-1800\" sbatch scripts/sbatch/sft_7b_stage1_box_resampler.sh
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"

mkdir -p logs "$OUTPUT_DIR"
export WANDB_RUN_NAME="$RUN_NAME"
MASTER_PORT="${MASTER_PORT:-29500}"
# 仅当 LVR_HEAD=True 时传 --lvr_head_type，避免 LVR_HEAD=False 时还要无意义地指定 type
[ "$LVR_HEAD" = "True" ] && LVR_HEAD_TYPE_ARG="--lvr_head_type ${LVR_HEAD_TYPE:-simple}" || LVR_HEAD_TYPE_ARG=""

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
    --lvr_head $LVR_HEAD \
    --latent_end_token $LATENT_END_TOKEN \
    --use_box_feature_resampler $USE_BOX_FEATURE_RESAMPLER \
    --num_latent_tokens $NUM_LATENT_TOKENS \
    --loss_lvr_resampler_lambda $LOSS_LVR_RESAMPLER_LAMBDA \
    --freeze_vision_tower True \
    --freeze_merger True \
    --freeze_llm False \
    --max_steps $MAX_STEPS \
    --learning_rate $LR \
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
# 使用本次训练的保存目录作为评估的 checkpoint 根目录，对所有 ck 跑一遍评估
export BASE_CHECKPOINT_DIR="${PWD}/${OUTPUT_DIR}"
# 可选：与训练保持一致的数据配置
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
