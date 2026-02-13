#!/bin/bash
# SFT with DiT-S/8 pixel reconstruction head: Latent Diffusion training.
# Training: bbox crop -> VAE encode -> scale latent -> add noise -> DiT predict noise (conditioned on LLM 8 tokens) -> MSE(noise_pred, noise) in latent space.
# Inference: pure noise -> DDIM denoising (conditioned on LLM 8 tokens) -> scale latent -> VAE decode -> image.
# Uses use_box_feature_resampler + fixed 8 latent tokens so sequence has 8 slots per bbox; DiT uses those 8 LLM tokens as condition.
#SBATCH --partition=cvr
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:hgx:8
#SBATCH --mem=640G
#SBATCH --qos=preemptive
#SBATCH --output=/comp_robot/zhoujiazhou/projects/Active-Coconut/logs/SFT_7b_b4_dit_recon_v2_resize_256_pretrained_%j.txt

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
LR=1e-5
LVR_HEAD=False

# BoxFeatureResampler: 8 latent slots per bbox (same as DiT condition length)
USE_BOX_FEATURE_RESAMPLER=True
NUM_LATENT_TOKENS=8
LATENT_END_TOKEN=False
LOSS_LVR_RESAMPLER_LAMBDA=0.1
LOSS_MODE_SWITCH_LAMBDA=0

# DiT pixel reconstruction head (Latent Diffusion)
# Training: predicts noise in latent space, no VAE decode during training
# Inference: uses DDIMScheduler for fast sampling (default 20 steps)
USE_DIT_RECONSTRUCTION=True
# Note: Currently no compatible DiT-S/8 pretrained weights available on HF
# Leave empty to train from scratch (recommended)
DIT_PRETRAINED_PATH="${DIT_PRETRAINED_PATH:-/comp_robot/zhoujiazhou/projects/Active-Coconut/data/DiT-XL-2-256x256.pt}"  # Empty by default - train from scratch
LOSS_DIT_RECON_LAMBDA=1.0  # Loss weight for latent space noise prediction MSE
DIT_NUM_INFERENCE_STEPS=20  # DDIM inference steps (faster than DDPM)
# 50/50 GT condition: probability of using Resampler GT tokens instead of LLM tokens as DiT condition
# Helps DiT learn denoising with clean condition signal. 0.0=always LLM, 1.0=always GT, 0.5=50/50
DIT_CONDITION_GT_PROB="${DIT_CONDITION_GT_PROB:-0.5}"
DIT_CROP_SIZE="${DIT_CROP_SIZE:-128}"  # Bbox crop size: 128->16x16 latent (faster), 256->32x32 latent

# Loss control
USE_MSE_LOSS=True
LVR_LOSS_FCT=mse
LAMBDA_LVR=0.1

MAX_TOKEN=5120
MIN_TOKEN=128

RUN_NAME="SFT_dit_recon_steps${MAX_STEPS}_b${MAX_INSTANCE_PER_BATCH}_dit${LOSS_DIT_RECON_LAMBDA}_resampler${LOSS_LVR_RESAMPLER_LAMBDA}_acc${GRAD_ACCUM_STEPS}_LATENT${NUM_LATENT_TOKENS}"
OUTPUT_DIR="result/dit_recon/${RUN_NAME}/"
# DiT 生成图片保存目录（推理时保存，训练时不进行 VAE decode）
export DIT_SAVE_DIR="${DIT_SAVE_DIR:-${PWD}/${OUTPUT_DIR}generated_images}"

# 如果需要从已有 checkpoint 继续训练，请在提交前设置 CHECKPOINT_PATH，例如：
# CHECKPOINT_PATH=\"/comp_robot/zhoujiazhou/projects/Active-Coconut/result/box_resampler/SFT_box_resampler_steps2500_b4_LVR0.1_resampler0.1_acc8/checkpoint-1800\" sbatch scripts/sbatch/sft_7b_box_resampler.sh
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"

mkdir -p logs "$OUTPUT_DIR"
export WANDB_RUN_NAME="$RUN_NAME"
MASTER_PORT="${MASTER_PORT:-29500}"
[ "$LVR_HEAD" = "True" ] && LVR_HEAD_TYPE_ARG="--lvr_head_type ${LVR_HEAD_TYPE:-simple}" || LVR_HEAD_TYPE_ARG=""

DIT_PRETRAINED_ARG=""
[ -n "$DIT_PRETRAINED_PATH" ] && DIT_PRETRAINED_ARG="--dit_pretrained_path \"$DIT_PRETRAINED_PATH\""

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
    --loss_mode_switch_lambda $LOSS_MODE_SWITCH_LAMBDA \
    --use_dit_reconstruction $USE_DIT_RECONSTRUCTION \
    $DIT_PRETRAINED_ARG \
    --loss_dit_recon_lambda $LOSS_DIT_RECON_LAMBDA \
    --dit_num_inference_steps $DIT_NUM_INFERENCE_STEPS \
    --dit_condition_gt_prob $DIT_CONDITION_GT_PROB \
    --dit_crop_size $DIT_CROP_SIZE \
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
bash /comp_robot/zhoujiazhou/projects/Active-Coconut/scripts/evaluation/evaluation_7b_SFT_all_ck.sh
EVAL_EXIT=$?
if [ $EVAL_EXIT -ne 0 ]; then
    echo "Evaluation completed with errors (exit code $EVAL_EXIT)"
    exit $EVAL_EXIT
fi
echo "Training and evaluation completed successfully."
