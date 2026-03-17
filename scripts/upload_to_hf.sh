#!/bin/bash
# Upload V-Reflection model and data to Hugging Face
# Usage: bash scripts/upload_to_hf.sh

set -e
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

REPO_ID="garlandchou/V-Reflection"
CHECKPOINT_DIR="result/stage2_distillation/SFT_stage2_distillation_steps2500_b4_LVR0.1_resampler0.5_attnTransfer1.0_acc8_latent8_lr1e-6/checkpoint-1500"

# =============================================================================
# Step 1: Create a temp dir with only model files (exclude DeepSpeed, rng, etc.)
# =============================================================================
UPLOAD_DIR="${PROJECT_ROOT}/.hf_upload_temp"
rm -rf "$UPLOAD_DIR"
mkdir -p "$UPLOAD_DIR"

echo "Copying model files (excluding DeepSpeed checkpoints)..."
for f in config.json generation_config.json preprocessor_config.json video_preprocessor_config.json \
         tokenizer.json tokenizer_config.json special_tokens_map.json merges.txt added_tokens.json \
         chat_template.jinja \
         model.safetensors.index.json model-00001-of-00004.safetensors model-00002-of-00004.safetensors \
         model-00003-of-00004.safetensors model-00004-of-00004.safetensors; do
    if [ -f "${CHECKPOINT_DIR}/${f}" ]; then
        cp "${CHECKPOINT_DIR}/${f}" "$UPLOAD_DIR/"
        echo "  Copied $f"
    fi
done

echo ""
echo "Uploading model to ${REPO_ID}..."
huggingface-cli upload "$REPO_ID" "$UPLOAD_DIR" "." --repo-type model

rm -rf "$UPLOAD_DIR"
echo "Model upload done."

# =============================================================================
# Step 2: Upload data (annotations; images see README/Visual CoT)
# =============================================================================
echo ""
echo "Uploading data annotations to ${REPO_ID}/data/..."
huggingface-cli upload "$REPO_ID" "data/" "data/" --repo-type model

echo ""
echo "Done! Model and data uploaded to https://huggingface.co/${REPO_ID}"
