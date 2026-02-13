# DiT 像素重建头 技术文档

## 概述

DiT 像素重建头（DiTReconstructionHead）是一个基于 **DiT-S/8** 的轻量级扩散头，采用 **Latent Diffusion** 训练方式，以 **LLM 在 8 个 `<lvr>` 位置的 hidden states** 为条件，在 VAE latent 空间预测噪声进行训练，推理时通过 DDIM 采样生成图像。用于增强模型对物体几何与外观的感知能力。

## 动机

- **几何感知**：仅靠 latent token MSE 难以显式约束“长什么样”；像素级重建迫使模型在 bbox 内生成与 GT 一致的区域。
- **条件生成**：用 LLM 的 8 个推理 token 作为 condition，使 DiT 学会“按语言/推理结果画出来”，与 BoxFeatureResampler 的 8-token 设计对齐。
- **轻量高效**：采用 DiT-S 规格（hidden=384, depth=12），配合 VAE 隐空间和 DDIM 快速采样（20 步），训练与推理成本可控。

## 架构

```
训练 (Latent Diffusion):
  原图 → bbox 裁剪 → resize 256×256 → VAE Encoder → latent 32×32×4
                                                    ↓ × 0.18215 (scaling_factor)
  随机 timestep t → 加噪 → noisy_latent → DiT-S/8 (condition: LLM 8 tokens)
                                                    ↓
  预测噪声 noise_pred → MSE(noise_pred, noise) [latent 空间]

推理 (DDIM Sampling):
  纯高斯噪声 → DDIM 多步去噪 (condition: LLM 8 tokens, 默认 20 步)
                                                    ↓
  denoised latent → ÷ 0.18215 → VAE Decoder → 图像
```

### 核心参数（DiT-S/8）

| 项目 | 值 |
|------|-----|
| 输入图像 | 256×256 RGB，归一化 [-1, 1] |
| VAE | stabilityai/sd-vae-ft-mse，输出 32×32×4 latent |
| Patch | 2×2 → 16×16 = 256 patches |
| Hidden | 384 |
| Depth | 12 层 Transformer |
| Heads | 6 |
| Condition | LLM 8 tokens → Linear(3584→384) |

### 条件注入

- LLM 在 8 个 `<lvr>` 位置的 `hidden_states`：(B, 8, 3584) → `condition_proj` → (B, 8, 384)。
- 每个 DiT 块内：Self-Attn → **Cross-Attn(Q=latent, K/V=condition)** → MLP。

## 训练流程

### 1. 数据

- 每条样本在 `lvr_sft_dataset_packed` 中提供 **cropped_bbox_images**：按 GT bbox 裁剪并 resize 到 256×256，归一化到 [-1, 1]，shape (num_bboxes, 3, 256, 256)。
- 与 BoxFeatureResampler 共用 8 个 `<lvr>` 槽位；forward 时从对应位置取 LLM 的 hidden states 作为 condition。

### 2. Forward 与 Loss

- 在 `monkey_patch_forward_lvr` 中，当 `use_dit_reconstruction=True` 且本 batch 有 `cropped_bbox_images` 时：
  - 取 8 个 lvr 位置的 `hidden_states` → (B, 8, 3584)。
  - 调用 `dit_recon_head(cropped_bbox_images, llm_condition_tokens)`。
- 训练时 head 内部：
  1. VAE 编码图像得到 latents，**乘以 scaling_factor (0.18215)** 修正方差
  2. 随机采样 timestep t，生成噪声 `noise = randn_like(latents)`
  3. 使用 `scheduler.add_noise` 得到 `noisy_latents`
  4. DiT 预测噪声 `noise_pred = DiT(noisy_latents, t, condition)`
  5. **直接在 latent 空间计算 MSE(noise_pred, noise)**，即 `loss_dit_recon`
  6. **不进行 VAE decode**，避免梯度不稳定和计算浪费

### 3. 总 Loss

```
Total Loss = loss_ce + λ_resampler * loss_lvr_resampler + λ_dit_recon * loss_dit_recon
```

`λ_dit_recon` 由 `--loss_dit_recon_lambda` 控制（默认 0.1）。

## 预训练权重

**当前没有可直接使用的 DiT-S/8 预训练权重。**

- **Hugging Face 现状**：Meta 官方仅发布了 **DiT-XL/2** 系列（如 `facebook/DiT-XL-2-256`、`facebook/DiT-XL-2-512`），对应 **XL** 规格（hidden/depth/heads 等与 DiT-S 不同），且为 **class-conditioned**，与本文的 **LLM 8-token condition** 结构不一致。
- **本实现**：DiT-S 规格（hidden=384, depth=12, num_heads=6, patch 2），且带自定义的 cross-attention condition 分支，与 HF 上的 DiT 权重**不兼容**，`_load_pretrained_dit` 即使指向 `facebook/DiT-XL-2-256` 也无法正确加载（参数名与 shape 对不上）。
- **建议**：将 `--dit_pretrained_path` 留空，**从头训练** DiT 与 `condition_proj`。若后续社区或官方发布与 DiT-S/8 结构兼容的权重，再在代码中做适配后可传入该参数。

## 配置参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--use_dit_reconstruction` | 启用 DiT 像素重建头 | False |
| `--dit_pretrained_path` | 预训练 DiT 权重路径或 HF 仓库（当前无可用 DiT-S/8，建议留空） | None |
| `--dit_vae_repo` | DiT 用 VAE 仓库 | stabilityai/sd-vae-ft-mse |
| `--dit_hidden_size` | DiT hidden 维度 | 384 |
| `--dit_num_latent_tokens` | 作为 condition 的 LLM token 数 | 8 |
| `--loss_dit_recon_lambda` | DiT 重建 loss 权重 | 0.1 |
| `--dit_num_inference_steps` | 推理时 DDIM 去噪步数 | 20 |

## 使用示例

### 训练脚本

```bash
# scripts/sbatch/sft_7b_dit_recon.sh
USE_BOX_FEATURE_RESAMPLER=True
NUM_LATENT_TOKENS=8
USE_DIT_RECONSTRUCTION=True
DIT_PRETRAINED_PATH=""   # 暂无 DiT-S/8 预训练权重，留空从头训练
LOSS_DIT_RECON_LAMBDA=0.1
DIT_NUM_INFERENCE_STEPS=20

deepspeed src/train/train_lvr.py \
    --use_box_feature_resampler $USE_BOX_FEATURE_RESAMPLER \
    --num_latent_tokens $NUM_LATENT_TOKENS \
    --use_dit_reconstruction $USE_DIT_RECONSTRUCTION \
    --loss_dit_recon_lambda $LOSS_DIT_RECON_LAMBDA \
    --dit_num_inference_steps $DIT_NUM_INFERENCE_STEPS \
    ...
```

## 文件结构

```
src/
├── model/
│   ├── lvr_heads.py          # DiTReconstructionHead, _DiTBlock
│   └── qwen_lvr_model.py     # _init_dit_reconstruction_head
├── train/
│   └── monkey_patch_forward_lvr.py  # 取 8 个 lvr hidden、调用 head、返回 loss_dit_recon
├── dataset/
│   └── lvr_sft_dataset_packed.py    # cropped_bbox_images 生成与 collator 拼接
├── trainer/
│   └── lvr_trainer.py        # 汇总 loss_dit_recon
└── params.py                 # use_dit_reconstruction, loss_dit_recon_lambda 等

scripts/sbatch/
└── sft_7b_dit_recon.sh       # DiT 重建训练入口
```

## 与其他模块的关系

| 模块 | 关系 |
|------|------|
| BoxFeatureResampler | **配套使用**。DiT 的 8-token condition 与 resampler 的 8 个 latent 槽位一致，通常同时开启。 |
| LVR Head | **独立**。可与 LVR head 同时或单独使用。 |
| VAE | **冻结**。仅用于 encode/decode，不参与梯度。 |
| LLM | **联合训练**。通过 DiT 重建 loss 约束 LLM 在 `<lvr>` 位置输出对生成有用的 condition。 |

## 注意事项

1. **Latent Diffusion 训练**：训练时在 **latent 空间** 直接计算 `MSE(noise_pred, noise)`，不进行 VAE decode。这避免了：
   - x0 重建公式中的 clamp 操作导致的梯度截断
   - 像素空间 Loss 量级不稳定导致的震荡
   - 训练时不必要的 VAE decode 计算开销
2. **VAE Scaling Factor**：训练和推理时都必须正确处理 VAE 的 scaling factor (0.18215)：
   - **训练**：VAE encode 后 `latent *= scaling_factor`
   - **推理**：VAE decode 前 `latent /= scaling_factor`
3. **推理调度器**：使用 **DDIMScheduler** 进行快速采样（默认 20 步），训练时仍使用 DDPMScheduler。
4. **VAE 冻结**：减少显存与不稳定，只训练 DiT 与 condition_proj。
5. **预训练**：目前无可用 DiT-S/8 权重，建议 `dit_pretrained_path` 留空；若将来有结构兼容的 checkpoint，可传入路径，代码会尝试拷贝匹配的参数。
6. **依赖**：需要 `diffusers`（VAE、DDPMScheduler、DDIMScheduler、可选 DiT 预训练加载）。
