#!/usr/bin/env python3
"""
DiT Reconstruction Sanity Check -- 一步验证法

目的: 用 Teacher Forcing 模式做一次完整 forward, 从 hidden_states 中取 <lvr> 位置的 token,
然后喂给 dit_recon_head.generate() 看生成的图片是否正常。
同时保存原始 cropped bbox 图片作为对比。

如果 Teacher Forcing 模式生成的图片能看(哪怕模糊), 说明训练是正确的, 问题在推理采样方式。
如果 Teacher Forcing 模式生成的图片也是乱码, 说明训练输入/输出处理有误。

用法:
    cd /comp_robot/zhoujiazhou/projects/Active-Coconut
    python scripts/debug_dit_sanity_check.py

环境变量:
    CHECKPOINT_PATH: 要测试的 checkpoint 路径 (默认: 最新的 LATENT12 checkpoint-2500)
    NUM_SAMPLES: 测试多少个样本 (默认: 5)
    DIT_STEPS: DiT 去噪步数 (默认: 20)
"""
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import json
import numpy as np
from PIL import Image
from transformers import AutoTokenizer, AutoProcessor, AutoConfig

from src.model.qwen_lvr_model import QwenWithLVR
from src.train.monkey_patch_forward_lvr import replace_qwen2_5_with_mixed_modality_forward_lvr
from src.constants import LVR_TOKEN, LVR_START_TOKEN, LVR_END_TOKEN, LVR_PLACEHOLDER
from src.dataset.lvr_sft_dataset_packed import crop_bbox_and_resize
from src.dataset.data_utils import map_image_path
from qwen_vl_utils import process_vision_info


def load_model(checkpoint_path):
    print(f"[1] 加载模型: {checkpoint_path}")
    config = AutoConfig.from_pretrained(checkpoint_path)
    
    use_box_feature_resampler = getattr(config, 'use_box_feature_resampler', False)
    use_dit_reconstruction = getattr(config, 'use_dit_reconstruction', False)
    
    print(f"    use_box_feature_resampler={use_box_feature_resampler}")
    print(f"    use_dit_reconstruction={use_dit_reconstruction}")
    print(f"    num_latent_tokens={getattr(config, 'num_latent_tokens', 8)}")
    print(f"    dit_num_latent_tokens={getattr(config, 'dit_num_latent_tokens', 8)}")
    
    # 不需要 monkey patch forward (我们手动调用, 不走 generate)
    model = QwenWithLVR.from_pretrained(
        checkpoint_path,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        low_cpu_mem_usage=True,
    )
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    processor = AutoProcessor.from_pretrained(checkpoint_path)
    
    # 获取 lvr_id
    lvr_id = processor.tokenizer.convert_tokens_to_ids(LVR_TOKEN)
    config.lvr_id = lvr_id
    print(f"    lvr_id={lvr_id} (token: {LVR_TOKEN})")
    
    # 检查 dit_recon_head 是否存在
    has_dit = hasattr(model, 'dit_recon_head') and model.dit_recon_head is not None
    print(f"    has dit_recon_head={has_dit}")
    if has_dit:
        print(f"    dit_recon_head type: {type(model.dit_recon_head).__name__}")
        # 确保 VAE 已加载
        model.dit_recon_head._ensure_vae_loaded()
        print(f"    VAE loaded: {model.dit_recon_head._vae_loaded}")
        print(f"    VAE scaling_factor: {model.dit_recon_head.vae.config.scaling_factor}")
    
    return model, processor, config, device


def load_samples(num_samples=5):
    """从训练数据中加载几个有 bbox 的样本"""
    print(f"\n[2] 加载训练数据样本 (num={num_samples})")
    
    data_path = os.path.join(PROJECT_ROOT, "data/viscot_363k_lvr_formatted.json")
    image_folder = "/comp_robot/zhoujiazhou/Datasets/Visual_cot/images"
    
    data = json.load(open(data_path, "r"))
    
    samples = []
    for d in data:
        if 'bboxes' not in d or len(d['bboxes']) == 0:
            continue
        
        # 使用 map_image_path 来正确映射图片路径 (和训练代码一致)
        img_path = d['image'][0] if isinstance(d['image'], list) else d['image']
        full_img_path = map_image_path(img_path, image_folder)
        if full_img_path is None or not os.path.exists(full_img_path):
            continue
        
        samples.append({
            'image_path': full_img_path,
            'bboxes': d['bboxes'],
            'question': d['conversations'][0]['value'],
            'answer': d['conversations'][1]['value'],
        })
        
        if len(samples) >= num_samples:
            break
    
    print(f"    找到 {len(samples)} 个有效样本")
    for i, s in enumerate(samples):
        print(f"    [{i}] {s['image_path']}  bbox={s['bboxes']}")
    return samples


def prepare_input(sample, processor, config, device):
    """准备模型输入, 模拟训练时的 Teacher Forcing 输入
    
    关键: 训练时 <lvr> 占位符会被替换成 <|lvr_start|><|lvr|>*N<|lvr_end|>
    num_latent_tokens 来自 config (12 for LATENT12 model)
    """
    img_path = sample['image_path']
    question = sample['question']
    answer = sample['answer']
    
    # 和训练代码一致: 把 <lvr> 替换成 <|lvr_start|><|lvr|>*N<|lvr_end|>
    num_latent_tokens = getattr(config, 'num_latent_tokens', 12)
    lvr_replacement = LVR_START_TOKEN + LVR_TOKEN * num_latent_tokens + LVR_END_TOKEN
    
    # answer 中的 <lvr> 需要替换
    answer_replaced = answer.replace(LVR_PLACEHOLDER, lvr_replacement)
    # question 中的 <image> 需要保留给 processor
    question_clean = question.replace("<image>\n", "").replace("<image>", "")
    
    # 构造 messages
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img_path},
                {"type": "text", "text": question_clean},
            ],
        },
        {
            "role": "assistant", 
            "content": [
                {"type": "text", "text": answer_replaced},
            ],
        }
    ]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    return inputs


@torch.no_grad()
def run_sanity_check(model, processor, config, device, samples, save_dir, dit_steps=20):
    """执行 Sanity Check"""
    os.makedirs(save_dir, exist_ok=True)
    
    lvr_id = config.lvr_id
    num_latent_tokens = getattr(config, 'dit_num_latent_tokens', getattr(config, 'num_latent_tokens', 8))
    
    print(f"\n[3] 开始 Sanity Check (num_latent_tokens={num_latent_tokens}, dit_steps={dit_steps})")
    print(f"    保存目录: {save_dir}")
    
    for idx, sample in enumerate(samples):
        print(f"\n  === 样本 {idx} ===")
        print(f"  图片: {sample['image_path']}")
        print(f"  BBox: {sample['bboxes']}")
        print(f"  问题: {sample['question'][:100]}...")
        
        # 1. 准备输入
        inputs = prepare_input(sample, processor, config, device)
        input_ids = inputs['input_ids']
        
        # 查找 <lvr> token 位置
        lvr_mask = (input_ids == lvr_id)
        num_lvr_tokens = lvr_mask.sum().item()
        print(f"  序列长度: {input_ids.shape[1]}, <lvr> token 数量: {num_lvr_tokens}")
        
        if num_lvr_tokens < num_latent_tokens:
            print(f"  [跳过] <lvr> token 数量不足 ({num_lvr_tokens} < {num_latent_tokens})")
            continue
        
        # 2. Teacher Forcing forward -- 获取 hidden_states
        print(f"  执行 Teacher Forcing forward...")
        
        # 获取 input embeddings
        inputs_embeds = model.model.get_input_embeddings()(input_ids)
        
        # 处理 image embeddings
        pixel_values = inputs.get('pixel_values', None)
        image_grid_thw = inputs.get('image_grid_thw', None)
        
        if pixel_values is not None:
            image_embeds = model.visual(pixel_values, grid_thw=image_grid_thw)
            if isinstance(image_embeds, (list, tuple)):
                image_embeds = torch.cat(image_embeds, dim=0)
            
            image_mask = input_ids == model.config.image_token_id
            image_mask_unsqueeze = image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask_unsqueeze, image_embeds)
        
        # 计算 position_ids
        position_ids, rope_deltas = model.get_rope_index(
            input_ids,
            image_grid_thw,
            None,  # video_grid_thw
            second_per_grid_ts=None,
            attention_mask=inputs.get('attention_mask', None),
        )
        
        # Forward through transformer
        outputs = model.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=inputs.get('attention_mask', None),
            past_key_values=None,
            inputs_embeds=inputs_embeds,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )
        
        hidden_states = outputs.last_hidden_state  # (1, seq_len, hidden_size)
        
        # 3. 提取 <lvr> 位置的 hidden states
        batch_indices, seq_positions = torch.nonzero(lvr_mask, as_tuple=True)
        
        # 取前 num_latent_tokens 个 <lvr> token 的 hidden states
        # 训练代码中是 seq_positions - 1 (取前一个位置的 hidden state)
        seq_positions_for_dit = seq_positions[:num_latent_tokens] - 1
        llm_tokens = hidden_states[0, seq_positions_for_dit, :]  # (num_latent_tokens, hidden_size)
        llm_tokens = llm_tokens.unsqueeze(0)  # (1, num_latent_tokens, hidden_size)
        
        print(f"  LLM condition tokens shape: {llm_tokens.shape}")
        print(f"  LLM condition tokens stats: mean={llm_tokens.mean().item():.4f}, std={llm_tokens.std().item():.4f}, "
              f"min={llm_tokens.min().item():.4f}, max={llm_tokens.max().item():.4f}")
        
        # 4. 用 DiT generate 生成图片
        print(f"  DiT generate (steps={dit_steps})...")
        generated_img = model.dit_recon_head.generate(
            llm_tokens,
            num_inference_steps=dit_steps,
            return_intermediate=False,
        )
        
        print(f"  生成图片 shape: {generated_img.shape}, "
              f"range: [{generated_img.min().item():.3f}, {generated_img.max().item():.3f}]")
        
        # 5. 保存生成的图片
        img_np = ((generated_img[0].permute(1, 2, 0).cpu().float().numpy() + 1) * 127.5).clip(0, 255).astype('uint8')
        gen_img = Image.fromarray(img_np)
        gen_path = os.path.join(save_dir, f"sample{idx:03d}_dit_generated.png")
        gen_img.save(gen_path)
        print(f"  保存 DiT 生成图: {gen_path}")
        
        # 6. 保存原始 cropped bbox 图片 (Ground Truth)
        pil_img = Image.open(sample['image_path']).convert('RGB')
        for bi, bbox in enumerate(sample['bboxes'][:1]):  # 只取第一个 bbox
            cropped_tensor = crop_bbox_and_resize(pil_img, bbox)  # (3, 256, 256) in [-1, 1]
            crop_np = ((cropped_tensor.permute(1, 2, 0).numpy() + 1) * 127.5).clip(0, 255).astype('uint8')
            crop_img = Image.fromarray(crop_np)
            crop_path = os.path.join(save_dir, f"sample{idx:03d}_gt_crop.png")
            crop_img.save(crop_path)
            print(f"  保存 GT 裁切图: {crop_path}")
        
        # 7. 保存原始完整图片 (缩小版)
        orig_img = pil_img.copy()
        orig_img.thumbnail((512, 512))
        orig_path = os.path.join(save_dir, f"sample{idx:03d}_original.png")
        orig_img.save(orig_path)
        print(f"  保存原始图: {orig_path}")
        
        # 8. 额外测试: 一步去噪法 (训练时的前向噪声预测验证)
        print(f"\n  --- 额外测试: 一步去噪验证 ---")
        cropped_tensor_for_dit = crop_bbox_and_resize(pil_img, sample['bboxes'][0])
        cropped_tensor_for_dit = cropped_tensor_for_dit.unsqueeze(0).to(device, dtype=llm_tokens.dtype)
        
        # 用 DiT forward (训练模式模拟)
        model.dit_recon_head.eval()
        
        # VAE encode
        with torch.no_grad():
            cropped_f32 = cropped_tensor_for_dit.float()
            latent_gt = model.dit_recon_head.vae.encode(cropped_f32).latent_dist.sample()
            latent_gt = latent_gt * model.dit_recon_head.vae.config.scaling_factor
            latent_gt = latent_gt.to(llm_tokens.dtype)
        
        print(f"  GT latent stats: mean={latent_gt.mean().item():.4f}, std={latent_gt.std().item():.4f}")
        
        # 添加少量噪声 (t=100, 比较轻)
        t_test = torch.tensor([100], device=device, dtype=torch.long)
        noise = torch.randn_like(latent_gt)
        noisy_latent = model.dit_recon_head.noise_scheduler.add_noise(latent_gt, noise, t_test)
        
        # DiT 前向: 预测噪声 (与训练/forward 完全一致的路径: x_embedder)
        cond = model.dit_recon_head.condition_proj(llm_tokens)
        t_emb = model.dit_recon_head._timestep_embed(t_test, device, llm_tokens.dtype)
        x = model.dit_recon_head.x_embedder(noisy_latent)  # (B, hidden_size, H_patch, W_patch)
        B_x, C_x, H_patch, W_patch = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, hidden_size)
        x = x + t_emb.unsqueeze(1)
        for blk in model.dit_recon_head.blocks:
            x = blk(x, cond)
        x = model.dit_recon_head.norm(x)
        noise_pred_patch = model.dit_recon_head.final_layer(x)
        noise_pred_full = model.dit_recon_head._unpatchify(noise_pred_patch, H_patch, W_patch)
        noise_pred = noise_pred_full[:, : model.dit_recon_head.in_channels, :, :]
        
        # 一步去噪: 从 noisy_latent 减去预测噪声
        latent_recon = noisy_latent - noise_pred
        print(f"  noise_pred stats: mean={noise_pred.mean().item():.4f}, std={noise_pred.std().item():.4f}")
        print(f"  latent_recon stats: mean={latent_recon.mean().item():.4f}, std={latent_recon.std().item():.4f}")
        
        # VAE decode
        latent_recon_f32 = latent_recon.float() / model.dit_recon_head.vae.config.scaling_factor
        recon_img = model.dit_recon_head.vae.decode(latent_recon_f32).sample
        recon_img = recon_img.clamp(-1, 1)
        
        recon_np = ((recon_img[0].permute(1, 2, 0).cpu().float().numpy() + 1) * 127.5).clip(0, 255).astype('uint8')
        recon_pil = Image.fromarray(recon_np)
        recon_path = os.path.join(save_dir, f"sample{idx:03d}_one_step_denoise.png")
        recon_pil.save(recon_path)
        print(f"  保存一步去噪图: {recon_path}")
        
        # 9. 测试: 全零 condition 生成 (检查 condition 是否有影响)
        print(f"\n  --- 额外测试: 全零 condition 生成 ---")
        zero_cond = torch.zeros_like(llm_tokens)
        generated_zero = model.dit_recon_head.generate(
            zero_cond,
            num_inference_steps=dit_steps,
            return_intermediate=False,
        )
        zero_np = ((generated_zero[0].permute(1, 2, 0).cpu().float().numpy() + 1) * 127.5).clip(0, 255).astype('uint8')
        zero_img = Image.fromarray(zero_np)
        zero_path = os.path.join(save_dir, f"sample{idx:03d}_zero_cond.png")
        zero_img.save(zero_path)
        print(f"  保存全零 condition 图: {zero_path}")
        print(f"  全零 condition 生成范围: [{generated_zero.min().item():.3f}, {generated_zero.max().item():.3f}]")
    
    print(f"\n[完成] 所有结果保存在: {save_dir}")
    print(f"\n对比说明:")
    print(f"  - *_gt_crop.png:          Ground Truth (原始 bbox 裁切)")
    print(f"  - *_dit_generated.png:    DiT 从纯噪声 + Teacher Forcing LLM tokens 生成")
    print(f"  - *_one_step_denoise.png: 一步去噪验证 (t=100 加噪后, DiT 预测噪声减掉)")
    print(f"  - *_zero_cond.png:        全零 condition 生成 (检查 condition 是否影响结果)")
    print(f"\n判读:")
    print(f"  如果 _dit_generated 和 _one_step_denoise 都是乱码 -> 训练本身有问题")
    print(f"  如果 _one_step_denoise 可以看到东西但 _dit_generated 不行 -> DDIM 采样有问题")
    print(f"  如果两者都能看到东西 -> 训练正确, 问题在推理时 Coconut 自回归的 condition 分布偏移")


def main():
    # 配置
    default_ckpt = "/comp_robot/zhoujiazhou/projects/Active-Coconut/result/dit_recon/SFT_dit_recon_steps2500_b4_dit1.0_resampler0.1_acc8_LATENT12/checkpoint-2500"
    checkpoint_path = os.environ.get("CHECKPOINT_PATH", default_ckpt)
    num_samples = int(os.environ.get("NUM_SAMPLES", "5"))
    dit_steps = int(os.environ.get("DIT_STEPS", "20"))
    
    save_dir = os.path.join(PROJECT_ROOT, "evaluation", "dit_sanity_check")
    
    print("=" * 70)
    print("DiT Reconstruction Sanity Check -- 一步验证法")
    print("=" * 70)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Num samples: {num_samples}")
    print(f"DiT steps: {dit_steps}")
    print("=" * 70)
    
    # 加载模型
    model, processor, config, device = load_model(checkpoint_path)
    
    # 加载样本
    samples = load_samples(num_samples)
    
    if not samples:
        print("错误: 没有找到有效样本!")
        return
    
    # 执行 sanity check
    run_sanity_check(model, processor, config, device, samples, save_dir, dit_steps)


if __name__ == "__main__":
    main()
