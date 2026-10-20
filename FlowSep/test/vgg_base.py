import sys
import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import json
import csv
import numpy as np
import logging
import warnings
from tqdm import tqdm
from pytorch_lightning import seed_everything

# ==============================================================================
# 0. 基础设置 & Monkey Patch (必须保留)
# ==============================================================================
torch.set_float32_matmul_precision('high')
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

# 请修改为你的实际路径
sys.path.append("/s7home/zyk524/FLOWSEP/FlowSep-main/src")

# --- Monkey Patch ---
import latent_diffusion.modules.diffusionmodules.util as diffusion_util
def dummy_checkpoint(func, inputs, params, flag):
    return func(*inputs)
diffusion_util.checkpoint = dummy_checkpoint

from latent_diffusion.modules.diffusionmodules.openaimodel import AttentionBlock
def custom_attention_forward(self, x):
    return self._forward(x)
AttentionBlock.forward = custom_attention_forward

from latent_diffusion.util import instantiate_from_config
from utilities.data.dataset import AudioDataset

# ==============================================================================
# 1. 验证函数
# ==============================================================================
def verify_checkpoint_loading(model, ckpt_path, model_name="Base Model"):
    print(f"\n[Verification] Checking {model_name} against {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(state_dict.keys())
    intersection = model_keys.intersection(ckpt_keys)
    print(f"  Matched Keys: {len(intersection)}")
    print(f"  [SUCCESS] {model_name} loaded.")

# ==============================================================================
# 2. 模型构建 (Baseline - 无 Adapter)
# ==============================================================================
def build_model(args, config):
    print("Instantiating Model...")
    if args.vae_ckpt:
        config['model']['params']['first_stage_config']['params']['reload_from_ckpt'] = args.vae_ckpt
    model = instantiate_from_config(config["model"])
    
    # === 保持与 PEFT 一致：关闭 EMA ===
    print("[Config] Disabling EMA for fair comparison...")
    model.use_ema = False
    if hasattr(model, 'model_ema'):
        del model.model_ema
        model.model_ema = None

    if args.vae_ckpt:
        verify_checkpoint_loading(model.first_stage_model, args.vae_ckpt, "VAE")

    print(f"Loading Base Checkpoint: {args.load_checkpoint}")
    ckpt = torch.load(args.load_checkpoint, map_location="cpu")
    if "state_dict" in ckpt: ckpt = ckpt["state_dict"]
    
    # 加载权重
    model.load_state_dict(ckpt, strict=False)
    verify_checkpoint_loading(model, args.load_checkpoint, "Latent Diffusion")
    
    # 注意：这里不再注入 Adapter
    
    model.to(args.device)
    model.eval()
    return model

# ==============================================================================
# 3. 推理逻辑 (保持修复后的版本)
# ==============================================================================
def run_inference(model, mix_wav_path, text, device, val_dataset, args):
    batch = {}
    batch["fname"] = [mix_wav_path]
    batch["text"] = [text]
    batch["caption"] = [text]
    
    # Dummy data
    batch["waveform"] = torch.rand(1, 1, 16000).to(device)
    batch["log_mel_spec"] = torch.rand(1, 1024, 64).to(device)
    batch["sampling_rate"] = torch.tensor([16000]).to(device)
    batch["label_vector"] = torch.rand(1, 527).to(device)
    batch["stft"] = torch.rand(1, 1024, 512).to(device)

    # 读取音频
    noise_waveform, _ = val_dataset.read_wav_file(mix_wav_path)
    
    # 转 Numpy
    if torch.is_tensor(noise_waveform):
        noise_waveform = noise_waveform.detach().cpu().numpy()
    
    if noise_waveform.ndim == 2:
        noise_waveform = noise_waveform[0]

    # 填充/裁剪
    target_len = 163840 
    if len(noise_waveform) > target_len:
        noise_waveform = noise_waveform[:target_len]
    else:
        pad_len = target_len - len(noise_waveform)
        noise_waveform = np.pad(noise_waveform, (0, pad_len))

    # 特征提取
    mixed_mel, stft = val_dataset.wav_feature_extraction(noise_waveform.reshape(1, -1))
    
    # 构造 Batch
    if isinstance(noise_waveform, np.ndarray):
        batch["mixed_waveform"] = torch.from_numpy(noise_waveform.reshape(1, 1, target_len)).to(device)
    else:
        batch["mixed_waveform"] = noise_waveform.reshape(1, 1, target_len).to(device)

    if torch.is_tensor(mixed_mel):
        batch["mixed_mel"] = mixed_mel.reshape(1, mixed_mel.shape[0], mixed_mel.shape[1]).to(device)
    else:
        batch["mixed_mel"] = torch.from_numpy(mixed_mel.reshape(1, mixed_mel.shape[0], mixed_mel.shape[1])).to(device)

    # 生成
    with torch.no_grad():
        waveform = model.generate_sample(
            [batch], 
            name="result", 
            unconditional_guidance_scale=1.0, 
            ddim_steps=args.infer_step, 
            n_gen=1, 
            save_mixed=False
        )
    
    # 后处理
    if isinstance(waveform, dict):
        if 'sample' in waveform: gen_wav = waveform['sample']
        elif 'wav' in waveform: gen_wav = waveform['wav']
        else: gen_wav = list(waveform.values())[0]
    else:
        gen_wav = waveform

    if torch.is_tensor(gen_wav):
        gen_wav = gen_wav.detach().cpu().numpy()
    
    gen_wav = gen_wav.astype(np.float32)
    return gen_wav.flatten()

# ==============================================================================
# 4. 指标计算
# ==============================================================================
def calculate_metrics(ref, est):
    ref = np.array(ref).flatten()
    est = np.array(est).flatten()
    
    min_len = min(len(ref), len(est))
    ref = ref[:min_len]
    est = est[:min_len]
    
    if np.sum(ref**2) < 1e-8: return -99, -99
    
    def get_sisdr(ref, est):
        eps = 1e-8
        ref = ref - np.mean(ref)
        est = est - np.mean(est)
        alpha = np.dot(est, ref) / (np.dot(ref, ref) + eps)
        target = alpha * ref
        noise = est - target
        return 10 * np.log10((np.dot(target, target) + eps) / (np.dot(noise, noise) + eps))

    def get_sdr(ref, est):
        eps = 1e-8
        num = np.sum(ref**2)
        den = np.sum((ref - est)**2)
        return 10 * np.log10((num + eps) / (den + eps))

    return get_sisdr(ref, est), get_sdr(ref, est)

# ==============================================================================
# 5. 主程序
# ==============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, default="/s7home/zyk524/FLOWSEP/FlowSep-main/lass_config/2channel_flow.yaml")
    parser.add_argument("--load_checkpoint", type=str, default="/s7home/zyk524/FLOWSEP/FlowSep-main/model_logs/v2_100k.ckpt")
    parser.add_argument("--vae_ckpt", type=str, default="/s7home/zyk524/FLOWSEP/FlowSep-main/model_logs/vae.ckpt")
    parser.add_argument("--csv_path", type=str, default="/s7home/zyk524/test/audio/evaluation/metadata/vggidx_eval.csv")
    parser.add_argument("--audios_dir", type=str, default="/dataset/AudioSep_eval/vggsound")
    parser.add_argument("--infer_step", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    seed_everything(42)
    config = yaml.load(open(args.config_yaml, "r"), Loader=yaml.FullLoader)
    
    # 构建基线模型
    model = build_model(args, config)
    val_dataset = AudioDataset(config, split="test", add_ons=[])

    sisdrs = []
    sdris = []
    
    print(f"\nStarting Baseline Evaluation on {args.csv_path}...")
    with open(args.csv_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        
    error_count = 0
    
    for row in tqdm(rows):
        mix_path = os.path.join(args.audios_dir, row['mix_wav'])
        src_path = os.path.join(args.audios_dir, row['s0_wav'])
        text = row['s0_text']
        
        if not os.path.exists(mix_path) or not os.path.exists(src_path): continue
        
        try:
            ref_wav, _ = torchaudio.load(src_path)
            mix_wav_raw, _ = torchaudio.load(mix_path)
            ref_np = ref_wav.squeeze().numpy()
            mix_np = mix_wav_raw.squeeze().numpy()
        except: continue

        # 计算 No-Sep SDR
        min_len = min(len(ref_np), len(mix_np))
        _, sdr_no_sep = calculate_metrics(ref_np[:min_len], mix_np[:min_len])
        
        try:
            # 运行推理 (Baseline)
            est_np = run_inference(model, mix_path, text, args.device, val_dataset, args)
            
            if est_np.shape[0] == 0: raise ValueError("Empty audio")
            
            # 计算指标
            curr_sisdr, curr_sdr = calculate_metrics(ref_np, est_np)
            curr_sdri = curr_sdr - sdr_no_sep
            
            if curr_sisdr > -100:
                sisdrs.append(curr_sisdr)
                sdris.append(curr_sdri)
                
        except Exception as e:
            error_count += 1
            if error_count <= 5:
                print(f"\n[ERROR] Sample: {row['mix_wav']}")
                print(f"  Error Details: {e}")
            pass

    print("\n" + "="*40)
    print(f"FlowSep Baseline Evaluation Results")
    print(f"  Evaluated Samples: {len(sisdrs)}")
    if len(sisdrs) > 0:
        print(f"  Avg SDRi:   {np.mean(sdris):.3f}")
        print(f"  Avg SI-SDR: {np.mean(sisdrs):.3f}")
    else:
        print("  No samples evaluated successfully.")
    print("="*40)

if __name__ == "__main__":
    main()
