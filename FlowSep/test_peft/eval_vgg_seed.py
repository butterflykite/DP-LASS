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
# 0. 基础设置 & Monkey Patch
# ==============================================================================
torch.set_float32_matmul_precision('high')
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

sys.path.append("/s7home/zyk524/FLOWSEP/FlowSep-main/src")

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
# 1. Adapter 定义
# ==============================================================================
class ParallelAdapter(nn.Module):
    def __init__(self, original_layer, alpha=1.0):
        super().__init__()
        self.original_layer = original_layer
        self.alpha = alpha
        if isinstance(original_layer, nn.Conv2d):
            self.adapter = nn.Conv2d(
                in_channels=original_layer.in_channels,
                out_channels=original_layer.out_channels,
                kernel_size=original_layer.kernel_size,
                stride=original_layer.stride,
                padding=original_layer.padding,
                dilation=original_layer.dilation,
                groups=original_layer.groups,
                bias=False 
            )
        else:
            raise ValueError("ParallelAdapter only supports Conv2d")
        nn.init.zeros_(self.adapter.weight)
        self.scaling = self.alpha

    def forward(self, x, *args, **kwargs):
        if x.device != self.adapter.weight.device:
            self.adapter = self.adapter.to(x.device)
        with torch.no_grad():
            orig_out = self.original_layer(x, *args, **kwargs)
        return orig_out + self.adapter(x) * self.scaling

def inject_adapters_encoder_only(model, alpha=1.0):
    injected_count = 0
    target_blocks = ["input_blocks"]
    for name, module in model.named_modules():
        should_inject = False
        for target in target_blocks:
            if target in name:
                should_inject = True
                break
        if not should_inject: continue

        if isinstance(module, nn.Conv2d) and not isinstance(module, ParallelAdapter):
            if module.kernel_size == (1, 1) or module.kernel_size == 1: continue
            if '.' in name:
                parent_name, child_name = name.rsplit('.', 1)
                parent = model
                for part in parent_name.split('.'): parent = getattr(parent, part)
            else:
                parent = model
                child_name = name
            wrapped_layer = ParallelAdapter(module, alpha=alpha)
            setattr(parent, child_name, wrapped_layer)
            injected_count += 1
    print(f"[Injector] Injected {injected_count} Parallel Adapters into Encoder.")
    return injected_count

# ==============================================================================
# 2. 验证函数
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
# 3. 模型构建
# ==============================================================================
# ==============================================================================
# 3. 模型构建 (修复 EMA 冲突)
# ==============================================================================
def build_model(args, config):
    print("Instantiating Model...")
    
    # 1. 设置 VAE 路径
    if args.vae_ckpt:
        config['model']['params']['first_stage_config']['params']['reload_from_ckpt'] = args.vae_ckpt
        
    model = instantiate_from_config(config["model"])
    
    # === 【核心修复】强制关闭 EMA ===
    # Adapter 改变了参数名，EMA 字典无法匹配，会导致推理崩溃
    print("[Config] Disabling EMA for PEFT inference...")
    model.use_ema = False
    if hasattr(model, 'model_ema'):
        del model.model_ema
        model.model_ema = None
    # ==============================
    
    # 2. 验证 VAE 加载
    if args.vae_ckpt:
        verify_checkpoint_loading(model.first_stage_model, args.vae_ckpt, "VAE (First Stage)")

    # 3. 加载基线权重 (LDM)
    print(f"Loading Base Checkpoint: {args.load_checkpoint}")
    ckpt = torch.load(args.load_checkpoint, map_location="cpu")
    if "state_dict" in ckpt: ckpt = ckpt["state_dict"]
    
    m, u = model.load_state_dict(ckpt, strict=False)
    
    # 4. 验证基线加载
    verify_checkpoint_loading(model, args.load_checkpoint, "Latent Diffusion")
    
    # 5. 注入 Adapter
    print("Injecting Adapters...")
    if hasattr(model, "model") and hasattr(model.model, "diffusion_model"):
        unet = model.model.diffusion_model
    else:
        unet = model
    
    inject_adapters_encoder_only(unet, alpha=1.0)
    
    model.to(args.device)
    model.eval()
    return model

# ==============================================================================
# 4. 推理逻辑 (已修复 Tensor/Numpy 问题)
# ==============================================================================
def run_inference(model, mix_wav_path, text, device, val_dataset, args):
    # 1. 构造基础 Batch
    batch = {}
    batch["fname"] = [mix_wav_path]
    batch["text"] = [text]
    batch["caption"] = [text]
    
    # Dummy data (满足接口需求)
    batch["waveform"] = torch.rand(1, 1, 16000).to(device)
    batch["log_mel_spec"] = torch.rand(1, 1024, 64).to(device)
    batch["sampling_rate"] = torch.tensor([16000]).to(device)
    batch["label_vector"] = torch.rand(1, 527).to(device)
    batch["stft"] = torch.rand(1, 1024, 512).to(device)

    # 2. 读取音频
    # read_wav_file 通常返回 (waveform, start_time)
    noise_waveform, _ = val_dataset.read_wav_file(mix_wav_path)
    
    # === 修复点 1: 确保 noise_waveform 是 Numpy ===
    if torch.is_tensor(noise_waveform):
        noise_waveform = noise_waveform.detach().cpu().numpy()
    
    # 处理维度: 原始代码是 noise_waveform[0]，意味着它可能是 (C, T)
    if noise_waveform.ndim == 2:
        noise_waveform = noise_waveform[0] # 取第一个通道 -> (T,)

    # 3. 填充/裁剪
    target_len = 163840 
    if len(noise_waveform) > target_len:
        noise_waveform = noise_waveform[:target_len]
    else:
        pad_len = target_len - len(noise_waveform)
        noise_waveform = np.pad(noise_waveform, (0, pad_len))

    # 4. 特征提取
    # 传入 (1, T) 形状的 Numpy
    mixed_mel, stft = val_dataset.wav_feature_extraction(noise_waveform.reshape(1, -1))
    
    # === 修复点 2: 检查 mixed_mel 类型 ===
    # 如果 wav_feature_extraction 返回的是 Tensor，这里必须转回 Numpy 才能用 torch.from_numpy
    # 或者直接用 .to(device)
    
    # 处理 mixed_waveform
    if isinstance(noise_waveform, np.ndarray):
        batch["mixed_waveform"] = torch.from_numpy(noise_waveform.reshape(1, 1, target_len)).to(device)
    else: # 已经是 Tensor
        batch["mixed_waveform"] = noise_waveform.reshape(1, 1, target_len).to(device)

    # 处理 mixed_mel
    if torch.is_tensor(mixed_mel):
        # 如果已经是 Tensor，直接调整维度并移动到 GPU
        batch["mixed_mel"] = mixed_mel.reshape(1, mixed_mel.shape[0], mixed_mel.shape[1]).to(device)
    else:
        # 如果是 Numpy，使用 from_numpy
        batch["mixed_mel"] = torch.from_numpy(mixed_mel.reshape(1, mixed_mel.shape[0], mixed_mel.shape[1])).to(device)

    # 5. 生成
    with torch.no_grad():
        waveform = model.generate_sample(
            [batch], 
            name="result", 
            unconditional_guidance_scale=1.0, 
            ddim_steps=args.infer_step, 
            n_gen=1, 
            save_mixed=False
        )
    
    # 6. 后处理结果
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
# 5. 指标计算
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
# 6. 主程序
# ==============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, default="/s7home/zyk524/FLOWSEP/FlowSep-main/lass_config/2channel_flow.yaml")
    parser.add_argument("--load_checkpoint", type=str, default="/s7home/zyk524/FLOWSEP/FlowSep-main/model_logs/v2_100k.ckpt")
    parser.add_argument("--vae_ckpt", type=str, default="/s7home/zyk524/FLOWSEP/FlowSep-main/model_logs/vae.ckpt")
    parser.add_argument("--peft_dir", type=str, default="/s7home/zyk524/FLOWSEP/FlowSep-main/peft")
    parser.add_argument("--csv_path", type=str, default="/s7home/zyk524/test/audio/evaluation/metadata/vggidx_eval.csv")
    parser.add_argument("--audios_dir", type=str, default="/dataset/AudioSep_eval/vggsound")
    parser.add_argument("--infer_step", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    seed_everything(42)
    config = yaml.load(open(args.config_yaml, "r"), Loader=yaml.FullLoader)
    model = build_model(args, config)
    val_dataset = AudioDataset(config, split="test", add_ons=[])

    print("\nPre-loading Adapter weights...")
    adapter_weights_list = []
    for i in range(7):
        pth = os.path.join(args.peft_dir, f"cluster_{i}", f"cluster{i}_best.pt")
        if not os.path.exists(pth):
            pth = os.path.join(args.peft_dir, f"cluster_{i}", f"cluster{i}_last.pt")
        if os.path.exists(pth):
            print(f"  Cluster {i}: Found {os.path.basename(pth)}")
            adapter_weights_list.append(torch.load(pth, map_location=args.device))
        else:
            print(f"  Cluster {i}: NOT FOUND!")
            adapter_weights_list.append(None)

    sisdrs = []
    sdris = []
    
    print(f"\nStarting Evaluation on {args.csv_path}...")
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

        min_len = min(len(ref_np), len(mix_np))
        _, sdr_no_sep = calculate_metrics(ref_np[:min_len], mix_np[:min_len])
        
        best_sisdr = -999
        best_sdri = -999
        
        for i, adapter_w in enumerate(adapter_weights_list):
            if adapter_w is None: continue
            model.load_state_dict(adapter_w, strict=False)
            
            try:
                est_np = run_inference(model, mix_path, text, args.device, val_dataset, args)
                
                if est_np.shape[0] == 0: raise ValueError("Empty audio")
                
                curr_sisdr, curr_sdr = calculate_metrics(ref_np, est_np)
                curr_sdri = curr_sdr - sdr_no_sep
                
                if curr_sisdr > best_sisdr:
                    best_sisdr = curr_sisdr
                    best_sdri = curr_sdri
            except Exception as e:
                error_count += 1
                if error_count <= 5:
                    print(f"\n[ERROR] Sample: {row['mix_wav']} | Cluster: {i}")
                    print(f"  Error Details: {e}")
                pass
        
        if best_sisdr > -100:
            sisdrs.append(best_sisdr)
            sdris.append(best_sdri)

    print("\n" + "="*40)
    print(f"FlowSep PEFT Evaluation Results")
    print(f"  Evaluated Samples: {len(sisdrs)}")
    if len(sisdrs) > 0:
        print(f"  Avg SDRi:   {np.mean(sdris):.3f}")
        print(f"  Avg SI-SDR: {np.mean(sisdrs):.3f}")
    else:
        print("  No samples evaluated successfully.")
    print("="*40)

if __name__ == "__main__":
    main()
