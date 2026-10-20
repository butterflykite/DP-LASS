import os
import yaml
import json
import random
import argparse
import time
from tqdm import tqdm
import numpy as np
import gc
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader, Dataset
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import DDIMScheduler
from transformers import AutoProcessor, ClapModel

# 禁用 cudnn benchmark 以保证复现性，甚至禁用 cudnn
torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.enabled = False # 如果需要极致的确定性可以取消注释

# 假设这些模块在你的路径下
from model.udit import UDiT
from vae_modules.autoencoder_wrapper import Autoencoder

# ==========================================
# 1. LoRA 模块与注入工具
# ==========================================

def print_model_stats(model):
    total_params = 0
    trainable_params = 0
    frozen_params = 0

    for param in model.parameters():
        num_params = param.numel()
        total_params += num_params
        if param.requires_grad:
            trainable_params += num_params
        else:
            frozen_params += num_params

    print("-" * 50)
    print(f"Model Statistics:")
    print(f"  - Total Params:      {total_params / 1e6:.2f} M")
    print(f"  - Frozen (Base):     {frozen_params / 1e6:.2f} M")
    print(f"  - Trainable (LoRA):  {trainable_params / 1e6:.2f} M")
    print(f"  - Trainable Ratio:   {trainable_params / total_params * 100:.4f} %")
    print("-" * 50)

class LoRALinear(nn.Module):
    def __init__(self, original_linear: nn.Linear, rank: int, alpha: int, dropout: float = 0.1):
        super().__init__()
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.rank = rank
        self.scaling = alpha / rank

        self.base_layer = original_linear
        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

        self.lora_A = nn.Parameter(torch.zeros(self.in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, self.out_features))
        self.lora_dropout = nn.Dropout(p=dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        dtype = self.lora_A.dtype
        x_casted = x.to(dtype)
        base_out = self.base_layer(x)
        lora_out = (self.lora_dropout(x_casted) @ self.lora_A @ self.lora_B) * self.scaling
        return base_out + lora_out.to(base_out.dtype)

def inject_lora(model, rank=16, alpha=16, dropout=0.1):
    target_modules = ['to_q', 'to_k', 'to_v', 'proj', 'fc1', 'fc2']
    
    for param in model.parameters():
        param.requires_grad = False
        
    injected_count = 0
    modules_to_replace = []

    # 收集需要替换的层
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            layer_name = name.split('.')[-1]
            if layer_name in target_modules:
                if '.' in name:
                    parent_name = name.rsplit('.', 1)[0]
                    parent = model.get_submodule(parent_name)
                else:
                    parent = model
                modules_to_replace.append((parent, layer_name))
    
    # 执行替换
    for parent, layer_name in modules_to_replace:
        original = getattr(parent, layer_name)
        if isinstance(original, LoRALinear): continue
        
        lora_layer = LoRALinear(original, rank=rank, alpha=alpha, dropout=dropout)
        setattr(parent, layer_name, lora_layer)
        injected_count += 1
        
    print(f"[LoRA] Injected {injected_count} layers.")
    return model

# ==========================================
# 2. 改进后的动态混合器 (修复 forward)
# ==========================================
class SegmentMixer(nn.Module):
    def __init__(self, noise_json_files, max_mix_num=2, lower_db=-10, higher_db=10, target_sr=48000, target_len_sec=2.0):
        super().__init__()
        self.max_mix_num = max_mix_num
        self.loudness_range = (lower_db, higher_db) 
        self.target_sr = target_sr
        self.target_len = int(target_sr * target_len_sec)
        
        self.audio_pool = []
        self._load_audio_paths(noise_json_files)

    def _load_audio_paths(self, json_files):
        self.audio_pool = []
        if not json_files: return
        
        for json_file in json_files:
            if not os.path.exists(json_file): continue
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                for item in data.get('data', []):
                    if 'wav' in item:
                        self.audio_pool.append(item['wav'])
            except Exception as e:
                print(f"Error loading {json_file}: {e}")
                
        print(f"[Mixer] Loaded {len(self.audio_pool)} noise samples from {len(json_files)} files.")

    def _load_audio(self, audio_path, device=None):
        try:
            wav, sr = torchaudio.load(audio_path)
            if sr != self.target_sr:
                wav = torchaudio.functional.resample(wav, sr, self.target_sr)
            if wav.shape[0] > 1: 
                wav = wav.mean(0, keepdim=True)
            
            curr_len = wav.shape[1]
            if curr_len > self.target_len:
                start = random.randint(0, curr_len - self.target_len)
                wav = wav[:, start:start+self.target_len]
            elif curr_len < self.target_len:
                wav = F.pad(wav, (0, self.target_len - curr_len))
                
            if device: wav = wav.to(device)
            return wav
        except: return None

    def get_energy(self, x):
        return torch.mean(x ** 2)

    def dynamic_loudnorm(self, noise, clean):
        e_clean = self.get_energy(clean)
        e_noise = max(self.get_energy(noise), 1e-10)
        
        ratio = (e_clean / e_noise) ** 0.5
        ratio = torch.clamp(ratio, 0.02, 50)
        noise = noise * ratio
        
        db = random.uniform(self.loudness_range[0], self.loudness_range[1])
        gain = 10 ** (db / 20.0)
        return noise * gain

    def mix_single(self, clean_wav):
        """处理单条音频混合 (供 Dataset 使用)"""
        mixture = clean_wav.clone()
        if not self.audio_pool:
            return mixture, clean_wav

        num_noise = random.randint(1, max(1, self.max_mix_num - 1))
        
        for _ in range(num_noise):
            noise_path = random.choice(self.audio_pool)
            noise_wav = self._load_audio(noise_path, device=clean_wav.device)
            if noise_wav is not None:
                scaled_noise = self.dynamic_loudnorm(noise_wav, clean_wav)
                mixture = mixture + scaled_noise
            
        max_val = mixture.abs().max()
        if max_val > 0.95:
            scale = 0.95 / max_val
            mixture = mixture * scale
            clean_wav = clean_wav * scale 
            
        return mixture, clean_wav

    def forward(self, waveforms):
        """处理 Batch 音频混合 (供 GPU 训练循环使用，如果需要)"""
        mixtures = []
        targets = []
        for i in range(waveforms.shape[0]):
            m, t = self.mix_single(waveforms[i])
            mixtures.append(m)
            targets.append(t)
        return torch.stack(mixtures), torch.stack(targets)

class ClusterDataset(Dataset):
    def __init__(self, target_json, noise_json_files, sample_rate=48000, duration=2.0):
        with open(target_json, 'r') as f:
            self.data = json.load(f)['data']
        
        self.mixer = SegmentMixer(
            noise_json_files=noise_json_files,
            max_mix_num=2,
            lower_db=-10, higher_db=10, 
            target_sr=sample_rate,
            target_len_sec=duration
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        path = item['wav']
        caption = item['caption']
        
        clean_wav = self.mixer._load_audio(path)
        # 使用 mix_single
        mixture_wav, target_wav = self.mixer.mix_single(clean_wav)
        
        return {
            "mixture": mixture_wav.squeeze(0),
            "target": target_wav.squeeze(0),
            "text": caption
        }

# ==========================================
# 3. 参数解析
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-data-dir', type=str, 
                        default='/s7home/zyk524/test/audio/SED/train_json/89checkpoint/7/train')
    parser.add_argument('--config', type=str, default='/s7home/zyk524/SoloAudio/SoloAudio-main/config/SoloAudio.yaml')
    parser.add_argument('--udit-ckpt', type=str,default='/s7home/zyk524/SoloAudio/SoloAudio-main/pretrained_models/soloaudio_v2.pt')
    parser.add_argument('--vae-path', type=str, default='/s7home/zyk524/SoloAudio/SoloAudio-main/pretrained_models/audio-vae.pt')
    parser.add_argument('--clap-path', type=str, default='/s7home/zyk524/SoloAudio/SoloAudio-main/clap')
    parser.add_argument('--output-base-dir', type=str, default='/s7home/zyk524/SoloAudio/SoloAudio-main/LORA_chongxu')

    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--precision', type=str, default='bf16', choices=['no', 'fp16', 'bf16'])
    parser.add_argument('--grad-clip', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)
    
    parser.add_argument('--lora-rank', type=int, default=16)
    parser.add_argument('--lora-alpha', type=int, default=16)
    
    return parser.parse_args()

# ==========================================
# 4. 主循环流程
# ==========================================
def main():
    args = parse_args()
    set_seed(args.seed)

    if args.precision == 'no':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    accelerator = Accelerator(mixed_precision=args.precision, log_with="tensorboard", project_dir=args.output_base_dir)
    
    with open(args.config, 'r') as fp:
        diff_config = yaml.safe_load(fp)

    # --- 2. 准备 VAE 和 CLAP (增加验证) ---
    print("\n[Init] Loading Aux Models...")
    
    autoencoder = Autoencoder(args.vae_path, 'stable_vae', quantization_first=True)
    autoencoder.eval()
    autoencoder.requires_grad_(False)
    autoencoder.to(accelerator.device)
    
    # 计算 VAE 初始 Checksum
    vae_sum = sum(p.sum().item() for p in autoencoder.parameters())
    print(f"[Verify] VAE Loaded. Checksum: {vae_sum:.4f}")

    clap_model = ClapModel.from_pretrained(args.clap_path).to(accelerator.device)
    clap_processor = AutoProcessor.from_pretrained(args.clap_path)
    clap_model.eval()
    clap_model.requires_grad_(False)
    
    # 计算 CLAP 初始 Checksum
    clap_sum = sum(p.sum().item() for p in clap_model.parameters())
    print(f"[Verify] CLAP Loaded. Checksum: {clap_sum:.4f}")

    # 3. 准备文件列表
    all_json_files = [os.path.join(args.base_data_dir, f"cluster{i}.json") for i in range(7)]
    CLUSTER_IDS = list(range(5,7))

    # ================= 循环训练 =================
    for cluster_id in CLUSTER_IDS:
        target_json = all_json_files[cluster_id]
        if not os.path.exists(target_json):
            print(f"Skipping Cluster {cluster_id}: File not found.")
            continue

        noise_jsons = [f for i, f in enumerate(all_json_files) if i != cluster_id]
        cluster_output_dir = os.path.join(args.output_base_dir, f"cluster{cluster_id}")
        
        if accelerator.is_main_process:
            os.makedirs(cluster_output_dir, exist_ok=True)
            print("\n" + "="*60)
            print(f"Training Cluster {cluster_id}")
            print("="*60 + "\n")

        gc.collect()
        torch.cuda.empty_cache()
        accelerator.free_memory()

        # --- A. 初始化 UDiT ---
        unet = UDiT(**diff_config['diffwrap']['UDiT'])
        ckpt = torch.load(args.udit_ckpt, map_location='cpu')
        if 'model' in ckpt: ckpt = ckpt['model']
        unet.load_state_dict(ckpt, strict=False)
        
        # --- B. 注入 LoRA ---
        unet = inject_lora(unet, rank=args.lora_rank, alpha=args.lora_alpha)
        unet.to(accelerator.device)
        unet.train()
        
        if accelerator.is_main_process:
            print_model_stats(unet)
            
        # --- C. 数据集 ---
        dataset = ClusterDataset(
            target_json=target_json,
            noise_json_files=noise_jsons,
            sample_rate=48000,
            duration=2.0
        )
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, 
                                num_workers=8, pin_memory=True, drop_last=True)

        # --- D. 优化器 ---
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, unet.parameters()),
            lr=args.lr,
            weight_decay=1e-2
        )
        noise_scheduler = DDIMScheduler(**diff_config["ddim"]['diffusers'])
        is_v_prediction = diff_config["ddim"].get("v_prediction", False)

        # --- E. Prepare ---
        unet, optimizer, dataloader = accelerator.prepare(unet, optimizer, dataloader)
        loss_func = nn.MSELoss()
        best_loss = float('inf')
        
        # --- 验证 VAE/CLAP 在训练循环中未被修改 ---
        current_vae_sum = sum(p.sum().item() for p in autoencoder.parameters())
        if abs(current_vae_sum - vae_sum) > 1e-4:
            print(f"[WARNING] VAE weights changed! {vae_sum} -> {current_vae_sum}")
        
        # --- F. Training Loop ---
        for epoch in range(args.epochs):
            unet.train()
            epoch_loss = 0.0
            
            pbar = tqdm(dataloader, disable=not accelerator.is_main_process, desc=f"Cluster {cluster_id} Epoch {epoch+1}")
            for step, batch in enumerate(pbar):
                with torch.no_grad():
                    mix_wav = batch['mixture'].unsqueeze(1).to(accelerator.device)
                    target_wav = batch['target'].unsqueeze(1).to(accelerator.device)
                    texts = batch['text']
                    
                    mixture_latent = autoencoder(audio=mix_wav)
                    target_latent = autoencoder(audio=target_wav)
                    
                    text_inputs = clap_processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(accelerator.device)
                    timbre_emb = clap_model.get_text_features(**text_inputs)

                noise = torch.randn_like(target_latent)
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, 
                                          (noise.shape[0],), device=accelerator.device).long()
                
                noisy_target = noise_scheduler.add_noise(target_latent, noise, timesteps)
                noisy_target.requires_grad_(True)
                
                pred = unet(x=noisy_target, timesteps=timesteps, mixture=mixture_latent, timbre=timbre_emb)
                
                if is_v_prediction:
                    target_v = noise_scheduler.get_velocity(target_latent, noise, timesteps)
                    loss = loss_func(pred, target_v)
                else:
                    loss = loss_func(pred, noise)

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(filter(lambda p: p.requires_grad, unet.parameters()), args.grad_clip)
                
                optimizer.step()
                optimizer.zero_grad()
                
                epoch_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})

            # End of Epoch
            avg_loss = epoch_loss / len(dataloader)
            
            # 等待所有进程同步后再保存
            accelerator.wait_for_everyone()
            
            if accelerator.is_main_process:
                unwrapped_model = accelerator.unwrap_model(unet)
                lora_state_dict = {k: v for k, v in unwrapped_model.state_dict().items() if "lora_" in k}

                last_path = os.path.join(cluster_output_dir, "last.pt")
                torch.save(lora_state_dict, last_path)

                if avg_loss < best_loss:
                    best_loss = avg_loss
                    best_path = os.path.join(cluster_output_dir, "best.pt")
                    torch.save(lora_state_dict, best_path)
                    print(f"-> New Best Model saved (Loss: {best_loss:.5f})")

        print(f"Finished Cluster {cluster_id}.\n")

    accelerator.end_training()

if __name__ == "__main__":
    main()