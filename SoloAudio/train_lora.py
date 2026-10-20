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
torch.backends.cudnn.benchmark = False
# 甚至可以强制禁用 cudnn（虽然会慢，但能验证问题）
torch.backends.cudnn.enabled = False
# 假设这些模块在你的路径下
from model.udit import UDiT
from vae_modules.autoencoder_wrapper import Autoencoder

# ==========================================
# 1. LoRA 模块与注入工具 (保持不变)
# ==========================================


def print_model_stats(model):
    """
    统计并打印模型的参数量：原始参数 vs LoRA参数
    """
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
    # print(f"[LoRA] Injecting targets: {target_modules} | Rank: {rank}")
    
    for param in model.parameters():
        param.requires_grad = False
        
    injected_count = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            layer_name = name.split('.')[-1]
            if layer_name in target_modules:
                parent_name = name.rsplit('.', 1)[0]
                parent = model.get_submodule(parent_name)
                
                original = getattr(parent, layer_name)
                lora_layer = LoRALinear(original, rank=rank, alpha=alpha, dropout=dropout)
                setattr(parent, layer_name, lora_layer)
                injected_count += 1
    return model

# ==========================================
# 2. 改进后的动态混合器 (支持 Target vs Others)
# ==========================================
class SegmentMixer(nn.Module):
    def __init__(self, noise_json_files, max_mix_num=2, lower_db=-10, higher_db=10, target_sr=24000, target_len_sec=2.0):
        super().__init__()
        self.max_mix_num = max_mix_num
        self.loudness_range = (lower_db, higher_db) 
        self.target_sr = target_sr
        self.target_len = int(target_sr * target_len_sec)
        
        self.audio_pool = []
        # 直接加载传入的文件列表，不再进行 exclude 判断
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
            # 统一重采样
            if sr != self.target_sr:
                wav = torchaudio.functional.resample(wav, sr, self.target_sr)
            # 统一单声道
            if wav.shape[0] > 1: 
                wav = wav.mean(0, keepdim=True)
            
            # 统一长度 (Padding/Crop)
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

    def mix(self, clean_wav):
        mixture = clean_wav.clone()
        if not self.audio_pool:
            return mixture, clean_wav

        num_noise = random.randint(1, max(1, self.max_mix_num - 1))
        
        for _ in range(num_noise):
            noise_path = random.choice(self.audio_pool)
            noise_wav = self._load_audio(noise_path)
            scaled_noise = self.dynamic_loudnorm(noise_wav, clean_wav)
            mixture = mixture + scaled_noise
            
        max_val = mixture.abs().max()
        if max_val > 0.95:
            scale = 0.95 / max_val
            mixture = mixture * scale
            clean_wav = clean_wav * scale 
            
        return mixture, clean_wav

class ClusterDataset(Dataset):
    def __init__(self, target_json, noise_json_files, sample_rate=24000, duration=2.0):
        # 加载目标 Target 数据
        with open(target_json, 'r') as f:
            self.data = json.load(f)['data']
        
        # 初始化 Mixer，传入噪声文件列表
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
        mixture_wav, target_wav = self.mixer.mix(clean_wav)
        
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
    # 路径配置
    parser.add_argument('--base-data-dir', type=str, 
                        default='/s7home/zyk524/test/audio/SED/train_json/89checkpoint/7/train',
                        help="Directory containing clusterX.json files")
    parser.add_argument('--config', type=str, default='/s7home/zyk524/SoloAudio/SoloAudio-main/config/SoloAudio.yaml')
    parser.add_argument('--udit-ckpt', type=str,default='/s7home/zyk524/SoloAudio/SoloAudio-main/pretrained_models/soloaudio_v2.pt', help="Pretrained UDiT checkpoint")
    parser.add_argument('--vae-path', type=str, default='/s7home/zyk524/SoloAudio/SoloAudio-main/pretrained_models/audio-vae.pt')
    parser.add_argument('--clap-path', type=str, default='/s7home/zyk524/SoloAudio/SoloAudio-main/clap')
    parser.add_argument('--output-base-dir', type=str, default='/s7home/zyk524/SoloAudio/SoloAudio-main/LORA_chongxu')

    # 训练超参数
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=5e-4) # LoRA 建议 LR 略大
    parser.add_argument('--precision', type=str, default='bf16', choices=['no', 'fp16', 'bf16'])
    parser.add_argument('--grad-clip', type=float, default=1.0)
    parser.add_argument('--save-every', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    
    # LoRA 参数
    parser.add_argument('--lora-rank', type=int, default=16)
    parser.add_argument('--lora-alpha', type=int, default=16)
    
    return parser.parse_args()

# ==========================================
# 4. 主循环流程
# ==========================================
def main():
    args = parse_args()
    
    # 全局固定种子
    set_seed(args.seed)

    # 0. 初始化 Accelerator (只初始化一次)
    if args.precision == 'no':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    accelerator = Accelerator(mixed_precision=args.precision, log_with="tensorboard", project_dir=args.output_base_dir)
    
    # 1. 加载 Config
    with open(args.config, 'r') as fp:
        diff_config = yaml.safe_load(fp)

    # 2. 准备 VAE 和 CLAP (全局冻结，加载一次，节省时间)
    # 注意：在 Accelerator 环境下，最好手动移到 device 或者让 accelerator prepare (但不优化)
    # 这里我们手动移到 device 并设为 eval
    autoencoder = Autoencoder(args.vae_path, 'stable_vae', quantization_first=True)
    autoencoder.eval()
    autoencoder.requires_grad_(False)
    autoencoder.to(accelerator.device)

    clap_model = ClapModel.from_pretrained(args.clap_path).to(accelerator.device)
    clap_processor = AutoProcessor.from_pretrained(args.clap_path)
    clap_model.eval()
    clap_model.requires_grad_(False)

    print("Loaded Aux Models (VAE, CLAP).")

    # 3. 准备文件列表
    # 假设文件名为 cluster0.json ... cluster6.json
    all_json_files = [os.path.join(args.base_data_dir, f"cluster{i}.json") for i in range(7)]
    CLUSTER_IDS = list(range(5,7))

    # ================= 循环训练 7 个 Cluster =================
    for cluster_id in CLUSTER_IDS:
        target_json = all_json_files[cluster_id]
        if not os.path.exists(target_json):
            print(f"Skipping Cluster {cluster_id}: File not found.")
            continue

        # 构造噪声文件列表：除了当前 cluster_id 之外的所有文件
        noise_jsons = [f for i, f in enumerate(all_json_files) if i != cluster_id]

        cluster_output_dir = os.path.join(args.output_base_dir, f"cluster{cluster_id}")
        if accelerator.is_main_process:
            os.makedirs(cluster_output_dir, exist_ok=True)
            print("\n" + "="*60)
            print(f"Training Cluster {cluster_id} | Target: {os.path.basename(target_json)}")
            print(f"Noise Sources: {[os.path.basename(f) for f in noise_jsons]}")
            print(f"Output Dir: {cluster_output_dir}")
            print("="*60 + "\n")

        # --- 显存清理 ---
        gc.collect()
        torch.cuda.empty_cache()
        accelerator.free_memory() # 如果 accelerator 版本支持

        # --- A. 初始化 UDiT (Fresh Model) ---
        unet = UDiT(**diff_config['diffwrap']['UDiT'])
        # 加载 Base Weights
        ckpt = torch.load(args.udit_ckpt, map_location='cpu')
        if 'model' in ckpt: ckpt = ckpt['model']
        unet.load_state_dict(ckpt, strict=False)
        
        # --- B. 注入 LoRA ---
        unet = inject_lora(unet, rank=args.lora_rank, alpha=args.lora_alpha)
        unet.to(accelerator.device) # 移至 GPU
        unet.train()
        if accelerator.is_main_process:
            print_model_stats(unet)
        # --- C. 数据集 ---
        dataset = ClusterDataset(
            target_json=target_json,
            noise_json_files=noise_jsons, # 传入排除后的列表
            sample_rate=24000,
            duration=2.0
        )
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, 
                                num_workers=8, pin_memory=True, drop_last=True)

        # --- D. 优化器 & Scheduler ---
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, unet.parameters()),
            lr=args.lr,
            weight_decay=1e-2
        )
        noise_scheduler = DDIMScheduler(**diff_config["ddim"]['diffusers'])
        is_v_prediction = diff_config["ddim"].get("v_prediction", False)

        # --- E. Prepare with Accelerator ---
        # 注意：每次循环都需要 prepare 新的模型和优化器
        unet, optimizer, dataloader = accelerator.prepare(unet, optimizer, dataloader)
        loss_func = nn.MSELoss()
        best_loss = float('inf')
        # --- F. Training Loop ---
        for epoch in range(args.epochs):
            unet.train()
            epoch_loss = 0.0
            
            # 使用 tqdm 显示进度
            pbar = tqdm(dataloader, disable=not accelerator.is_main_process, desc=f"Cluster {cluster_id} Epoch {epoch+1}")
            for step, batch in enumerate(pbar):
                with torch.no_grad():
                    # 数据准备 (Raw -> Latent)
                    mix_wav = batch['mixture'].unsqueeze(1).to(accelerator.device)
                    target_wav = batch['target'].unsqueeze(1).to(accelerator.device)
                    texts = batch['text']
                    
                    mixture_latent = autoencoder(audio=mix_wav)
                    target_latent = autoencoder(audio=target_wav)
                    
                    text_inputs = clap_processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(accelerator.device)
                    timbre_emb = clap_model.get_text_features(**text_inputs)

                # Diffusion
                noise = torch.randn_like(target_latent)
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, 
                                          (noise.shape[0],), device=accelerator.device).long()
                
                noisy_target = noise_scheduler.add_noise(target_latent, noise, timesteps)
                noisy_target.requires_grad_(True)
                # Forward
                pred = unet(x=noisy_target, timesteps=timesteps, mixture=mixture_latent, timbre=timbre_emb)
                
                # Loss
                if is_v_prediction:
                    target_v = noise_scheduler.get_velocity(target_latent, noise, timesteps)
                    loss = loss_func(pred, target_v)
                else:
                    loss = loss_func(pred, noise)

                # Backward
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(filter(lambda p: p.requires_grad, unet.parameters()), args.grad_clip)
                
                optimizer.step()
                optimizer.zero_grad()
                
                epoch_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})

            # End of Epoch
            avg_loss = epoch_loss / len(dataloader)
            if accelerator.is_main_process:
                unwrapped_model = accelerator.unwrap_model(unet)
                lora_state_dict = {k: v for k, v in unwrapped_model.state_dict().items() if "lora_" in k}

                # 1. 保存 Last
                last_path = os.path.join(cluster_output_dir, "last.pt")
                torch.save(lora_state_dict, last_path)

                # 2. 保存 Best
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    best_path = os.path.join(cluster_output_dir, "best.pt")
                    torch.save(lora_state_dict, best_path)
                    print(f"-> New Best Model saved (Loss: {best_loss:.5f})")

        print(f"Finished Cluster {cluster_id}.\n")

    accelerator.end_training()

if __name__ == "__main__":
    main()
