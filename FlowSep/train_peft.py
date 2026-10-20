import sys
import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import json
import random
import logging
import gc
import types
import warnings
import numpy as np
from typing import List

from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import Trainer, seed_everything, LightningDataModule, Callback
from pytorch_lightning.callbacks import ModelCheckpoint

# ==============================================================================
# 0. 基础设置 & 关键修复
# ==============================================================================
torch.set_float32_matmul_precision('high') # FP32 精度
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

sys.path.append("/s7home/zyk524/FLOWSEP/FlowSep-main/src")
from latent_diffusion.util import instantiate_from_config

# --- 【核心修复1】全局禁用 Checkpoint ---
import latent_diffusion.modules.diffusionmodules.util as diffusion_util
def dummy_checkpoint(func, inputs, params, flag):
    return func(*inputs)
diffusion_util.checkpoint = dummy_checkpoint
print("[System] Globally disabled gradient checkpointing.")

# --- 【核心修复2】Monkey Patch AttentionBlock (双重保险) ---
from latent_diffusion.modules.diffusionmodules.openaimodel import AttentionBlock
def custom_attention_forward(self, x):
    return self._forward(x)
AttentionBlock.forward = custom_attention_forward
# ----------------------------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger('fsspec').setLevel(logging.ERROR)

# ==============================================================================
# 1. Parallel Adapter (仅 Conv2d)
# ==============================================================================
class ParallelAdapter(nn.Module):
    def __init__(self, original_layer, alpha=1.0):
        super().__init__()
        self.original_layer = original_layer
        self.alpha = alpha
        
        for param in self.original_layer.parameters():
            param.requires_grad = False

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

def inject_adapters_conv2d_only(model, target_blocks: List[str], alpha=1.0):
    injected_count = 0
    for name, module in model.named_modules():
        should_inject = False
        for target in target_blocks:
            if target in name:
                should_inject = True
                break
        if not should_inject: continue

        if isinstance(module, nn.Conv2d) and not isinstance(module, ParallelAdapter):
            if '.' in name:
                parent_name, child_name = name.rsplit('.', 1)
                parent = model
                for part in parent_name.split('.'):
                    parent = getattr(parent, part)
            else:
                parent = model
                child_name = name
            
            wrapped_layer = ParallelAdapter(module, alpha=alpha)
            setattr(parent, child_name, wrapped_layer)
            injected_count += 1
            
    return injected_count

# ==============================================================================
# 2. 数据集 (SNR 混合 + 2.56s Padding)
# ==============================================================================
class SNRClusterDataset(Dataset):
    def __init__(self, target_json_path, noise_json_paths, sr=16000, max_len_sec=2.56):
        self.sr = sr
        self.max_len = int(sr * max_len_sec)
        
        with open(target_json_path, 'r') as f:
            self.target_data = json.load(f)['data']
            
        self.noise_pool = []
        for json_file in noise_json_paths:
            if not os.path.exists(json_file): continue
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                for item in data['data']:
                    if 'wav' in item and os.path.exists(item['wav']):
                        self.noise_pool.append(item['wav'])
            except: pass

    def __len__(self):
        return len(self.target_data)

    def _load_and_process_wav(self, path):
        try:
            wav, sr = torchaudio.load(path)
            if wav.shape[0] > 1: wav = wav.mean(0, keepdim=True)
            if sr != self.sr: wav = torchaudio.functional.resample(wav, sr, self.sr)
            
            current_len = wav.shape[1]
            if current_len > self.max_len:
                start = random.randint(0, current_len - self.max_len)
                wav = wav[:, start:start+self.max_len]
            elif current_len < self.max_len:
                wav = F.pad(wav, (0, self.max_len - current_len))
            
            wav = wav - torch.mean(wav)
            max_val = torch.max(torch.abs(wav))
            if max_val > 0: wav = wav / (max_val + 1e-8)
            wav = wav * 0.5 
            return wav
        except:
            return None

    def _mix_snr(self, clean_wav, noise_wav):
        min_len = min(clean_wav.shape[1], noise_wav.shape[1])
        clean_wav = clean_wav[:, :min_len]
        noise_wav = noise_wav[:, :min_len]

        clean_power = torch.mean(clean_wav ** 2)
        noise_power = torch.mean(noise_wav ** 2)
        snr = torch.randint(-15, 15, (1,)).item()
        
        if noise_power > 0:
            desired_noise_power = clean_power / (10 ** (snr / 10))
            scaling_factor = torch.sqrt(desired_noise_power / (noise_power + 1e-8))
        else:
            scaling_factor = 0.0

        noise_wav = noise_wav * scaling_factor
        mixed_wav = clean_wav + noise_wav

        max_val = torch.max(torch.abs(mixed_wav))
        if max_val > 1: mixed_wav *= 0.9 / max_val
        return mixed_wav

    def __getitem__(self, idx):
        item = self.target_data[idx]
        clean_wav = self._load_and_process_wav(item['wav'])
        if clean_wav is None: return self.__getitem__(random.randint(0, len(self.target_data)-1))

        noise_wav = None
        for _ in range(5):
            if not self.noise_pool: break
            noise_path = random.choice(self.noise_pool)
            if noise_path == item['wav']: continue 
            noise_wav = self._load_and_process_wav(noise_path)
            if noise_wav is not None: break
        
        if noise_wav is not None:
            mixed_wav = self._mix_snr(clean_wav, noise_wav)
        else:
            mixed_wav = clean_wav.clone()

        return {
            "clean_wav": clean_wav,
            "mixed_wav": mixed_wav,
            "text": [item['caption']],
            "fname": item['wav']
        }

class LASSDataModule(LightningDataModule):
    def __init__(self, target_json, noise_json_files, batch_size, num_workers, sr=16000, max_len_sec=2.56):
        super().__init__()
        self.target_json = target_json
        self.noise_json_files = noise_json_files
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.sr = sr
        self.max_len_sec = max_len_sec
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr, n_fft=1024, win_length=1024, hop_length=160, 
            f_min=0, f_max=8000, n_mels=64, power=1.0, normalized=False
        )

    def setup(self, stage=None):
        self.train_dataset = SNRClusterDataset(self.target_json, self.noise_json_files, sr=self.sr, max_len_sec=self.max_len_sec)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, 
                          num_workers=self.num_workers, collate_fn=self.collate_fn, pin_memory=True)

    def collate_fn(self, batch):
        clean_wavs = torch.stack([item['clean_wav'] for item in batch])
        mixed_wavs = torch.stack([item['mixed_wav'] for item in batch])
        texts = [item['text'][0] for item in batch]
        fnames = [item['fname'] for item in batch]
        
        clean_mels = self.mel_transform(clean_wavs)
        mixed_mels = self.mel_transform(mixed_wavs)

        target_len = 256
        current_len = clean_mels.shape[-1]
        pad_amount = target_len - current_len
        
        if pad_amount > 0:
            clean_mels = F.pad(clean_mels, (0, pad_amount))
            mixed_mels = F.pad(mixed_mels, (0, pad_amount))
        elif pad_amount < 0:
            clean_mels = clean_mels[..., :target_len]
            mixed_mels = mixed_mels[..., :target_len]

        clean_log_mels = torch.log(torch.clamp(clean_mels, min=1e-5))
        mixed_log_mels = torch.log(torch.clamp(mixed_mels, min=1e-5))
        
        return {
            "fbank": clean_log_mels,         
            "mixed_mel": mixed_log_mels,     
            "caption": texts,                
            "text": texts,                   
            "waveform": clean_wavs,          
            "fname": fnames,                 
            "label_vector": torch.zeros(clean_wavs.shape[0], 1), 
            "stft": torch.zeros(clean_wavs.shape[0], 1)          
        }

# ==============================================================================
# 3. Callback
# ==============================================================================
class PeftSaverCallback(Callback):
    def __init__(self, save_dir, cluster_id):
        super().__init__()
        self.save_dir = save_dir
        self.cluster_id = cluster_id
        self.best_loss = float('inf')
        
    def _save_adapter(self, pl_module, filename):
        adapter_dict = {k: v for k, v in pl_module.state_dict().items() if "adapter" in k}
        save_path = os.path.join(self.save_dir, filename)
        torch.save(adapter_dict, save_path)
        print(f"[PeftSaver] Saved {filename}")

    def on_train_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        current_loss = metrics.get('train/loss') or metrics.get('loss') or metrics.get('train_loss')
        if current_loss is not None and current_loss < self.best_loss:
            self.best_loss = current_loss
            self._save_adapter(pl_module, f"cluster{self.cluster_id}_best.pt")

    def on_fit_end(self, trainer, pl_module):
        self._save_adapter(pl_module, f"cluster{self.cluster_id}_last.pt")

# ==============================================================================
# 4. 训练逻辑
# ==============================================================================
def train_one_cluster(cluster_id, args, config, all_json_files):
    seed_everything(args.seed)
    print(f"\n{'='*40}")
    print(f"STARTING TRAINING FOR CLUSTER {cluster_id}")
    print(f"{'='*40}\n")

    if args.vae_ckpt:
        config['model']['params']['first_stage_config']['params']['reload_from_ckpt'] = args.vae_ckpt

    # 强制关闭 Checkpoint
    if 'unet_config' in config['model']['params']:
        config['model']['params']['unet_config']['params']['use_checkpoint'] = False

    print("Loading Base Model...")
    model = instantiate_from_config(config["model"])
    
    # ==========================================================================
    # 【关键修复】强制关闭 EMA
    # ==========================================================================
    model.use_ema = False
    if hasattr(model, 'model_ema'):
        del model.model_ema
        model.model_ema = None
    print("[System] Disabled EMA for PEFT training.")
    # ==========================================================================

    model.set_log_dir(args.workspace, "peft_training", f"cluster_{cluster_id}")

    # 优化器
    def configure_optimizers_override(self):
        lr = 5e-5 
        print(f"[Optimizer] Configuring AdamW with lr={lr} for ADAPTER parameters only.")
        params = [p for p in self.parameters() if p.requires_grad]
        return torch.optim.AdamW(params, lr=lr, weight_decay=0.01)

    model.configure_optimizers = types.MethodType(configure_optimizers_override, model)
    
    # 自定义 get_input
    def custom_get_input(self, batch, k, return_first_stage_encode=True, return_decoding_output=False, return_encoder_input=False, return_encoder_output=False, unconditional_prob_cfg=0.1):
        x = batch[k].to(self.device)

        if return_first_stage_encode:
            encoder_posterior = self.encode_first_stage(x)
            if isinstance(encoder_posterior, torch.Tensor):
                z = encoder_posterior
            else:
                z = self.get_first_stage_encoding(encoder_posterior).detach()

            z.requires_grad = True 

            if self.extra_channels: 
                extra = batch[self.extra_channel_key].to(self.device)
                if len(extra.shape) == 3:
                    extra = extra.unsqueeze(1)
                
                extra_posterior = self.encode_first_stage(extra)
                e = self.get_first_stage_encoding(extra_posterior).detach()
                e.requires_grad = True 
                z = torch.cat([z,e],dim = 1)
        else:
            z = None

        cond_dict = {}
        if(len(self.cond_stage_model_metadata.keys()) > 0):
            unconditional_cfg = False 
            for cond_model_key in self.cond_stage_model_metadata.keys():
                cond_stage_key = self.cond_stage_model_metadata[cond_model_key]["cond_stage_key"]
                if cond_stage_key != "all":
                    xc = batch[cond_stage_key] 
                else:
                    xc = batch
                c = self.get_learned_conditioning(xc, key=cond_model_key, unconditional_cfg=unconditional_cfg)
                if(isinstance(c, dict)):
                    for k_sub in c.keys(): cond_dict[k_sub] = c[k_sub]
                else:
                    cond_dict[cond_model_key] = c

        out = [z, cond_dict]
        return out

    model.get_input = types.MethodType(custom_get_input, model)

    # 加载权重
    if args.load_checkpoint:
        print(f"Loading LDM weights from {args.load_checkpoint}")
        ckpt = torch.load(args.load_checkpoint, map_location="cpu")
        if "state_dict" in ckpt: ckpt = ckpt["state_dict"]
        model.load_state_dict(ckpt, strict=False)
    
    # 注入 Adapter
    target_blocks = ["input_blocks", "middle_block", "output_blocks"]
    
    if hasattr(model, "model") and hasattr(model.model, "diffusion_model"):
        unet = model.model.diffusion_model
    else:
        unet = model
    
    count = inject_adapters_conv2d_only(unet, target_blocks=target_blocks, alpha=1.0)
    print(f"Injected {count} Conv2d adapters.")

    # 冻结参数
    trainable_params = 0
    all_params = 0
    for name, param in model.named_parameters():
        all_params += param.numel()
        if "adapter" in name:
            param.requires_grad = True
            trainable_params += param.numel()
        else:
            param.requires_grad = False
    print(f"Total Params: {all_params:,}")
    print(f"Trainable Params (Adapters): {trainable_params:,} ({trainable_params/all_params:.2%})")

    # 数据
    target_json = all_json_files[cluster_id]
    noise_json_files = [f for i, f in enumerate(all_json_files) if i != cluster_id]
    
    datamodule = LASSDataModule(
        target_json=target_json,
        noise_json_files=noise_json_files,
        batch_size=args.batch_size,
        num_workers=2,
        max_len_sec=2.56 
    )

    cluster_workspace = os.path.join(args.workspace, f"cluster_{cluster_id}")
    os.makedirs(cluster_workspace, exist_ok=True)

    peft_saver = PeftSaverCallback(save_dir=cluster_workspace, cluster_id=cluster_id)

    trainer = Trainer(
        accelerator="gpu",
        devices=1,         
        strategy="auto",   
        max_epochs=args.max_epochs,
        callbacks=[peft_saver], 
        log_every_n_steps=10,
        precision="32-true", # FP32
        default_root_dir=cluster_workspace,
        enable_checkpointing=False 
    )

    trainer.fit(model, datamodule)
    
    del model, trainer, datamodule
    gc.collect()
    torch.cuda.empty_cache()
    print(f"Finished Cluster {cluster_id}. Memory cleared.")

def main(args):
    config = yaml.load(open(args.config_yaml, "r"), Loader=yaml.FullLoader)
    base_json_dir = args.train_json_dir
    all_json_files = [os.path.join(base_json_dir, f"cluster{i}.json") for i in range(7)]
    
    for f in all_json_files:
        if not os.path.exists(f):
            print(f"Warning: File not found {f}")

    for cluster_id in range(0,7):
        train_one_cluster(cluster_id, args, config, all_json_files)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, default="/s7home/zyk524/FLOWSEP/FlowSep-main/lass_config/2channel_flow.yaml")
    parser.add_argument("--load_checkpoint", type=str, default="/s7home/zyk524/FLOWSEP/FlowSep-main/model_logs/v2_100k.ckpt")
    parser.add_argument("--vae_ckpt", type=str, default="/s7home/zyk524/FLOWSEP/FlowSep-main/model_logs/vae.ckpt")
    parser.add_argument("--train_json_dir", type=str, default="/s7home/zyk524/test/audio/SED/train_json/89checkpoint/7/train")
    parser.add_argument("--workspace", type=str, default="/s7home/zyk524/FLOWSEP/FlowSep-main/peft")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)
