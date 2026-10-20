import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
import random
import numpy as np
import logging
import gc
from torch.utils.data import DataLoader, Dataset
import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger

os.environ["TOKENIZERS_PARALLELISM"] = "false"
from sam_audio import SAMAudio, SAMAudioConfig

torch.set_float32_matmul_precision('high')

# ==========================================
# 1. LoRA 动态注入工具
# ==========================================

class TextLogCallback(Callback):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx % trainer.log_every_n_steps == 0:
            loss = outputs['loss']
            print(f"[Epoch {trainer.current_epoch}] Step {batch_idx}: loss={loss.item():.4f}", flush=True)

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
    if trainable_params > 0:
        print(f"  -> [预估] 保存的 LoRA 权重大小: {(trainable_params * 4) / (1024 * 1024):.2f} MB")
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
        
    # 增加属性伪装，防止核心网络中有其他代码读取 weight 报错
    @property
    def weight(self):
        return self.base_layer.weight

    @property
    def bias(self):
        return self.base_layer.bias

def inject_lora(model, rank=16, alpha=16, dropout=0.1):
    #(Attention + MLP)
    target_modules = [
        'wq', 'wk', 'wv', 'proj',  # 对应原来的 to_q, to_k, to_v, proj (注意力层)
        'w1', 'w2', 'w3'           # 对应原来的 fc1, fc2 (MLP前馈层)
    ]
    
    for param in model.parameters():
        param.requires_grad = False
        
    injected_count = 0
    modules_to_replace = []
    
    for name, module in model.named_modules():
        # 严格跳过 T5 文本编码器
        if "text_encoder" in name:
            continue
            
        if isinstance(module, nn.Linear):
            layer_name = name.split('.')[-1]
            # ⚠️ 注意这里：因为名字很短(比如 w1)，为了防止误杀，
            # 建议改回精确等于 (==) 或者 in 列表，而不是子串匹配
            if layer_name in target_modules:
                if '.' in name:
                    parent_name = name.rsplit('.', 1)[0]
                    parent = model.get_submodule(parent_name)
                else:
                    parent = model
                modules_to_replace.append((parent, layer_name))

    for parent, layer_name in modules_to_replace:
        original = getattr(parent, layer_name)
        if isinstance(original, LoRALinear):
            continue
        lora_layer = LoRALinear(original, rank=rank, alpha=alpha, dropout=dropout)
        setattr(parent, layer_name, lora_layer)
        injected_count += 1
        
    print(f"[LoRA] Successfully injected {injected_count} layers (Text Encoder excluded).")
    return model

# ==========================================
# 2. SegmentMixer
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
        mixtures = []
        targets = []
        for i in range(waveforms.shape[0]):
            wav = waveforms[i]
            mix_wav, target_wav = self.mix_single(wav)
            mixtures.append(mix_wav)
            targets.append(target_wav)
        return torch.stack(mixtures), torch.stack(targets)

# ==========================================
# 3. Dataset
# ==========================================
class ClusterDataset(Dataset):
    def __init__(self, target_json, noise_json_files, sr=48000, max_len_sec=2.0):
        with open(target_json, 'r') as f:
            self.data = json.load(f)['data']
        self.sr = sr
        self.max_len = int(sr * max_len_sec)
        
    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        try:
            wav, sr = torchaudio.load(item['wav'])
            if wav.shape[0] > 1: wav = wav.mean(0, keepdim=True)
            if sr != self.sr: wav = torchaudio.functional.resample(wav, sr, self.sr)
            
            current_len = wav.shape[1]
            target_len = self.max_len
            
            if current_len > target_len:
                start = random.randint(0, current_len - target_len)
                wav = wav[:, start:start+target_len]
            elif current_len < target_len:
                wav = F.pad(wav, (0, target_len - current_len))
            
            return {"waveform": wav, "text": item['caption']}
        except:
            return {"waveform": torch.zeros(1, self.max_len), "text": "error"}

# ==========================================
# 4. Lightning Module
# ==========================================
class SAMAudioLoRATrainer(pl.LightningModule):
    def __init__(self, config, ckpt_path, segment_mixer, lora_rank=8, lora_alpha=8, lr=1e-4):
        super().__init__()
        self.save_hyperparameters(ignore=['segment_mixer'])
        
        self.model = SAMAudio(config)
        
        if hasattr(self.model, 'text_encoder'):
            t5_weight_sum = 0
            for p in self.model.text_encoder.parameters():
                t5_weight_sum += p.data.sum().item()
                break
            print(f"\n[Verify] T5 initialized from config. Weight Checksum: {t5_weight_sum:.4f}")
        
        print("[Checkpoint] Loading checkpoint from:", ckpt_path)
        state_dict = torch.load(ckpt_path, map_location='cpu')
        filtered_state_dict = {k: v for k, v in state_dict.items() if "vision_encoder" not in k}
        
        load_result = self.model.load_state_dict(filtered_state_dict, strict=False)
        print(f"[Checkpoint] Missing keys (Expected for T5): {len(load_result.missing_keys)}")
        
        if hasattr(self.model, 'text_encoder'):
            t5_weight_sum_after = 0
            for p in self.model.text_encoder.parameters():
                t5_weight_sum_after += p.data.sum().item()
                break
            if t5_weight_sum == t5_weight_sum_after:
                print(f"[Verify] T5 weights unchanged after checkpoint load. (Correct behavior)")
            else:
                print(f"[Warning] T5 weights changed! Checksum: {t5_weight_sum_after:.4f}")

        print("="*60 + "\n")
        
        # 注入 LoRA (此时 T5 会被自动跳过)
        self.model = inject_lora(
            self.model, 
            rank=lora_rank, 
            alpha=lora_alpha, 
            dropout=0.1
        )
        print_model_stats(self.model)
        self.segment_mixer = segment_mixer
        self.lr = lr

    def on_train_start(self):
        self.freeze_base_model_bn()

    def freeze_base_model_bn(self):
        frozen_count = 0
        # 1. 冻结核心网络中的 BN/LN
        for name, module in self.model.named_modules():
            if 'lora' in name.lower() or 'text_encoder' in name:
                continue
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, 
                                  nn.SyncBatchNorm, nn.LayerNorm)):
                module.eval()
                for param in module.parameters():
                    param.requires_grad = False
                frozen_count += 1
        
        # 2. ⚠️ 核心修改：全局严格冻结整个 T5 文本编码器
        if hasattr(self.model, 'text_encoder'):
            self.model.text_encoder.eval() # 强制 eval 模式，防止 Dropout 和 BN/LN 漂移
            for param in self.model.text_encoder.parameters():
                param.requires_grad = False # 强制锁死梯度
            print("[Freeze] Text Encoder (T5) is strictly frozen (eval mode & no grad).")
                        
        if frozen_count > 0:
            print(f"[Freeze] Base model: {frozen_count} BN/LN layers frozen (eval mode).")

    def training_step(self, batch, batch_idx):
        clean_wavs = batch['waveform']
        texts = batch['text']
        
        with torch.no_grad():
            mixtures, targets = self.segment_mixer(clean_wavs)
            
            target_latents = self.model.audio_codec(targets).transpose(1, 2)
            target_latents = torch.cat([target_latents, target_latents], dim=2) 
            
            mixture_latents = self.model.audio_codec(mixtures).transpose(1, 2)
            mixture_latents = torch.cat([mixture_latents, mixture_latents], dim=2)
            
            # T5 已经被全局设为 eval()，这里提取特征非常安全
            text_feat, text_mask = self.model.text_encoder(texts)
            masked_video_features = self.model._get_video_features(None, mixture_latents)

        B, T, C = target_latents.shape 
        t = torch.rand(B, device=self.device)
        x_0 = torch.randn_like(target_latents)
        x_1 = target_latents
        
        t_expanded = t.view(B, 1, 1)
        x_t = (1 - t_expanded) * x_0 + t_expanded * x_1
        v_target = x_1 - x_0
        
        v_pred = self.model.forward(
            noisy_audio=x_t,
            audio_features=mixture_latents,
            text_features=text_feat,
            time=t,
            masked_video_features=masked_video_features,
            text_mask=text_mask,
            audio_pad_mask=None
        )
        
        loss = F.mse_loss(v_pred, v_target)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr)

# ==========================================
# 5. Callback
# ==========================================
class LoRASaveCallback(Callback):
    def __init__(self, save_dir):
        super().__init__()
        self.save_dir = save_dir
        self.best_loss = float('inf')

    def _save_lora(self, pl_module, filename):
        model = pl_module.model
        if hasattr(model, 'model'):
            model = model.model
        lora_state_dict = {k: v for k, v in model.state_dict().items() if "lora_" in k}
        
        if len(lora_state_dict) == 0:
            return
        save_path = os.path.join(self.save_dir, filename)
        torch.save(lora_state_dict, save_path)
        print(f"[Save] Saved LoRA params to {filename}")

    def on_train_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        current_loss = metrics.get('train_loss_epoch', metrics.get('train_loss'))

        if current_loss is None: return

        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self._save_lora(pl_module, "best_lora.pt")

        self._save_lora(pl_module, "last_lora.pt")

# ==========================================
# 6. 主程序
# ==========================================
def main():
    # --- 路径配置 ---
    SAM_AUDIO_DIR = "/s7home/zyk524/SAM_Audio/sam-audio-large"
    CONFIG_PATH = os.path.join(SAM_AUDIO_DIR, "config.json")
    CKPT_PATH = os.path.join(SAM_AUDIO_DIR, "checkpoint.pt")
    LOCAL_T5_DIR = "/s7home/zyk524/SAM_Audio/sam-audio-large/huggingface/hub/models--t5-base/snapshots/a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1"
    
    BASE_DATA_DIR = "/s7home/zyk524/test/audio/SED/train_json/89checkpoint/7/train"
    WORKSPACE = "/s7home/zyk524/SAM_Audio/sam-audio-large/7cluster_fix_not5_attention_mlp"

    # --- 关键超参数 ---
    BATCH_SIZE = 32  # 如果显存不够，请调小
    MAX_EPOCHS = 150
    LR = 5e-4
    LORA_RANK = 16
    LORA_ALPHA = 16
    AUDIO_LEN_SEC = 2.0
    CLUSTER_IDS = list(range(0, 1))

    # 加载 Config
    with open(CONFIG_PATH) as f: cfg = json.load(f)
    cfg['text_encoder']['name'] = LOCAL_T5_DIR
    config = SAMAudioConfig(**cfg)

    all_json_files = [os.path.join(BASE_DATA_DIR, f"cluster{i}.json") for i in range(7)]

    for cluster_id in CLUSTER_IDS:
        print("\n" + "="*60)
        print(f"========== Training Cluster {cluster_id} (2s, Rank{LORA_RANK}) ==========")
        print("="*60 + "\n")

        target_json = all_json_files[cluster_id]
        if not os.path.exists(target_json):
            print(f"Skipping Cluster {cluster_id}: File not found.")
            continue

        noise_jsons = [f for i, f in enumerate(all_json_files) if i != cluster_id]

        gc.collect()
        torch.cuda.empty_cache()
        
        segment_mixer = SegmentMixer(
            noise_json_files=noise_jsons,
            max_mix_num=2,
            lower_db=-10, higher_db=10, 
        )
        
        model = SAMAudioLoRATrainer(
            config=config,
            ckpt_path=CKPT_PATH,
            segment_mixer=segment_mixer,
            lora_rank=LORA_RANK,
            lora_alpha=LORA_ALPHA,
            lr=LR
        )
        
        dataset = ClusterDataset(
            target_json=target_json,
            noise_json_files=noise_jsons,
            sr=config.audio_codec.sample_rate,
            max_len_sec=AUDIO_LEN_SEC
        )
        
        dataloader = DataLoader(
            dataset, 
            batch_size=BATCH_SIZE, 
            shuffle=True, 
            num_workers=8, 
            pin_memory=True, 
            drop_last=True
        )
        
        save_dir = os.path.join(WORKSPACE, f"cluster{cluster_id}")
        os.makedirs(save_dir, exist_ok=True)
        
        logger = TensorBoardLogger(save_dir=WORKSPACE, name=f"cluster{cluster_id}")
        lora_saver = LoRASaveCallback(save_dir=save_dir)
        
        trainer = pl.Trainer(
            accelerator="gpu",
            devices=1,
            max_epochs=MAX_EPOCHS,
            precision="32-true",
            enable_progress_bar=False,
            callbacks=[lora_saver, LearningRateMonitor(logging_interval='step'), TextLogCallback()],
            logger=logger,
            gradient_clip_val=1.0,
            log_every_n_steps=5,
            enable_checkpointing=False 
        )
        
        trainer.fit(model, train_dataloaders=dataloader)
        print(f"Cluster {cluster_id} Done.")

if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    os.environ["SAM_AUDIO_TEXT_ONLY"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    
    seed = 42
    pl.seed_everything(seed)
    
    main()
