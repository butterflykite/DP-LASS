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
from data2.waveform_mixers_zidong_7cluster import SegmentMixer
# 引入 SAM-Audio
from sam_audio import SAMAudio, SAMAudioConfig

# ==========================================
# 1. LoRA 动态注入工具
# ==========================================
class LoRALinear(nn.Module):
    def __init__(self, original_linear: nn.Linear, rank: int = 8, alpha: int = 16, dropout: float = 0.1):
        super().__init__()
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.rank = rank
        self.scaling = alpha / rank
        
        # 冻结原始层
        self.base_layer = original_linear
        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False
            
        # LoRA 参数
        self.lora_A = nn.Parameter(torch.zeros(self.in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, self.out_features))
        self.lora_dropout = nn.Dropout(p=dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        base_out = self.base_layer(x)
        lora_out = (self.lora_dropout(x) @ self.lora_A @ self.lora_B) * self.scaling
        return base_out + lora_out

def inject_lora_and_freeze(model, rank=16, alpha=32, dropout=0.1):
    """在运行时将 Linear 替换为 LoRALinear"""
    # 1. 冻结全网
    for param in model.parameters():
        param.requires_grad = False
        
    # 2. 替换 Transformer 中的 Linear
    for name, module in model.named_modules():
        if "transformer.layers" in name and isinstance(module, nn.Linear):
            target_layers = ['wq', 'wk', 'wv', 'wo', 'w1', 'w2', 'w3']
            if any(t in name for t in target_layers):
                parent_name = name.rsplit('.', 1)[0]
                child_name = name.rsplit('.', 1)[1]
                parent = model.get_submodule(parent_name)
                
                original = getattr(parent, child_name)
                lora_layer = LoRALinear(original, rank=rank, alpha=alpha, dropout=dropout)
                setattr(parent, child_name, lora_layer)
                
    # 3. 打印状态
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[LoRA] Injected. Trainable Params: {trainable/1e6:.2f}M / {total/1e6:.2f}M ({trainable/total:.2%})")
    return model

# ==========================================
# 2. SegmentMixer
# ==========================================
class SegmentMixer(nn.Module):
    def __init__(self, max_mix_num, lower_db, higher_db, exclude_cluster_id=None, json_files=None):
        super().__init__()
        self.max_mix_num = max_mix_num
        self.loudness_param = {'lower_db': lower_db, 'higher_db': higher_db}
        self.exclude_cluster_id = exclude_cluster_id
        self.base_json_files = json_files
        self.audio_pool = []
        self._load_audio_paths()

    def _load_audio_paths(self):
        self.audio_pool = []
        for json_file in self.base_json_files:
            if self.exclude_cluster_id is not None and f"cluster{self.exclude_cluster_id}.json" in json_file:
                continue
            if not os.path.exists(json_file): continue
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                for item in data['data']:
                    if 'wav' in item and os.path.exists(item['wav']):
                        self.audio_pool.append(item['wav'])
            except: pass
        print(f"[Mixer] Loaded {len(self.audio_pool)} noise samples (Excluded Cluster {self.exclude_cluster_id})")

    def _load_audio(self, audio_path, device=None):
        try:
            wav, sr = torchaudio.load(audio_path)
            if wav.shape[0] > 1: wav = wav.mean(0, keepdim=True)
            if device: wav = wav.to(device)
            return wav
        except: return None

    def __call__(self, waveforms):
        batch_size = waveforms.shape[0]
        device = waveforms.device
        mixtures = []
        segments = []

        for n in range(batch_size):
            segment = waveforms[n].clone()
            noise = torch.zeros_like(segment)
            mix_num = random.randint(2, self.max_mix_num)

            for _ in range(1, mix_num):
                if not self.audio_pool: break
                for _ in range(10): 
                    noise_path = random.choice(self.audio_pool)
                    noise_wav = self._load_audio(noise_path, device)
                    if noise_wav is not None:
                        if noise_wav.shape[1] < segment.shape[1]:
                            repeats = (segment.shape[1] // noise_wav.shape[1]) + 1
                            noise_wav = noise_wav.repeat(1, repeats)
                        noise_wav = noise_wav[:, :segment.shape[1]]
                        noise += self.dynamic_loudnorm(noise_wav, segment)
                        break
            
            mixture = segment + self.dynamic_loudnorm(noise, segment)
            max_val = torch.max(torch.abs(mixture))
            if max_val > 1:
                scale = 0.9 / max_val
                segment *= scale
                mixture *= scale
            
            mixtures.append(mixture)
            segments.append(segment)

        return torch.stack(mixtures), torch.stack(segments)

    def get_energy(self, x): return torch.mean(x ** 2)
    
    def dynamic_loudnorm(self, audio, reference):
        e1 = self.get_energy(audio)
        e2 = max(self.get_energy(reference), 1e-10)
        ratio = (e1 / e2) ** 0.5
        ratio = torch.clamp(ratio, 0.02, 50)
        audio = audio / ratio
        gain = 10 ** (random.uniform(self.loudness_param['lower_db'], self.loudness_param['higher_db']) / 20.0)
        return audio * gain

# ==========================================
# 3. Dataset & DataModule
# ==========================================
class ClusterDataset(Dataset):
    def __init__(self, json_file, sr=48000, max_len=5):
        with open(json_file, 'r') as f:
            self.data = json.load(f)['data']
        self.sr = sr
        self.max_len = int(sr * max_len)

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        try:
            wav, sr = torchaudio.load(item['wav'])
            if wav.shape[0] > 1: wav = wav.mean(0, keepdim=True)
            if sr != self.sr: wav = torchaudio.functional.resample(wav, sr, self.sr)
            
            if wav.shape[1] > self.max_len:
                start = random.randint(0, wav.shape[1] - self.max_len)
                wav = wav[:, start:start+self.max_len]
            else:
                wav = F.pad(wav, (0, self.max_len - wav.shape[1]))
            return {"waveform": wav, "text": item['caption']}
        except:
            return {"waveform": torch.zeros(1, self.max_len), "text": "error"}

class ClusterDataModule(pl.LightningDataModule):
    def __init__(self, train_json, batch_size=4, num_workers=4, sr=48000):
        super().__init__()
        self.train_json = train_json
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.sr = sr

    def train_dataloader(self):
        ds = ClusterDataset(self.train_json, sr=self.sr)
        return DataLoader(ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, drop_last=True, pin_memory=True)

# ==========================================
# 4. Lightning Module (Flow Matching)
# ==========================================
class SAMAudioLoRATrainer(pl.LightningModule):
    def __init__(self, config, ckpt_path, segment_mixer, lora_rank=16, lr=1e-4):
        super().__init__()
        self.save_hyperparameters(ignore=['segment_mixer'])
        
        self.model = SAMAudio(config)
        state_dict = torch.load(ckpt_path, map_location='cpu')
        self.model.load_state_dict(state_dict, strict=False)
        
        self.model = inject_lora_and_freeze(self.model, rank=lora_rank)
        
        self.segment_mixer = segment_mixer
        self.lr = lr

    def training_step(self, batch, batch_idx):
        clean_wavs = batch['waveform']
        texts = batch['text']
        
        with torch.no_grad():
            mixtures, targets = self.segment_mixer(clean_wavs)
            
            # 1. 编码 Target (x_1)
            target_latents = self.model.audio_codec(targets).transpose(1, 2)
            # 【关键修正】Target 必须是双倍通道，以匹配 in_channels=768
            target_latents = torch.cat([target_latents, target_latents], dim=2) # [B, T, 256]
            
            # 2. 编码 Mixture (Condition)
            mixture_latents = self.model.audio_codec(mixtures).transpose(1, 2)
            mixture_latents = torch.cat([mixture_latents, mixture_latents], dim=2) # [B, T, 256]
            
            # 3. 编码 Text
            text_feat, text_mask = self.model.text_encoder(texts)
            
            # 4. Video (None)
            masked_video_features = self.model._get_video_features(None, mixture_latents)

        # 3. Flow Matching
        B, T, C = target_latents.shape # C 应该是 256
        t = torch.rand(B, device=self.device)
        x_0 = torch.randn_like(target_latents)
        x_1 = target_latents
        
        t_expanded = t.view(B, 1, 1)
        x_t = (1 - t_expanded) * x_0 + t_expanded * x_1
        v_target = x_1 - x_0
        
        # 4. 预测
        # 输入维度检查:
        # noisy_audio (x_t): 256
        # audio_features (mixture): 256
        # zeros: 256
        # Total: 768 -> 匹配 config.in_channels!
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
# 5. Callback: 保存 Best 和 Last
# ==========================================
class LoRASaveCallback(Callback):
    def __init__(self, save_dir):
        super().__init__()
        self.save_dir = save_dir
        self.best_loss = float('inf')

    def _save_lora(self, pl_module, filename):
        lora_state_dict = {k: v for k, v in pl_module.model.state_dict().items() if "lora_" in k}
        save_path = os.path.join(self.save_dir, filename)
        torch.save(lora_state_dict, save_path)

    def on_train_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        current_loss = metrics.get('train_loss_epoch', metrics.get('train_loss'))

        if current_loss is None: return

        if current_loss < self.best_loss:
            self.best_loss = current_loss
            print(f"\n[Checkpoint] New best loss: {self.best_loss:.4f}. Saving best_lora.pt...")
            self._save_lora(pl_module, "best_lora.pt")

        self._save_lora(pl_module, "last_lora.pt")

# ==========================================
# 6. 主程序
# ==========================================
def main():
    # --- 配置 ---
    SAM_AUDIO_DIR = "/s7home/zyk524/SAM_Audio/sam-audio-large"
    CONFIG_PATH = os.path.join(SAM_AUDIO_DIR, "config.json")
    CKPT_PATH = os.path.join(SAM_AUDIO_DIR, "checkpoint.pt")
    LOCAL_T5_DIR = "/s7home/zyk524/SAM_Audio/sam-audio-large/huggingface/hub/models--t5-base/snapshots/a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1"
    
    BASE_DATA_DIR = "/s7home/zyk524/MSCLAP/AudioSet_Auido/xianfenge/danyi_eval/lora"
    WORKSPACE = "/s7home/zyk524/SAM_Audio/sam-audio-large/7cluster"

    # 训练参数
    BATCH_SIZE = 4
    MAX_EPOCHS = 150
    LR = 1e-4
    LORA_RANK = 16

    # 待训练的 Cluster 列表
    CLUSTER_IDS = list(range(0, 7))

    # --- 1. 加载 Config ---
    with open(CONFIG_PATH) as f: cfg = json.load(f)
    cfg['text_encoder']['name'] = LOCAL_T5_DIR
    config = SAMAudioConfig(**cfg)

    # 获取所有 JSON 文件列表
    all_json_files = [os.path.join(BASE_DATA_DIR, f"cluster{i}.json") for i in range(7)]

    for cluster_id in CLUSTER_IDS:
        print("\n" + "="*60)
        print(f"========== 开始训练 Cluster {cluster_id} ==========")
        print("="*60 + "\n")
        
        gc.collect()
        torch.cuda.empty_cache()
        
        segment_mixer = SegmentMixer(
            max_mix_num=2, 
            lower_db=-10, higher_db=10, 
            exclude_cluster_id=cluster_id,
            json_files=all_json_files
        )
        
        model = SAMAudioLoRATrainer(
            config=config,
            ckpt_path=CKPT_PATH,
            segment_mixer=segment_mixer,
            lora_rank=LORA_RANK,
            lr=LR
        )
        
        train_json = os.path.join(BASE_DATA_DIR, f"cluster{cluster_id}.json")
        if not os.path.exists(train_json):
            print(f"Skipping Cluster {cluster_id}: JSON not found.")
            continue
            
        dm = ClusterDataModule(train_json, batch_size=BATCH_SIZE, sr=config.audio_codec.sample_rate)
        
        save_dir = os.path.join(WORKSPACE, f"cluster{cluster_id}")
        os.makedirs(save_dir, exist_ok=True)
        
        logger = TensorBoardLogger(save_dir=WORKSPACE, name=f"cluster{cluster_id}")
        lora_saver = LoRASaveCallback(save_dir=save_dir)
        
        trainer = pl.Trainer(
            accelerator="gpu",
            devices=1,
            max_epochs=MAX_EPOCHS,
            precision="16-mixed",
            callbacks=[lora_saver, LearningRateMonitor(logging_interval='step')],
            logger=logger,
            gradient_clip_val=1.0,
            log_every_n_steps=10,
            enable_checkpointing=False 
        )
        
        trainer.fit(model, datamodule=dm)
        
        print(f"Cluster {cluster_id} 训练完成。")

if __name__ == "__main__":
    os.environ["SAM_AUDIO_TEXT_ONLY"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    main()