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
# 假设 data2 和 sam_audio 都在 python path 下，或者你根据实际路径调整 import
from data2.waveform_mixers_zidong_7cluster import SegmentMixer
from sam_audio import SAMAudio, SAMAudioConfig
torch.set_float32_matmul_precision('high')
# ==========================================
# 1. LoRA 动态注入工具 (Target Module 优化版)
# ==========================================

class TextLogCallback(Callback):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx % trainer.log_every_n_steps == 0:
            loss = outputs['loss']
            print(f"[Epoch {trainer.current_epoch}] Step {batch_idx}: loss={loss.item():.4f}", flush=True)

class LoRALinear(nn.Module):
    def __init__(self, original_linear: nn.Linear, rank: int = 8, alpha: int = 8, dropout: float = 0.1):
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
        # A 使用 Kaiming 初始化，B 使用 0 初始化
        # 这样初始状态下 LoRA 输出为 0，模型行为与预训练模型完全一致
        self.lora_A = nn.Parameter(torch.zeros(self.in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, self.out_features))
        self.lora_dropout = nn.Dropout(p=dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        base_out = self.base_layer(x)
        # 显式转换类型，防止混合精度训练时出错
        lora_out = (self.lora_dropout(x) @ self.lora_A @ self.lora_B) * self.scaling
        return base_out + lora_out

def inject_lora_and_freeze(model, rank=8, alpha=8, dropout=0.05, target_modules=['wq', 'wv']):
    """
    只对 Transformer 中的 target_modules 进行 LoRA 注入
    """
    print(f"[LoRA] Injecting Rank={rank}, Alpha={alpha} into {target_modules}...")
    
    # 1. 冻结全网参数
    for param in model.parameters():
        param.requires_grad = False
        
    # 2. 替换指定层
    injected_count = 0
    for name, module in model.named_modules():
        if "transformer.layers" in name and isinstance(module, nn.Linear):
            # 获取层的名称，例如 'wq', 'wo', 'w1' 等
            child_name = name.rsplit('.', 1)[1]
            
            if child_name in target_modules:
                parent_name = name.rsplit('.', 1)[0]
                parent = model.get_submodule(parent_name)
                
                original = getattr(parent, child_name)
                # 创建 LoRA 层
                lora_layer = LoRALinear(original, rank=rank, alpha=alpha, dropout=dropout)
                setattr(parent, child_name, lora_layer)
                injected_count += 1
                
    # 3. 打印统计
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[LoRA] Injected {injected_count} layers.")
    print(f"[LoRA] Trainable Params: {trainable/1e6:.4f}M / {total/1e6:.2f}M ({trainable/total:.4%})")
    return model

# ==========================================
# 2. SegmentMixer (保持不变，逻辑通用)
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
        if not self.base_json_files: return
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
        print(f"[Mixer] Loaded {len(self.audio_pool)} noise samples.")

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
                        # 动态适配当前 segment 的长度 (2s)
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
# 3. Dataset (适配 2秒 短音频)
# ==========================================
class ClusterDataset(Dataset):
    def __init__(self, json_file, sr=48000, max_len_sec=2.0):
        with open(json_file, 'r') as f:
            self.data = json.load(f)['data']
        self.sr = sr
        self.max_len = int(sr * max_len_sec) # 48000 * 2 = 96000

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
                # 随机裁剪到 2s
                start = random.randint(0, current_len - target_len)
                wav = wav[:, start:start+target_len]
            elif current_len < target_len:
                # 如果不足 2s，补零 (Pad)
                # 对于极短音频，补零比 Loop 更安全，避免相位问题，且 2s 已经很短了
                wav = F.pad(wav, (0, target_len - current_len))
            
            return {"waveform": wav, "text": item['caption']}
        except:
            return {"waveform": torch.zeros(1, self.max_len), "text": "error"}

class ClusterDataModule(pl.LightningDataModule):
    def __init__(self, train_json, batch_size=4, num_workers=2, sr=48000, max_len_sec=2.0):
        super().__init__()
        self.train_json = train_json
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.sr = sr
        self.max_len_sec = max_len_sec

    def train_dataloader(self):
        ds = ClusterDataset(self.train_json, sr=self.sr, max_len_sec=self.max_len_sec)
        return DataLoader(ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, drop_last=True, pin_memory=True)

# ==========================================
# 4. Lightning Module
# ==========================================
class SAMAudioLoRATrainer(pl.LightningModule):
    def __init__(self, config, ckpt_path, segment_mixer, lora_rank=8, lora_alpha=8, lr=1e-4):
        super().__init__()
        self.save_hyperparameters(ignore=['segment_mixer'])
        
        self.model = SAMAudio(config)
        # 加载预训练权重
        state_dict = torch.load(ckpt_path, map_location='cpu')
        self.model.load_state_dict(state_dict, strict=False)
        
        # 注入 LoRA (只针对 wq, wv)
        self.model = inject_lora_and_freeze(
            self.model, 
            rank=lora_rank, 
            alpha=lora_alpha, 
            dropout=0.05,
            target_modules=['wq', 'wv'] # 仅训练 Query 和 Value
        )
        
        self.segment_mixer = segment_mixer
        self.lr = lr

    def training_step(self, batch, batch_idx):
        clean_wavs = batch['waveform']
        texts = batch['text']
        
        with torch.no_grad():
            # 混合噪声
            mixtures, targets = self.segment_mixer(clean_wavs)
            
            # 1. 编码 Target (x_1)
            target_latents = self.model.audio_codec(targets).transpose(1, 2)
            target_latents = torch.cat([target_latents, target_latents], dim=2) 
            
            # 2. 编码 Mixture (Condition)
            mixture_latents = self.model.audio_codec(mixtures).transpose(1, 2)
            mixture_latents = torch.cat([mixture_latents, mixture_latents], dim=2)
            
            # 3. 编码 Text
            text_feat, text_mask = self.model.text_encoder(texts)
            
            # 4. Video (None)
            masked_video_features = self.model._get_video_features(None, mixture_latents)

        # Flow Matching 过程
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
        # 只优化 requires_grad=True 的参数 (即 LoRA 参数)
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
        # 只保存包含 "lora_" 的参数，文件极小
        lora_state_dict = {k: v for k, v in pl_module.model.state_dict().items() if "lora_" in k}
        save_path = os.path.join(self.save_dir, filename)
        torch.save(lora_state_dict, save_path)

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
    WORKSPACE = "/s7home/zyk524/SAM_Audio/sam-audio-large/7cluster_510-5" # 修改 workspace 区分实验

    # --- 关键超参数 (基于小样本短音频优化) ---
    BATCH_SIZE = 32
    MAX_EPOCHS = 100       # 数据少，epoch 可以适当减少，或者配合 early stopping
    LR = 5e-5              # LoRA 通常可以承受比全量微调稍大的 LR
    LORA_RANK = 8          # 降低 Rank 防止过拟合
    LORA_ALPHA = 8         # Alpha=Rank (Scale=1) 保持稳定
    AUDIO_LEN_SEC = 2.0    # 核心修改：2秒

    CLUSTER_IDS = list(range(5, 7))

    # 加载 Config
    with open(CONFIG_PATH) as f: cfg = json.load(f)
    cfg['text_encoder']['name'] = LOCAL_T5_DIR
    config = SAMAudioConfig(**cfg)

    all_json_files = [os.path.join(BASE_DATA_DIR, f"cluster{i}.json") for i in range(7)]

    for cluster_id in CLUSTER_IDS:
        print("\n" + "="*60)
        print(f"========== Training Cluster {cluster_id} (2s, Rank{LORA_RANK}) ==========")
        print("="*60 + "\n")
        
        # 清理显存
        gc.collect()
        torch.cuda.empty_cache()
        
        # 1. 准备 Mixer
        segment_mixer = SegmentMixer(
            max_mix_num=2, 
            lower_db=-10, higher_db=10, 
            exclude_cluster_id=cluster_id,
            json_files=all_json_files
        )
        
        # 2. 准备模型
        model = SAMAudioLoRATrainer(
            config=config,
            ckpt_path=CKPT_PATH,
            segment_mixer=segment_mixer,
            lora_rank=LORA_RANK,
            lora_alpha=LORA_ALPHA,
            lr=LR
        )
        
        # 3. 准备数据
        train_json = os.path.join(BASE_DATA_DIR, f"cluster{cluster_id}.json")
        if not os.path.exists(train_json):
            print(f"Skipping Cluster {cluster_id}: JSON not found.")
            continue
            
        dm = ClusterDataModule(
            train_json, 
            batch_size=BATCH_SIZE, 
            sr=config.audio_codec.sample_rate,
            max_len_sec=AUDIO_LEN_SEC # 传入 2.0
        )
        
        # 4. 训练器设置
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
            callbacks=[lora_saver, LearningRateMonitor(logging_interval='step'),TextLogCallback()],
            logger=logger,
            gradient_clip_val=1.0,
            log_every_n_steps=5, # 数据少，频繁打印 log
            enable_checkpointing=False 
        )
        
        trainer.fit(model, datamodule=dm)
        print(f"Cluster {cluster_id} Done.")

if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    os.environ["SAM_AUDIO_TEXT_ONLY"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    
    seed = 42
    pl.seed_everything(seed)
    
    main()
