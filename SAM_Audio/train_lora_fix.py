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
from sam_audio import SAMAudio, SAMAudioConfig
torch.set_float32_matmul_precision('high')
# ==========================================
# 1. LoRA 动态注入工具 (Target Module 优化版)
# ==========================================

# class TextLogCallback(Callback):
#     def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
#         if batch_idx % trainer.log_every_n_steps == 0:
#             loss = outputs['loss']
#             print(f"[Epoch {trainer.current_epoch}] Step {batch_idx}: loss={loss.item():.4f}", flush=True)


# def freeze_bn(model):
#     """
#     冻结模型中的所有 BatchNorm 层：
#     1. 冻结参数（requires_grad = False）
#     2. 设置为 eval 模式（冻结 running_mean 和 running_var，防止统计信息漂移）
#     """
#     bn_count = 0
#     for name, module in model.named_modules():
#         if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, 
#                               nn.SyncBatchNorm)):
#             # 冻结参数
#             for param in module.parameters():
#                 param.requires_grad = False
#             # 设置为 eval 模式，冻结统计信息
#             module.eval()
#             bn_count += 1
        
#         # LayerNorm 也需要设置为 eval 模式
#         elif isinstance(module, nn.LayerNorm):
#             for param in module.parameters():
#                 param.requires_grad = False
#             module.eval()
    
#     if bn_count > 0:
#         print(f"[Freeze] Frozen {bn_count} BatchNorm layers (eval mode).")


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
        dtype = self.lora_A.dtype
        x_casted = x.to(dtype)
        base_out = self.base_layer(x)
        # 显式转换类型，防止混合精度训练时出错
        lora_out = (self.lora_dropout(x_casted) @ self.lora_A @ self.lora_B) * self.scaling
        return base_out + lora_out.to(base_out.dtype)

# def inject_lora_and_freeze(model, rank=8, alpha=8, dropout=0.05, target_modules=['wq', 'wv']):
#     """
#     只对 Transformer 中的 target_modules 进行 LoRA 注入
#     """
#     print(f"[LoRA] Injecting Rank={rank}, Alpha={alpha} into {target_modules}...")
    
#     # 1. 冻结全网参数
#     for param in model.parameters():
#         param.requires_grad = False
        
#     # 2. 替换指定层
#     injected_count = 0
#     for name, module in model.named_modules():
#         if "transformer.layers" in name and isinstance(module, nn.Linear):
#             # 获取层的名称，例如 'wq', 'wo', 'w1' 等
#             child_name = name.rsplit('.', 1)[1]
            
#             if child_name in target_modules:
#                 parent_name = name.rsplit('.', 1)[0]
#                 parent = model.get_submodule(parent_name)
                
#                 original = getattr(parent, child_name)
#                 # 创建 LoRA 层
#                 lora_layer = LoRALinear(original, rank=rank, alpha=alpha, dropout=dropout)
#                 setattr(parent, child_name, lora_layer)
#                 injected_count += 1
                
#     # 3. 打印统计
#     trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     total = sum(p.numel() for p in model.parameters())
#     print(f"[LoRA] Injected {injected_count} layers.")
#     print(f"[LoRA] Trainable Params: {trainable/1e6:.4f}M / {total/1e6:.2f}M ({trainable/total:.4%})")
#     return model
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
# 2. SegmentMixer (保持不变，逻辑通用)
# ==========================================
# class SegmentMixer(nn.Module):
#     def __init__(self, max_mix_num, lower_db, higher_db, exclude_cluster_id=None, json_files=None):
#         super().__init__()
#         self.max_mix_num = max_mix_num
#         self.loudness_param = {'lower_db': lower_db, 'higher_db': higher_db}
#         self.exclude_cluster_id = exclude_cluster_id
#         self.base_json_files = json_files
#         self.audio_pool = []
#         self._load_audio_paths()

#     def _load_audio_paths(self):
#         self.audio_pool = []
#         if not self.base_json_files: return
#         for json_file in self.base_json_files:
#             if self.exclude_cluster_id is not None and f"cluster{self.exclude_cluster_id}.json" in json_file:
#                 continue
#             if not os.path.exists(json_file): continue
#             try:
#                 with open(json_file, 'r') as f:
#                     data = json.load(f)
#                 for item in data['data']:
#                     if 'wav' in item and os.path.exists(item['wav']):
#                         self.audio_pool.append(item['wav'])
#             except: pass
#         print(f"[Mixer] Loaded {len(self.audio_pool)} noise samples.")

#     def _load_audio(self, audio_path, device=None):
#         try:
#             wav, sr = torchaudio.load(audio_path)
#             if wav.shape[0] > 1: wav = wav.mean(0, keepdim=True)
#             if device: wav = wav.to(device)
#             return wav
#         except: return None

#     def __call__(self, waveforms):
#         batch_size = waveforms.shape[0]
#         device = waveforms.device
#         mixtures = []
#         segments = []

#         for n in range(batch_size):
#             segment = waveforms[n].clone()
#             noise = torch.zeros_like(segment)
#             mix_num = random.randint(2, self.max_mix_num)

#             for _ in range(1, mix_num):
#                 if not self.audio_pool: break
#                 for _ in range(10): 
#                     noise_path = random.choice(self.audio_pool)
#                     noise_wav = self._load_audio(noise_path, device)
#                     if noise_wav is not None:
#                         # 动态适配当前 segment 的长度 (2s)
#                         if noise_wav.shape[1] < segment.shape[1]:
#                             repeats = (segment.shape[1] // noise_wav.shape[1]) + 1
#                             noise_wav = noise_wav.repeat(1, repeats)
#                         noise_wav = noise_wav[:, :segment.shape[1]]
#                         noise += self.dynamic_loudnorm(noise_wav, segment)
#                         break
            
#             mixture = segment + self.dynamic_loudnorm(noise, segment)
#             max_val = torch.max(torch.abs(mixture))
#             if max_val > 1:
#                 scale = 0.9 / max_val
#                 segment *= scale
#                 mixture *= scale
            
#             mixtures.append(mixture)
#             segments.append(segment)

#         return torch.stack(mixtures), torch.stack(segments)

#     def get_energy(self, x): return torch.mean(x ** 2)
    
#     def dynamic_loudnorm(self, audio, reference):
#         e1 = self.get_energy(audio)
#         e2 = max(self.get_energy(reference), 1e-10)
#         ratio = (e1 / e2) ** 0.5
#         ratio = torch.clamp(ratio, 0.02, 50)
#         audio = audio / ratio
#         gain = 10 ** (random.uniform(self.loudness_param['lower_db'], self.loudness_param['higher_db']) / 20.0)
#         return audio * gain
class SegmentMixer(nn.Module):
    def __init__(self, noise_json_files, max_mix_num=2, lower_db=-10, higher_db=10, target_sr=48000, target_len_sec=2.0):
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
# ==========================================
# 3. Dataset (适配 2秒 短音频)
# ==========================================
class ClusterDataset(Dataset):
    def __init__(self,target_json,noise_json_files,sr=48000, max_len_sec=2.0):
        with open(json_file, 'r') as f:
            self.data = json.load(f)['data']
        self.sr = sr
        self.max_len = int(sr * max_len_sec) # 48000 * 2 = 96000
        self.mixer = SegmentMixer(
            noise_json_files=noise_json_files,
            max_mix_num=2,
            lower_db=-10, higher_db=10, 
            target_sr=sample_rate,
            target_len_sec=duration
        )
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
        # item = self.data[idx]
        # try:
        #     wav, sr = torchaudio.load(item['wav'])
        #     if wav.shape[0] > 1: wav = wav.mean(0, keepdim=True)
        #     if sr != self.sr: wav = torchaudio.functional.resample(wav, sr, self.sr)
            
        #     current_len = wav.shape[1]
        #     target_len = self.max_len
            
        #     if current_len > target_len:
        #         # 随机裁剪到 2s
        #         start = random.randint(0, current_len - target_len)
        #         wav = wav[:, start:start+target_len]
        #     elif current_len < target_len:
        #         # 如果不足 2s，补零 (Pad)
        #         # 对于极短音频，补零比 Loop 更安全，避免相位问题，且 2s 已经很短了
        #         wav = F.pad(wav, (0, target_len - current_len))
        #     clean_wav = self.mixer._load_audio(wav)
        #     mixture_wav, target_wav = self.mixer.mix(clean_wav)
        #     return {            "mixture": mixture_wav.squeeze(0),
        #     "target": target_wav.squeeze(0),
        #      "text": item['caption']}
        # except:
        #     return {"mixture": torch.zeros(1, self.max_len), "target": "error","text": "error"}
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
        # self.model.load_state_dict(state_dict, strict=False)
                # 检查checkpoint的键
        print("\n" + "="*60)
        print("[Checkpoint] Loading checkpoint from:", ckpt_path)
        print("[Checkpoint] Total keys in checkpoint:", len(state_dict.keys()))
        
        # 尝试加载并检查键匹配情况
        load_result = self.model.load_state_dict(state_dict, strict=False)
        
        # 检查缺失的键（checkpoint有但模型没有）
        missing_keys = load_result.missing_keys
        # 检查未匹配的键（模型有但checkpoint没有）
        unexpected_keys = load_result.unexpected_keys
        
        if missing_keys:
            print(f"[Warning] Missing keys ({len(missing_keys)}):")
            for key in missing_keys[:10]:  # 只打印前10个
                print(f"  - {key}")
            if len(missing_keys) > 10:
                print(f"  ... and {len(missing_keys) - 10} more")
        else:
            print("[Checkpoint] No missing keys - all checkpoint weights loaded!")
            
        if unexpected_keys:
            print(f"[Warning] Unexpected keys ({len(unexpected_keys)}):")
            for key in unexpected_keys[:10]:  # 只打印前10个
                print(f"  - {key}")
            if len(unexpected_keys) > 10:
                print(f"  ... and {len(unexpected_keys) - 10} more")
        else:
            print("[Checkpoint] No unexpected keys!")
            
        print("="*60 + "\n")
        # 注入 LoRA (只针对 wq, wv)
        self.model = inject_lora(
            self.model, 
            rank=lora_rank, 
            alpha=lora_alpha, 
            dropout=0.1
            # target_modules=['wq', 'wv'] # 仅训练 Query 和 Value
        )
        print_model_stats(self.model)
        self.segment_mixer = segment_mixer
        self.lr = lr
    
        # 检查T5编码器是否正确加载
        if hasattr(self.model, 'text_encoder'):
            print("\n[T5] Text encoder found in model.")
            # 尝试检查text_encoder的参数数量
            t5_params = sum(p.numel() for p in self.model.text_encoder.parameters())
            print(f"[T5] Text encoder parameters: {t5_params/1e6:.2f}M")
        else:
            print("[Warning] No text_encoder found in model!")
        
        # 保存T5编码器的原始状态，用于验证
        self._text_encoder_original_params = None
        if hasattr(self.model, 'text_encoder'):
            self._text_encoder_original_params = {n: p.clone() for n, p in self.model.text_encoder.named_parameters()}

    def on_train_start(self):
        """
        在训练开始前冻结基线模型的BN层，这是正确的位置
        """
        self.freeze_base_model_bn()
        print("[Train] Training started, base model BN frozen.")
        
        # 验证T5编码器参数未被修改
        if hasattr(self.model, 'text_encoder') and self._text_encoder_original_params is not None:
            current_params = {n: p for n, p in self.model.text_encoder.named_parameters()}
            changed = []
            for n, orig_p in self._text_encoder_original_params.items():
                if not torch.equal(orig_p, current_params[n]):
                    changed.append(n)
            if changed:
                print(f"[Warning] T5 parameters changed before training: {len(changed)} params")
            else:
                print("[Verify] T5 parameters unchanged - correctly frozen!")

    def freeze_base_model_bn(self):
        """
        冻结基线模型的BN层（包括T5编码器），防止统计信息漂移
        """
        frozen_count = 0
        for name, module in self.model.named_modules():
            # 跳过LoRA层
            if 'lora' in name.lower():
                continue
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, 
                                  nn.SyncBatchNorm, nn.LayerNorm)):
                module.eval()
                for param in module.parameters():
                    param.requires_grad = False
                frozen_count += 1
        # 特别处理：如果text_encoder是独立的T5模型，也需要冻结
        if hasattr(self.model, 'text_encoder'):
            t5_bn_count = 0
            for name, module in self.model.text_encoder.named_modules():
                if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, 
                                      nn.SyncBatchNorm, nn.LayerNorm)):
                    module.eval()
                    for param in module.parameters():
                        param.requires_grad = False
                    t5_bn_count += 1
            if t5_bn_count > 0:
                print(f"[Freeze] T5 encoder: {t5_bn_count} BN/LN layers frozen.")
        # print("[Freeze] Base model BN layers frozen (eval mode).")
        if frozen_count > 0:
            print(f"[Freeze] Base model: {frozen_count} BN/LN layers frozen (eval mode).")
    def training_step(self, batch, batch_idx):
        # self.freeze_base_model_bn()

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
        model = pl_module.model
        # 确保获取到正确的模型引用
        if hasattr(model, 'model'):
            model = model.model
        lora_state_dict = {k: v for k, v in model.state_dict().items() if "lora_" in k}
        
        if len(lora_state_dict) == 0:
            print(f"[Warning] No LoRA parameters found to save!")
            return
        save_path = os.path.join(self.save_dir, filename)
        torch.save(lora_state_dict, save_path)
        print(f"[Save] Saved {len(lora_state_dict)} LoRA params to {filename}")
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
    WORKSPACE = "/s7home/zyk524/SAM_Audio/sam-audio-large/7cluster_fix" # 修改 workspace 区分实验

    # --- 关键超参数 (基于小样本短音频优化) ---
    BATCH_SIZE = 32
    MAX_EPOCHS = 150       # 数据少，epoch 可以适当减少，或者配合 early stopping
    LR = 5e-4              # LoRA 通常可以承受比全量微调稍大的 LR
    LORA_RANK = 16          # 降低 Rank 防止过拟合
    LORA_ALPHA = 16         # Alpha=Rank (Scale=1) 保持稳定
    AUDIO_LEN_SEC = 2.0    # 核心修改：2秒
    CLUSTER_IDS = list(range(0, 2))

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

        # 构造噪声文件列表：除了当前 cluster_id 之外的所有文件
        noise_jsons = [f for i, f in enumerate(all_json_files) if i != cluster_id]

        
        
    
            
        # 清理显存
        gc.collect()
        torch.cuda.empty_cache()
        
    
        segment_mixer = SegmentMixer(
            noise_json_files=noise_jsons,
            max_mix_num=2,
            lower_db=-10, higher_db=10, 
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
        dataset = ClusterDataset(
            target_json=target_json,
            noise_json_files=noise_jsons, # 传入排除后的列表
            sample_rate=48000,
            duration=2.0
        )
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, 
                                num_workers=8, pin_memory=True, drop_last=True)
        # 3. 准备数据
        # train_json = os.path.join(BASE_DATA_DIR, f"cluster{cluster_id}.json")
        # if not os.path.exists(train_json):
        #     print(f"Skipping Cluster {cluster_id}: JSON not found.")
        #     continue
            
        # dm = ClusterDataModule(
        #     train_json, 
        #     batch_size=BATCH_SIZE, 
        #     sr=config.audio_codec.sample_rate,
        #     max_len_sec=AUDIO_LEN_SEC # 传入 2.0
        # )
        
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
        
        trainer.fit(model, datamodule=dataloader)
        print(f"Cluster {cluster_id} Done.")

if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    os.environ["SAM_AUDIO_TEXT_ONLY"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    
    seed = 42
    pl.seed_everything(seed)
    
    main()
