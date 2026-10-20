import os
import json
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import Dataset, DataLoader
from typing import List, Optional, Tuple, Dict, Any
import logging
import gc
from clapsep.csv_dataset_for_eval import CSVAudioTextDatasetForEval
from CLAPSep_new import CLAPSepModule
from clapsep.waveform_mixers1 import SegmentMixer


# 设置环境变量
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def seed_everything(seed):
    """设置随机种子以确保实验可重复性"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def clear_memory():
    """清理内存"""
    gc.collect()
    torch.cuda.empty_cache()


class MemoryMonitorCallback(pl.Callback):
    """内存监控回调"""
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx % 100 == 0:
            gc.collect()
            torch.cuda.empty_cache()


class LossMonitorCallback(pl.Callback):
    """损失监控回调"""
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx % 50 == 0:
            loss = outputs.item() if isinstance(outputs, torch.Tensor) else outputs['loss'].item()
            print(f"Batch {batch_idx}, Loss: {loss:.4f}")


class ModelSaveCallback(pl.Callback):
    """模型保存回调"""
    def __init__(self, save_dir, save_freq=5):
        super().__init__()
        self.save_dir = save_dir
        self.save_freq = save_freq
        os.makedirs(save_dir, exist_ok=True)

    def on_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % self.save_freq == 0:
            epoch = trainer.current_epoch + 1
            save_path = os.path.join(self.save_dir, f"model_epoch_{epoch}.ckpt")
            trainer.save_checkpoint(save_path)
            print(f"Model saved to {save_path}")


class CustomAudioTextDataset(Dataset):
    """自定义音频文本数据集，与train_adapter.py保持一致"""
    def __init__(self, json_file, sample_rate=32000, max_clip_len=2):
        self.sampling_rate = sample_rate
        self.max_clip_len = max_clip_len
        self.data = self._load_and_filter_data(json_file)

    def _load_and_filter_data(self, json_file):
        with open(json_file, 'r') as f:
            data_dict = json.load(f)

        filtered_data = []
        if 'data' in data_dict:
            for item in data_dict['data']:
                filtered_data.append(item)
        else:
            # 兼容不同格式的JSON文件
            filtered_data = data_dict

        print(f"加载了 {len(filtered_data)} 条数据")
        return filtered_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 尝试加载有效的音频，如果当前索引的音频损坏，则尝试下一个
        max_attempts = 10  # 最多尝试10次
        for attempt in range(max_attempts):
            current_idx = (idx + attempt) % len(self.data)  # 循环索引，避免越界
            item = self.data[current_idx]
            wav_path = item['wav']
            caption = item['caption']
            
            try:
                # 使用torchaudio加载音频文件
                audio_data, audio_rate = torchaudio.load(wav_path, channels_first=True)
                
                # 检查音频是否为空或损坏
                if audio_data.numel() == 0 or torch.isnan(audio_data).any() or torch.isinf(audio_data).any():
                    print(f"音频文件为空或损坏: {wav_path}")
                    continue  # 尝试下一个音频
                
                # 转换为单声道
                if audio_data.shape[0] > 1:
                    audio_data = (audio_data[0] + audio_data[1]) / 2
                    audio_data = audio_data.unsqueeze(0)

                # 重采样音频
                if audio_rate != self.sampling_rate:
                    audio_data = torchaudio.functional.resample(audio_data, orig_freq=audio_rate,
                                                                new_freq=self.sampling_rate)

                # 裁剪或填充音频
                max_length = self.sampling_rate * self.max_clip_len
                if audio_data.size(1) > max_length:
                    # 随机裁剪
                    start = random.randint(0, audio_data.size(1) - max_length)
                    audio_data = audio_data[:, start:start + max_length]
                else:
                    # 填充
                    temp_wav = torch.zeros(1, max_length)
                    temp_wav[:, :audio_data.size(1)] = audio_data
                    audio_data = temp_wav
                    
                # 检查处理后的音频是否有效
                if torch.isnan(audio_data).any() or torch.isinf(audio_data).any():
                    print(f"处理后的音频数据包含NaN或Inf值: {wav_path}")
                    continue  # 尝试下一个音频
                    
                # 音频有效，返回结果
                return {
                    'audio_text': {
                        'text': caption,
                        'waveform': audio_data,
                        'modality': 'audio_text'
                    }
                }
                
            except Exception as e:
                print(f"加载音频文件 {wav_path} 时出错: {e}")
                continue  # 尝试下一个音频
        
        # 如果尝试了最大次数仍未找到有效音频，则创建随机数据
        print(f"尝试{max_attempts}次后仍未找到有效音频，使用随机数据")
        audio_data = torch.randn(1, self.sampling_rate * self.max_clip_len)
        
        return {
            'audio_text': {
                'text': caption,
                'waveform': audio_data,
                'modality': 'audio_text'
            }
        }



class DataModule(pl.LightningDataModule):
    """数据模块，与train_adapter.py保持一致"""
    def __init__(self, train_json, val_csv=None, batch_size=16, num_workers=4, sample_rate=32000, max_clip_len=2):
        super().__init__()
        self.train_json = train_json
        self.val_csv = val_csv
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.sample_rate = sample_rate
        self.max_clip_len = max_clip_len
        
        # 初始化训练数据集
        self.train_dataset = CustomAudioTextDataset(
            json_file=self.train_json,
            sample_rate=self.sample_rate,
            max_clip_len=self.max_clip_len
        )
        
        # 初始化验证数据集（如果提供了验证CSV文件）
        self.val_dataset = None
        if self.val_csv:
            self.val_dataset = CSVAudioTextDatasetForEval(
                csv_file=self.val_csv,
                sampling_rate=self.sample_rate
            )
        
        # 初始化波形混合器
        self.waveform_mixer = SegmentMixer(
            max_mix_num=2,  
            lower_db=-10,   
            higher_db=10,   
        )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True  # 加速数据传输到GPU
        )

    def val_dataloader(self):
        if self.val_dataset is not None:
            return DataLoader(
                dataset=self.val_dataset,
                batch_size=1,  # 验证时使用批大小为1
                shuffle=False,
                drop_last=False,
                pin_memory=True  # 加速数据传输到GPU
            )
        return None


def train_clapsep_lora(args):
    """训练 CLAPSep 模型，使用 LoRA 微调 CLAP 的音频编码器"""
    # 设置随机种子
    seed_everything(args.seed)
    
    # 创建数据模块
    data_module = DataModule(
        train_json=args.train_json,
        val_csv=args.val_csv,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sample_rate=args.sample_rate,
        max_clip_len=args.max_clip_len
    )
    
    # 创建模型
    model = CLAPSepModule(
        clap_ckpt_path=args.clap_ckpt_path,
        sample_rate=args.sample_rate,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        win_length=args.win_length,
        window=args.window,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        patch_size=args.patch_size,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lr=args.lr,
        use_lora=args.use_lora
    )
    print(f"CLAPSepModule initialized on device: {next(model.parameters()).device}")
    
    # 创建回调
    callbacks = [
        ModelCheckpoint(
            dirpath=args.checkpoint_dir,
            filename='{epoch:02d}-{val_loss:.2f}',
            save_top_k=3,
            monitor='val_loss',
            mode='min',
            save_last=True
        ),
        LearningRateMonitor(logging_interval='epoch'),
        MemoryMonitorCallback(),
        LossMonitorCallback(),
        ModelSaveCallback(save_dir=args.checkpoint_dir, save_freq=args.save_freq)
    ]
    
    # 创建日志记录器
    logger = TensorBoardLogger(save_dir=args.log_dir, name=args.experiment_name)
    
    # 自动检测设备配置
    if torch.cuda.is_available() and args.accelerator == 'gpu':
        devices = args.devices
        accelerator = 'gpu'
        strategy = args.strategy
        
        # 显式地将模型转移到GPU设备上
        device = torch.device(f"cuda:{0 if devices == 1 else 0}")
        model = model.to(device)
        print(f"模型已转移到设备: {next(model.parameters()).device}")
        print(f"CLAP_Encoder转移到设备: {next(model.clap_encoder.parameters()).device}")
        print(f"Audio branch转移到设备: {next(model.audio_branch.parameters()).device}")
        print(f"Decoder model转移到设备: {next(model.decoder_model.parameters()).device}")
        if hasattr(model.stft, 'device'):
            print(f"STFT转移到设备: {model.stft.device}")
    else:
        devices = 'auto'
        accelerator = 'cpu'
        strategy = 'auto'  # CPU训练时使用auto策略
        print(f"警告: CUDA不可用或未指定GPU加速器，将使用CPU训练")
    
    # 创建训练器
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        devices=devices,
        accelerator=accelerator,
        strategy=args.strategy,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=args.log_every_n_steps,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        precision=args.precision,
        gradient_clip_val=args.gradient_clip_val
    )
    
    # 训练模型
    trainer.fit(model, data_module)
    
    # 保存最终模型
    trainer.save_checkpoint(os.path.join(args.checkpoint_dir, 'final_model.ckpt'))
    
    # 仅保存 LoRA 参数
    if args.save_lora_only:
        checkpoint = torch.load(os.path.join(args.checkpoint_dir, 'final_model.ckpt'))
        weights = checkpoint['state_dict']
        new_dict = {}
        for k, v in weights.items():
            # 保存loralib的LoRA参数和解码器参数
            if any(e in k for e in ['lora_A', 'lora_B', 'decoder_model']):
                new_dict[k] = v
        checkpoint['state_dict'] = new_dict
        torch.save(checkpoint, os.path.join(args.checkpoint_dir, 'final_model_lora_only.ckpt'))
    
    return model


def main():
    parser = argparse.ArgumentParser(description='Train CLAPSep model with LoRA')
    
    # 数据参数
    parser.add_argument('--train_json', type=str, default='/s1home/zyk524/AudioSep/LORA/s4/cluster0.json', help='Path to training JSON file')
    parser.add_argument('--val_csv', type=str, default='/s1home/zyk524/AudioSep/LORA/s4/csv/cluster0.csv', help='Path to validation CSV file')
    parser.add_argument('--sample_rate', type=int, default=32000, help='Audio sample rate')
    parser.add_argument('--max_clip_len', type=int, default=2, help='Maximum audio clip length in seconds')
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=None,
        help="要使用的GPU数量，默认为自动检测"
    )
    # 模型参数
    parser.add_argument('--clap_ckpt_path', type=str,default='/s1home/zyk524/study/experiment/CLAPSep/checkpoint/music_audioset_epoch_15_esc_90.14.pt' , help='Path to CLAP checkpoint')
    parser.add_argument('--n_fft', type=int, default=2048, help='FFT size')
    parser.add_argument('--hop_length', type=int, default=512, help='Hop length')
    parser.add_argument('--win_length', type=int, default=2048, help='Window length')
    parser.add_argument('--window', type=str, default='hann', help='Window type')
    parser.add_argument('--embed_dim', type=int, default=768, help='Embedding dimension')
    parser.add_argument('--num_heads', type=int, default=12, help='Number of attention heads')
    parser.add_argument('--patch_size', type=int, nargs=2, default=[4, 4], help='Patch size')
    parser.add_argument('--lora_rank', type=int, default=4, help='LoRA rank')
    parser.add_argument('--lora_alpha', type=float, default=1.0, help='LoRA alpha')
    parser.add_argument('--use_lora', action='store_false',default=True, help='Use LoRA for fine-tuning')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--max_epochs', type=int, default=150, help='Maximum number of epochs')
    parser.add_argument('--devices', type=int, default=1, help='Number of GPUs')
    parser.add_argument('--accelerator', type=str, default='gpu', help='Accelerator type')
    parser.add_argument('--strategy', type=str, default='single_device', help='Training strategy')
    parser.add_argument('--log_every_n_steps', type=int, default=50, help='Log every N steps')
    parser.add_argument('--check_val_every_n_epoch', type=int, default=1, help='Check validation every N epochs')
    parser.add_argument('--precision', type=int, default=32, help='Precision')
    parser.add_argument('--gradient_clip_val', type=float, default=1.0, help='Gradient clipping value')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # 保存参数
    parser.add_argument('--checkpoint_dir', type=str, default='/s1home/zyk524/study/experiment/CLAPSep/LORA_cluster/checkpoint', help='Checkpoint directory')
    parser.add_argument('--log_dir', type=str, default='/s1home/zyk524/study/experiment/CLAPSep/LORA_cluster/log', help='Log directory')
    parser.add_argument('--experiment_name', type=str, default='clapsep_lora0', help='Experiment name')
    parser.add_argument('--save_freq', type=int, default=5, help='Save frequency in epochs')
    parser.add_argument('--save_lora_only', action='store_false',default=True, help='Save only LoRA parameters')
    
    args = parser.parse_args()
    
    # 创建目录
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # 训练模型
    train_clapsep_lora(args)


if __name__ == '__main__':
    main()
