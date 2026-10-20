#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：CLAPSep
@File    ：train_multiple_lora.py
@Author  ：Based on original code by Aisaka/Hao Ma @SDU
@Date    ：2024
'''

import os
import logging
import torch
import pytorch_lightning as pl
import laion_clap
from model.CLAPSep_decoder import HTSAT_Decoder
from model.CLAPSep import LightningModule
import argparse
from helpers import utils as local_utils
import json
import random
import numpy as np
import copy
import torchaudio
from torch.utils.data import Dataset, DataLoader
from data.csv_dataset_for_eval import CSVAudioTextDatasetForEval
from data.waveform_mixers1 import SegmentMixer

# 设置日志格式
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

# 设置随机种子以确保结果可重现
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 清理内存
def clear_memory():
    """清理GPU和CPU内存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc
    gc.collect()

class CustomAudioTextDataset(Dataset):
    """从JSON文件加载音频和文本数据的数据集"""

    def __init__(self, json_file, sampling_rate=32000, max_clip_len=2):
        self.sampling_rate = sampling_rate
        self.max_clip_len = max_clip_len
        self.data = self._load_and_filter_data(json_file)

    def _load_and_filter_data(self, json_file):
        with open(json_file, 'r') as f:
            data_dict = json.load(f)

        filtered_data = []
        for item in data_dict['data']:
            filtered_data.append(item)

        logging.info(f"加载了 {len(filtered_data)} 条数据")
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
                    logging.warning(f"音频文件为空或损坏: {wav_path}")
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
                    logging.warning(f"处理后的音频数据包含NaN或Inf值: {wav_path}")
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
                logging.warning(f"加载音频文件 {wav_path} 时出错: {e}")
                continue  # 尝试下一个音频
        
        # 如果尝试了最大次数仍未找到有效音频，则创建随机数据
        logging.warning(f"尝试{max_attempts}次后仍未找到有效音频，使用随机数据")
        audio_data = torch.randn(1, self.sampling_rate * self.max_clip_len)
        
        return {
            'audio_text': {
                'text': caption,
                'waveform': audio_data,
                'modality': 'audio_text'
            }
        }

class DataModule(pl.LightningDataModule):
    """数据模块，用于加载和处理数据"""

    def __init__(self, train_dataset, val_dataset, num_workers=2, batch_size=1):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.num_workers = num_workers
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,  # 加速数据传输到GPU
        )

    def val_dataloader(self):
        if self.val_dataset is not None:
            return DataLoader(
                dataset=self.val_dataset,
                batch_size=1,
                shuffle=False,
                drop_last=False,
                pin_memory=True,  # 加速数据传输到GPU
            )
        return None

# 训练单个LORA模型
def train_lora(args, lora_name, train_data_path, val_data_path):
    """训练单个LORA模型"""
    logging.info(f"开始训练LORA模型: {lora_name}")
    
    # 加载配置文件
    with open(args.config, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 创建目录
    dirs = {}
    dirs['checkpoint'] = os.path.join(args.exp_dir, args.checkpoint_dir, lora_name)
    dirs['logs'] = os.path.join(args.exp_dir, args.log_dir, lora_name)
    dirs['tensorboard'] = os.path.join(args.exp_dir, args.tensorboard_dir, lora_name)
    
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    # 创建波形混合器
    segment_mixer = SegmentMixer(
        max_mix_num=config.get('max_mix_num', 4),
        lower_db=config.get('lower_db', -10),
        higher_db=config.get('higher_db', 10)
    )
    logging.info("已初始化SegmentMixer用于音频混合")
    
    # 加载数据集
    train_dataset = CustomAudioTextDataset(
        json_file=train_data_path,
        sampling_rate=config['train_data'].get('sr', 32000),
        max_clip_len=config['train_data'].get('segment_seconds', 2)
    )
    logging.info(f"加载训练数据集: {train_data_path}, 包含 {len(train_dataset)} 个元素")
    
    # 加载验证数据集
    val_dataset = CSVAudioTextDatasetForEval(
        csv_file=val_data_path,
        sampling_rate=config['val_data'].get('sr', 32000)
    )
    logging.info(f"加载验证数据集: {val_data_path}, 包含 {len(val_dataset)} 个元素")
    
    # 创建数据模块
    data_module = DataModule(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        num_workers=config.get('n_workers', 4),
        batch_size=config.get('batch_size', 8)
    )
    
    # 加载CLAP模型
    logging.info("加载CLAP预训练模型...")
    # 初始化CLAP模型时明确指定device='cpu'
    clap_model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base', device='cpu')
    clap_model.load_ckpt(config['clap_path'])
    logging.info("CLAP预训练模型加载完成，参数已冻结")
    
    # 创建解码器
    decoder = HTSAT_Decoder(**config['model'])
    
    # 创建Lightning模块
    lightning_module = LightningModule(
        clap_model, 
        decoder, 
        lr=config['optim']['lr'],
        use_lora=True,  # 确保使用LORA
        rank=config.get('lora_rank', 16),
        nfft=config.get('nfft', 1024),
        waveform_mixer=segment_mixer
    )
    
    # 将模型移动到GPU
    # 使用torch.device对象而不是字符串，并确保检查CUDA可用性
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 确保整个模型都移动到正确的设备上
    lightning_module = lightning_module.to(device)

    # 创建训练器
    trainer = pl.Trainer(
        default_root_dir=dirs['checkpoint'],
        devices=args.gpus,
        accelerator="gpu",
        benchmark=True,
        gradient_clip_val=5.0,
        precision='bf16-mixed',
        limit_train_batches=1.0,
        max_epochs=args.max_epochs,
        strategy="ddp" if args.gpus > 1 else "single_device",
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                dirpath=dirs['checkpoint'],
                filename=f"{lora_name}-{{epoch}}-{{val_loss:.4f}}",
                save_top_k=3,
                monitor="val_loss",
                mode="max"
            ),
            pl.callbacks.LearningRateMonitor(logging_interval="epoch")
        ],
        logger=pl.loggers.TensorBoardLogger(dirs['tensorboard'])
    )
    
    # 开始训练
    logging.info("开始训练LoRA模型...")
    trainer.fit(model=lightning_module, datamodule=data_module)
    
    # 保存最终模型
    final_model_path = os.path.join(dirs['checkpoint'], f"{lora_name}_final.pt")
    # 只保存LoRA参数
    lora_state_dict = {}
    for name, param in lightning_module.state_dict().items():
        if 'lora_' in name:  # 只保存LoRA相关参数
            lora_state_dict[name] = param
    
    torch.save({
        'lora_state_dict': lora_state_dict,
        'epoch': trainer.current_epoch,
        'global_step': trainer.global_step
    }, final_model_path)
    logging.info(f"LoRA模型已保存到: {final_model_path}")
    
    # 清理内存
    del clap_model, decoder, lightning_module, trainer, data_module
    clear_memory()
    
    return final_model_path

# 主函数
def main(args):
    torch.set_float32_matmul_precision('medium')
    
    # 设置随机种子
    set_seed(args.seed if hasattr(args, 'seed') else 114514)
    
    # 获取单个LORA模型的配置
    lora_name = args.lora_name
    train_data_path = args.train_data_path
    val_data_path = args.val_data_path
    
    logging.info(f"开始训练LORA模型: {lora_name}")
    logging.info(f"训练数据路径: {train_data_path}")
    logging.info(f"验证数据路径: {val_data_path}")
    
    # 检查文件是否存在
    if not os.path.exists(train_data_path):
        logging.error(f"训练数据文件不存在: {train_data_path}")
        return
        
    if not os.path.exists(val_data_path):
        logging.error(f"验证数据文件不存在: {val_data_path}")
        return
    
    try:
        model_path = train_lora(args, lora_name, train_data_path, val_data_path)
        logging.info(f"LORA模型 {lora_name} 训练完成，保存到: {model_path}")
    except Exception as e:
        logging.error(f"训练LORA模型 {lora_name} 时出错: {e}")
    
    # 训练后清理内存
    clear_memory()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='训练单个LORA模型')
    parser.add_argument('--lora_name', type=str, default='lora_0', help='LORA模型名称')
    parser.add_argument('--train_data_path', type=str, default='/s1home/zyk524/AudioSep/LORA/s4/cluster0.json', help='训练数据JSON文件路径')
    parser.add_argument('--val_data_path', type=str, default='/s1home/zyk524/AudioSep/LORA/s4/csv/cluster0.csv', help='验证数据文件路径')
    parser.add_argument('--config', type=str, default='/s1home/zyk524/study/experiment/CLAPSep/experiments/CLAPSep_base/config.json', help='模型配置文件路径')
    parser.add_argument('--exp_dir', type=str,
                         default='/s1home/zyk524/study/experiment/CLAPSep/LORA_cluster',
                         help="保存检查点和日志的路径")
    parser.add_argument('--checkpoint_dir', type=str, default='/s1home/zyk524/study/experiment/CLAPSep/LORA_cluster/checkpoint', help='检查点保存目录')
    parser.add_argument('--log_dir', type=str, default='/s1home/zyk524/study/experiment/CLAPSep/LORA_cluster/log ', help='日志保存目录')
    parser.add_argument('--tensorboard_dir', type=str, default='/s1home/zyk524/study/experiment/CLAPSep/LORA_cluster/dir', help='TensorBoard日志目录')
    parser.add_argument('--max_epochs', type=int, default=150, help='最大训练轮数')
    parser.add_argument('--gpus', type=int, default=1, help='使用的GPU数量')
    parser.add_argument('--seed', type=int, default=114514,
                         help="随机种子")

    args = parser.parse_args()

    # 设置随机种子以确保结果可重现
    pl.seed_everything(args.seed)
    
    # 设置检查点
    if not os.path.exists(args.exp_dir):
        os.makedirs(args.exp_dir)

    # 加载模型和训练参数
    params = local_utils.Params(os.path.join(args.exp_dir, 'config.json'))
    for k, v in params.__dict__.items():
        vars(args)[k] = v
        
    main(args)