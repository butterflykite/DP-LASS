#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：CLAPSep
@File    ：inference_multiple_lora.py
@Author  ：Based on original code by Aisaka/Hao Ma @SDU
@Date    ：2024
'''

import os
import torch
import laion_clap
from model.CLAPSep_decoder import HTSAT_Decoder
from model.CLAPSep import LightningModule
import argparse
from helpers import utils as local_utils
import torchaudio
import librosa
import numpy as np
import logging

# 设置日志格式
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

class LoraModelManager:
    """管理多个LORA模型的加载和切换"""
    
    def __init__(self, config_path, device='cuda'):
        """初始化LORA模型管理器
        
        Args:
            config_path: 配置文件路径
            device: 运行设备
        """
        self.device = device
        self.config = local_utils.Params(config_path)
        
        # 加载CLAP模型
        self.clap_model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base', device='cpu')
        self.clap_model.load_ckpt(self.config.clap_path)
        
        # 创建解码器
        self.decoder = HTSAT_Decoder(**self.config.model)
        
        # 创建基础模型
        self.base_model = LightningModule(
            self.clap_model, 
            self.decoder, 
            lr=self.config.optim['lr'],
            use_lora=True,
            rank=self.config.lora_rank,
            nfft=self.config.nfft
        )
        
        # 将模型移动到指定设备
        self.base_model = self.base_model.to(device)
        self.base_model.eval()
        
        # 存储已加载的LORA模型状态字典
        self.lora_models = {}
        self.current_lora = None
    
    def load_lora_model(self, lora_path, lora_name=None):
        """加载LORA模型
        
        Args:
            lora_path: LORA模型路径
            lora_name: LORA模型名称，如果为None，则使用文件名
        """
        if lora_name is None:
            lora_name = os.path.basename(lora_path).split('.')[0]
        
        # 加载模型状态字典
        state_dict = torch.load(lora_path, map_location=self.device)
        
        # 过滤出LORA相关参数
        lora_state_dict = {}
        for k, v in state_dict.items():
            if any(e in k for e in ['lora', 'attn.qkv.bias', 'attn.proj.bias']):
                lora_state_dict[k] = v
        
        # 存储LORA状态字典
        self.lora_models[lora_name] = lora_state_dict
        logging.info(f"已加载LORA模型: {lora_name}")
        
        return lora_name
    
    def switch_lora(self, lora_name):
        """切换到指定的LORA模型
        
        Args:
            lora_name: LORA模型名称
        """
        if lora_name not in self.lora_models:
            raise ValueError(f"LORA模型 {lora_name} 未加载")
        
        # 加载LORA状态字典
        self.base_model.load_state_dict(self.lora_models[lora_name], strict=False)
        self.current_lora = lora_name
        logging.info(f"已切换到LORA模型: {lora_name}")
    
    def separate_audio(self, audio_path, text_query, output_path=None):
        """使用当前LORA模型分离音频
        
        Args:
            audio_path: 输入音频路径
            text_query: 文本查询
            output_path: 输出音频路径，如果为None，则不保存
            
        Returns:
            分离后的音频波形
        """
        if self.current_lora is None:
            raise ValueError("请先切换到一个LORA模型")
        
        # 加载音频
        audio_data, sr = torchaudio.load(audio_path)
        
        # 确保音频是单声道
        if audio_data.shape[0] > 1:
            audio_data = torch.mean(audio_data, dim=0, keepdim=True)
        
        # 重采样到模型所需采样率
        if sr != self.config.train_data['sr']:
            resampler = torchaudio.transforms.Resample(sr, self.config.train_data['sr'])
            audio_data = resampler(audio_data)
        
        # 裁剪或填充音频
        max_length = self.config.train_data['sr'] * 10  # 假设最大长度为10秒
        if audio_data.size(1) > max_length:
            audio_data = audio_data[:, :max_length]
        else:
            temp_wav = torch.zeros(1, max_length)
            temp_wav[:, :audio_data.size(1)] = audio_data
            audio_data = temp_wav
        
        # 准备输入
        mixed = audio_data.to(self.device)
        mixed_resample = torchaudio.functional.resample(
            mixed, 
            self.config.train_data['sr'], 
            self.config.train_data['resample_rate']
        )
        
        # 推理
        with torch.no_grad():
            # 获取文本嵌入
            embed_pos = self.clap_model.get_text_embedding([text_query], use_tensor=True)
            embed_neg = self.clap_model.get_text_embedding(["background noise"], use_tensor=True)
            embed = torch.concat([embed_pos, embed_neg], dim=-1)
            
            # 计算STFT
            stft = self.base_model.stft
            real, imag = stft(mixed)
            mag, cos, sin = self.base_model.features[0] = mag
            
            # 清除特征缓存
            del self.base_model.features[:]
            self.base_model.features.append(mag)
            
            # 运行音频分支
            self.base_model.audio_branch({"waveform": mixed_resample})
            
            # 生成掩码
            mask = self.base_model.decoder_model(
                hidden_state=self.base_model.features[-1], 
                skip_features=self.base_model.features[:-1], 
                embed=embed
            )
            
            # 重建波形
            pred = self.base_model.wav_reconstruct(mask, mag, cos, sin, length=mixed.size(-1))
        
        # 将结果转换为CPU
        pred = pred.cpu()
        
        # 保存结果
        if output_path is not None:
            torchaudio.save(output_path, pred.unsqueeze(0), self.config.train_data['sr'])
            logging.info(f"已保存分离结果到: {output_path}")
        
        return pred


def main():
    parser = argparse.ArgumentParser(description="使用多个LORA模型进行音频分离")
    parser.add_argument('--config', type=str, default='./experiments/CLAPSep_multiple_lora/config.json',
                        help="配置文件路径")
    parser.add_argument('--lora_dir', type=str, default='./experiments/CLAPSep_multiple_lora/checkpoints',
                        help="LORA模型目录")
    parser.add_argument('--input', type=str, required=True,
                        help="输入音频文件路径")
    parser.add_argument('--output_dir', type=str, default='./output',
                        help="输出目录")
    parser.add_argument('--query', type=str, required=True,
                        help="文本查询")
    parser.add_argument('--lora_model', type=str,
                        help="指定使用的LORA模型名称，如果不指定，则使用所有模型")
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 创建LORA模型管理器
    manager = LoraModelManager(args.config)
    
    # 加载LORA模型
    if args.lora_model:
        # 加载指定模型
        lora_path = os.path.join(args.lora_dir, args.lora_model, f"{args.lora_model}_final.pt")
        if not os.path.exists(lora_path):
            # 尝试查找最新的检查点
            import glob
            checkpoints = glob.glob(os.path.join(args.lora_dir, args.lora_model, f"{args.lora_model}-*.pt"))
            if checkpoints:
                checkpoints.sort(key=os.path.getmtime, reverse=True)
                lora_path = checkpoints[0]
            else:
                raise FileNotFoundError(f"找不到LORA模型: {args.lora_model}")
        
        lora_name = manager.load_lora_model(lora_path, args.lora_model)
        manager.switch_lora(lora_name)
        
        # 分离音频
        output_path = os.path.join(args.output_dir, f"{os.path.basename(args.input).split('.')[0]}_{args.lora_model}.wav")
        manager.separate_audio(args.input, args.query, output_path)
    else:
        # 加载所有模型
        import glob
        lora_dirs = [d for d in os.listdir(args.lora_dir) if os.path.isdir(os.path.join(args.lora_dir, d))]
        
        for lora_dir in lora_dirs:
            # 查找最终模型或最新的检查点
            final_model = os.path.join(args.lora_dir, lora_dir, f"{lora_dir}_final.pt")
            if os.path.exists(final_model):
                lora_path = final_model
            else:
                # 尝试查找最新的检查点
                checkpoints = glob.glob(os.path.join(args.lora_dir, lora_dir, f"{lora_dir}-*.pt"))
                if not checkpoints:
                    logging.warning(f"找不到LORA模型: {lora_dir}，跳过")
                    continue
                checkpoints.sort(key=os.path.getmtime, reverse=True)
                lora_path = checkpoints[0]
            
            # 加载并切换LORA模型
            lora_name = manager.load_lora_model(lora_path, lora_dir)
            manager.switch_lora(lora_name)
            
            # 分离音频
            output_path = os.path.join(args.output_dir, f"{os.path.basename(args.input).split('.')[0]}_{lora_name}.wav")
            manager.separate_audio(args.input, args.query, output_path)


if __name__ == '__main__':
    main()