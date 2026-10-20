#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：CLAPSep
@File ：train_7_decoder_lora_clusters.py
@Description：训练7个独立的解码器LoRA模块（cluster0到cluster6）
"""

import argparse
import logging
import os
import json
import random
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor, Callback
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import Dataset, DataLoader
import torchaudio
import gc
import laion_clap
from model.CLAPSep_decoder1 import HTSAT_Decoder
from model.CLAPSep import LightningModule
from data.waveform_mixers_zidong_bi_fast_7 import SegmentMixer

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
    gc.collect()

# ==========================================
# 只基于训练损失保存模型的 Callback
# ==========================================
class ModelSaveCallback(Callback):
    """
    只根据训练损失(train_loss)的最优值保存模型
    """
    def __init__(self, save_dir, model_name_prefix="decoder_lora"):
        super().__init__()
        self.save_dir = save_dir
        self.model_name_prefix = model_name_prefix
        self.best_train_loss = float('inf')
        self.last_best_train_file = None

    def _delete_old_file(self, file_path):
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
                logging.info(f"已删除旧模型文件: {file_path}")
            except Exception as e:
                logging.warning(f"删除旧文件失败 {file_path}: {str(e)}")

    def _save_model(self, trainer, pl_module, loss_value=None):
        if loss_value is not None:
            # 保存最优训练损失模型
            formatted_loss = f"{loss_value:.4f}".replace('.', '_')
            file_name = f"{self.model_name_prefix}_best_train_loss.pt"
        else:
            # 保存最终模型
            file_name = f"{self.model_name_prefix}_final.pt"
        
        save_path = os.path.join(self.save_dir, file_name)
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 只保存解码器LoRA参数
        decoder_lora_state_dict = {}
        for name, param in pl_module.decoder_model.state_dict().items():
            if 'lora_' in name:
                decoder_lora_state_dict[f'decoder_model.{name}'] = param
        
        torch.save({
            'decoder_lora_state_dict': decoder_lora_state_dict,
            'epoch': trainer.current_epoch,
            'global_step': trainer.global_step
        }, save_path)
        
        if loss_value is not None:
             logging.info(f"Epoch {trainer.current_epoch}: 最优损失模型已保存到 {save_path}(损失值: {loss_value:.4f})")
        else:
            logging.info(f"训练完成，最终模型已保存到 {save_path}")
        return save_path

    def on_train_epoch_end(self, trainer, pl_module):
        """在每个训练epoch结束时检查"""
        current_train_loss = trainer.callback_metrics.get("train_loss")
        if current_train_loss is not None and current_train_loss < self.best_train_loss:
            # 更新最优值
            self.best_train_loss = current_train_loss.item()
            logging.info(f"发现新的最优训练损失: {self.best_train_loss:.4f}")
            
            # 删除旧的最优文件
            self._delete_old_file(self.last_best_train_file)
            
            # 保存新模型
            self.last_best_train_file = self._save_model(
                trainer, 
                pl_module, 
                self.best_train_loss
            )
            
    def on_fit_end(self, trainer, pl_module):
        """训练结束后保存最终模型"""
        logging.info("训练完成，保存最终模型。")
        self._save_model(trainer, pl_module)


class CustomAudioTextDataset(Dataset):
    """从JSON文件加载音频和文本数据的数据集"""

    def __init__(self, json_file, sampling_rate=32000, max_clip_len=10):
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
        max_attempts = 10
        for attempt in range(max_attempts):
            current_idx = (idx + attempt) % len(self.data)
            item = self.data[current_idx]
            wav_path = item['wav']
            caption = item['caption']
            
            try:
                # 使用torchaudio加载音频文件
                audio_data, audio_rate = torchaudio.load(wav_path, channels_first=True)
                
                # 检查音频是否为空或损坏
                if audio_data.numel() == 0 or torch.isnan(audio_data).any() or torch.isinf(audio_data).any():
                    logging.warning(f"音频文件为空或损坏: {wav_path}")
                    continue
                
                # 转换为单声道
                if audio_data.shape[0] > 1:
                    audio_data = (audio_data[0] + audio_data[1]) / 2
                    audio_data = audio_data.unsqueeze(0)

                # 重采样音频
                if audio_rate != self.sampling_rate:
                    audio_data = torchaudio.functional.resample(audio_data, orig_freq=audio_rate, new_freq=self.sampling_rate)

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
                    continue
                
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
                continue
        
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

# ==========================================
# DataModule 移除验证集
# ==========================================
class DataModule(pl.LightningDataModule):
    def __init__(self, train_dataset, num_workers=4, batch_size=32):
        super().__init__()
        self.train_dataset = train_dataset
        self.num_workers = num_workers
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
        )


# ==========================================
# 训练单个解码器LoRA模型
# ==========================================
def train_decoder_lora(
    cluster_id, 
    workspace, 
    config_path,
    clap_checkpoint_path,
    clapsep_checkpoint_path,
    max_epochs,
    train_json_path
):
    """训练单个解码器LoRA模型"""
    set_seed(114514)
    
    # 创建目录
    cluster_name = f"cluster{cluster_id}"
    checkpoints_dir = os.path.join(workspace, "checkpoints", cluster_name)
    logs_dir = os.path.join(workspace, "logs", cluster_name)
    tf_logs_dir = os.path.join(workspace, "tf_logs", cluster_name)
    
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(tf_logs_dir, exist_ok=True)
    
    logging.info(f"--- 开始为聚类 {cluster_id} 进行训练 ---")
    logging.info(f"Workspace: {workspace}")
    logging.info(f"Train JSON: {train_json_path}")
    
    # 加载配置文件
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 创建波形混合器 - 排除当前cluster_id的数据用于混合噪声
    segment_mixer = SegmentMixer(
        max_mix_num=config.get('max_mix_num', 2),
        lower_db=config.get('lower_db', -10),
        higher_db=config.get('higher_db', 10),
        exclude_cluster_id=cluster_id
    )
    logging.info(f"已初始化SegmentMixer用于音频混合，排除聚类 {cluster_id}")
    
    # 加载数据集
    train_dataset = CustomAudioTextDataset(
        json_file=train_json_path,
        sampling_rate=config['train_data'].get('sr', 32000),
        max_clip_len=config['train_data'].get('segment_seconds', 10)
    )
    logging.info(f"加载训练数据集: {train_json_path}, 包含 {len(train_dataset)} 个元素")
    
    # 创建数据模块
    data_module = DataModule(
        train_dataset=train_dataset,
        num_workers=config.get('n_workers', 4),
        batch_size=config.get('batch_size', 32)
    )
    
    # 加载CLAP模型
    logging.info(f"加载CLAP预训练模型: {clap_checkpoint_path}")
    clap_model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base', device='cpu')
    clap_model.load_ckpt(clap_checkpoint_path)
    logging.info("CLAP预训练模型加载完成")
    
    # 创建解码器
    decoder = HTSAT_Decoder(**config['model'])
    
    # 创建LightningModule - 使用解码器LoRA
    lightning_module = LightningModule(
        clap_model, 
        decoder, 
        lr=1e-5,
        use_lora=True,
        rank=config.get('lora_rank', 16),
        nfft=config.get('nfft', 1024),
        waveform_mixer=segment_mixer,
        pretrained_checkpoint=clapsep_checkpoint_path,
        use_lora_decoder=True
    )
    
    # 计算并记录可训练参数数量
    trainable_params = sum(p.numel() for p in lightning_module.parameters() if p.requires_grad)
    logging.info(f"可训练参数数量: {trainable_params}")
    
    # 创建回调
    callbacks = [
        # 1. 使用标准的 EarlyStopping，监控 train_loss
        #EarlyStopping(
            #monitor="train_loss",
            #min_delta=0.0,
            #patience=15,
           #verbose=True,
            #mode="min",
            #check_on_train_epoch_end=True
        #),
        
        # 2. 自动保存最新检查点 (Last)
        ModelCheckpoint(
            dirpath=checkpoints_dir,
            filename=f"{cluster_name}_last",
            save_top_k=1,
            save_last=True,
        ),
        
        # 3. 学习率监控
        LearningRateMonitor(logging_interval='step'),
        
        # 4. 保存最优模型 (只看 train_loss)
        ModelSaveCallback(
            save_dir=checkpoints_dir,
            model_name_prefix=cluster_name
        ),
    ]
    
    # 创建日志记录器
    logger = TensorBoardLogger(save_dir=tf_logs_dir, name=cluster_name)
    
    # 创建训练器
    trainer = pl.Trainer(
        default_root_dir=checkpoints_dir,
        accelerator='cuda',
        devices='auto',
        strategy='auto',
        max_epochs=max_epochs,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=50,
        enable_checkpointing=True,
        use_distributed_sampler=False,
        enable_progress_bar=True,
        enable_model_summary=True,
        sync_batchnorm=True,
        precision='bf16-mixed',
        gradient_clip_val=1.0,
        inference_mode=False,
        detect_anomaly=False
    )
    
    # 开始训练
    logging.info(f"开始训练聚类 {cluster_id} 的解码器LoRA模型...")
    trainer.fit(model=lightning_module, datamodule=data_module)
    
    # 保存最终模型
    final_save_path = os.path.join(checkpoints_dir, f"{cluster_name}_final.pt")
    decoder_lora_state_dict = {}
    for name, param in lightning_module.decoder_model.state_dict().items():
        if 'lora_' in name:
            decoder_lora_state_dict[f'decoder_model.{name}'] = param
    
    torch.save({
        'decoder_lora_state_dict': decoder_lora_state_dict,
        'epoch': trainer.current_epoch,
        'global_step': trainer.global_step
    }, final_save_path)
    logging.info(f"最终模型已保存到: {final_save_path}")
    
    # 清理内存
    del clap_model, decoder, lightning_module, trainer, data_module
    clear_memory()
    
    return final_save_path


def main():
    parser = argparse.ArgumentParser(description="训练7个独立的解码器LoRA模块")
    parser.add_argument("--workspace", type=str, 
                       default="/s7home/zyk524/CLAPSEP/CLAPSep/decoder_only/1e-5", 
                       help="工作目录路径")
    parser.add_argument("--base_data_dir", type=str, 
                       default="/s7home/zyk524/test/audio/SED/train_json/89checkpoint/7/train", 
                       help="包含所有聚类数据父目录的基础路径")
    parser.add_argument("--config_path", type=str, 
                       default="/s7home/zyk524/CLAPSEP/CLAPSep/experiments/decoder_lora_example/config.json", 
                       help="配置文件路径")
    parser.add_argument("--clap_checkpoint_path", type=str, 
                       default="/s7home/zyk524/test/audio/checkpoint/music_audioset_epoch_15_esc_90.14.pt", 
                       help="CLAP预训练模型检查点路径")
    parser.add_argument("--clapsep_checkpoint_path", type=str, 
                       default="/s7home/zyk524/CLAPSEP/CLAPSep/checkpoint/best_model.ckpt", 
                       help="CLAPSep预训练模型检查点路径（包含编码器LoRA）")
    parser.add_argument("--max_epochs", type=int, default=150, help="最大训练轮数")
    
    args = parser.parse_args()
    
    set_seed(114514)
    
    # 训练 cluster0 到 cluster6
    cluster_ids = list(range(0, 7))
    
    for cluster_id in cluster_ids:
        print("\n" + "="*60)
        print(f"========== 开始训练聚类 {cluster_id} 的解码器LoRA模型 ==========")
        print("="*60 + "\n")
        
        current_workspace = args.workspace
        train_json_path = os.path.join(args.base_data_dir, f"cluster{cluster_id}.json")
        
        if not os.path.exists(train_json_path):
            logging.error(f"训练文件未找到: {train_json_path}，跳过聚类 {cluster_id}")
            continue
        
        try:
            train_decoder_lora(
                cluster_id=cluster_id,
                workspace=current_workspace,
                config_path=args.config_path,
                clap_checkpoint_path=args.clap_checkpoint_path,
                clapsep_checkpoint_path=args.clapsep_checkpoint_path,
                max_epochs=args.max_epochs,
                train_json_path=train_json_path
            )
            
            print("\n" + "="*60)
            print(f"========== 聚类 {cluster_id} 解码器LoRA模型训练完成! ==========")
            print("="*60 + "\n")
            
        except Exception as e:
            logging.error(f"训练聚类 {cluster_id} 时出错: {e}")
            import traceback
            traceback.print_exc()
        
        clear_memory()
    
    print("所有聚类的解码器LoRA模型均已训练完成！")


if __name__ == "__main__":
     main()
