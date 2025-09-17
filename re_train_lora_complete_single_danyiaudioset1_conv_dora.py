import argparse
import logging
import os
os.environ["HF_HUB_DISABLE_SSL_VERIFICATION"] = "1"  # 禁用 SSL 验证
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import pathlib
import re
import json
from typing import List, NoReturn
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
from torch.utils.data import DataLoader, Dataset
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
import gc
from models.resunet import ResUNet30
from doramodel.conv_dora import DoRAManager
from models.audiosep import AudioSep, get_model_class
from models.clap_encoder import CLAP_Encoder
from data.waveform_mixers1 import SegmentMixer
from losses import get_loss_function
from utils import create_logging, parse_yaml, calculate_sdr, calculate_sisdr, get_mean_sdr_from_dict
from lightning.pytorch.callbacks import Callback
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.lr_scheduler import SequentialLR, ReduceLROnPlateau
from optimizers.lr_schedulers import get_lr_lambda,constant_warm_up
torch.set_float32_matmul_precision('high')
from data.csv_dataset_for_eval import CSVAudioTextDatasetForEval
def seed_everything(seed=42):
    """设置随机种子以确保结果可重现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def clear_memory():
    """清理GPU内存"""
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class MemoryMonitorCallback(Callback):
    """监控GPU内存使用情况的回调函数"""

    def __init__(self, frequency=100):
        super().__init__()
        self.frequency = frequency

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx % self.frequency == 0 and torch.cuda.is_available():
            clear_memory()
class LossMonitorCallback(Callback):
    """监控训练和验证损失，记录与历史最低值的比较信息"""

    def __init__(self, monitor_train: str = "train_loss", monitor_val: str = "val_loss", mode: str = "min"):
        super().__init__()
        self.monitor_train = monitor_train
        self.monitor_val = monitor_val
        self.mode = mode
        self.train_losses = []  # 存储每个epoch的训练损失
        self.val_losses = []    # 存储每个epoch的验证损失
        self.train_wait = 0     # 训练损失未改善的等待计数
        self.val_wait = 0       # 验证损失未改善的等待计数
        self._last_epoch_processed = -1  # 防止重复处理同一epoch
    
    def on_train_epoch_end(self, trainer, pl_module):
        # 确保每个epoch只处理一次
        if trainer.current_epoch == self._last_epoch_processed:
            return
        self._last_epoch_processed = trainer.current_epoch
        
        # 获取当前训练损失
        train_loss = trainer.callback_metrics.get(self.monitor_train)
        val_loss = trainer.callback_metrics.get(self.monitor_val)
        
        if train_loss is not None:
            current_train = train_loss.item()
            self.train_losses.append(current_train)  # 将当前损失添加到历史列表
            
            # 判断当前损失是否相比历史最低值有所改进
            if len(self.train_losses) == 1:
                # 第一个epoch，无可比性
                train_improved = True
                train_best = current_train
            else:
                if self.mode == "min":
                    train_best = min(self.train_losses[:-1])
                    train_improved = current_train < train_best
                else:
                    train_best = max(self.train_losses[:-1])
                    train_improved = current_train > train_best
            
            # 更新等待计数器
            if train_improved:
                self.train_wait = 0
            else:
                self.train_wait += 1
            
            # 日志记录当前状态
            logging.info(
                f"Epoch {trainer.current_epoch}: "
                f"train_loss={current_train:.4f}, "
                f"best_train_loss={train_best:.4f}, "
                f"train_wait={self.train_wait} epochs"
            )
        
        if val_loss is not None:
            current_val = val_loss.item()
            self.val_losses.append(current_val)  # 将当前损失添加到历史列表
            
            # 判断当前损失是否相比历史最低值有所改进
            if len(self.val_losses) == 1:
                # 第一个epoch，无可比性
                val_improved = True
                val_best = current_val
            else:
                if self.mode == "min":
                    val_best = min(self.val_losses[:-1])
                    val_improved = current_val < val_best
                else:
                    val_best = max(self.val_losses[:-1])
                    val_improved = current_val > val_best
            
            # 更新等待计数器
            if val_improved:
                self.val_wait = 0
            else:
                self.val_wait += 1
            
            # 日志记录当前状态
            logging.info(
                f"Epoch {trainer.current_epoch}: "
                f"val_loss={current_val:.4f}, "
                f"best_val_loss={val_best:.4f}, "
                f"val_wait={self.val_wait} epochs"
            )
class ModelSaveCallback(Callback):
    """在达到指定epoch后每个epoch结束时保存模型"""

    def __init__(self, save_dir, model_name_prefix="dora_model", start_epoch=0):
        super().__init__()
        self.save_dir = save_dir
        self.model_name_prefix = model_name_prefix
        self.start_epoch = start_epoch
    
    def on_train_epoch_end(self, trainer, pl_module):
        # 只有在达到指定epoch后才开始保存
        if trainer.current_epoch >= self.start_epoch:
            # 创建保存目录（如果不存在）
            os.makedirs(self.save_dir, exist_ok=True)
            
            # 构建保存路径
            save_path = os.path.join(
                self.save_dir, 
                f"{self.model_name_prefix}_epoch_{trainer.current_epoch}.pt"
            )
            
            # 保存模型（只保存DoRA模型的状态字典）
            torch.save({
                'dora_state_dict': pl_module.dora_manager.dora_model.state_dict(),
                # 'epoch': trainer.current_epoch,
                # 'global_step': trainer.global_step
            }, save_path)
            
            logging.info(f"已保存模型到 {save_path}")
class CustomAudioTextDataset(Dataset):
 

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
    """
    数据模块，用于加载和处理数据
    """

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



class DoRAAudioSep(pl.LightningModule):
    """
    使用DoRA的AudioSep模型，用于微调
    """

    def __init__(
            self,
            base_model,
            batch_size,
            dora_manager,
            waveform_mixer,
            query_encoder,
            loss_function,
            # reduce_lr_steps=None,
            learning_rate=0.0005137044879409329,
            use_text_ratio=1.0,
            # warm_up_steps =None
    ):
        self.sync_dist = torch.cuda.device_count() > 1  # 自动检测环境
        super().__init__()
        self.base_model = base_model
        self.dora_manager = dora_manager
        self.waveform_mixer = waveform_mixer
        self.query_encoder = query_encoder
        self.query_encoder_type = self.query_encoder.encoder_type
        self.loss_function = loss_function
        self.learning_rate = learning_rate
        self.use_text_ratio = use_text_ratio
        self.batch_size=batch_size
    def forward(self, x):
        return self.dora_manager.forward(x)

    def training_step(self, batch_data_dict, batch_idx):
        # 固定随机种子以确保结果可重现
        random.seed(batch_idx)

        batch_audio_text_dict = batch_data_dict['audio_text']
        batch_text = batch_audio_text_dict['text']
        batch_audio = batch_audio_text_dict['waveform']
        device = batch_audio.device

        # 确保waveform_mixer中的所有组件都在正确的设备上
        if hasattr(self.waveform_mixer, 'lora_adapter') and next(self.waveform_mixer.lora_adapter.parameters()).device != device:
            self.waveform_mixer.lora_adapter = self.waveform_mixer.lora_adapter.to(device)

        # 混合波形
        mixtures, segments = self.waveform_mixer(waveforms=batch_audio)
        # 确保mixtures和segments有梯度并在正确的设备上
        mixtures = mixtures.detach().to(device).requires_grad_(True)
        segments = segments.detach().to(device).requires_grad_(True)
        
        # 计算条件嵌入
        if self.query_encoder_type == 'CLAP':
            # 确保query_encoder在正确的设备上
            if hasattr(self.query_encoder, 'model') and next(self.query_encoder.model.parameters()).device != device:
                self.query_encoder.model = self.query_encoder.model.to(device)
                
            conditions = self.query_encoder.get_query_embed(
                modality='text',
                text=batch_text,
                audio=segments.squeeze(1),
                use_text_ratio=self.use_text_ratio,
                device=device
            )
            conditions = conditions.to(device)
        # 准备输入
        input_dict = {
            'mixture': mixtures[:, None, :].squeeze(1).to(device),
            'condition': conditions.to(device),
        }

        # 目标
        target_dict = {
            'segment': segments.squeeze(1).to(device),
        }

        # 确保dora_manager在正确的设备上
        if hasattr(self.dora_manager, 'dora_model') and next(self.dora_manager.dora_model.parameters()).device != device:
            self.dora_manager.dora_model = self.dora_manager.dora_model.to(device)

        output_dict = self.forward(input_dict)
        sep_segment = output_dict['waveform'].squeeze().to(device)
        # 确保输出张量有梯度
        if not sep_segment.requires_grad:
            logging.warning("输出张量没有梯度，这可能导致反向传播失败")
            # 尝试强制设置requires_grad
            sep_segment = sep_segment.detach().to(device).requires_grad_(True)
        # 准备输出
        output_dict_for_loss = {
            'segment': sep_segment,
        }

        # 计算损失
        loss = self.loss_function(output_dict_for_loss, target_dict)

        # 检查损失是否有梯度
        if not loss.requires_grad:
            logging.error("损失没有梯度，无法进行反向传播")
            # 这种情况下，我们需要创建一个有梯度的损失
            # 通过乘以一个需要梯度的张量
            dummy = torch.ones(1, device=device, requires_grad=True)
            loss = loss * dummy

        # 记录损失
        self.log('train_loss', loss, on_step=False, on_epoch=True, sync_dist=self.sync_dist, prog_bar=True, batch_size=self.batch_size)
        return loss

    def __init_val_metrics(self):
        """初始化验证指标存储结构"""
        if not hasattr(self, 'val_sdris_dict'):
            self.val_sdris_dict = {}
            self.val_sisdrs_dict = {}
            self.val_samples_processed = 0
            self.val_total_samples = 0
            
    def validation_step(self, batch_data_dict, batch_idx):
        # 初始化验证指标存储结构
        self.__init_val_metrics()
        
        # 使用CSVAudioTextDataset提供的预加载音频数据
        batch_audio_text_dict = batch_data_dict['audio_text']
        batch_text = batch_audio_text_dict['text']
        batch_audio = batch_audio_text_dict['waveform']  # 干净音频
        batch_mixture = batch_audio_text_dict['mixture']  # 混合音频
        device = batch_audio.device
        
        # 获取类别ID（如果有）- 确保使用正确的idx字段
        class_id = None
        if 'idx' in batch_audio_text_dict:
            class_id = batch_audio_text_dict['idx']
        else:
            # 尝试从数据集的原始数据中获取idx
            try:
                # 获取当前批次在数据集中的索引
                dataset_idx = batch_idx
                if hasattr(self.trainer.datamodule.val_dataset, 'data'):
                    class_id = self.trainer.datamodule.val_dataset.data[dataset_idx]['idx']
            except (AttributeError, IndexError, KeyError):
                # 如果无法获取，则使用batch_idx作为备用
                class_id = str(batch_idx)
            
        # 直接使用预加载的音频数据
        segments = batch_audio.detach().to(device)
        mixtures = batch_mixture.detach().to(device)
        
        # 转换为numpy数组用于计算SDR - 与evaluate_single_dora_single_danyiaduioset.py保持一致的处理方式
        source = segments.squeeze().cpu().numpy()
        mixture = mixtures.squeeze().cpu().numpy()
        
        # 计算基准SDR - 确保与evaluate_single_dora_single_danyiaduioset.py一致
        sdr_no_sep = calculate_sdr(ref=source, est=mixture)
        
        # 模型推理
        with torch.no_grad():
            # 准备输入 - 与evaluate_single_dora_single_danyiaduioset.py保持一致的格式
            input_dict = {
                "mixture": torch.Tensor(mixture)[None, None, :].to(device),
                "condition": self.query_encoder.get_query_embed(
                    modality='text',
                    text=batch_text,
                    device=device
                ),
            }
            output_dict = self.dora_manager.dora_model(input_dict)
            # 处理输出 - 与evaluate_single_dora_single_danyiaduioset.py完全一致
            sep_segment = output_dict["waveform"]
            sep_segment = sep_segment.squeeze(0).squeeze(0).cpu().numpy()

            # 计算评估指标 - 确保与evaluate_single_dora_single_danyiaduioset.py一致
            sdr = calculate_sdr(ref=source, est=sep_segment)
            sdri = sdr - sdr_no_sep
            sisdr = calculate_sisdr(ref=source, est=sep_segment)

            # 确保class_id是字符串类型，便于一致性比较
            if class_id is None:
                class_id = str(batch_idx)
            else:
                class_id = str(class_id)
                
            # 存储指标到类属性中
            if class_id not in self.val_sdris_dict:
                self.val_sdris_dict[class_id] = []
                self.val_sisdrs_dict[class_id] = []

            self.val_sdris_dict[class_id].append(sdri)
            self.val_sisdrs_dict[class_id].append(sisdr)
            
            # 更新处理的样本数
            self.val_samples_processed += 1
            
            # 在每个验证步骤结束时检查是否所有样本都已处理完毕
            # 获取验证数据集的总样本数
            if self.val_total_samples == 0 and self.trainer is not None and self.trainer.datamodule is not None:
                val_dataloader = self.trainer.datamodule.val_dataloader()
                if val_dataloader is not None:
                    self.val_total_samples = len(val_dataloader)
            
            # 如果所有样本都已处理完毕，计算最终指标
            if self.val_samples_processed >= self.val_total_samples and self.val_total_samples > 0:
                self._compute_validation_metrics()
    
    def _compute_validation_metrics(self):
        """计算验证指标并记录到日志"""
        # 检查是否收集到指标
        if self.val_sdris_dict:
            # 按类别ID分组并计算每个类别的中位数SDRi
            class_median_sdris = []
            class_median_sisdrs = []
            
            # 记录每个类别的指标
            for class_id, sdris in self.val_sdris_dict.items():
                if sdris:  # 确保有数据
                    median_sdri = np.nanmedian(sdris)
                    class_median_sdris.append(median_sdri)
                    # 只记录样本数量，不记录详细的SDRi值
                    # logging.info(f"类别 {class_id} 的样本数量: {len(sdris)}")

            if class_median_sdris:
                # 计算所有类别中位数的平均值 - 使用与evaluate_single_dora_single_danyiaduioset.py相同的函数
                # 将中位数字典转换为与get_mean_sdr_from_dict函数兼容的格式
                sdris_dict_for_mean = {class_id: np.nanmedian(sdris) for class_id, sdris in self.val_sdris_dict.items() if sdris}
                avg_class_median_sdri = get_mean_sdr_from_dict(sdris_dict_for_mean)
                
                # 安全获取设备
                try:
                    device = self.device
                except AttributeError:
                    device = next(self.parameters()).device
                    
                # 使用与evaluate_single_dora_single_danyiaduioset.py完全一致的损失计算方式
                # SDRi越高越好，所以损失应该是-SDRi，确保绝对值一致
                val_loss = -torch.tensor(avg_class_median_sdri, device=device)

                # 记录验证损失
                self.log('val_loss', val_loss, on_step=False, on_epoch=True, sync_dist=self.sync_dist, prog_bar=True)
                logging.info(f"验证损失: {val_loss.item():.4f} (平均SDRi: {avg_class_median_sdri:.4f} dB)")
                # 输出与evaluate_single_dora_single_danyiaduioset.py一致的格式
                print(f"Model  Avg SDRi: {avg_class_median_sdri:.4f}, SISDR: {get_mean_sdr_from_dict({class_id: np.nanmedian(sisdrs) for class_id, sisdrs in self.val_sisdrs_dict.items() if sisdrs}):.4f}")
                
                # 不再记录每个类别的详细指标，减少日志输出
            else:
                # 如果没有足够数据，返回默认损失
                default_loss = torch.tensor(0.0, device=self.device)
                self.log('val_loss', default_loss, on_step=False, on_epoch=True, sync_dist=self.sync_dist,
                         prog_bar=True)
                logging.info("未收集到足够的类别数据，使用默认损失值 0.0")
        else:
            logging.warning("验证完成，但未收集到任何SDR指标，请检查验证数据集是否正确配置")
            self.log('val_loss', torch.tensor(float('inf'), device=self.device), on_step=False, on_epoch=True, 
                     sync_dist=self.sync_dist, prog_bar=True)
        
        # 清空指标字典，准备下一个epoch
        self.val_sdris_dict = {}
        self.val_sisdrs_dict = {}
        self.val_samples_processed = 0
        
    def on_validation_epoch_start(self):
        """在验证epoch开始时初始化指标存储"""
        self.__init_val_metrics()
        
    def on_validation_epoch_end(self):
        """确保在验证epoch结束时计算指标（以防validation_step中未能计算）"""
        if hasattr(self, 'val_sdris_dict') and self.val_sdris_dict and self.val_samples_processed > 0:
            self._compute_validation_metrics()

    def configure_optimizers(self):
        # 只优化DoRA参数
        optimizer = optim.AdamW(
            [p for p in self.dora_manager.dora_model.parameters() if p.requires_grad],
            lr=self.learning_rate,
            amsgrad=True,
            # betas=(0.8709968372372442, 0.9648747006299911),
            weight_decay=0.0,
        )
        # 创建学习率调度器
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.7,
            patience=25

        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
            },

        }

def get_dirs(workspace, filename, config_yaml):
    """获取目录和路径"""
    os.makedirs(workspace, exist_ok=True)

    yaml_name = pathlib.Path(config_yaml).stem

    # 检查点目录
    checkpoints_dir = os.path.join(
        workspace,
        "checkpoints",
        "dora_models",
        yaml_name
    )
    os.makedirs(checkpoints_dir, exist_ok=True)

    # 日志目录
    logs_dir = os.path.join(
        workspace,
        "logs",
        "dora_models",
        yaml_name
    )
    os.makedirs(logs_dir, exist_ok=True)

    # TensorBoard日志目录
    tf_logs_dir = os.path.join(
        workspace,
        "tf_logs",
        "dora_models",
        yaml_name
    )
    os.makedirs(tf_logs_dir, exist_ok=True)

    return checkpoints_dir, logs_dir, tf_logs_dir


def train_dora(args):
    """为特定聚类训练DoRA模型"""
    # 设置随机种子
    seed_everything(42)

    # 参数和配置
    workspace = args.workspace
    config_yaml = args.config_yaml
    checkpoint_path = args.checkpoint_path
    max_epochs = args.max_epochs

    # 获取目录
    checkpoints_dir, logs_dir, tf_logs_dir = get_dirs(
        workspace, "train_dora", config_yaml
    )

    # 创建日志
    create_logging(logs_dir, filemode="w")
    logging.info(args)

    # 读取配置
    configs = parse_yaml(config_yaml)

    # 数据配置
    max_mix_num = configs['data']['max_mix_num']
    sampling_rate = configs['data']['sampling_rate']
    lower_db = configs['data']['loudness_norm']['lower_db']
    higher_db = configs['data']['loudness_norm']['higher_db']

    # 模型配置
    model_type = configs['model']['model_type']
    input_channels = configs['model']['input_channels']
    output_channels = configs['model']['output_channels']
    condition_size = configs['model']['condition_size']
    use_text_ratio = configs['model']['use_text_ratio']

    # 训练配置
    original_batch_size = configs['train']['batch_size_per_device']
    num_workers = configs['train']['num_workers']
    loss_type = configs['train']['loss_type']

    batch_size = original_batch_size
    # if torch.cuda.is_available():
    #     free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
    #     free_memory_gb = free_memory / (1024 ** 3)
    #
    #     # 根据可用内存调整batch_size
    #     if free_memory_gb < 4:  # 小于4GB可用内存
    #         batch_size = max(1, original_batch_size // 8)
    #     elif free_memory_gb < 8:  # 小于8GB可用内存
    #         batch_size = max(1, original_batch_size // 4)
    #     elif free_memory_gb < 12:  # 小于12GB可用内存
    #         batch_size = max(2, original_batch_size // 2)
    #
    #     logging.info(f"使用batch_size: {batch_size}, 原始batch_size: {original_batch_size}")
    # 加载数据集

    json_file = os.path.join( f"/s1home/zyk524/AudioSep/LORA/s4/cluster6.json")
    dataset = CustomAudioTextDataset(
        json_file=json_file,
        sampling_rate=sampling_rate,
        max_clip_len=configs['data']['segment_seconds']
    )

    # 使用训练数据集
    train_dataset = dataset

    
    val_dataset = CSVAudioTextDatasetForEval(
        csv_file=f"/s1home/zyk524/AudioSep/LORA/s4/csv/cluster6.csv",
        sampling_rate=sampling_rate,
    )
    
    # 创建数据模块
    data_module = DataModule(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        num_workers=num_workers,
        batch_size=batch_size
    )

    # 创建基础模型
    base_model = get_model_class(model_type)(
        input_channels=input_channels,
        output_channels=output_channels,
        condition_size=condition_size,
    )
    # 清理内存
    clear_memory()
    # 加载预训练权重
    logging.info(f"加载预训练模型权重: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # 检查checkpoint的结构，确定如何加载权重
    if 'state_dict' in checkpoint:
        # 如果是Lightning模型保存的checkpoint
        state_dict = {}
        for key, value in checkpoint['state_dict'].items():
            if key.startswith('ss_model.'):
                # 去掉'ss_model.'前缀
                state_dict[key[9:]] = value
        base_model.load_state_dict(state_dict, strict=False)
    else:
        # 尝试直接加载
        try:
            base_model.load_state_dict(checkpoint, strict=False)
        except Exception as e:
            logging.error(f"加载预训练模型失败: {e}")
            raise ValueError("无法加载预训练模型权重")

    # 冻结基础模型参数
    for param in base_model.parameters():
        param.requires_grad = False
    dora_manager = DoRAManager(base_model)
    k = sum(p.numel() for p in dora_manager.dora_model.parameters() if p.requires_grad)  # 计算可训练参数总数
    print('# dora_model of parameters:', k)  # 打印参数数量
    loss_function = get_loss_function(loss_type)

    # 创建波形混合器
    segment_mixer = SegmentMixer(
        max_mix_num=max_mix_num,
        lower_db=lower_db,
        higher_db=higher_db
    )

    # 创建查询编码器
    query_encoder = CLAP_Encoder()

    # 创建DoRA模型
    pl_model = DoRAAudioSep(
        base_model=base_model,
        dora_manager=dora_manager,
        waveform_mixer=segment_mixer,
        query_encoder=query_encoder,
        loss_function=loss_function,
        batch_size=batch_size,
        learning_rate=float(configs['train']['optimizer']['learning_rate']),
        use_text_ratio=use_text_ratio
    )

    # 创建TensorBoard日志记录器
    logger = TensorBoardLogger(save_dir=tf_logs_dir, name=f"cluster0")

    # 创建回调函数
    callbacks = [
        # 监控训练和验证损失
        LossMonitorCallback(
            monitor_train="train_loss",
            monitor_val="val_loss",
            mode="min"
        ),
        # 自动保存最新检查点用于断点续训
        ModelCheckpoint(
            dirpath=checkpoints_dir,
            filename=f"dora0_last",
            save_top_k=1,
            save_last=True,
        ),
        # 学习率监控
        LearningRateMonitor(logging_interval='step'),
        # 在达到100个epoch后每个epoch保存一次模型
        ModelSaveCallback(
            save_dir=checkpoints_dir,
            model_name_prefix="dora0_model",
            start_epoch=149
        ),
    ]
    # 创建训练器
    trainer = pl.Trainer(
        logger=logger,
        accelerator='cuda',
        devices=args.num_gpus if args.num_gpus is not None else 'auto',
        strategy=args.strategy,
        max_epochs=max_epochs,
        callbacks=callbacks,
        log_every_n_steps=50,
        enable_checkpointing=True,  # 确保启用检查点保存
        use_distributed_sampler=False,
        enable_progress_bar=True,
        enable_model_summary=True,
        sync_batchnorm=True,  # 在多GPU训练时同步批归一化层的统计信息
        precision="32-true",
        # num_sanity_val_steps=2,
        # accumulate_grad_batches=4,  # 梯度累积，减少内存使用
        gradient_clip_val=1.0,  # 梯度裁剪，防止梯度爆炸
        inference_mode=False,  # 关闭推理模式，避免某些操作的内存泄漏
        # 添加自动恢复训练的配置
        # auto_lr_find=False,  # 禁用自动学习率查找，以避免与恢复训练冲突 已弃用
        # auto_scale_batch_size=False,  # 禁用自动批量大小缩放，以避免与恢复训练冲突 已弃用
        detect_anomaly=False  # 关闭异常检测，减少内存使用
    )


    last_checkpoint_path = os.path.join(checkpoints_dir, f"last.ckpt")
    resume_checkpoint = None

    # 根据命令行参数决定是否从检查点恢复训练
    if args.resume and os.path.exists(last_checkpoint_path):
        logging.info(f"找到上次训练的检查点: {last_checkpoint_path}，将从该检查点恢复训练")
        resume_checkpoint = last_checkpoint_path

        # 加载检查点信息以记录日志
        ckpt_info = torch.load(last_checkpoint_path, map_location='cpu')
        if 'epoch' in ckpt_info:
            logging.info(f"恢复训练：从第 {ckpt_info['epoch']} 轮继续训练")
        if 'global_step' in ckpt_info:
            logging.info(f"恢复训练：从第 {ckpt_info['global_step']} 步继续训练")
    else:
        if args.resume and not os.path.exists(last_checkpoint_path):
            logging.warning(f"未找到上次训练的检查点: {last_checkpoint_path}，将从头开始训练")
        else:
            logging.info(f"未启用断点续训或未找到检查点，将从头开始训练")

    # 开始训练

    trainer.fit(model=pl_model, datamodule=data_module, ckpt_path=resume_checkpoint)

    # 保存最终模型
    final_save_path = os.path.join(checkpoints_dir, f"dora_cluster0_final.pt")
    torch.save({
        'dora_state_dict': dora_manager.dora_model.state_dict(),
    }, final_save_path)
    logging.info(f"已保存最终模型到 {final_save_path}")

    return dora_manager


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="使用DoRA对AudioSep模型进行微调训练")
    parser.add_argument(
        "--workspace",
        type=str,
        default="/s1home/zyk524/AudioSep/LORA/DORA/CONDORA/cluster6",
        help="工作目录路径"
    )
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="/s1home/zyk524/AudioSep/config/audiosep_base.yaml",
        help="配置文件路径"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="/s1home/zyk524/AudioSep/checkpoint/audiosep_base_4M_steps.ckpt",
        help="预训练模型检查点路径"
    )

    parser.add_argument(
        "--max_epochs",
        type=int,
        default=150,
        help="最大训练轮数"
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=None,
        help="要使用的GPU数量，默认为自动检测"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="ddp",
        help="分布式训练策略，可选：ddp, deepspeed, fsdp等"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="是否从上次训练的检查点恢复训练"
    )

    args = parser.parse_args()

    # 设置全局随机种子
    seed_everything(42)

    # 记录是否从检查点恢复训练
    if args.resume:
        logging.info("启用断点续训功能，将尝试从上次训练的检查点恢复训练")
    else:
        logging.info("未启用断点续训功能，将从头开始训练")

    #

    train_dora(args)
    print(f"聚类DoRA模型训练完成!")

    # 在每个聚类训练完成后清理内存
    clear_memory()


if __name__ == "__main__":
     main()
