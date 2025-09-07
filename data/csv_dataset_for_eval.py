import os
import csv
import torch
import torchaudio
import librosa
from torch.utils.data import Dataset
import logging

class CSVAudioTextDatasetForEval(Dataset):
    """
    从CSV文件加载音频和文本数据的数据集，专用于评估
    不进行裁剪和填充，直接加载完整音频，与evaluate_single_dora_single_danyiaduioset.py保持一致
    CSV文件格式应包含以下列：idx,caption,src_wav,noise_wav
    """

    def __init__(self, csv_file, sampling_rate=32000):
        self.sampling_rate = sampling_rate
        self.data = self._load_data(csv_file)

    def _load_data(self, csv_file):
        data_list = []
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                data_list.append({
                    'idx': row.get('idx', ''),
                    'caption': row.get('caption', ''),
                    'src_wav': row.get('src_wav', ''),
                    'noise_wav': row.get('noise_wav', '')
                })
        
        logging.info(f"从CSV文件加载了 {len(data_list)} 条数据")
        return data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 尝试加载有效的音频对，如果当前索引的音频对损坏，则尝试下一个
        max_attempts = 10  # 最多尝试10次
        for attempt in range(max_attempts):
            current_idx = (idx + attempt) % len(self.data)  # 循环索引，避免越界
            item = self.data[current_idx]
            caption = item['caption']
            src_wav_path = item['src_wav'].strip('"')  # 移除可能的引号
            noise_wav_path = item['noise_wav'].strip('"')  # 移除可能的引号
            
            # 如果文件不存在，跳过
            if not os.path.exists(src_wav_path) or not os.path.exists(noise_wav_path):
                logging.warning(f"文件不存在: {src_wav_path} 或 {noise_wav_path}")
                continue

            try:
                # 使用librosa加载音频，与evaluate_single_dora_single_danyiaduioset.py保持一致
                source, _ = librosa.load(src_wav_path, sr=self.sampling_rate, mono=True)
                mixture, _ = librosa.load(noise_wav_path, sr=self.sampling_rate, mono=True)
                
                # 检查音频是否有效
                if len(source) == 0 or len(mixture) == 0:
                    logging.warning(f"音频文件为空: {src_wav_path} 或 {noise_wav_path}")
                    continue
                
                # 转换为torch张量
                source_tensor = torch.tensor(source).unsqueeze(0)  # 添加通道维度
                mixture_tensor = torch.tensor(mixture).unsqueeze(0)  # 添加通道维度
                
                # 返回结果，不进行裁剪或填充
                return {
                    'audio_text': {
                        'text': caption,
                        'waveform': source_tensor,  # 干净音频
                        'mixture': mixture_tensor,  # 混合音频
                        'modality': 'audio_text',
                        'idx': item['idx']  # 添加idx用于分组评估
                    }
                }
                
            except Exception as e:
                logging.error(f"加载音频文件时出错: {e}")
                continue  # 尝试下一个音频
        
        # 如果尝试了最大次数仍未找到有效音频对，则创建随机数据
        # 注意：这里创建的随机数据长度是固定的，仅作为应急措施
        logging.warning(f"尝试{max_attempts}次后仍未找到有效音频，使用随机数据")
        random_length = self.sampling_rate * 5  # 5秒随机数据
        source_tensor = torch.randn(1, random_length)
        mixture_tensor = torch.randn(1, random_length)
        
        return {
            'audio_text': {
                'text': caption,
                'waveform': source_tensor,
                'mixture': mixture_tensor,
                'modality': 'audio_text',
                'idx': item['idx']
            }
        }
