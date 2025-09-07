import random
import sre_compile
import numpy as np
import torch
import torch.nn as nn
import pyloudnorm as pyln
import json
import os
import sys


class SegmentMixer(nn.Module):
    def __init__(self, max_mix_num, lower_db, higher_db, json_files=None):
        super(SegmentMixer, self).__init__()

        self.max_mix_num = max_mix_num
        self.loudness_param = {
            'lower_db': lower_db,
            'higher_db': higher_db,
        }
        
        # 存储6个JSON文件的路径
        self.json_files = json_files if json_files else [
            "/home/zyk524/MSCLAP/audioset_eval/cluster7/cluster6.json",
            "/home/zyk524/MSCLAP/audioset_eval/cluster7/cluster5.json",
            "/home/zyk524/MSCLAP/audioset_eval/cluster7/cluster3.json",
            "/home/zyk524/MSCLAP/audioset_eval/cluster7/cluster0.json",
            "/home/zyk524/MSCLAP/audioset_eval/cluster7/cluster4.json",
            "/home/zyk524/MSCLAP/audioset_eval/cluster7/cluster1.json"
        ]
        
        # 加载所有JSON文件中的音频路径
        self.audio_pool = []
        self._load_audio_paths()

    def _load_audio_paths(self):
        """加载所有JSON文件中的音频路径"""
        
        self.audio_pool = []
        
        for json_file in self.json_files:


            full_path = json_file  # 如果提供的是绝对路径
                
            if os.path.exists(full_path):
                try:
                    with open(full_path, 'r') as f:
                        data = json.load(f)
                        if 'data' in data and isinstance(data['data'], list):
                            for item in data['data']:
                                if 'wav' in item:
                                    self.audio_pool.append(item['wav'])
                except Exception as e:
                    print(f"Error loading {full_path}: {e}")
            else:
                print(f"Warning: JSON file not found: {full_path}")
        
        print(f"Loaded {len(self.audio_pool)} audio paths from JSON files")
        
    def reload_audio_pool(self, json_files=None):
        """重新加载音频池，可以指定新的JSON文件列表"""
        if json_files is not None:
            self.json_files = json_files
        self._load_audio_paths()
    
    def _load_audio(self, audio_path, device=None):
        """加载音频文件并返回波形"""
        import torchaudio
        import os
        import torch.nn.functional as F
        
        try:
            # 处理文件路径
            if not os.path.exists(audio_path) and not audio_path.startswith('/'):
                # 尝试使用绝对路径
                base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                full_path = os.path.join(base_dir, audio_path)
                if os.path.exists(full_path):
                    audio_path = full_path
            
            if not os.path.exists(audio_path):
                print(f"Audio file not found: {audio_path}")
                return None
                
            waveform, sr = torchaudio.load(audio_path)
            # 检查音频是否为空或损坏
            if waveform.numel() == 0 or torch.isnan(waveform).any() or torch.isinf(waveform).any():
                print(f"Audio file is empty or corrupted: {audio_path}")
                return None
            # 确保音频是单声道的
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
                # 如果指定了设备，将波形移动到该设备上
            if device is not None:
                waveform = waveform.to(device)
            
            return waveform
        except Exception as e:
            print(f"Error loading audio {audio_path}: {e}")
            return None
    
    def __call__(self, waveforms):
        
        batch_size = waveforms.shape[0]
        device = waveforms.device  # 获取输入张量的设备
        data_dict = {
            'segment': [],
            'mixture': [],
        }

        for n in range(0, batch_size):

            segment = waveforms[n].clone()

            # create zero tensors as the background template
            noise = torch.zeros_like(segment)

            mix_num = random.randint(2, self.max_mix_num)
            assert mix_num >= 2

            for i in range(1, mix_num):
                # 从音频池中随机选择一个音频路径
                if len(self.audio_pool) > 0:
                    # 尝试最多10次加载有效的音频
                    max_attempts = 10
                    next_segment = None
                    for attempt in range(max_attempts):
                        random_audio_path = random.choice(self.audio_pool)
                        # 传递设备信息给_load_audio方法
                        next_segment = self._load_audio(random_audio_path, device=device)

                        # 如果加载成功，跳出循环
                        if next_segment is not None:
                            # 确保音频形状与目标段匹配
                            if next_segment.shape[1] != segment.shape[1]:
                                # 如果音频太短，则重复填充；如果太长，则截断
                                if next_segment.shape[1] < segment.shape[1]:
                                    # 计算需要重复的次数
                                    repeat_times = (segment.shape[1] // next_segment.shape[1]) + 1
                                    next_segment = next_segment.repeat(1, repeat_times)
                                # 截断到目标长度
                                next_segment = next_segment[:, :segment.shape[1]]
                            # 确保在正确的设备上
                            next_segment = next_segment.to(device)
                            break

                    # 如果尝试多次后仍然加载失败，使用当前批次中的音频
                    if next_segment is None:
                        print(f"尝试{max_attempts}次后仍未找到有效音频，使用批次中的备用音频")
                        next_segment = waveforms[(n + i) % batch_size]
                else:
                    # 如果音频池为空，使用当前批次中的音频
                    next_segment = waveforms[(n + i) % batch_size]

                rescaled_next_segment = dynamic_loudnorm(audio=next_segment, reference=segment, **self.loudness_param)
                # 确保在正确的设备上
                rescaled_next_segment = rescaled_next_segment.to(device)
                noise += rescaled_next_segment

            # randomly normalize background noise
            noise = dynamic_loudnorm(audio=noise, reference=segment, **self.loudness_param)
            noise = noise.to(device)
            # create audio mixyure
            mixture = segment + noise

            # declipping if need be
            max_value = torch.max(torch.abs(mixture))
            if max_value > 1:
                segment *= 0.9 / max_value
                mixture *= 0.9 / max_value

            data_dict['segment'].append(segment)
            data_dict['mixture'].append(mixture)

        for key in data_dict.keys():
            data_dict[key] = torch.stack(data_dict[key], dim=0).to(device)

        # return data_dict
        return data_dict['mixture'], data_dict['segment']


def rescale_to_match_energy(segment1, segment2):

    ratio = get_energy_ratio(segment1, segment2)
    rescaled_segment1 = segment1 / ratio
    return rescaled_segment1 


def get_energy(x):
    return torch.mean(x ** 2)


def get_energy_ratio(segment1, segment2):

    energy1 = get_energy(segment1)
    energy2 = max(get_energy(segment2), 1e-10)
    ratio = (energy1 / energy2) ** 0.5
    ratio = torch.clamp(ratio, 0.02, 50)
    return ratio


def dynamic_loudnorm(audio, reference, lower_db=-10, higher_db=10): 
    rescaled_audio = rescale_to_match_energy(audio, reference)
    
    delta_loudness = random.randint(lower_db, higher_db)

    gain = np.power(10.0, delta_loudness / 20.0)

    device = audio.device
    return (gain * rescaled_audio).to(device)


def torch_to_numpy(tensor):
    """Convert a PyTorch tensor to a NumPy array."""
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    else:
        raise ValueError("Input must be a PyTorch tensor.")


def numpy_to_torch(array):
    """Convert a NumPy array to a PyTorch tensor."""
    if isinstance(array, np.ndarray):
        return torch.from_numpy(array)
    else:
        raise ValueError("Input must be a NumPy array.")


# decayed
def random_loudness_norm(audio, lower_db=-35, higher_db=-15, sr=32000):
    device = audio.device
    audio = torch_to_numpy(audio.squeeze(0))
    # randomly select a norm volume
    norm_vol = random.randint(lower_db, higher_db)

    # measure the loudness first 
    meter = pyln.Meter(sr) # create BS.1770 meter
    loudness = meter.integrated_loudness(audio)
    # loudness normalize audio
    normalized_audio = pyln.normalize.loudness(audio, loudness, norm_vol)

    normalized_audio = numpy_to_torch(normalized_audio).unsqueeze(0)
    
    return normalized_audio.to(device)
    
