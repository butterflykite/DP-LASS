import librosa
import torchaudio
import numpy as np
import torch
from models.htsat import HTSAT_Swin_Transformer
from sed_model import SEDWrapper
import htsat_config
import sys
sys.path.append("/s7home/zyk524/test/audio/SED") 
sed_model = HTSAT_Swin_Transformer(
    spec_size=htsat_config.htsat_spec_size,
    patch_size=htsat_config.htsat_patch_size,
    in_chans=1,
    num_classes=htsat_config.classes_num,
    window_size=htsat_config.htsat_window_size,
    config = htsat_config,
    depths = htsat_config.htsat_depth,
    embed_dim = htsat_config.htsat_dim,
    patch_stride=htsat_config.htsat_stride,
    num_heads=htsat_config.htsat_num_head
)
at_model = SEDWrapper(
    sed_model = sed_model,
    config = htsat_config,
    dataset = None
)
# load the checkpoint
ckpt = torch.load(htsat_config.resume_checkpoint, map_location="cpu")
at_model.load_state_dict(ckpt["state_dict"])
at_model.eval() # 切换到评估模式
for param in at_model.parameters():
    param.requires_grad = False
def segment_audio(waveform, sample_rate=32000, segment_duration=10, overlap=0.5):
    """
    waveform: torch.Tensor of shape (1, L)
    Returns: list of (segment_waveform, start_time_in_seconds)
    """
    seg_len = int(segment_duration * sample_rate)
    step = int(seg_len * (1 - overlap))  # 50% overlap by default
    segments = []
    L = waveform.shape[1]

    start = 0
    while start < L:
        end = min(start + seg_len, L)
        seg = waveform[:, start:end]
        # Pad if too short (optional, but helps consistency)
        if seg.shape[1] < seg_len:
            seg = torch.nn.functional.pad(seg, (0, seg_len - seg.shape[1]))
        segments.append((seg, start / sample_rate))
        start += step
    return segments
all_events = []
audio_path = '/dataset/AudioSep_eval/clotho/clotho/mixture-0.wav'
waveform, orig_sr = torchaudio.load(audio_path)  # shape: (channels, length)

# 转为单声道（如果多通道）
if waveform.shape[0] > 1:
    waveform = torch.mean(waveform, dim=0, keepdim=True)  # (1, L)

# 重采样到 32000 Hz
if orig_sr != 32000:
    resampler = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=32000)
    waveform = resampler(waveform)
segments = segment_audio(waveform, sample_rate=32000, segment_duration=8.0, overlap=0.5)

for seg_wave, start_time in segments:
    # 推理
    output_dict = at_model.inference(seg_wave.cpu().numpy())  # 确保内部 infer_mode=True

    # 获取帧级输出: shape (T, num_classes)
    framewise = output_dict['framewise_output']  # (T, 527)
    print("framewise shape:", framewise.shape)
    framewise = framewise.squeeze(0)
    # 计算每帧对应的真实时间
    # 假设模型输出帧率 = 32000 / 320 / 4 = 25 Hz（需根据你的模型确认！）
    # 更通用方式：从 segment 长度和 framewise.shape[0] 反推
    seg_duration = seg_wave.shape[1] / 32000  
    frame_times = np.linspace(start_time, start_time + seg_duration, framewise.shape[0], endpoint=False)

    # 保存：(时间, 概率向量)
    for t, probs in zip(frame_times, framewise):
        all_events.append((t, probs))

all_events = np.array(all_events, dtype=object)
times = np.array([e[0] for e in all_events])
probs = np.stack([e[1] for e in all_events])  # (N, 527)



threshold = 0.3
sed_events = []

for cls_idx in range(probs.shape[1]):
    active = probs[:, cls_idx] > threshold
    # 简单合并连续帧（可改用更 sophisticated 的后处理）
    if not active.any():
        continue
    # 找连续段
    changes = np.diff(active.astype(int))
    start_indices = np.where(changes == 1)[0] + 1
    end_indices = np.where(changes == -1)[0] + 1
    
    if bool(active[0].item()):
        start_indices = np.r_[0, start_indices]
    if bool(active[-1].item()):
        end_indices = np.r_[end_indices, len(active)]
    
    for s, e in zip(start_indices, end_indices):
        sed_events.append({
            'start': float(times[s]),
            'end': float(times[e-1] + (times[1]-times[0]) if len(times)>1 else times[0]+0.04),
            'label': cls_idx,
            'confidence': float(probs[s:e, cls_idx].max())
        })
print("\n=== Detected Sound Events (by class index) ===")
if not sed_events:
    print("No sound events detected above threshold.")
else:
    for i, event in enumerate(sed_events):
        print(f"[{i+1}] Class {event['label']}")
        print(f"    Time: {event['start']:.2f}s - {event['end']:.2f}s")
        print(f"    Confidence: {event['confidence']:.3f}")
        print()
