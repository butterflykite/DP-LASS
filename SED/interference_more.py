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
            'label': [cls_idx],
            'confidence': float(probs[s:e, cls_idx].max())
        })


audio_path = '/dataset/AudioSep_eval/audiocaps/segment-4.wav' 
target_sr = 32000

# 1. 使用 librosa 加载音频
# sr=None 保证以原始采样率加载，以便后续正确重采样
waveform, sr = librosa.load(audio_path, sr=None, mono=True)

# 2. 检查并重采样到模型需要的 32000 Hz
if sr != target_sr:
    print(f"原始采样率是 {sr} Hz, 正在重采样到 {target_sr} Hz...")
    waveform = librosa.resample(waveform, orig_sr=sr, target_sr=target_sr)

# 3. 构造成 (batch_size, num_samples) 的形状
# 我们只有一个文件，所以 batch_size = 1
# np.newaxis 会增加一个新的维度
input_waveform = waveform[np.newaxis, :]
input_waveform = input_waveform.astype(np.float32)

print(f"输入波形的形状: {input_waveform.shape}") 
# 预期输出: 输入波形的形状: (1, 160000)  (因为 5s * 32000 = 160000)

output_dict = at_model.inference(input_waveform)

print("推理完成！")

# 1. 分析 'clipwise_output'
clipwise_preds = output_dict['clipwise_output'][0] # 取出 batch 里的第一个结果
print(f"\n片段级预测 (前5个最高概率):")
# 找到概率最高的几个类别的索引
top_indices = np.argsort(clipwise_preds)[-5:][::-1]
for i in top_indices:
    # 假设我们知道索引对应的标签
    print(f"  类别索引 {i}: 概率 = {clipwise_preds[i]:.4f}")
# 预期会看到 'Dog' (索引74) 和 'Door' (索引315) 的概率非常高

# 2. 分析 'framewise_output'
framewise_preds = output_dict['framewise_output'][0] # 形状 (time_steps, 527)
print(f"\n帧级预测形状: {framewise_preds.shape}")
# 我们可以查看 'Dog' 类别在时间上的概率变化
dog_prob_over_time = framewise_preds[:, 74] 
# 如果狗在音频的开头叫，那么 dog_prob_over_time 数组的前半部分值会比较高
# 这可以用来画图，可视化声音事件发生的时间
# 方法A: 查看特定时间帧上最可能的事件
# 比如，我们看看第 300 帧（大约在音频的 1/3 处）发生了什么
frame_index_to_check = 300
frame_predictions = framewise_preds[frame_index_to_check] # 这是一个 (527,) 的向量

print(f"\n在第 {frame_index_to_check} 帧的预测 (前3个最高概率):")
top_indices_frame = np.argsort(frame_predictions)[-3:][::-1]
for i in top_indices_frame:
    print(f"  类别:  (索引 {i}): 概率 = {frame_predictions[i]:.4f}")


# 方法B: 找到整段音频中瞬时概率最高的事件
# 这可以帮你找到即使持续时间很短但很突出的声音
# 沿着时间轴（axis=0）找到每个类别的最大概率
peak_probs = np.max(framewise_preds, axis=0)
print(f"\n整段音频中各事件的峰值概率 (前5个):")
top_indices_peak = np.argsort(peak_probs)[-5:][::-1]
for i in top_indices_peak:
    # 找到这个峰值发生在哪一帧
    peak_frame = np.argmax(framewise_preds[:, i])
    print(f"  类别: (索引 {i}: 峰值概率 = {peak_probs[i]:.4f} (发生在第 {peak_frame} 帧)")

# 3. 获取 'latent_output'
latent_vector = output_dict['latent_output'][0] # 形状 (2048,)
print(f"\n潜在嵌入向量形状: {latent_vector.shape}")
print(f"向量的前5个值: {latent_vector[:5]}")
# 这个 latent_vector 就是描述 "test_audio.wav" 声学内容的“指纹”
# 在源分离任务中，这个向量会被传递给 ZeroShotASP 模型，作为分离的依据。
