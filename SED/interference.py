import librosa
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
audio_path = '/dataset/AudioCaps/train/1.wav' 
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

# 3. 获取 'latent_output'
latent_vector = output_dict['latent_output'][0] # 形状 (2048,)
print(f"\n潜在嵌入向量形状: {latent_vector.shape}")
print(f"向量的前5个值: {latent_vector[:5]}")
# 这个 latent_vector 就是描述 "test_audio.wav" 声学内容的“指纹”
# 在源分离任务中，这个向量会被传递给 ZeroShotASP 模型，作为分离的依据。
