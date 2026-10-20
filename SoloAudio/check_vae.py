import torch
import sys

path = "/s7home/zyk524/SoloAudio/SoloAudio-main/pretrained_models/audio-vae.pt"

print(f"正在尝试加载: {path}")
try:
    # map_location='cpu' 可以避免显存干扰，纯粹检查文件完整性
    ckpt = torch.load(path, map_location='cpu')
    print("✅ 文件读取成功！Key 数量:", len(ckpt.keys()))
except Exception as e:
    print(f"❌ 文件损坏: {e}")
except:
    print("❌ 发生严重错误 (Bus Error 等)")