import torch

path = '/s7home/zyk524/SoloAudio/SoloAudio-main/pretrained_models/audio-vae.pt'
print(f"正在检查: {path}")

data = torch.load(path, map_location='cpu')
print(f"数据类型: {type(data)}")

if isinstance(data, dict):
    print("这是一个字典 (state_dict)。")
    print("Keys:", data.keys())
    if 'state_dict' in data:
        print("包含 'state_dict' 键。")
    if 'model' in data:
        print("包含 'model' 键。")
else:
    print("这不是一个字典，可能是一个完整的模型对象 (TorchScript 或 Pickle)。")
    print(data)