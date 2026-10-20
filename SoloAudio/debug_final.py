import argparse
import os
import sys
import torch
import yaml
import builtins
from transformers import AutoProcessor, ClapModel
from vae_modules.autoencoder_wrapper import Autoencoder
import functools

# ==========================================
# 核心 Debug 逻辑：劫持 torch.load
# ==========================================
original_load = torch.load

def safe_load_hook(*args, **kwargs):
    # 获取正在加载的文件路径
    file_path = args[0] if len(args) > 0 else "Unknown"
    
    print(f"\033[93m[HOOK] 拦截到 torch.load 请求！\033[0m")
    print(f"\033[93m[HOOK] 目标文件: {file_path}\033[0m")
    print(f"\033[93m[HOOK] 原始参数: map_location={kwargs.get('map_location', 'Not Set')}\033[0m")
    
    # 强制修改 map_location 为 cpu
    if 'map_location' not in kwargs or kwargs['map_location'] != 'cpu':
        print(f"\033[92m[FIX] 正在强制将 map_location 修改为 'cpu' 以避免 Bus Error...\033[0m")
        kwargs['map_location'] = 'cpu'
    
    try:
        return original_load(*args, **kwargs)
    except Exception as e:
        print(f"\033[91m[HOOK] 加载失败: {e}\033[0m")
        raise e

# 应用劫持
torch.load = safe_load_hook
# ==========================================

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--autoencoder-path', type=str, default='/s7home/zyk524/SoloAudio/SoloAudio-main/pretrained_models/audio-vae.pt')
args = parser.parse_args()

def test_vae_loading():
    print("\n" + "="*40)
    print("开始 VAE 加载测试 (GPU模式)")
    print("="*40)

    # 1. 检查 GPU
    if not torch.cuda.is_available():
        print("❌ 没有检测到 GPU，无法复现 GPU Bus Error。")
        return

    device = torch.device(args.device)
    print(f"目标设备: {device}")

    # 2. 实例化 Autoencoder
    # 注意：这里我们不手动 .to(device)，看它内部会不会自己转，或者我们手动转
    print(">>> 步骤 1: 实例化 Autoencoder 类...")
    try:
        # 这里会触发被我们劫持的 torch.load
        autoencoder = Autoencoder(args.autoencoder_path, 'stable_vae', quantization_first=True)
        print(">>> 实例化成功！(此时模型应该在 CPU)")
    except Exception as e:
        print(f"❌ 实例化阶段崩溃: {e}")
        return

    # 3. 移动到 GPU
    print(">>> 步骤 2: 将模型移动到 GPU...")
    try:
        autoencoder.to(device)
        print(">>> 移动成功！")
    except Exception as e:
        print(f"❌ 移动到 GPU 阶段崩溃: {e}")
        return

    # 4. 运行一次前向传播
    print(">>> 步骤 3: 运行 Dummy Forward Pass...")
    try:
        dummy_input = torch.randn(1, 1, 24000).to(device)
        with torch.no_grad():
            _ = autoencoder(audio=dummy_input)
        print("✅✅✅ 测试全部通过！问题已解决！")
        print("结论：原代码在加载 VAE 时直接读取到 GPU 导致了 NFS Bus Error。强制 CPU 加载可解。")
    except Exception as e:
        print(f"❌ 前向传播崩溃: {e}")
        # 这里如果崩溃，才可能是 Flash Attention 或算子问题（虽然 VAE 不太可能用 Flash Attn）

if __name__ == '__main__':
    test_vae_loading()