import torch
import sys
import os
from vae_modules.autoencoder_wrapper import Autoencoder
import torch.nn.utils.weight_norm as weight_norm_fn

# ================= 配置 =================
vae_path = '/s7home/zyk524/SoloAudio/SoloAudio-main/pretrained_models/audio-vae.pt'
device = 'cuda'
# =======================================

def remove_all_weight_norms(module):
    """
    递归移除模型中所有的 weight_norm 钩子。
    这会将 weight_g 和 weight_v 合并成最终的 weight，并删除钩子。
    """
    found_count = 0
    for name, child in module.named_children():
        found_count += remove_all_weight_norms(child)
    
    # 检查当前模块是否有 weight_norm
    for hook in module._forward_pre_hooks.values():
        if "WeightNorm" in str(hook):
            try:
                # 使用 PyTorch 官方 API 移除
                torch.nn.utils.remove_weight_norm(module)
                found_count += 1
            except Exception as e:
                print(f"⚠️ 移除失败: {e}")
    return found_count

def test_vae():
    print(f"正在加载 VAE (CPU)...")
    try:
        # 1. 先加载到 CPU
        autoencoder = Autoencoder(vae_path, 'stable_vae', quantization_first=True)
        autoencoder.eval()
        print(f"✅ CPU 加载成功")
    except Exception as e:
        print(f"❌ CPU 加载就失败了: {e}")
        return

    # 2. 尝试移除 weight_norm
    print(f"-"*30)
    print(f"正在尝试移除 weight_norm 钩子...")
    try:
        count = remove_all_weight_norms(autoencoder)
        print(f"✅ 共移除了 {count} 个 weight_norm 钩子")
    except Exception as e:
        print(f"⚠️ 移除过程出现异常 (可能模型里没有): {e}")

    # 3. 移动到 GPU
    print(f"-"*30)
    print(f"正在移动到 GPU: {device} ...")
    try:
        autoencoder.to(device)
        print(f"✅ 移动成功")
    except Exception as e:
        print(f"❌ 移动到 GPU 失败: {e}")
        return

    # 4. 前向传播测试
    print(f"-"*30)
    print(f"开始前向传播测试...")
    try:
        dummy_audio = torch.randn(1, 1, 24000).to(device)
        with torch.no_grad():
            output = autoencoder(audio=dummy_audio)
        print(f"✅✅✅ 成功！前向传播完成！")
        print(f"结论：确实是 weight_norm 导致的兼容性问题。")
    except Exception as e:
        print(f"❌ 前向传播依然崩溃: {e}")

if __name__ == '__main__':
    # 强制使用你之前成功的 GPU ID (比如 1)
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    test_vae()