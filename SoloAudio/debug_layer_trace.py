import torch
import os
import sys
from vae_modules.autoencoder_wrapper import Autoencoder

# ================= 配置 =================
# 请确认路径正确
vae_path = '/s7home/zyk524/SoloAudio/SoloAudio-main/pretrained_models/audio-vae.pt'
device = 'cuda'
# =======================================

def register_hooks(model):
    """给模型的每一层注册 forward hook，监控执行进度"""
    def pre_hook(module, input):
        # 获取模块的名称（通过 id 反查，或者简单打印类名）
        print(f"➡️ [进入] {module.__class__.__name__}")
    
    def post_hook(module, input, output):
        print(f"⬅️ [离开] {module.__class__.__name__}")

    # 递归注册
    for name, layer in model.named_modules():
        # 跳过顶层容器，只看子模块
        if name == '': continue
        layer.register_forward_pre_hook(pre_hook)
        layer.register_forward_hook(post_hook)

def debug_kernel_conflict():
    print(f"正在加载 VAE (直接从 NFS)...")
    try:
        # 1. 加载模型
        autoencoder = Autoencoder(vae_path, 'stable_vae', quantization_first=True)
        autoencoder.eval()
        
        # 2. 强制将所有参数设为内存连续 (Contiguous)
        # 这是解决 CUDA 算子冲突的第一招：修复内存碎片
        print("正在整理内存布局 (make contiguous)...")
        for param in autoencoder.parameters():
            param.data = param.data.contiguous()
            
        autoencoder.to(device)
        print(f"✅ 模型已加载并移动到 {device}")
    except Exception as e:
        print(f"❌ 加载阶段就崩了: {e}")
        return

    # 3. 注册监控钩子
    print(f"正在注册层级监控钩子...")
    register_hooks(autoencoder)

    # 4. 运行前向传播
    print(f"🚀 开始前向传播 (CUDA Kernel 测试)...")
    print(f"如果程序崩溃，请看最后打印的是哪一层。")
    
    try:
        # 构造一个假输入
        dummy_audio = torch.randn(1, 1, 24000).to(device)
        
        # 启用 CUDA 错误同步，确保报错位置准确
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        
        with torch.no_grad():
            output = autoencoder(audio=dummy_audio)
            
        print(f"✅✅✅ 奇迹发生了：前向传播成功！")
    except Exception as e:
        print(f"❌ Python 层面捕获到错误: {e}")

if __name__ == '__main__':
    # 换一个没问题的卡，或者继续用你觉得有问题的卡
    # 建议加上 CUDA_LAUNCH_BLOCKING=1 环境变量
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1' 
    debug_kernel_conflict()