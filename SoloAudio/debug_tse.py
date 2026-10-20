import yaml
import random
import argparse
import os
import sys
import torch
import librosa
from tqdm import tqdm
from diffusers import DDIMScheduler
from transformers import AutoProcessor, ClapModel, ClapProcessor
from model.udit import UDiT
from utils import save_audio
import shutil
from vae_modules.autoencoder_wrapper import Autoencoder
import torchaudio
import traceback

# ================= 配置参数 =================
parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--output_dir', type=str, default='/s7home/zyk524/SoloAudio/SoloAudio-main/output-languageTSE/')
parser.add_argument('--mixture', type=str, default='/dataset/AudioSep_eval/audioset/class_id=0/class_id=0,index=000,mixture.wav')
parser.add_argument('--enrollment', type=str, default='Speech')
parser.add_argument('--autoencoder-path', type=str, default='/s7home/zyk524/SoloAudio/SoloAudio-main/pretrained_models/audio-vae.pt')
parser.add_argument("--num_infer_steps", type=int, default=50)
parser.add_argument('--diffusion-config', type=str, default='/s7home/zyk524/SoloAudio/SoloAudio-main/config/SoloAudio.yaml')
parser.add_argument('--diffusion-ckpt', type=str, default='/s7home/zyk524/SoloAudio/SoloAudio-main/pretrained_models/soloaudio_v2.pt')
parser.add_argument('--random-seed', type=int, default=2024)
# 新增 debug 参数
parser.add_argument('--disable_flash_attn', action='store_true', help='强制禁用 Flash Attention')
args = parser.parse_args()

with open(args.diffusion_config, 'r') as fp:
    args.diff_config = yaml.safe_load(fp)
args.v_prediction = args.diff_config["ddim"]["v_prediction"]

# ================= Debug 工具函数 =================
def print_debug(msg):
    print(f"\033[93m[DEBUG] {msg}\033[0m")

def check_gpu_status(device_name):
    if 'cuda' in device_name:
        try:
            print_debug(f"正在检查 GPU: {device_name}")
            props = torch.cuda.get_device_properties(device_name)
            print_debug(f"GPU 型号: {props.name}")
            print_debug(f"总显存: {props.total_memory / 1024**3:.2f} GB")
            # 做一个简单的张量计算测试
            x = torch.randn(1024, 1024).to(device_name)
            y = x @ x
            print_debug("GPU 基础张量计算测试通过 ✅")
        except Exception as e:
            print_debug(f"GPU 基础测试失败 ❌: {e}")
            traceback.print_exc()
            sys.exit(1)

@torch.no_grad()
def sample_diffusion(args, unet, autoencoder, scheduler,
                     mixture, timbre, device, ddim_steps=50, eta=0, seed=2023):
    unet.eval()
    scheduler.set_timesteps(ddim_steps)
    generator = torch.Generator(device=device).manual_seed(seed)
    noise = torch.randn(mixture.shape, generator=generator, device=device)
    pred = noise

    print_debug("开始 Diffusion 循环...")
    for i, t in enumerate(tqdm(scheduler.timesteps)):
        # 每一部都打印，看是不是某一步崩了
        if i == 0: print_debug(f"进入第 1 步去噪，Timestep: {t}")
        
        pred = scheduler.scale_model_input(pred, t)
        
        # 这里是核心崩溃点，加上 try-catch
        try:
            model_output = unet(x=pred, timesteps=t, mixture=mixture, timbre=timbre)
        except RuntimeError as e:
            print_debug(f"UNet 前向传播在 step {i} 崩溃! 错误信息: {e}")
            raise e

        pred = scheduler.step(model_output=model_output, timestep=t, sample=pred,
                              eta=eta, generator=generator).prev_sample

    print_debug("Diffusion 循环结束，开始 VAE 解码...")
    pred = autoencoder(embedding=pred).squeeze(1)
    return pred

if __name__ == '__main__':
    # 1. 检查 GPU
    check_gpu_status(args.device)

    os.makedirs(args.output_dir, exist_ok=True)
    
    # 2. 加载 CLAP
    print_debug("正在加载 CLAP 模型...")
    local_clap_path = "/s7home/zyk524/SoloAudio/SoloAudio-main/clap" # 确保这里路径对
    try:
        clapmodel = ClapModel.from_pretrained(local_clap_path).to(args.device)
        processor = AutoProcessor.from_pretrained(local_clap_path)
        print_debug("CLAP 加载成功 ✅")
    except Exception as e:
        print_debug(f"CLAP 加载失败 ❌: {e}")
        sys.exit(1)

    # 3. 加载 VAE
    print_debug("正在加载 Autoencoder (VAE)...")
    try:
        autoencoder = Autoencoder(args.autoencoder_path, 'stable_vae', quantization_first=True)
        autoencoder.eval()
        autoencoder.to(args.device)
        # 测试 VAE
        dummy_audio = torch.randn(1, 1, 24000).to(args.device)
        _ = autoencoder(audio=dummy_audio)
        print_debug("VAE 加载及测试成功 ✅")
    except Exception as e:
        print_debug(f"VAE 加载或测试失败 ❌: {e}")
        sys.exit(1)

    # 4. 加载 UDiT
    print_debug("正在加载 UDiT (Diffusion Model)...")
    try:
        unet = UDiT(**args.diff_config['diffwrap']['UDiT']).to(args.device)
        unet.load_state_dict(torch.load(args.diffusion_ckpt)['model'])
        total = sum([param.nelement() for param in unet.parameters()])
        print("Number of parameter: %.2fM" % (total / 1e6))
        
        # 单独测试 UDiT 一次前向传播 (使用假数据)
        print_debug("正在进行 UDiT 假数据测试 (Dummy Forward Pass)...")
        dummy_x = torch.randn(1, 8, 128, 16).to(args.device) # 假设 latent shape
        dummy_t = torch.tensor([100]).to(args.device)
        dummy_mix = torch.randn(1, 8, 128, 16).to(args.device)
        dummy_timbre = torch.randn(1, 512).to(args.device)
        
        # 尝试运行一次
        _ = unet(x=dummy_x, timesteps=dummy_t, mixture=dummy_mix, timbre=dummy_timbre)
        print_debug("UDiT 假数据前向传播测试成功 ✅")
        
    except Exception as e:
        print_debug(f"UDiT 加载或测试失败 ❌: {e}")
        print("如果是 Shape 错误请忽略，重点看是不是 Bus Error")
        # 这里不退出，因为假数据的 shape 我是猜的，可能不对，继续往下跑真实数据

    # 5. 准备 Scheduler
    if args.v_prediction:
        print('v prediction')
        noise_scheduler = DDIMScheduler(**args.diff_config["ddim"]['diffusers'])
    else:
        print('noise prediction')
        noise_scheduler = DDIMScheduler(**args.diff_config["ddim"]['diffusers'])
    
    # 6. 数据预处理
    print_debug("正在处理输入音频和文本...")
    with torch.no_grad():
        mixture, _ = librosa.load(args.mixture, sr=24000)
        mixture = torch.tensor(mixture).unsqueeze(0).to(args.device)
        mixture = autoencoder(audio=mixture.unsqueeze(1))

        text = args.enrollment
        text_inputs = processor(text=[text], max_length=10, padding='max_length', truncation=True, return_tensors="pt")
        inputs = {key: value.to(args.device) for key, value in text_inputs.items() if key in ["input_ids", "attention_mask"]}
        timbre = clapmodel.get_text_features(**inputs)

    # 7. 运行推理 (核心 Debug 区域)
    print_debug("准备进入推理阶段...")
    
    # 这里的 Context Manager 可以强制禁用 Flash Attention
    # 如果 args.disable_flash_attn 为 True，则禁用 Flash Attention
    # 否则使用默认设置
    
    ctx = torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=True) if args.disable_flash_attn else torch.no_grad() # 这里的 else 只是占位，实际上我们外面已经有 no_grad 了
    
    if args.disable_flash_attn:
        print_debug("⚠️ 已强制禁用 Flash Attention (使用 Math/Mem-Efficient Attention)")
        context_manager = torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=True)
    else:
        print_debug("ℹ️ 使用默认 Attention 模式 (可能包含 Flash Attention)")
        context_manager = torch.no_grad() # 空上下文，不做改变

    try:
        with context_manager:
            pred = sample_diffusion(args, unet, autoencoder, noise_scheduler, mixture, timbre, args.device, ddim_steps=args.num_infer_steps, eta=0, seed=args.random_seed)
        
        savename = args.mixture.split('/')[-1].split('.wav')[0]
        shutil.copyfile(args.mixture, f'{args.output_dir}/{savename}_mix.wav')
        save_audio(f'{args.output_dir}/{savename}_pred.wav', 24000, pred)
        print(f'the prediction is save to {savename}_pred.wav')
        print_debug("程序成功运行完毕! 🎉")
        
    except Exception as e:
        print_debug(f"推理过程中发生错误 ❌: {e}")
        traceback.print_exc()