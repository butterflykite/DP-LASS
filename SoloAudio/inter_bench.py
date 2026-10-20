import yaml
import random
import argparse
import os
import sys
import json
import torch
import librosa
import shutil
import torchaudio
import numpy as np
from tqdm import tqdm
from diffusers import DDIMScheduler
from transformers import AutoProcessor, ClapModel
from model.udit import UDiT
from utils import save_audio
from vae_modules.autoencoder_wrapper import Autoencoder
sys.path.append('/s7home/zyk524/SoloAudio/SoloAudio-main')
# 禁用 cudnn 自动调优
torch.backends.cudnn.benchmark = False 
torch.backends.cudnn.enabled = False

# ==============================================================================
# 1. 参数设置
# ==============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda')
# 输入输出相关
parser.add_argument('--input_json', type=str, default='/s7home/zyk524/SAM_Audio/sam-audio-bench/json_freeform/General_SFX_freeform.json', help="Path to the input json file containing audio paths and captions")
parser.add_argument('--output_dir', type=str, default='/s7home/zyk524/SAM_Audio/sam-audio-bench/soloaudio/General_SFX/old_free')

# 预训练模型路径
parser.add_argument('--autoencoder-path', type=str, default='/s7home/zyk524/SoloAudio/SoloAudio-main/pretrained_models/audio-vae.pt')
parser.add_argument("--num_infer_steps", type=int, default=50)

# 模型配置
parser.add_argument('--diffusion-config', type=str, default='/s7home/zyk524/SoloAudio/SoloAudio-main/config/SoloAudio.yaml')
parser.add_argument('--diffusion-ckpt', type=str, default='/s7home/zyk524/SoloAudio/SoloAudio-main/pretrained_models/soloaudio_v2.pt')

# 日志和随机种子
parser.add_argument('--random-seed', type=int, default=2024)

args = parser.parse_args()

# 加载 YAML 配置
with open(args.diffusion_config, 'r') as fp:
    args.diff_config = yaml.safe_load(fp)

args.v_prediction = args.diff_config["ddim"]["v_prediction"]

# ==============================================================================
# 2. 核心推理函数 (保持不变)
# ==============================================================================
@torch.no_grad()
def sample_diffusion(args, unet, autoencoder, scheduler,
                     mixture, timbre, device, ddim_steps=50, eta=0, seed=2023):
    unet.eval()
    scheduler.set_timesteps(ddim_steps)
    generator = torch.Generator(device=device).manual_seed(seed)
    # init noise
    noise = torch.randn(mixture.shape, generator=generator, device=device)
    pred = noise

    for t in tqdm(scheduler.timesteps, desc="Sampling", leave=False):
        pred = scheduler.scale_model_input(pred, t)
        model_output = unet(x=pred, timesteps=t, mixture=mixture, timbre=timbre)
        pred = scheduler.step(model_output=model_output, timestep=t, sample=pred,
                              eta=eta, generator=generator).prev_sample

    pred = autoencoder(embedding=pred).squeeze(1)
    return pred

# ==============================================================================
# 3. 主程序
# ==============================================================================
if __name__ == '__main__':
    # 设置随机种子
    if args.random_seed is not None:
        random.seed(args.random_seed)
        np.random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output Directory: {args.output_dir}")

    # ------------------------------------------------------
    # 模型加载 (保持不变)
    # ------------------------------------------------------
    print("Loading Models...")
    local_clap_path = "/s7home/zyk524/SoloAudio/SoloAudio-main/clap"
    clapmodel = ClapModel.from_pretrained(local_clap_path).to(args.device)
    processor = AutoProcessor.from_pretrained(local_clap_path)
    
    autoencoder = Autoencoder(args.autoencoder_path, 'stable_vae', quantization_first=True)
    autoencoder.eval()
    autoencoder.to(args.device)

    unet = UDiT(
        **args.diff_config['diffwrap']['UDiT']
    ).to(args.device)
    unet.load_state_dict(torch.load(args.diffusion_ckpt)['model'])

    total = sum([param.nelement() for param in unet.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))
    
    if args.v_prediction:
        print('Using v prediction')
        noise_scheduler = DDIMScheduler(**args.diff_config["ddim"]['diffusers'])
    else:
        print('Using noise prediction')
        noise_scheduler = DDIMScheduler(**args.diff_config["ddim"]['diffusers'])
    
    # 初始化 scheduler 类型
    latents = torch.randn((1, 128, 128), device=args.device)
    noise = torch.randn(latents.shape).to(latents.device)
    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps,
                              (noise.shape[0],), device=latents.device).long()
    _ = noise_scheduler.add_noise(latents, noise, timesteps)

    # ------------------------------------------------------
    # 数据加载与批量推理 (参考 FlowSep 逻辑)
    # ------------------------------------------------------
    print(f"Loading data from {args.input_json}")
    with open(args.input_json, 'r') as f:
        data_list = json.load(f)

    dataset_root = os.path.dirname(args.input_json)
    results_metadata = []
    success_count = 0

    for idx, item in enumerate(tqdm(data_list, desc="Processing Batch")):
        # 1. 解析 JSON 字段 (兼容多种常见命名)
        mix_path = item.get("audio") or item.get("original_audio") or item.get("mix_wav") or item.get("mixture")
        text = item.get("caption") or item.get("text") or item.get("enrollment")
        
        if not mix_path:
            print(f"[Warning] Item {idx} has no audio path key.")
            continue
        
        # 2. 处理路径 (如果是相对路径，拼接 JSON 所在目录)
        if not os.path.isabs(mix_path):
            mix_path = os.path.join(dataset_root, mix_path)

        if not os.path.exists(mix_path):
            print(f"[ERROR] File not found: {mix_path}")
            continue

        # 3. 定义保存路径
        original_name = os.path.basename(mix_path)
        savename = os.path.splitext(original_name)[0]
        save_path = os.path.join(args.output_dir, f'{savename}_pred.wav')

        try:
            with torch.no_grad():
                # --- 预处理 (保持 SoloAudio 逻辑) ---
                # 加载音频，重采样到 24000
                mixture_audio, _ = librosa.load(mix_path, sr=24000)
                mixture_tensor = torch.tensor(mixture_audio).unsqueeze(0).to(args.device)
                
                # VAE Encode
                # 注意：这里假设 VAE 需要 (B, 1, T) 格式
                mixture_latent = autoencoder(audio=mixture_tensor.unsqueeze(1))

                # CLAP Text Encode
                text_inputs = processor(
                    text=[text],
                    max_length=10, 
                    padding='max_length',
                    truncation=True,
                    return_tensors="pt"
                )
                inputs = {
                    "input_ids": text_inputs["input_ids"][0].unsqueeze(0),
                    "attention_mask": text_inputs["attention_mask"][0].unsqueeze(0),
                }
                inputs = {key: value.to(args.device) for key, value in inputs.items()}
                timbre = clapmodel.get_text_features(**inputs)

                # --- 推理 ---
                pred = sample_diffusion(
                    args, unet, autoencoder, noise_scheduler, 
                    mixture_latent, timbre, args.device, 
                    ddim_steps=args.num_infer_steps, eta=0, seed=args.random_seed
                )
            
            # --- 保存结果 ---
            # 保存生成的音频
            save_audio(save_path, 24000, pred)
            
            # 可选：复制原始混合音频以便对比
            # shutil.copyfile(mix_path, os.path.join(args.output_dir, f'{savename}_mix.wav'))

            success_count += 1

            # 添加到元数据列表
            results_metadata.append({
                "original_audio": os.path.abspath(mix_path),
                "generated_audio": os.path.abspath(save_path),
                "caption": text
            })

        except Exception as e:
            print(f"\n[ERROR] Processing {mix_path}: {e}")
            import traceback
            traceback.print_exc()

    # ------------------------------------------------------
    # 保存最终的 JSON 结果文件
    # ------------------------------------------------------
    #json_output_path = os.path.join(args.output_dir, "results.json")
    json_output_path =  "/s7home/zyk524/SAM_Audio/sam-audio-bench/soloaudio/General_SFX/old_free_json/results.json"
    with open(json_output_path, "w") as f:
        json.dump(results_metadata, f, indent=4)

    print("\n" + "="*40)
    print(f"Inference Finished.")
    print(f"Successfully processed: {success_count} / {len(data_list)}")
    print(f"Audio saved to: {args.output_dir}")
    print(f"JSON saved to:  {json_output_path}")
    print("="*40)
