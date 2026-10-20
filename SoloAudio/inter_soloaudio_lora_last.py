import yaml
import random
import argparse
import os
import sys
import json
import torch
import torch.nn as nn
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

# 禁用 cudnn 自动调优
torch.backends.cudnn.benchmark = False 
torch.backends.cudnn.enabled = False

# ==============================================================================
# 1. LoRA 定义 (必须与训练代码完全一致)
# ==============================================================================
class LoRALinear(nn.Module):
    def __init__(self, original_linear: nn.Linear, rank: int, alpha: int, dropout: float = 0.1):
        super().__init__()
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.rank = rank
        self.scaling = alpha / rank

        self.base_layer = original_linear
        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

        self.lora_A = nn.Parameter(torch.zeros(self.in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, self.out_features))
        self.lora_dropout = nn.Dropout(p=dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        dtype = self.lora_A.dtype
        x_casted = x.to(dtype)
        base_out = self.base_layer(x)
        lora_out = (self.lora_dropout(x_casted) @ self.lora_A @ self.lora_B) * self.scaling
        return base_out + lora_out.to(base_out.dtype)

def inject_lora(model, rank=16, alpha=16, dropout=0.1):
    target_modules = ['to_q', 'to_k', 'to_v', 'proj', 'fc1', 'fc2']
    # print(f"[LoRA] Injecting targets: {target_modules} | Rank: {rank}")
    
    for param in model.parameters():
        param.requires_grad = False
        
    injected_count = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            layer_name = name.split('.')[-1]
            if layer_name in target_modules:
                parent_name = name.rsplit('.', 1)[0]
                parent = model.get_submodule(parent_name)
                
                original = getattr(parent, layer_name)
                lora_layer = LoRALinear(original, rank=rank, alpha=alpha, dropout=dropout)
                setattr(parent, layer_name, lora_layer)
                injected_count += 1
    print(f"[LoRA] Injected {injected_count} layers.")
    return model

def print_parameter_statistics(model):
    """
    打印模型参数统计信息：基座参数 vs LoRA参数
    """
    total_params = 0
    base_params = 0
    lora_params = 0

    for name, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params
        
        # 判断是否为 LoRA 参数 (根据 LoRALinear 的定义，参数名包含 lora_)
        if "lora_" in name:
            lora_params += num_params
        else:
            base_params += num_params

    print("\n" + "=" * 50)
    print(f"Model Parameter Statistics:")
    print(f"  - Base Model Params:   {base_params / 1e6:.2f} M")
    print(f"  - Single LoRA Params:  {lora_params / 1e6:.2f} M")
    print(f"  - Total Params:        {total_params / 1e6:.2f} M")
    print(f"  - LoRA Ratio:          {lora_params / total_params * 100:.4f} %")
    print("=" * 50 + "\n")
def load_lora_safely(model, state_dict, cluster_id):
    """
    安全加载 LoRA 权重并验证完整性
    """
    # strict=False 是必须的，因为 state_dict 里没有 Base 参数
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
    # 1. 检查是否有 LoRA 参数缺失
    # 逻辑：missing_keys 中应该全是 Base 参数，不应该包含任何 "lora_" 开头的参数
    lora_missing = [k for k in missing_keys if "lora_" in k]
    
    # 2. 检查是否有未知的参数 (Unexpected)
    # 逻辑：state_dict 中的所有 key 都应该能对应到 model 中
    
    is_valid = True
    
    if len(lora_missing) > 0:
        print(f"\n[ERROR] Cluster {cluster_id}: LoRA loading FAILED!")
        print(f"  Missing {len(lora_missing)} LoRA keys (Example: {lora_missing[:3]}...)")
        is_valid = False
        
    if len(unexpected_keys) > 0:
        print(f"\n[WARNING] Cluster {cluster_id}: Found {len(unexpected_keys)} unexpected keys in checkpoint!")
        print(f"  Example: {unexpected_keys[:3]}...")
        # 通常 unexpected keys 不会致命，但值得注意
    
    if is_valid:
        # print(f"[Success] Cluster {cluster_id}: LoRA weights loaded perfectly.")
        pass
        
    return is_valid
# ==============================================================================
# 2. 核心推理函数 (SoloAudio 原生逻辑)
# ==============================================================================
@torch.no_grad()
def sample_diffusion(unet, autoencoder, scheduler,
                     mixture, timbre, device, ddim_steps=50, eta=0, seed=2023):
    unet.eval()
    scheduler.set_timesteps(ddim_steps)
    generator = torch.Generator(device=device).manual_seed(seed)
    
    # init noise
    noise = torch.randn(mixture.shape, generator=generator, device=device)
    pred = noise

    # 这里的 tqdm 可以关掉 leave=False 以避免刷屏
    for t in scheduler.timesteps:
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    
    # 输入输出
    parser.add_argument('--input_json', type=str, default='/s7home/zyk524/SAM_Audio/mrfakename-sam-audio-musdb18hq-test/data.json')
    parser.add_argument('--output_dir', type=str, default='/s7home/zyk524/SAM_Audio/mrfakename-sam-audio-musdb18hq-test/soloaudio_peft_last')
    
    # LoRA 路径配置 (这里硬编码了你提供的路径列表，也可以通过参数传入)
    parser.add_argument('--lora_rank', type=int, default=16)
    parser.add_argument('--lora_alpha', type=int, default=16)

    # 预训练模型路径
    parser.add_argument('--autoencoder-path', type=str, default='/s7home/zyk524/SoloAudio/SoloAudio-main/pretrained_models/audio-vae.pt')
    parser.add_argument('--diffusion-config', type=str, default='/s7home/zyk524/SoloAudio/SoloAudio-main/config/SoloAudio.yaml')
    parser.add_argument('--diffusion-ckpt', type=str, default='/s7home/zyk524/SoloAudio/SoloAudio-main/pretrained_models/soloaudio_v2.pt')
    parser.add_argument('--clap-path', type=str, default='/s7home/zyk524/SoloAudio/SoloAudio-main/clap')
    
    parser.add_argument("--num_infer_steps", type=int, default=50)
    parser.add_argument('--random-seed', type=int, default=2024)

    args = parser.parse_args()

    # 设置随机种子
    if args.random_seed is not None:
        random.seed(args.random_seed)
        np.random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)

    os.makedirs(args.output_dir, exist_ok=True)
    with open(args.diffusion_config, 'r') as fp:
        args.diff_config = yaml.safe_load(fp)
    args.v_prediction = args.diff_config["ddim"]["v_prediction"]

    # ------------------------------------------------------
    # 1. 加载基础模型 (VAE, CLAP, UDiT Base)
    # ------------------------------------------------------
    print("Loading Base Models...")
    
    # CLAP
    clapmodel = ClapModel.from_pretrained(args.clap_path).to(args.device)
    processor = AutoProcessor.from_pretrained(args.clap_path)
    
    # VAE
    autoencoder = Autoencoder(args.autoencoder_path, 'stable_vae', quantization_first=True)
    autoencoder.eval()
    autoencoder.to(args.device)

    # UDiT Base
    unet = UDiT(**args.diff_config['diffwrap']['UDiT'])
    base_ckpt = torch.load(args.diffusion_ckpt, map_location='cpu')
    if 'model' in base_ckpt: base_ckpt = base_ckpt['model']
     # 使用 strict=True，如果基线参数不匹配，这里直接报错退出
    try:
        unet.load_state_dict(base_ckpt, strict=True)
        print("✅ [SUCCESS] Base Model loaded successfully (Strict Match).")
    except RuntimeError as e:
        print("❌ [ERROR] Base Model loading failed!")
        print(e)
        sys.exit(1)
    
    # ------------------------------------------------------
    # 2. 注入 LoRA 结构
    # ------------------------------------------------------
    print("Injecting LoRA Structure...")
    unet = inject_lora(unet, rank=args.lora_rank, alpha=args.lora_alpha)
    unet.to(args.device)
    unet.eval()

    print_parameter_statistics(unet)
    # Scheduler
    if args.v_prediction:
        noise_scheduler = DDIMScheduler(**args.diff_config["ddim"]['diffusers'])
    else:
        noise_scheduler = DDIMScheduler(**args.diff_config["ddim"]['diffusers'])

    # ------------------------------------------------------
    # 3. 预加载 LoRA 权重
    # ------------------------------------------------------
    # 你提供的路径列表
    LORA_PATHS = [
        "/s7home/zyk524/SoloAudio/SoloAudio-main/LORA/cluster0/last.pt",
        "/s7home/zyk524/SoloAudio/SoloAudio-main/LORA_chongxu/cluster1/last.pt",  
        "/s7home/zyk524/SoloAudio/SoloAudio-main/LORA_chongxu/cluster2/last.pt",
        "/s7home/zyk524/SoloAudio/SoloAudio-main/LORA_chongxu/cluster3/last.pt",  
        "/s7home/zyk524/SoloAudio/SoloAudio-main/LORA/cluster4/last.pt",       
        "/s7home/zyk524/SoloAudio/SoloAudio-main/LORA_chongxu/cluster5/last.pt", 
        "/s7home/zyk524/SoloAudio/SoloAudio-main/LORA_chongxu/cluster6/last.pt"
    ]

    print("\nPre-loading LoRA weights...")
    adapter_weights_list = []
    adapter_paths_list = []
    
    for i, pth in enumerate(LORA_PATHS):
        if os.path.exists(pth):
            print(f"  Cluster {i}: Found {os.path.basename(pth)}")
            state_dict = torch.load(pth, map_location=args.device)
            
            # === 验证 2: 预检查 LoRA 权重完整性 ===
            # 我们在这里临时加载一次，验证是否匹配，验证完不影响后续逻辑
            if not load_lora_safely(unet, state_dict, i):
                print(f"❌ [CRITICAL] Cluster {i} weights mismatch! Exiting to prevent wrong inference.")
                sys.exit(1)
            else:
                print(f"  -> ✅ Cluster {i} weights verified.")

            adapter_weights_list.append(state_dict)
            adapter_paths_list.append(os.path.abspath(pth))
        else:
            print(f"  Cluster {i}: NOT FOUND at {pth}")
            adapter_weights_list.append(None)
            adapter_paths_list.append(None)

    # ------------------------------------------------------
    # 4. 数据加载与批量推理
    # ------------------------------------------------------
    print(f"\nLoading data from {args.input_json}")
    with open(args.input_json, 'r') as f:
        data_list = json.load(f)

    dataset_root = os.path.dirname(args.input_json)
    results_metadata = []
    success_count = 0

    print(f"Starting Inference on {len(data_list)} samples x 7 Clusters...")

    for idx, item in enumerate(tqdm(data_list, desc="Processing Audio")):
        mix_path = item.get("audio") or item.get("original_audio") or item.get("mix_wav") or item.get("mixture")
        text = item.get("caption") or item.get("text") or item.get("enrollment")
        
        if not mix_path: continue

        # 处理相对路径
        if not os.path.isabs(mix_path):
            mix_path = os.path.join(dataset_root, mix_path)

        if not os.path.exists(mix_path):
            print(f"[ERROR] File not found: {mix_path}")
            continue

        original_name = os.path.basename(mix_path)
        base_name = os.path.splitext(original_name)[0]

        # --- 预处理 (一次性完成，避免重复计算) ---
        try:
            with torch.no_grad():
                # 1. Audio -> VAE Latent
                mixture_audio, _ = librosa.load(mix_path, sr=24000)
                mixture_tensor = torch.tensor(mixture_audio).unsqueeze(0).to(args.device)
                mixture_latent = autoencoder(audio=mixture_tensor.unsqueeze(1))

                # 2. Text -> CLAP Embedding
                text_inputs = processor(
                    text=[text],
                    max_length=10,
                    padding='max_length',
                    truncation=True,
                    return_tensors="pt"
                )
                inputs = {k: v.to(args.device) for k, v in text_inputs.items() if k in ["input_ids", "attention_mask"]}
                timbre = clapmodel.get_text_features(**inputs)
        except Exception as e:
            print(f"Error preprocessing {mix_path}: {e}")
            continue

        # --- 遍历 7 个 Cluster 进行推理 ---
        for i, adapter_w in enumerate(adapter_weights_list):
            if adapter_w is None: continue
            
            # 1. 动态加载 LoRA 权重
            # strict=False 非常重要：它会忽略模型中的 Base 参数（因为 adapter_w 里只有 LoRA 参数）
            # 同时也会忽略 adapter_w 里没有的参数（即 Base 参数保持不变）
            unet.load_state_dict(adapter_w, strict=False)
            
            # 2. 构造保存路径
            save_name = f"{base_name}_peft{i}.wav"
            save_path = os.path.join(args.output_dir, save_name)
            
            try:
                with torch.no_grad():
                    # 3. 推理
                    pred = sample_diffusion(
                        unet, autoencoder, noise_scheduler, 
                        mixture_latent, timbre, args.device, 
                        ddim_steps=args.num_infer_steps, eta=0, seed=args.random_seed
                    )
                
                # 4. 保存音频
                save_audio(save_path, 24000, pred)
                
                # 5. 记录元数据
                results_metadata.append({
                    "original_audio": os.path.abspath(mix_path),
                    "generated_audio": os.path.abspath(save_path),
                    "caption": text,
                    "peft_id": i,
                    "peft_path": adapter_paths_list[i]
                })
                
            except Exception as e:
                print(f"\n[ERROR] Item {idx} Cluster {i}: {e}")
                import traceback
                traceback.print_exc()

        success_count += 1

    # ------------------------------------------------------
    # 保存最终结果
    # ------------------------------------------------------
    json_output_path = os.path.join(args.output_dir, "results.json")
    with open(json_output_path, "w") as f:
        json.dump(results_metadata, f, indent=4)

    print("\n" + "="*40)
    print(f"Inference Finished.")
    print(f"Processed Items: {success_count} / {len(data_list)}")
    print(f"Total Generated Files: {len(results_metadata)}")
    print(f"JSON saved to: {json_output_path}")
    print("="*40)
