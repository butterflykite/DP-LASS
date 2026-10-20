import sys
import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import json
import numpy as np
import logging
import warnings
from tqdm import tqdm
from pytorch_lightning import seed_everything

# ==============================================================================
# 0. 基础设置 & Monkey Patch
# ==============================================================================
torch.set_float32_matmul_precision('high')
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

sys.path.append("/s7home/zyk524/FLOWSEP/FlowSep-main/src")

import latent_diffusion.modules.diffusionmodules.util as diffusion_util
def dummy_checkpoint(func, inputs, params, flag):
    return func(*inputs)
diffusion_util.checkpoint = dummy_checkpoint

from latent_diffusion.modules.diffusionmodules.openaimodel import AttentionBlock
def custom_attention_forward(self, x):
    return self._forward(x)
AttentionBlock.forward = custom_attention_forward

from latent_diffusion.util import instantiate_from_config
from utilities.data.dataset import AudioDataset

# ==============================================================================
# 1. 验证函数
# ==============================================================================
def verify_checkpoint_loading(model, ckpt_path, model_name="Base Model"):
    print(f"\n[Verification] Checking {model_name} against {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(state_dict.keys())
    intersection = model_keys.intersection(ckpt_keys)
    print(f"  Matched Keys: {len(intersection)}")
    print(f"  [SUCCESS] {model_name} loaded.")

# ==============================================================================
# 2. 模型构建
# ==============================================================================
def build_model(args, config):
    print("Instantiating Model...")
    if args.vae_ckpt:
        config['model']['params']['first_stage_config']['params']['reload_from_ckpt'] = args.vae_ckpt
    model = instantiate_from_config(config["model"])
    
    print("[Config] Disabling EMA...")
    model.use_ema = False
    if hasattr(model, 'model_ema'):
        del model.model_ema
        model.model_ema = None

    if args.vae_ckpt:
        verify_checkpoint_loading(model.first_stage_model, args.vae_ckpt, "VAE")

    print(f"Loading Base Checkpoint: {args.load_checkpoint}")
    ckpt = torch.load(args.load_checkpoint, map_location="cpu")
    if "state_dict" in ckpt: ckpt = ckpt["state_dict"]
    
    model.load_state_dict(ckpt, strict=False)
    verify_checkpoint_loading(model, args.load_checkpoint, "Latent Diffusion")
    
    model.to(args.device)
    model.eval()
    return model

# ==============================================================================
# 3. 推理逻辑
# ==============================================================================
def run_inference(model, mix_wav_path, text, device, val_dataset, args):
    batch = {}
    batch["fname"] = [mix_wav_path]
    batch["text"] = [text]
    batch["caption"] = [text]
    
    batch["waveform"] = torch.rand(1, 1, 16000).to(device)
    batch["log_mel_spec"] = torch.rand(1, 1024, 64).to(device)
    batch["sampling_rate"] = torch.tensor([16000]).to(device)
    batch["label_vector"] = torch.rand(1, 527).to(device)
    batch["stft"] = torch.rand(1, 1024, 512).to(device)

    noise_waveform, _ = val_dataset.read_wav_file(mix_wav_path)
    
    if torch.is_tensor(noise_waveform):
        noise_waveform = noise_waveform.detach().cpu().numpy()
    
    if noise_waveform.ndim == 2:
        noise_waveform = noise_waveform[0]

    target_len = 163840 
    if len(noise_waveform) > target_len:
        noise_waveform = noise_waveform[:target_len]
    else:
        pad_len = target_len - len(noise_waveform)
        noise_waveform = np.pad(noise_waveform, (0, pad_len))

    mixed_mel, stft = val_dataset.wav_feature_extraction(noise_waveform.reshape(1, -1))
    
    batch["mixed_waveform"] = torch.from_numpy(noise_waveform.reshape(1, 1, target_len)).to(device)
    
    # 修复 Tensor/Numpy 问题
    if isinstance(mixed_mel, np.ndarray):
        batch["mixed_mel"] = torch.from_numpy(mixed_mel.reshape(1, mixed_mel.shape[0], mixed_mel.shape[1])).to(device)
    else:
        batch["mixed_mel"] = mixed_mel.reshape(1, mixed_mel.shape[0], mixed_mel.shape[1]).to(device)

    with torch.no_grad():
        waveform = model.generate_sample(
            [batch], 
            name="result", 
            unconditional_guidance_scale=1.0, 
            ddim_steps=args.infer_step, 
            n_gen=1, 
            save_mixed=False
        )
    
    if isinstance(waveform, dict):
        if 'sample' in waveform: gen_wav = waveform['sample']
        elif 'wav' in waveform: gen_wav = waveform['wav']
        else: gen_wav = list(waveform.values())[0]
    else:
        gen_wav = waveform

    if not torch.is_tensor(gen_wav):
        gen_wav = torch.from_numpy(gen_wav)
    
    gen_wav = gen_wav.detach().cpu().float()
    
    if gen_wav.dim() == 1:
        gen_wav = gen_wav.unsqueeze(0)
    elif gen_wav.dim() == 3:
        gen_wav = gen_wav.squeeze(0)
        
    return gen_wav

# ==============================================================================
# 4. 主程序
# ==============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, default="/s7home/zyk524/FLOWSEP/FlowSep-main/lass_config/2channel_flow.yaml")
    parser.add_argument("--load_checkpoint", type=str, default="/s7home/zyk524/FLOWSEP/FlowSep-main/model_logs/v2_100k.ckpt")
    parser.add_argument("--vae_ckpt", type=str, default="/s7home/zyk524/FLOWSEP/FlowSep-main/model_logs/vae.ckpt")
    
    parser.add_argument("--input_json", type=str, default="/s7home/zyk524/SAM_Audio/sam-audio-bench/json_freeform/General_SFX_freeform.json")
    parser.add_argument("--output_dir", type=str, default="/s7home/zyk524/SAM_Audio/sam-audio-bench/flowsep/General_SFX/old_free")
    
    parser.add_argument("--infer_step", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    seed_everything(42)
    
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output Directory: {args.output_dir}")

    config = yaml.load(open(args.config_yaml, "r"), Loader=yaml.FullLoader)
    model = build_model(args, config)
    val_dataset = AudioDataset(config, split="test", add_ons=[])

    print(f"Loading data from {args.input_json}")
    with open(args.input_json, 'r') as f:
        data_list = json.load(f)

    dataset_root = os.path.dirname(args.input_json)

    # === 用于存储结果 JSON 的列表 ===
    results_metadata = []
    success_count = 0

    for idx, item in enumerate(tqdm(data_list)):
        mix_path = item.get("audio") or item.get("original_audio") or item.get("mix_wav")
        caption = item.get("caption") or item.get("text")
        
        if not mix_path:
            print(f"[Warning] Item {idx} has no audio path key.")
            continue

        if not os.path.isabs(mix_path):
            mix_path = os.path.join(dataset_root, mix_path)

        if not os.path.exists(mix_path):
            if idx < 3: 
                print(f"\n[ERROR] File not found: {mix_path}")
            continue

        original_name = os.path.basename(mix_path)
        save_name = f"{os.path.splitext(original_name)[0]}_sep.wav"
        save_path = os.path.join(args.output_dir, save_name)
        
        try:
            gen_wav = run_inference(model, mix_path, caption, args.device, val_dataset, args)
            torchaudio.save(save_path, gen_wav, 16000)
            success_count += 1
            
            # === 添加到结果列表 ===
            results_metadata.append({
                "original_audio": os.path.abspath(mix_path),
                "generated_audio": os.path.abspath(save_path),
                "caption": caption
            })
            
        except Exception as e:
            print(f"\n[ERROR] Processing {mix_path}: {e}")

    # === 保存结果 JSON ===
    json_output_path = '/s7home/zyk524/SAM_Audio/sam-audio-bench/flowsep/General_SFX/old_free_json/results.json'
    with open(json_output_path, "w") as f:
        json.dump(results_metadata, f, indent=4)

    print("\n" + "="*40)
    print(f"Inference Finished.")
    print(f"Successfully processed: {success_count} / {len(data_list)}")
    print(f"Audio saved to: {args.output_dir}")
    print(f"JSON saved to:  {json_output_path}")
    print("="*40)

if __name__ == "__main__":
    main()
