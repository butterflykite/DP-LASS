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
from typing import List
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
# 1. Adapter 定义
# ==============================================================================
class ParallelAdapter(nn.Module):
    def __init__(self, original_layer, alpha=1.0):
        super().__init__()
        self.original_layer = original_layer
        self.alpha = alpha
        if isinstance(original_layer, nn.Conv2d):
            self.adapter = nn.Conv2d(
                in_channels=original_layer.in_channels,
                out_channels=original_layer.out_channels,
                kernel_size=original_layer.kernel_size,
                stride=original_layer.stride,
                padding=original_layer.padding,
                dilation=original_layer.dilation,
                groups=original_layer.groups,
                bias=False 
            )
        else:
            raise ValueError("ParallelAdapter only supports Conv2d")
        nn.init.zeros_(self.adapter.weight)
        self.scaling = self.alpha

    def forward(self, x, *args, **kwargs):
        if x.device != self.adapter.weight.device:
            self.adapter = self.adapter.to(x.device)
        with torch.no_grad():
            orig_out = self.original_layer(x, *args, **kwargs)
        return orig_out + self.adapter(x) * self.scaling

def inject_adapters_encoder_only(model, alpha=1.0):
    injected_count = 0
    target_blocks = ["input_blocks"]
    for name, module in model.named_modules():
        should_inject = False
        for target in target_blocks:
            if target in name:
                should_inject = True
                break
        if not should_inject: continue

        if isinstance(module, nn.Conv2d) and not isinstance(module, ParallelAdapter):
            if module.kernel_size == (1, 1) or module.kernel_size == 1: continue
            if '.' in name:
                parent_name, child_name = name.rsplit('.', 1)
                parent = model
                for part in parent_name.split('.'): parent = getattr(parent, part)
            else:
                parent = model
                child_name = name
            wrapped_layer = ParallelAdapter(module, alpha=alpha)
            setattr(parent, child_name, wrapped_layer)
            injected_count += 1
    print(f"[Injector] Injected {injected_count} Parallel Adapters into Encoder.")
    return injected_count

# ==============================================================================
# 2. OLA 辅助函数
# ==============================================================================
def compute_chunk_starts(total: int, chunk: int, overlap: int) -> List[int]:
    if total <= chunk:
        return [0]
    step = chunk - overlap
    if step <= 0:
        raise ValueError("chunk must be larger than overlap")
    starts = list(range(0, total, step))
    if starts[-1] + chunk < total:
        starts.append(total - chunk)
    return sorted(list(set(max(0, s) for s in starts)))

def make_ola_window(length: int, overlap: int, fade_in: bool, fade_out: bool) -> torch.Tensor:
    w = torch.ones(length, dtype=torch.float32)
    if overlap <= 0 or length <= 1:
        return w
    ov = min(overlap, length)
    if fade_in and ov > 0:
        w[:ov] *= torch.linspace(0.0, 1.0, steps=ov, dtype=torch.float32)
    if fade_out and ov > 0:
        w[-ov:] *= torch.linspace(1.0, 0.0, steps=ov, dtype=torch.float32)
    return w

# ==============================================================================
# 3. 模型构建
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
    
    print("Injecting Adapters...")
    if hasattr(model, "model") and hasattr(model.model, "diffusion_model"):
        unet = model.model.diffusion_model
    else:
        unet = model
    inject_adapters_encoder_only(unet, alpha=1.0)
    
    model.to(args.device)
    model.eval()
    return model

# ==============================================================================
# 4. OLA 推理逻辑 (严格长度控制版)
# ==============================================================================
def run_inference_ola(model, mix_wav_path, text, device, val_dataset, args):
    CHUNK_SIZE = 163840  # 10.24s @ 16kHz
    OVERLAP_SIZE = int(CHUNK_SIZE * 0.2) 

    wav, sr = torchaudio.load(mix_wav_path)
    if wav.shape[0] > 1: wav = wav.mean(dim=0, keepdim=True)
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        wav = resampler(wav)
    
    full_wav = wav.squeeze().detach().cpu().numpy()
    true_total_len = len(full_wav)
    
    starts = compute_chunk_starts(true_total_len, CHUNK_SIZE, OVERLAP_SIZE)
    
    buffer_len = starts[-1] + CHUNK_SIZE
    out_wav = torch.zeros(buffer_len, dtype=torch.float32, device="cpu")
    weight_wav = torch.zeros(buffer_len, dtype=torch.float32, device="cpu")
    
    for i, s in enumerate(starts):
        e = s + CHUNK_SIZE
        chunk_np = full_wav[s:e]
        if len(chunk_np) < CHUNK_SIZE:
            pad_len = CHUNK_SIZE - len(chunk_np)
            chunk_np = np.pad(chunk_np, (0, pad_len))
        
        batch = {}
        batch["fname"] = [mix_wav_path]
        batch["text"] = [text]
        batch["caption"] = [text]
        batch["waveform"] = torch.rand(1, 1, 16000).to(device)
        batch["log_mel_spec"] = torch.rand(1, 1024, 64).to(device)
        batch["sampling_rate"] = torch.tensor([16000]).to(device)
        batch["label_vector"] = torch.rand(1, 527).to(device)
        batch["stft"] = torch.rand(1, 1024, 512).to(device)

        mixed_mel, _ = val_dataset.wav_feature_extraction(chunk_np.reshape(1, -1))
        batch["mixed_waveform"] = torch.from_numpy(chunk_np.reshape(1, 1, CHUNK_SIZE)).to(device)
        
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
            if 'sample' in waveform: gen_chunk = waveform['sample']
            elif 'wav' in waveform: gen_chunk = waveform['wav']
            else: gen_chunk = list(waveform.values())[0]
        else:
            gen_chunk = waveform
        
        if not torch.is_tensor(gen_chunk):
            gen_chunk = torch.from_numpy(gen_chunk)
        
        gen_chunk = gen_chunk.detach().cpu().float().view(-1) 

        if len(gen_chunk) > CHUNK_SIZE:
            gen_chunk = gen_chunk[:CHUNK_SIZE]

        is_first = (i == 0)
        is_last = (i == len(starts) - 1)
        w = make_ola_window(CHUNK_SIZE, OVERLAP_SIZE, fade_in=not is_first, fade_out=not is_last)
        
        out_wav[s:e] += gen_chunk * w
        weight_wav[s:e] += w

    weight_wav = weight_wav.clamp_min(1e-6)
    final_wav = out_wav / weight_wav
    final_wav = final_wav[:true_total_len]
    
    return final_wav.unsqueeze(0)

# ==============================================================================
# 5. 主程序
# ==============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, default="/s7home/zyk524/FLOWSEP/FlowSep-main/lass_config/2channel_flow.yaml")
    parser.add_argument("--load_checkpoint", type=str, default="/s7home/zyk524/FLOWSEP/FlowSep-main/model_logs/v2_100k.ckpt")
    parser.add_argument("--vae_ckpt", type=str, default="/s7home/zyk524/FLOWSEP/FlowSep-main/model_logs/vae.ckpt")
    parser.add_argument("--peft_dir", type=str, default="/s7home/zyk524/FLOWSEP/FlowSep-main/peft")
    
    parser.add_argument("--input_json", type=str, default="/s7home/zyk524/SAM_Audio/sam-audio-bench/json_freeform/General_SFX_freeform.json")
    parser.add_argument("--output_dir", type=str, default="/s7home/zyk524/SAM_Audio/sam-audio-bench/flowsep/General_SFX/new_free")
    
    parser.add_argument("--infer_step", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    seed_everything(42)
    
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output Directory: {args.output_dir}")

    config = yaml.load(open(args.config_yaml, "r"), Loader=yaml.FullLoader)
    model = build_model(args, config)
    val_dataset = AudioDataset(config, split="test", add_ons=[])

    # 预加载 Adapter
    print("\nPre-loading Adapter weights...")
    adapter_weights_list = []
    adapter_paths_list = [] # 记录路径用于 JSON
    
    for i in range(7):
        pth = os.path.join(args.peft_dir, f"cluster_{i}", f"cluster{i}_best.pt")
        if not os.path.exists(pth):
            pth = os.path.join(args.peft_dir, f"cluster_{i}", f"cluster{i}_last.pt")
        
        if os.path.exists(pth):
            print(f"  Cluster {i}: Found {os.path.basename(pth)}")
            adapter_weights_list.append(torch.load(pth, map_location=args.device))
            adapter_paths_list.append(os.path.abspath(pth))
        else:
            print(f"  Cluster {i}: NOT FOUND!")
            adapter_weights_list.append(None)
            adapter_paths_list.append(None)

    print(f"\nLoading data from {args.input_json}")
    with open(args.input_json, 'r') as f:
        data_list = json.load(f)

    dataset_root = os.path.dirname(args.input_json)
    results_metadata = []
    success_count = 0

    print(f"Starting Inference on {len(data_list)} samples x 7 Clusters...")

    for idx, item in enumerate(tqdm(data_list)):
        mix_path = item.get("audio") or item.get("original_audio") or item.get("mix_wav")
        caption = item.get("caption") or item.get("text")
        
        if not mix_path: continue

        if not os.path.isabs(mix_path):
            mix_path = os.path.join(dataset_root, mix_path)

        if not os.path.exists(mix_path):
            if idx < 3: print(f"\n[ERROR] File not found: {mix_path}")
            continue

        original_name = os.path.basename(mix_path)
        base_name = os.path.splitext(original_name)[0]

        # === 遍历 7 个 Cluster ===
        for i, adapter_w in enumerate(adapter_weights_list):
            if adapter_w is None: continue
            
            # 1. 动态加载 Adapter
            model.load_state_dict(adapter_w, strict=False)
            
            # 2. 构造保存路径
            save_name = f"{base_name}_{caption}_peft{i}.wav"
            save_path = os.path.join(args.output_dir, save_name)
            
            try:
                # 3. 推理
                gen_wav = run_inference_ola(model, mix_path, caption, args.device, val_dataset, args)
                torchaudio.save(save_path, gen_wav, 16000)
                
                # 4. 记录元数据
                results_metadata.append({
                    "original_audio": os.path.abspath(mix_path),
                    "generated_audio": os.path.abspath(save_path),
                    "caption": caption,
                    "peft_id": i,
                    "peft_path": adapter_paths_list[i]
                })
                
            except Exception as e:
                print(f"\n[ERROR] Item {idx} Cluster {i}: {e}")

        success_count += 1

    json_output_path =  "/s7home/zyk524/SAM_Audio/sam-audio-bench/flowsep/General_SFX/new_free_json/results.json"
    with open(json_output_path, "w") as f:
        json.dump(results_metadata, f, indent=4)

    print("\n" + "="*40)
    print(f"Inference Finished.")
    print(f"Processed Items: {success_count} / {len(data_list)}")
    print(f"Total Generated Files: {len(results_metadata)}")
    print(f"JSON saved to: {json_output_path}")
    print("="*40)

if __name__ == "__main__":
    main()
