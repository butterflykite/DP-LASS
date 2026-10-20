# Author: Haohe Liu
# Email: haoheliu@gmail.com
# Date: 11 Feb 2023


import sys

sys.path.append("/s7home/zyk524/FLOWSEP/FlowSep-main/src")
import shutil
import os
# please modify the following settings to use wandb or setup the cache folder
# os.environ["HF_HOME"] = ""
# os.environ["WANDB_API_KEY"] = ""
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["HUGGINGFACE_HUB_CACHE"] = ""
# os.environ["TORCH_HOME"] = ""

# import os

from tqdm import tqdm
import argparse
import yaml
import torch
import ipdb
from utilities.data.dataset import AudioDataset

from torch.utils.data import WeightedRandomSampler
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer, seed_everything
import wandb
from latent_diffusion.util import instantiate_from_config
import torchaudio
import transformers
transformers.logging.set_verbosity_error()

def print_on_rank0(msg):
    if torch.distributed.get_rank() == 0:
        print(msg)
        
def main(configs, exp_group_name, exp_name,text,wav):
    seed_everything(0)

    log_path = configs["log_directory"]

    save_dir =  "/s7home/zyk524/FLOWSEP/FlowSep-main/inference_results"
    os.makedirs(save_dir, exist_ok=True)
    print(f"Results will be saved to: {save_dir}")

    val_dataset = AudioDataset(configs, split="test", add_ons=[])

    latent_diffusion = instantiate_from_config(configs["model"]).to("cuda")

    resume_from_checkpoint = args.load_checkpoint
    if resume_from_checkpoint is not None:
        ckpt = torch.load(resume_from_checkpoint)["state_dict"]

    latent_diffusion.load_state_dict(ckpt, strict=True)

    print("the checkpoint is",resume_from_checkpoint)

    count = 0
    for cur_text in text:
        cur_wav = wav[count]

        batch = {}

        batch["fname"] = [cur_wav]
        batch["text"] = [cur_text]
        batch["caption"] = [cur_text]
        batch["waveform"] = torch.rand(1,1,163840).cuda()
        batch["log_mel_spec"] = torch.rand(1,1024,64).cuda()
        batch["sampling_rate"] = torch.tensor([16000]).cuda()
        batch["label_vector"] = torch.rand(1,527).cuda()
        batch["stft"] = torch.rand(1,1024,512).cuda()
        noise_waveform, random_start = val_dataset.read_wav_file(cur_wav)

        noise_waveform = noise_waveform[0][:163840]

        mixed_mel, stft = val_dataset.wav_feature_extraction(noise_waveform.reshape(1,-1))
        
        batch["mixed_waveform"] = torch.from_numpy(noise_waveform.reshape(1,1,163840))
        batch["mixed_mel"] = mixed_mel.reshape(1,mixed_mel.shape[0],mixed_mel.shape[1])

        waveform = latent_diffusion.generate_sample([batch],name="lass_result",unconditional_guidance_scale=1.0,ddim_steps=args.infer_step,n_gen=1,save_mixed = args.no_mixed)
        
        # ================== 修正后的保存逻辑 ==================
        # 1. 提取音频数据 (处理可能的字典返回)
        if isinstance(waveform, dict):
            # 尝试常见的键名，或者直接取第一个值
            gen_wav = waveform.get('sample', waveform.get('wav', list(waveform.values())[0]))
        else:
            gen_wav = waveform

        # 2. 处理数据类型 (解决 AttributeError)
        # 如果是 numpy 数组，转为 Tensor；如果是 Tensor，则 detach().cpu()
        if not torch.is_tensor(gen_wav):
            gen_wav = torch.from_numpy(gen_wav)
        else:
            gen_wav = gen_wav.detach().cpu()

        # 3. 处理维度 [Batch, Channel, Time] -> [Channel, Time]
        # torchaudio.save 需要 (Channel, Time) 格式
        if gen_wav.dim() == 3:
            gen_wav = gen_wav.squeeze(0)  # 去掉 Batch 维度
        elif gen_wav.dim() == 1:
            gen_wav = gen_wav.unsqueeze(0) # 增加 Channel 维度

        # 4. 保存
        file_name = f"gen_{count}_{exp_name}.wav"
        save_path = os.path.join(save_dir, file_name)
        
        # 确保数据是 float 类型
        gen_wav = gen_wav.float()
        
        torchaudio.save(save_path, gen_wav, 16000)
        print(f"Saved generated audio to {save_path}")
        
        count+=1


def set_yaml_config(config_yaml, cfg_scale, ddim, n_cand):
    config_yaml["model"]["params"]["evaluation_params"]["unconditional_guidance_scale"] = cfg_scale
    config_yaml["model"]["params"]["evaluation_params"]["ddim_sampling_steps"] = ddim
    config_yaml["model"]["params"]["evaluation_params"]["n_candidates_per_samples"] = n_cand
    return config_yaml

if __name__ == "__main__":

    # python3 ~/submit.py -i config/4_audiomae_cond/2023_05_20_audiomae_crossattn_audiocaps_pool_rand.sh --cpu 32 --gpu 4 --mem 96 --group_id 122 --cluster_id 17 --note audiomae_pool_rand_audiocaps --elas
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config_yaml",
        type=str,
        default="/s7home/zyk524/FLOWSEP/FlowSep-main/lass_config/2channel_flow.yaml",
        help="path to config .yaml file",
    )

    parser.add_argument(
        "-t",
        "--text",
        type=str,
        default="Speech",
        help="text of the prompt",
    )

    parser.add_argument(
        "-a",
        "--audio",
        type=str,
        default="/dataset/AudioSep_eval/audioset/class_id=0/class_id=0,index=000,mixture.wav",
        help="path to config .yaml file",
    )
    parser.add_argument(
        "-l",
        "--load_checkpoint",
        type=str,
        default="/s7home/zyk524/FLOWSEP/FlowSep-main/model_logs/v2_100k.ckpt",
        help="path to the checkpoint",
    )

    parser.add_argument(
        "-s",
        "--infer_step",
        type=int,
        default="20",
        help="the steps for inference",
    )
    

    parser.add_argument(
        "-m",
        "--no_mixed",
        #type=bool,
        action='store_false',
        help="saving the mixed waveform as well",
    )
    
    args = parser.parse_args()
    # torch._dynamo.config.suppress_errors = True

    assert torch.cuda.is_available(), "CUDA is not available"

    config_yaml = args.config_yaml

    exp_name = os.path.basename(config_yaml.split(".")[0])
    exp_group_name = os.path.basename(os.path.dirname(config_yaml))

    config_yaml_path = os.path.join(config_yaml)
    config_yaml = yaml.load(open(config_yaml_path, "r"), Loader=yaml.FullLoader)
    
    text = [args.text]

    wav = [args.audio]

    
    
    main(config_yaml,exp_group_name, exp_name,text,wav)
