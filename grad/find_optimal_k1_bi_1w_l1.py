"""  
Determine Optimal K for AudioSep Experts based on Gradient Conflict
========================================  
1. Cross-Cluster Mixing: Clean audio from cluster_i, noise from others.
2. Computes Cosine Similarity (Eq 11) and/or Influence Score (Eq 12).
3. Automatically determines the optimal K by finding the exact point where 
   destructive interference (negative similarity/influence) disappears.
"""

import os
import sys
sys.path.append('/audiosep/audiosep/audio')
import json  
import random  
import pickle  
import argparse  
from collections import defaultdict  
from typing import Dict, List, Tuple, Optional  

import numpy as np  
import torch  
import torch.nn as nn  
import torch.nn.functional as F  
from torch.utils.data import Dataset, DataLoader  
import torchaudio  
from tqdm import tqdm  

from scipy.spatial.distance import squareform 
from scipy.cluster.hierarchy import linkage, fcluster 
from losses_sdr import get_loss_function
loss_type = 'l1_wav'
import warnings  
warnings.filterwarnings('ignore')  

# ============================================================  
# 0. Cross-Cluster SegmentMixer
# ============================================================
def get_energy(x): return torch.mean(x ** 2)

def get_energy_ratio(segment1, segment2):
    energy1 = get_energy(segment1)
    energy2 = max(get_energy(segment2), 1e-10)
    return torch.clamp((energy1 / energy2) ** 0.5, 0.02, 50)

def rescale_to_match_energy(segment1, segment2):
    return segment1 / get_energy_ratio(segment1, segment2)

def dynamic_loudnorm(audio, reference, lower_db=-10, higher_db=10): 
    rescaled_audio = rescale_to_match_energy(audio, reference)
    gain = 10 ** (random.uniform(lower_db, higher_db) / 20.0)
    return gain * rescaled_audio

class SegmentMixer(nn.Module):
    def __init__(self, max_mix_num, lower_db, higher_db, exclude_cluster_id=None, json_files=None):
        super().__init__()
        self.max_mix_num = max_mix_num
        self.loudness_param = {'lower_db': lower_db, 'higher_db': higher_db}
        self.exclude_cluster_id = exclude_cluster_id
        self.base_json_files = json_files
        self.audio_pool = []
        self._load_audio_paths()

    def _load_audio_paths(self):
        self.audio_pool = []
        for json_file in self.base_json_files:
            if self.exclude_cluster_id is not None and os.path.basename(json_file) == f"cluster{self.exclude_cluster_id}.json":
                continue
            if not os.path.exists(json_file): continue
            try:
                with open(json_file, 'r') as f: data = json.load(f)
                if 'data' not in data: continue
                json_dir = os.path.dirname(json_file)
                for item in data['data']:
                    if 'wav' in item:
                        raw_path = item['wav']
                        self.audio_pool.append(raw_path if raw_path.startswith('/') else os.path.join(json_dir, raw_path))
            except Exception: pass

    def _load_audio(self, audio_path, device=None):
        try:
            if not os.path.exists(audio_path): return None
            waveform, sr = torchaudio.load(audio_path)
            if waveform.numel() == 0 or torch.isnan(waveform).any(): return None
            if waveform.shape[0] > 1: waveform = torch.mean(waveform, dim=0, keepdim=True)
            return waveform.to(device) if device else waveform
        except Exception: return None
    
    def forward(self, waveforms):
        batch_size, device = waveforms.shape[0], waveforms.device
        out_segments, out_mixtures = [], []
        if not self.audio_pool: return waveforms, waveforms

        for n in range(batch_size):
            segment = waveforms[n].clone()
            noise = torch.zeros_like(segment)
            mix_num = random.randint(2, self.max_mix_num)
            
            for i in range(1, mix_num):
                next_segment = None
                for _ in range(10): 
                    loaded = self._load_audio(random.choice(self.audio_pool), device)
                    if loaded is not None:
                        next_segment = loaded
                        break
                if next_segment is None: next_segment = waveforms[(n + i) % batch_size].clone()
                
                if next_segment.shape[1] < segment.shape[1]:
                    next_segment = next_segment.repeat(1, (segment.shape[1] // next_segment.shape[1]) + 1)
                next_segment = next_segment[:, :segment.shape[1]]
                noise += dynamic_loudnorm(next_segment, segment, **self.loudness_param).to(device)

            noise = dynamic_loudnorm(noise, segment, **self.loudness_param).to(device)
            mixture = segment + noise
            
            max_val = torch.max(torch.abs(mixture))
            if max_val > 1:
                segment *= 0.9 / max_val
                mixture *= 0.9 / max_val
            
            out_segments.append(segment)
            out_mixtures.append(mixture)
        return torch.stack(out_mixtures), torch.stack(out_segments)

# ============================================================  
# 1. Dataset & Gradient Extraction
# ============================================================  
class CategoryAudioDataset(Dataset):  
    def __init__(self, file_list, sample_rate=32000, segment_length=32000*2):  
        self.file_list, self.sr, self.length = file_list, sample_rate, segment_length  
    def __len__(self): return len(self.file_list)  
    def __getitem__(self, idx):  
        try:  
            w, sr = torchaudio.load(self.file_list[idx])  
            if sr != self.sr: w = torchaudio.transforms.Resample(sr, self.sr)(w)  
            if w.shape[0] > 1: w = w.mean(dim=0, keepdim=True)  
            if w.shape[-1] < self.length: w = F.pad(w, (0, self.length - w.shape[-1]))  
            else:  
                start = random.randint(0, w.shape[-1] - self.length)  
                w = w[:, start:start + self.length]  
            return w  
        except Exception: return torch.zeros(1, self.length)  

def flatten_gradients(model, target_modules=None):  
    grads = [p.grad.detach().flatten() for n, p in model.named_parameters() 
             if p.grad is not None and (target_modules is None or any(tm in n for tm in target_modules))]
    return torch.cat(grads) if grads else None  

def compute_category_gradient(model, query_encoder, mixer, loss_fn, data, name, device, bs=4, tm=None):  
    loader = DataLoader(CategoryAudioDataset(data), batch_size=bs, shuffle=True)  
    acc_grad, count = None, 0  
    model.eval() 
    for w in loader:  
        w = w.to(device)  
        model.zero_grad()  
        mixtures, targets = mixer(w)  
        with torch.no_grad():  
            cond = query_encoder.get_query_embed(modality='text', text=[name]*w.shape[0], audio=None, use_text_ratio=1.0)  
        out = model({'mixture': mixtures, 'condition': cond})['waveform']
        if out.dim() == 3: out = out.squeeze(1)
        loss = loss_fn({'segment': out}, {'segment': targets.squeeze(1)})
        loss.backward()  
        grad = flatten_gradients(model, tm)  
        if grad is not None:  
            acc_grad = grad.clone() if acc_grad is None else acc_grad + grad  
            count += 1  
    return acc_grad / count if acc_grad is not None and count > 0 else None  

# ============================================================  
# 2. Eq 11: Cosine Similarity
# ============================================================  
def compute_cosine_similarity(grad_dict):  
    cats = sorted(grad_dict.keys())  
    G = torch.stack([grad_dict[c] for c in cats])  
    G_norm = G / G.norm(dim=1, keepdim=True).clamp(min=1e-8)  
    return torch.mm(G_norm, G_norm.t()).cpu().numpy(), cats, G_norm  

# ============================================================  
# 3. Eq 12: Causal Influence Score (Optional, Very Slow)
# ============================================================  
def evaluate_loss(model, query_encoder, mixer, loss_fn, data, name, device, num_samples=8):
    loader = DataLoader(CategoryAudioDataset(data[:num_samples]), batch_size=4, shuffle=False)
    total_loss, count = 0.0, 0
    model.eval()
    with torch.no_grad():
        for w in loader:
            w = w.to(device)
            mixtures, targets = mixer(w)
            cond = query_encoder.get_query_embed(modality='text', text=[name]*w.shape[0], audio=None, use_text_ratio=1.0)
            out = model({'mixture': mixtures, 'condition': cond})['waveform']
            if out.dim() == 3: out = out.squeeze(1)
            loss = loss_fn({'segment': out}, {'segment': targets.squeeze(1)})
            total_loss += loss.item()
            count += 1
    return total_loss / max(count, 1)

def compute_influence_matrix(model, G_norm, cats, all_files, query_encoder, mixer, loss_fn, device, tm=None, lambda_lr=0.01):
    N = len(cats)
    influence_matrix = np.zeros((N, N))
    orig_params = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}
    
    # --- 核心修复：通过一次 Dummy Backward 找出真正产生梯度的活跃参数 ---
    print("\n[0/2] 正在探测活跃参数 (Dummy Backward)...")
    model.zero_grad()
    dummy_w = torch.randn(2, 1, 32000).to(device)
    dummy_mix, dummy_tgt = mixer(dummy_w)
    cond = query_encoder.get_query_embed(modality='text', text=["dummy"]*2, audio=None, use_text_ratio=1.0)
    out = model({'mixture': dummy_mix, 'condition': cond})['waveform']
    if out.dim() == 3: out = out.squeeze(1)
    loss = loss_fn({'segment': out}, {'segment': dummy_tgt.squeeze(1)}, dummy_mix.squeeze(1))
    loss.backward()
    
    active_param_names = set()
    for n, p in model.named_parameters():
        if p.grad is not None and (tm is None or any(t in n for t in tm)):
            active_param_names.add(n)
    model.zero_grad() # 清理 dummy 梯度
    print(f"探测完毕：共有 {len(active_param_names)} 个参数张量实际参与了梯度更新。")
    # ----------------------------------------------------------------------
    
    print("\n[1/2] 计算基线 Loss 和自更新 Loss (Delta_i L_i)...")
    base_loss, self_delta = {}, {}
    for i, cat_i in enumerate(tqdm(cats)):
        l_orig = evaluate_loss(model, query_encoder, mixer, loss_fn, all_files[cat_i], cat_i, device)
        base_loss[cat_i] = l_orig
        
        g_i = G_norm[i].to(device)
        idx = 0
        for n, p in model.named_parameters():
            if n in active_param_names:  # <--- 修复：只更新活跃参数
                numel = p.numel()
                p.data = orig_params[n] - lambda_lr * g_i[idx:idx+numel].view_as(p.data)
                idx += numel
                
        l_new = evaluate_loss(model, query_encoder, mixer, loss_fn, all_files[cat_i], cat_i, device)
        self_delta[cat_i] = l_orig - l_new 
        for n, p in model.named_parameters():
            if p.requires_grad: p.data = orig_params[n].clone()

    print("\n[2/2] 计算交叉影响分数 (Influence Score I_{i,j})...")
    for j, cat_j in enumerate(tqdm(cats, desc="Simulating updates for task j")):
        g_j = G_norm[j].to(device)
        idx = 0
        for n, p in model.named_parameters():
            if n in active_param_names:  # <--- 修复：只更新活跃参数
                numel = p.numel()
                p.data = orig_params[n] - lambda_lr * g_j[idx:idx+numel].view_as(p.data)
                idx += numel
                
        for i, cat_i in enumerate(cats):
            if i == j:
                influence_matrix[i, j] = 1.0
                continue
            l_new_i = evaluate_loss(model, query_encoder, mixer, loss_fn, all_files[cat_i], cat_i, device)
            delta_j_L_i = base_loss[cat_i] - l_new_i
            denom = abs(self_delta[cat_i]) + 1e-6
            influence_matrix[i, j] = delta_j_L_i / denom
            
        for n, p in model.named_parameters():
            if p.requires_grad: p.data = orig_params[n].clone()
            
    return influence_matrix

# ============================================================  
# 4. 核心算法：基于物理法则自动确定 K 值
# ============================================================  
def determine_optimal_k_by_conflict(affinity_matrix: np.ndarray, categories: List[str], conflict_threshold: float = 0.0):
    """
    通过不断增加 K，直到所有专家组内部都不存在破坏性干扰（相似度/影响分数 >= 0），从而确定最优 K。
    """
    print(f"\n开始基于梯度冲突法则寻找最优 K 值 (冲突容忍度: {conflict_threshold})...")
    
    # 将亲和度矩阵转换为距离矩阵 (Complete Linkage 需要距离)
    # Affinity 越大 (越协同)，Distance 越小
    dist_matrix = 1.0 - affinity_matrix  
    np.fill_diagonal(dist_matrix, 0.0)  
    dist_matrix = np.maximum(dist_matrix, 0.0)  
    dist_matrix = (dist_matrix + dist_matrix.T) / 2  
    condensed_dist = squareform(dist_matrix)  
    
    # 使用 Complete Linkage 构建层次树
    # Complete Linkage 保证：合并两个簇时，看的是它们之间最冲突的两个类别的距离。
    Z = linkage(condensed_dist, method='complete')  
    
    optimal_k = None
    optimal_labels = None
    max_possible_k = len(categories)
    
    print(f"{'K值':<5} | {'组内最差亲和度 (Worst Affinity)':<30} | {'状态'}")
    print("-" * 60)
    
    # 从 K=2 开始，不断细分，直到组内没有冲突
    for k in range(2, max_possible_k + 1):
        labels = fcluster(Z, t=k, criterion='maxclust')
        worst_affinity_overall = 1.0  
        
        for c in range(1, k + 1):
            mask = (labels == c)
            if mask.sum() > 1:
                sub_affinity = affinity_matrix[np.ix_(mask, mask)]
                triu_idx = np.triu_indices_from(sub_affinity, k=1)
                vals = sub_affinity[triu_idx]
                
                min_affinity_in_cluster = np.min(vals)
                if min_affinity_in_cluster < worst_affinity_overall:
                    worst_affinity_overall = min_affinity_in_cluster
        
        status = "❌ 存在破坏性干扰" if worst_affinity_overall < conflict_threshold else "✅ 协同 (Synergistic)"
        print(f"{k:<5} | {worst_affinity_overall:<30.4f} | {status}")
        
        # 找到第一个满足无冲突条件的 K
        if worst_affinity_overall >= conflict_threshold:
            optimal_k = k
            optimal_labels = labels
            break
            
    if optimal_k is None:
        optimal_k = max_possible_k
        optimal_labels = np.arange(max_possible_k)
        
    print("-" * 60)
    print(f"🎉 结论: 为了彻底消除破坏性干扰，最优的专家组数量 K = {optimal_k}")
    
    cluster_info = defaultdict(list)
    for cat, label in zip(categories, optimal_labels):
        cluster_info[int(label)].append(cat)
        
    return optimal_k, dict(cluster_info)

# ============================================================  
# 5. Main Pipeline  
# ============================================================  
def load_train_json(json_path):  
    with open(json_path, 'r') as f: data = json.load(f)  
    if 'data' in data: data = data['data']  
    cat_files = defaultdict(list)  
    for item in data: cat_files[item['caption'].strip()].append(item['wav'])  
    return dict(cat_files)  

def load_audiosep_checkpoint(model, path):
    ckpt = torch.load(path, map_location='cpu')
    sd = {k[9:]: v for k, v in ckpt.get('state_dict', ckpt).items() if k.startswith('ss_model.')} or ckpt
    model.load_state_dict(sd, strict=False)
    return model

def main():  
    parser = argparse.ArgumentParser()  
    parser.add_argument('--base_data_dir', type=str, default='/audiosep/audiosep/zhunbei/70cluster/train')
    parser.add_argument('--checkpoint', type=str, default='/audiosep/audiosep/audio/checkpoint/audiosep_base_4M_steps.ckpt')
    parser.add_argument('--output_dir', type=str, default='/audiosep/audiosep/527class/gradient_expert_k/1w/2/l1')
    parser.add_argument('--metric', type=str, choices=['cosine', 'influence'], default='cosine', 
                        help='使用哪种指标来确定K值。cosine极快，influence极慢。')
    parser.add_argument('--conflict_threshold', type=float, default=0.0, 
                        help='冲突容忍度。0.0表示不允许任何负相关。')
    parser.add_argument('--device', type=str, default=None)  
    parser.add_argument('--target_modules', type=str, nargs='+', default=None)
    args = parser.parse_args()  
      
    os.makedirs(args.output_dir, exist_ok=True)  
    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
      
    print("Building model...")
    from models.resunet import ResUNet30  
    from models.clap_encoder import CLAP_Encoder  
      
    ss_model = ResUNet30(input_channels=1, output_channels=1, condition_size=512).to(device) 
    query_encoder = CLAP_Encoder().eval().to(device)  
    if os.path.exists(args.checkpoint): ss_model = load_audiosep_checkpoint(ss_model, args.checkpoint)

    ss_model.eval()
    for n, p in ss_model.named_parameters():
        p.requires_grad = (args.target_modules is None or any(tm in n for tm in args.target_modules))
    for p in query_encoder.parameters(): p.requires_grad = False
    
    loss_fn = get_loss_function(loss_type) 
    
    # ---- 1. 收集梯度 ----
    save_path = os.path.join(args.output_dir, 'gradient_cache.pkl')
    all_json_files = [os.path.join(args.base_data_dir, f"cluster{i}.json") for i in range(70)]
    global_category_files = defaultdict(list)
    
    if os.path.exists(save_path):
        print(f"\nLoading cached gradients from {save_path}")
        with open(save_path, 'rb') as f: final_gradient_dict = pickle.load(f)
        if args.metric == 'influence':
            for jf in all_json_files:
                if os.path.exists(jf):
                    for c, fs in load_train_json(jf).items(): global_category_files[c].extend(fs)
    else:
        print("\n开始跨聚类梯度收集...")
        cat_grads = defaultdict(list)
        for cid in range(70):
            if not os.path.exists(all_json_files[cid]): continue
            cat_files = load_train_json(all_json_files[cid])
            if not cat_files: continue
            
            for c, fs in cat_files.items(): global_category_files[c].extend(fs)
                
            mixer = SegmentMixer(2, -10, 10, cid, all_json_files).to(device).eval()
            for cat, files in tqdm(cat_files.items(), desc=f"Cluster {cid}"):
                grad = compute_category_gradient(ss_model, query_encoder, mixer, loss_fn, files, cat, device, 4, args.target_modules)
                if grad is not None: cat_grads[cat].append(grad.cpu())
        
        final_gradient_dict = {c: torch.stack(gs).mean(dim=0) for c, gs in cat_grads.items()}
        with open(save_path, 'wb') as f: pickle.dump(final_gradient_dict, f)
            
    # ---- 2. 计算亲和度矩阵 (Cosine 或 Influence) ----  
    print("\n计算梯度余弦相似度 (Eq 11)...")  
    sim_matrix, categories, G_norm = compute_cosine_similarity(final_gradient_dict)  
    np.save(os.path.join(args.output_dir, 'cosine_similarity.npy'), sim_matrix)  
    
    affinity_matrix = sim_matrix
    
    if args.metric == 'influence':
        print("\n计算因果影响分数 (Eq 12)... 这可能需要几天时间！")
        eval_mixer = SegmentMixer(2, -10, 10, None, all_json_files).to(device).eval()
        inf_matrix = compute_influence_matrix(
            ss_model, G_norm, categories, global_category_files, 
            query_encoder, eval_mixer, loss_fn, device, args.target_modules
        )
        np.save(os.path.join(args.output_dir, 'influence_score.npy'), inf_matrix)
        
        # 影响分数是不对称的，为了寻找 K，我们取最小值（最悲观的冲突估计）
        # 如果 A 破坏 B，或者 B 破坏 A，我们都认为它们之间存在冲突
        affinity_matrix = np.minimum(inf_matrix, inf_matrix.T)
      
    # ---- 3. 自动确定最优 K 值 ----  
    optimal_k, expert_groups = determine_optimal_k_by_conflict(
        affinity_matrix=affinity_matrix, 
        categories=categories, 
        conflict_threshold=args.conflict_threshold
    )
      
    # 保存结果
    with open(os.path.join(args.output_dir, f'optimal_experts_k{optimal_k}.json'), 'w') as f:  
        json.dump(expert_groups, f, indent=2, ensure_ascii=False)
      
    print(f"\n✓ 最终专家分组结果已保存至 {args.output_dir}/optimal_experts_k{optimal_k}.json")  
  
if __name__ == '__main__':  
    main()
