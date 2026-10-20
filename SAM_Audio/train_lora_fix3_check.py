import os
import json
import torch
import torch.nn as nn

# 禁用 tokenizer 并行警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["SAM_AUDIO_TEXT_ONLY"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

from sam_audio import SAMAudio, SAMAudioConfig

# ==========================================
# 1. LoRA 模块定义 (与你之前的一致)
# ==========================================
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

def print_model_stats(model, stage=""):
    total_params = 0
    trainable_params = 0
    frozen_params = 0

    for param in model.parameters():
        num_params = param.numel()
        total_params += num_params
        if param.requires_grad:
            trainable_params += num_params
        else:
            frozen_params += num_params

    print(f"\n{'-'*20} {stage} Model Statistics {'-'*20}")
    print(f"  - Total Params:      {total_params / 1e6:.2f} M")
    print(f"  - Frozen Params:     {frozen_params / 1e6:.2f} M")
    print(f"  - Trainable Params:  {trainable_params / 1e6:.2f} M")
    print(f"  - Trainable Ratio:   {trainable_params / total_params * 100:.4f} %")
    
    if trainable_params > 0 and "LoRA" in stage:
        # 预估保存的 .pt 文件大小 (假设 float32, 每个参数 4 bytes)
        estimated_mb = (trainable_params * 4) / (1024 * 1024)
        print(f"  -> [预估] 保存的 LoRA 权重文件大小约为: {estimated_mb:.2f} MB")
    print("-" * 65)

# ==========================================
# 2. 改进版的 LoRA 注入逻辑
# ==========================================
def inject_lora_improved(model, target_modules, rank=16, alpha=16, dropout=0.1):
    # 1. 冻结所有基础参数
    for param in model.parameters():
        param.requires_grad = False
        
    injected_count = 0
    modules_to_replace = []
    
    # 2. 寻找匹配的层
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            layer_suffix = name.split('.')[-1]
            
            # 只要层的后缀包含在 target_modules 中，或者部分匹配，就进行替换
            # 例如：layer_suffix 是 'q_proj'，target_modules 里有 'q_proj'，就会匹配成功
            if any(target in layer_suffix for target in target_modules):
                if '.' in name:
                    parent_name = name.rsplit('.', 1)[0]
                    parent = model.get_submodule(parent_name)
                else:
                    parent = model
                modules_to_replace.append((parent, layer_suffix, name))

    # 3. 执行替换
    for parent, layer_suffix, full_name in modules_to_replace:
        original = getattr(parent, layer_suffix)
        if isinstance(original, LoRALinear):
            continue
        lora_layer = LoRALinear(original, rank=rank, alpha=alpha, dropout=dropout)
        setattr(parent, layer_suffix, lora_layer)
        injected_count += 1
        
    print(f"\n[LoRA Injection] 成功注入了 {injected_count} 个 Linear 层！")
    return model

# ==========================================
# 3. 主检查流程
# ==========================================
def main():
    # --- 路径配置 (与你代码一致) ---
    SAM_AUDIO_DIR = "/s7home/zyk524/SAM_Audio/sam-audio-large"
    CONFIG_PATH = os.path.join(SAM_AUDIO_DIR, "config.json")
    LOCAL_T5_DIR = "/s7home/zyk524/SAM_Audio/sam-audio-large/huggingface/hub/models--t5-base/snapshots/a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1"
    
    print(">>> 正在加载 SAMAudio Config 并初始化模型 (仅在 CPU 上，不加载权重)...")
    with open(CONFIG_PATH) as f: 
        cfg = json.load(f)
    cfg['text_encoder']['name'] = LOCAL_T5_DIR
    config = SAMAudioConfig(**cfg)
    
    # 初始化模型 (不需要 load_state_dict，我们只看结构)
    model = SAMAudio(config)
    
    # ==========================================
    # 步骤 A：扫描并打印所有真实的 Linear 层名字
    # ==========================================
    print("\n>>> [步骤 A] 扫描 SAMAudio 中的所有 nn.Linear 层...")
    linear_suffixes = set()
    example_paths = []
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            suffix = name.split('.')[-1]
            linear_suffixes.add(suffix)
            if len(example_paths) < 10: # 收集前10个完整路径作为示例
                example_paths.append(name)
                
    print(f"\n[发现的 Linear 层后缀集合]:")
    print(list(linear_suffixes))
    print(f"\n[部分完整路径示例]:")
    for p in example_paths:
        print(f"  - {p}")

    # ==========================================
    # 步骤 B：打印基线模型参数量
    # ==========================================
    print_model_stats(model, stage="[Baseline]")

    # ==========================================
    # 步骤 C：使用扩充的 target_modules 注入 LoRA
    # ==========================================
    # ⚠️ 这里我为你准备了一个“天罗地网”式的 target 列表，涵盖了绝大多数大模型的命名习惯
    expanded_target_modules = [
        # UDiT / 传统习惯
        'to_q', 'to_k', 'to_v', 'proj', 'fc1', 'fc2', 
        # HuggingFace LLaMA / Qwen / T5 习惯
        'q', 'k', 'v', 'o', 'wi_0', 'wi_1', 'wo', # T5 常见
        'q_proj', 'k_proj', 'v_proj', 'out_proj', 'up_proj', 'down_proj', 'gate_proj',
        # 其他常见
        'qkv', 'dense', 'linear_1', 'linear_2'
    ]
    
    print(f"\n>>> [步骤 C] 开始注入 LoRA (Rank=16)...")
    print(f"使用的目标匹配列表: {expanded_target_modules}")
    
    model = inject_lora_improved(
        model, 
        target_modules=expanded_target_modules, 
        rank=16, 
        alpha=16
    )

    # ==========================================
    # 步骤 D：打印注入后的参数量与预估文件大小
    # ==========================================
    print_model_stats(model, stage="[After LoRA]")

if __name__ == "__main__":
    main()