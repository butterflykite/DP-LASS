# CLAPSep 多LORA模型训练与推理指南

本文档提供了如何使用CLAPSep框架训练和使用多个可拆卸的LORA模型的详细说明。

## 概述

本实现基于原始CLAPSep项目，扩展了其功能以支持训练多个可拆卸的LORA模型。每个LORA模型可以使用不同的训练数据进行训练，并且可以根据需要在推理时切换。

主要特点：
- 使用与原始项目相同的CLAP文本编码器和音频编码器架构
- LORA的位置与原始项目保持一致
- 支持训练多个独立的LORA模型
- 提供简单的接口用于在推理时切换不同的LORA模型

## 文件结构

- `train_multiple_lora.py`: 用于训练多个LORA模型的主脚本
- `inference_multiple_lora.py`: 用于使用训练好的LORA模型进行推理的脚本
- `experiments/lora_config.json`: 配置文件，指定多个LORA模型的训练数据路径
- `experiments/CLAPSep_multiple_lora/config.json`: 模型和训练参数配置文件

## 准备工作

### 1. 数据准备

为每个LORA模型准备独立的训练和验证数据集：

#### 训练数据

训练数据应以JSON格式准备，结构如下：

```json
{
  "data": [
    {
      "wav": "音频文件的绝对路径",
      "caption": "音频描述文本"
    },
    ...
  ]
}
```

建议的目录结构：

```
data/
  ├── cluster1/
  │   ├── cluster1.json  # 训练数据
  │   └── cluster1.csv   # 验证数据
  ├── cluster2/
  │   ├── cluster2.json
  │   └── cluster2.csv
  └── ...
```

#### 验证数据

验证数据应以CSV格式准备，包含以下列：

```
idx,caption,src_wav,noise_wav
1,"钢琴声","/path/to/piano.wav","/path/to/mixture.wav"
...
```

- `idx`: 样本ID
- `caption`: 音频描述文本
- `src_wav`: 源音频文件的绝对路径
- `noise_wav`: 混合音频文件的绝对路径

### 2. 配置LORA模型

编辑`experiments/lora_config.json`文件，指定每个LORA模型的名称和训练/验证数据路径：

```json
[
    {
        "name": "lora_model_1",
        "train_data_path": "./data/cluster1/cluster1.json",
        "val_data_path": "./data/cluster1/cluster1.csv"
    },
    {
        "name": "lora_model_2",
        "train_data_path": "./data/cluster2/cluster2.json",
        "val_data_path": "./data/cluster2/cluster2.csv"
    },
    ...
]
```

注意：
- `train_data_path` 应指向JSON格式的训练数据文件
- `val_data_path` 应指向CSV格式的验证数据文件

### 3. 配置训练参数

编辑`experiments/CLAPSep_multiple_lora/config.json`文件，设置模型和训练参数：

```json
{
    "model": {
        "lan_embed_dim": 768,
        "depths": [2, 2, 2, 2],
        "num_heads": [4, 8, 16, 32],
        "window_size": 8,
        "phase": false
    },
    "train_data": {
        "sr": 32000,
        "segment_seconds": 10
    },
    "val_data": {
        "sr": 32000
    },

    "optim": {
        "lr": 0.0001
    },
    "clap_path": "./pretrained/music_audioset_epoch_15_esc_90.14.pt",
    "lora": true,
    "lora_rank": 16,
    "nfft": 1024,
    "epochs": 100,
    "batch_size": 32,
    "eval_batch_size": 64,
    "n_workers": 4
}
```

## 训练多个LORA模型

使用以下命令启动训练：

```bash
python train_multiple_lora.py --exp_dir ./experiments/CLAPSep_multiple_lora --lora_config ./experiments/lora_config.json --seed 42 --max_epochs 100 --gpus 1
```

参数说明：
- `--exp_dir`: 实验目录，用于保存检查点和日志
- `--lora_config`: LORA配置文件路径
- `--seed`: 随机种子，用于确保结果可重现
- `--max_epochs`: 最大训练轮次
- `--gpus`: 使用的GPU数量

训练过程中，每个LORA模型的检查点和日志将保存在以下目录结构中：

```
experiments/CLAPSep_multiple_lora/
  ├── checkpoints/
  │   ├── lora_model_1/
  │   │   └── [检查点文件]
  │   ├── lora_model_2/
  │   │   └── [检查点文件]
  │   └── ...
  ├── logs/
  │   ├── lora_model_1/
  │   ├── lora_model_2/
  │   └── ...
  └── tensorboard/
      ├── lora_model_1/
      ├── lora_model_2/
      └── ...
```

## 使用训练好的LORA模型进行推理

### 使用单个LORA模型

```bash
python inference_multiple_lora.py --input ./test_audio.wav --query "piano music" --lora_model lora_model_1 --output_dir ./output
```

参数说明：
- `--input`: 输入音频文件路径
- `--query`: 文本查询，用于指定要分离的声音
- `--lora_model`: 指定使用的LORA模型名称
- `--output_dir`: 输出目录

### 使用所有LORA模型

```bash
python inference_multiple_lora.py --input ./test_audio.wav --query "piano music" --output_dir ./output
```

如果不指定`--lora_model`参数，脚本将使用所有可用的LORA模型进行推理，并将结果保存在输出目录中。

## 在代码中使用LORA模型管理器

您也可以在自己的代码中使用`LoraModelManager`类来管理和使用多个LORA模型：

```python
from inference_multiple_lora import LoraModelManager

# 创建LORA模型管理器
manager = LoraModelManager('./experiments/CLAPSep_multiple_lora/config.json')

# 加载LORA模型
lora_name = manager.load_lora_model('./experiments/CLAPSep_multiple_lora/checkpoints/lora_model_1/lora_model_1_final.pt')

# 切换到指定的LORA模型
manager.switch_lora(lora_name)

# 分离音频
separated_audio = manager.separate_audio('./test_audio.wav', 'piano music', './output/separated.wav')
```

## 注意事项

1. **内存管理**: 训练多个LORA模型可能需要大量内存，特别是在GPU上。如果遇到内存不足的问题，可以考虑减小批次大小或串行训练模型。

2. **数据准备**: 确保每个LORA模型的训练数据质量良好，并且与模型的预期用途相匹配。

3. **模型保存**: 默认情况下，每个LORA模型的最终状态将保存为`[lora_name]_final.pt`文件。此外，训练过程中还会保存性能最佳的几个检查点。

4. **推理效率**: 在推理时，切换不同的LORA模型只需要加载LORA参数，而不需要重新加载整个模型，因此非常高效。

5. **兼容性**: 本实现与原始CLAPSep项目兼容，可以使用原始项目中的预训练模型和数据处理方法。