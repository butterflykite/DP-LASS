```markdown
# DP-LASS

Official implementation of Domain Partitioning Meets Parameter-Efficient Fine-Tuning: A Novel Method for Improved Language-Queried Audio Source Separation and its extended version Language-Queried Audio Source Separation Based on Data Domain Partitioning and Subdomain Fine-Tuning.

Repository: [https://github.com/butterflykite/DP-LASS](https://github.com/butterflykite/DP-LASS)

This repository contains two generations of our subdomain fine-tuning framework for Language-Queried Audio Source Separation (LASS):

- AudioSep-SFT v1 / DP-LASS v1: domain partitioning + subdomain fine-tuning with a fixed number of 7 subdomains.
- AudioSep-SFT v2 / DP-LASS v2: extended version with gradient-conflict-based optimal subdomain count, SED-based data refinement, improved subdomain classifier, lightweight fine-tuning modules, hybrid loss, and generalization to other LASS backbones.

---

## 🚀 Environment Setup Guide

### ✅ Step 1: Create the Conda Environment

Place the `environment.yml` file in your project root directory and run:

```bash
conda env create -f environment.yml
```

### ✅ Step 2: Activate the Environment

```bash
conda activate DP-LASS
```

### ✅ Step 3: Verify Installation

Check that PyTorch and CUDA are correctly installed:

```bash
python -c "import sys, torch; print(f'Python: {sys.version.split()[0]} | PyTorch: {torch.__version__} | CUDA: {torch.cuda.is_available()}')"
```

You should see output similar to:

```text
Python: 3.10.16 | PyTorch: 1.13.1 | CUDA: True
```

---

## 💻 Model Training

This section provides a comprehensive guide to training the DP-LASS model on AudioSet or your own datasets.

### ✅ Step 1: Prepare Data (JSON Format)

Your training data should be defined in a single JSON file. This file must contain a list named `data`, where each element is an object specifying the path to the audio file (`wav`) and its corresponding text description (`caption`).

**Example File Structure (`train_data.json`):**

```json
{
  "data": [
    {
      "wav": "/path/to/your/dataset/audio1.wav",
      "caption": "a person is speaking"
    },
    {
      "wav": "/path/to/your/dataset/audio2.wav",
      "caption": "a dog is barking"
    },
    {
      "wav": "/path/to/your/dataset/audio3.wav",
      "caption": "the sound of rain"
    }
  ]
}
```

### ✅ Step 1.5: Downloading AudioSet

To train on AudioSet:

1. Visit the official download page: [https://research.google.com/audioset/download.html](https://research.google.com/audioset/download.html)
2. Download the segment CSV files (`balanced_train_segments.csv`, etc.) and `class_labels_indices.csv`.
3. Use a tool like `yt-dlp` to download audio clips from YouTube using the provided video IDs and timestamps.
4. Convert label IDs (e.g., `/m/07rwj`) to human-readable captions using the `class_labels_indices.csv`, then format your data into the required JSON structure (Step 1).

> ⚠️ Note: AudioSet does not provide direct audio downloads — you must retrieve clips from YouTube, and availability is not guaranteed.

For DP-LASS v2, we further provide an **SED-based data construction pipeline** in [`SED/`](#-extension-2-sed-based-data-refinement-sed) to build the SED-2.25M and SED-32K datasets.

### ✅ Step 2: Prepare the Configuration File

All hyperparameters and path settings for the training process are managed through a YAML configuration file.

1. **Locate the Config File**: An example configuration file is provided in the repository, such as `config/audiosep_base.yaml`.
2. **Customize Your Configuration**: Create a copy of the example file (e.g., `my_training_config.yaml`) and modify the key parameters to match your setup:
   - **`data`**: Data-related settings, including `sampling_rate` and `segment_seconds`.
   - **`model`**: Model-specific configurations like `model_type`.
   - **`train`**: Training parameters, including `batch_size_per_device`, `num_workers` for the data loader, and the optimizer's `learning_rate`.

### ✅ Step 3: Prepare Pre-trained Base Models

Our training methodology fine-tunes a pre-trained AudioSep model. You must download the necessary base models before starting.

1. **Download Models**: The pretrained AudioSep and CLAP models are available at the official AudioSep Hugging Face repository.
   - **Download URL**: [Hugging Face](https://huggingface.co/spaces/Audio-AGI/AudioSep/tree/main/checkpoint)
2. **Model Origin**: These models are provided by the official AudioSep project. For more details, visit their GitHub repository: [https://github.com/Audio-AGI/AudioSep](https://github.com/Audio-AGI/AudioSep).
3. **Place Models**: After downloading the checkpoint file (e.g., `audiosep_base_4M_steps.ckpt`), place it in a convenient location, such as a `pretrained_models/` directory in your project root. You will need to provide the path to this file when running the training script.

### ✅ Step 4: Run the Training Script

Once your data and configuration are ready, you can start the training process by executing the `train.py` script from your terminal.

Below is a complete example of the training command for DP-LASS v1:

```bash
python train.py \
    --workspace /path/to/your/output/directory \
    --config_yaml /path/to/your/config.yaml \
    --checkpoint_path /path/to/your/pretrained_models/audiosep_base_4M_steps.ckpt \
    --max_epochs 150 \
    --num_gpus 2 \
    --strategy ddp \
    --resume
```

#### Parameter Explanation

- `--workspace`: **(Required)** The directory where all training artifacts (model checkpoints, logs, etc.) will be saved.
- `--config_yaml`: **(Required)** The path to the YAML configuration file you prepared in Step 2.
- `--checkpoint_path`: **(Required)** The path to the pre-trained AudioSep model checkpoint you downloaded in Step 3.
- `--max_epochs`: The total number of epochs to train for. Defaults to `150`.
- `--num_gpus`: The number of GPUs to use for training. Defaults to auto-detection if not specified.
- `--strategy`: The distributed training strategy. `ddp` is recommended for multi-GPU training.
- `--resume`: An optional flag. If included, the script will automatically search for the latest checkpoint (`last.ckpt`) in the `--workspace` directory and resume training from that point.

For DP-LASS v2, set the subdomain count according to the gradient analysis result:

```bash
python train.py \
    --workspace /path/to/your/output/directory \
    --config_yaml /path/to/your/config.yaml \
    --checkpoint_path /path/to/your/pretrained_models/audiosep_base_4M_steps.ckpt \
    --data_json /path/to/your/SED/SED-2.25M.json \
    --num_subdomains 70 \
    --max_epochs 150 \
    --num_gpus 2 \
    --strategy ddp
```

> 💡 The theoretical zero-conflict point derived in the extended paper is `K = 71`. The final deployed model uses `K = 70`, which performs marginally better in practice.

### ✅ Step 5: Monitor Training and Find Results

After launching the script, you can monitor its progress and find the results in the directory specified by `--workspace`:

- `checkpoints/`: Contains all saved model checkpoints.
- `logs/`: Contains detailed text log files that record loss values and other important training information.
- `tf_logs/`: Contains TensorBoard log files. You can visualize the training progress in real-time by running:

```bash
tensorboard --logdir /path/to/your/workspace/tf_logs
```

---

## 🔬 Extension 1: Gradient Conflict Analysis and Optimal Subdomain Count (`grad/`)

DP-LASS v1 fixes the subdomain count at 7 to align with AudioSet's top-level categories. In v2, we analyze gradient conflicts across 527 AudioSet sound classes and determine the optimal subdomain count via **gradient affinity matrix + Hierarchical Agglomerative Clustering (HAC)**.

### ✅ Key Idea

- For each of the 527 sound classes, extract and average the expected gradient vector over the ResUNet30 backbone of AudioSep.
- Compute the gradient affinity between class `i` and class `j` as the cosine similarity of their expected gradient vectors.
- Apply HAC with complete-linkage criterion and search for the minimal `K` where the worst-case intra-subdomain gradient affinity becomes non-negative.
- The theoretical zero-conflict point is `K = 71`; the final deployed model uses `K = 70`.

### ✅ Reproduce the Analysis

```bash
cd grad/

# Step 1: Extract class-wise gradients and compute the affinity matrix
python extract_gradients.py \
    --checkpoint_path /path/to/audiosep_base_4M_steps.ckpt \
    --audioset_root /path/to/AudioSet \
    --output_dir ./grad_outputs

# Step 2: HAC clustering and optimal-K search
python cluster_gradients.py \
    --affinity_matrix ./grad_outputs/affinity_matrix.npy \
    --output_dir ./grad_outputs

# Step 3: Visualize the gradient affinity matrix and K-performance curve
python visualize_conflicts.py \
    --grad_outputs ./grad_outputs
```

Expected outputs include:

- `grad_outputs/optimal_K.txt`: should report `71`
- `grad_outputs/affinity_matrix_reordered.png`: corresponds to Fig. 3(b) in the extended paper
- `grad_outputs/worst_affinity_vs_K.png`: corresponds to Fig. 4 in the extended paper

> ⚠️ `grad/` is an offline analysis module. It does not participate in the training loop. The resulting subdomain partition can be used as the basis for `--num_subdomains 70` in `train.py`.

---

## 🗂️ Extension 2: SED-Based Data Refinement (`SED/`)

DP-LASS v1 randomly crops 2-second segments from AudioSet, which may miss the target sound event. In v2, we use **ST-SED weakly supervised temporal localization** to locate the peak activation frame of the target class and crop a 2-second segment centered at this peak. This expands the training set from about 800K to **2.25M (SED-2.25M)** and also supports multi-class audio clips.

### ✅ Data Construction Pipeline

1. **Input**: all 10-second AudioSet recordings, including single-class and multi-class clips.
2. **ST-SED inference**: generate frame-level probability matrices and localize the peak activation frame of the target class.
3. **Cropping**: crop a 2-second segment centered at the peak frame.
4. **Output**: JSON format compatible with `train.py --data_json`.

### ✅ Build SED-2.25M and SED-32K

```bash
cd SED/

# Step 1: Prepare the ST-SED checkpoint
# Place the ST-SED checkpoint at SED/pretrained_models/stsed.ckpt

# Step 2: Localize target events and crop segments
python build_sed_dataset.py \
    --audioset_json /path/to/AudioSet/labeled_segments.json \
    --audioset_audio_root /path/to/AudioSet/audio \
    --stsed_checkpoint ./pretrained_models/stsed.ckpt \
    --output_json ./SED-2.25M.json \
    --output_audio_root ./SED-2.25M_audio \
    --segment_seconds 2.0 \
    --num_workers 16

# Step 3: Sample SED-32K for fine-tuning other LASS backbones
python sample_subset.py \
    --input_json ./SED-2.25M.json \
    --output_json ./SED-32K.json \
    --num_samples 32772 \
    --max_class_ratio 10
```

**Usage with training**:

```bash
python train.py \
    --data_json /path/to/SED/SED-2.25M.json \
    --num_subdomains 70 \
    ...
```

---

## 🔁 Extension 3: Generalization to Other LASS Backbones

The subdomain fine-tuning framework in v2 is adapted to four representative LASS models. Each model has a dedicated folder in the repository.

| Folder | Model Type | Fine-tuning Strategy | Extra Params | Reported Gain |
|---|---|---|---|---|
| `clapsep/` | Discriminative | Decoder-only Attention LoRA in Swin-Transformer decoder | +2.05 M | SDRi 7.55 → **8.48** |
| `flowsep/` | Generative (RFM) | Encoder-only ReConv-Adapter + skip 1×1 convolutions | +22.87 M | SAJ 2.24 → **2.68** |
| `soloaudio/` | Generative (Latent Diffusion) | LoRA (r=16, α=16) injected into DiT QKV / Attn-Out / MLP | +2.88 M | SAJ 2.81 → **3.41** |
| `samaudio/` | Generative (Flow Matching) | Text-only adaptation + LoRA (r=16, α=16) in DiT linear projections | +27.79 M | SAJ 3.19 → **3.60** |

### ✅ CLAPSep-SFT

CLAPSep-SFT reuses the subdomain classifier from DP-LASS v2 for routing.

```bash
cd clapsep/
python train_clapsep_sft.py \
    --sed_32k_json ../SED/SED-32K.json \
    --num_subdomains 7 \
    --classifier_ckpt ../pretrained_models/DP-LASS-v2/classifier.pth \
    --batch_size 32 \
    --lr 1e-4 \
    --epochs 100
```

### ✅ FlowSep-SFT

FlowSep-SFT uses encoder-only ReConv-Adapter injection and skips 1×1 convolutions.

```bash
cd flowsep/
python train_flowsep_sft.py \
    --sed_32k_json ../SED/SED-32K.json \
    --num_subdomains 7 \
    --adapter_type reconv_encoder_only \
    --skip_1x1_conv \
    --batch_size 64 \
    --lr 5e-5 \
    --epochs 100
```

### ✅ SoloAudio-SFT

SoloAudio-SFT injects LoRA modules into the DiT blocks and uses mixed-precision training.

```bash
cd soloaudio/
python train_soloaudio_sft.py \
    --sed_32k_json ../SED/SED-32K.json \
    --num_subdomains 7 \
    --lora_rank 16 \
    --lora_alpha 16 \
    --batch_size 256 \
    --fp16 \
    --epochs 100
```

### ✅ SAM Audio-SFT

SAM Audio-SFT keeps only the text prompt and injects LoRA into the DiT linear projections.

```bash
cd samaudio/
python train_samaudio_sft.py \
    --sed_32k_json ../SED/SED-32K.json \
    --text_only \
    --num_subdomains 7 \
    --lora_rank 16 \
    --lora_alpha 16 \
    --batch_size 16 \
    --lr 4e-4 \
    --epochs 100
```

> ⚠️ For generative models (FlowSep, SoloAudio, SAM Audio), inference does **not** use the subdomain classifier. Instead, the model traverses all fine-tuning modules and selects the output with the highest SAJ score. For discriminative models (AudioSep-SFT, CLAPSep-SFT), the subdomain classifier is used for routing.

---

## 📊 Evaluation

### ✅ Step 1: Prepare Data and Pre-trained Models

Before running the evaluation, you need to download all the necessary datasets and model weights.

#### 1.1 Evaluation Datasets

We provide all the datasets used for evaluation, including AudioSet, AudioCaps, Clotho, ESC50, VGGSound, and Music.

- **Download Link**: [Google Drive](https://drive.google.com/drive/folders/1PbCsuvdrzwAZZ_fwIzF0PeVGZkTk0-kL?usp=sharing)
- **Data Source**: All evaluation datasets are provided by the official AudioSep project. For more information, please refer to their repository: [https://github.com/Audio-AGI/AudioSep](https://github.com/Audio-AGI/AudioSep).
- **Action**: After downloading and extracting the files, we recommend placing them in a unified directory, such as `data/`.

#### 1.2 Pre-trained Models

The evaluation process requires two types of pre-trained models:

1. **Base Models**: The original AudioSep and CLAP models.
   - **Download Link**: [Hugging Face](https://huggingface.co/spaces/Audio-AGI/AudioSep/tree/main/checkpoint)

2. **Fine-tuned DP-LASS Models**:
   - DP-LASS v1 (ReConv-Adapter, K=7): [Google Drive](https://drive.google.com/drive/folders/1AdSQNIwIoV78RcHGQTTF7RlUQte21aZO?usp=sharing)
   - DP-LASS v2 (K=70, classifier, lightweight/full variants): please refer to the extended paper and repository release.

### ✅ Step 2: Recommended Directory Structure

To ensure the scripts run smoothly, we suggest organizing your downloaded files using the following structure:

```text
your_project_root/
├── data/
│   ├── AudioCaps_evaluation/
│   ├── AudioSet_evaluation/
│   ├── Clotho_evaluation/
│   └── ... (and other datasets)
│
├── pretrained_models/
│   ├── audiosep_base_4M_steps.ckpt       # Base model
│   ├── DP-LASS/                          # v1: 7 fine-tuned models
│   │   ├── cluster0_model.pt
│   │   ├── cluster1_model.pt
│   │   └── ...
│   └── DP-LASS-v2/                       # v2: 70 fine-tuned models + classifier
│       ├── cluster0_model.pt
│       ├── cluster1_model.pt
│       ├── ...
│       └── classifier.pth
│
├── grad/                                 # Gradient conflict analysis
│   ├── extract_gradients.py
│   ├── cluster_gradients.py
│   ├── visualize_conflicts.py
│   └── grad_outputs/
│
├── SED/                                  # SED-based data refinement
│   ├── build_sed_dataset.py
│   ├── sample_subset.py
│   ├── pretrained_models/
│   │   └── stsed.ckpt
│   └── SED-2.25M.json / SED-32K.json
│
├── clapsep/                              # CLAPSep-SFT
├── flowsep/                              # FlowSep-SFT
├── soloaudio/                            # SoloAudio-SFT
├── samaudio/                             # SAM Audio-SFT
│
└── evaluation/
    ├── metadata/
    ├── evaluator_audiocaps_sdri.py
    ├── evaluator_audioset_sdri.py
    └── ... (and other evaluation scripts)
```

### ✅ Step 3: Run the Evaluation Scripts

Within the `evaluation/` directory, we provide a separate evaluation script for each dataset. For DP-LASS v1, the scripts automatically load all 7 fine-tuned models. For DP-LASS v2, the scripts can load 70 fine-tuned models and optionally use the subdomain classifier or oracle subdomain labels.

#### 3.1 General Command Format for DP-LASS v1

```bash
python evaluation/<script_name>.py \
    --metadata_csv evaluation/metadata/<metadata_file>.csv \
    --audio_dir /path/to/your/data/<dataset_folder> \
    --base_checkpoint /path/to/your/pretrained_models/audiosep_base_4M_steps.ckpt \
    --dora_checkpoints \
        /path/to/your/DP-LASS/cluster0_model.pt \
        /path/to/your/DP-LASS/cluster1_model.pt \
        /path/to/your/DP-LASS/cluster2_model.pt \
        /path/to/your/DP-LASS/cluster3_model.pt \
        /path/to/your/DP-LASS/cluster4_model.pt \
        /path/to/your/DP-LASS/cluster5_model.pt \
        /path/to/your/DP-LASS/cluster6_model.pt \
    --config_yaml config/audiosep_base.yaml
```

#### 3.2 Command Parameter Explanation

- `--metadata_csv`: Path to the metadata file required for evaluation (located in `evaluation/metadata/`).
- `--audio_dir`: Path to the directory where you stored the audio files for the evaluation dataset.
- `--base_checkpoint`: Path to the original AudioSep **base model** checkpoint file.
- `--dora_checkpoints`: **A list of paths** to the fine-tuned DP-LASS models.
- `--config_yaml`: Path to the project's YAML configuration file.

#### 3.3 Example: Evaluating DP-LASS v1 on AudioCaps

Assuming you have organized your files according to the recommended directory structure, the command to run the evaluation on the AudioCaps dataset would be:

```bash
python evaluation/evaluator_audiocaps_sdri.py \
    --metadata_csv evaluation/metadata/audiocaps_eval.csv \
    --audio_dir data/AudioCaps_evaluation \
    --base_checkpoint pretrained_models/audiosep_base_4M_steps.ckpt \
    --dora_checkpoints \
        pretrained_models/DP-LASS/cluster0_model.pt \
        pretrained_models/DP-LASS/cluster1_model.pt \
        pretrained_models/DP-LASS/cluster2_model.pt \
        pretrained_models/DP-LASS/cluster3_model.pt \
        pretrained_models/DP-LASS/cluster4_model.pt \
        pretrained_models/DP-LASS/cluster5_model.pt \
        pretrained_models/DP-LASS/cluster6_model.pt \
    --config_yaml config/audiosep_base.yaml
```

To evaluate other datasets, simply change the script name, the `--metadata_csv` file, and the `--audio_dir` path accordingly.

#### 3.4 Example: Evaluating DP-LASS v2 on AudioCaps

```bash
python evaluation/evaluator_audiocaps_sdri_v2.py \
    --metadata_csv evaluation/metadata/audiocaps_eval.csv \
    --audio_dir data/AudioCaps_evaluation \
    --base_checkpoint pretrained_models/audiosep_base_4M_steps.ckpt \
    --dora_checkpoints \
        pretrained_models/DP-LASS-v2/cluster0_model.pt \
        pretrained_models/DP-LASS-v2/cluster1_model.pt \
        ... \
        pretrained_models/DP-LASS-v2/cluster69_model.pt \
    --classifier_ckpt pretrained_models/DP-LASS-v2/classifier.pth \
    --config_yaml config/audiosep_base.yaml
```

Oracle evaluation, where the ground-truth subdomain is known:

```bash
python evaluation/evaluator_audiocaps_sdri_v2.py \
    --oracle \
    --metadata_csv evaluation/metadata/audiocaps_eval.csv \
    --audio_dir data/AudioCaps_evaluation \
    --base_checkpoint pretrained_models/audiosep_base_4M_steps.ckpt \
    --dora_checkpoints \
        pretrained_models/DP-LASS-v2/cluster0_model.pt \
        pretrained_models/DP-LASS-v2/cluster1_model.pt \
        ... \
        pretrained_models/DP-LASS-v2/cluster69_model.pt \
    --config_yaml config/audiosep_base.yaml
```

### ✅ Reference Results

For DP-LASS v2 Full on six evaluation datasets, the extended paper reports:

- Average **SDRi: 10.67 dB**
- Average **SI-SDR: 10.17 dB**
- Music subset **SDRi: 13.51 dB**
- Oracle average **SDRi: 11.41 dB**

Compared with the vanilla AudioSep baseline, DP-LASS v2 achieves an average SDRi gain of **1.92 dB**. Compared with DP-LASS v1, it achieves an average SDRi gain of **0.91 dB**.

---

## 📁 Repository Structure Overview

```text
DP-LASS/
├── config/
│   └── audiosep_base.yaml
├── evaluation/
│   ├── metadata/
│   ├── evaluator_audiocaps_sdri.py
│   ├── evaluator_audioset_sdri.py
│   └── ...
├── grad/                     # v2: gradient conflict analysis
├── SED/                      # v2: SED-based data refinement
├── clapsep/                  # v2: CLAPSep-SFT
├── flowsep/                  # v2: FlowSep-SFT
├── soloaudio/                # v2: SoloAudio-SFT
├── samaudio/                 # v2: SAM Audio-SFT
├── train.py                  # main training script
├── environment.yml
└── README.md
```

---

## 📝 Notes

- **v1 vs. v2**: DP-LASS v1 uses 7 subdomains and 7 ReConv-Adapter modules. DP-LASS v2 uses 70 subdomains, SED-refined data, an improved multimodal classifier with soft-label optimization, a lightweight fine-tuning module, and a hybrid SI-SDRi + L1 loss.
- **Generative vs. discriminative inference**: AudioSep-SFT and CLAPSep-SFT use the subdomain classifier for routing. FlowSep-SFT, SoloAudio-SFT, and SAM Audio-SFT traverse all fine-tuning modules and select the best output by SAJ score.
- **Lightweight variant**: DP-LASS v2 Light applies the ReConv-Adapter only to the first encoder layer, reducing the fine-tuning module size to about **18.4K** parameters while maintaining competitive performance.

---

## 📖 Citation

If you find this repository useful, please cite the corresponding papers.

```bibtex
@inproceedings{zhang2026domain,
  title={Domain Partitioning Meets Parameter-Efficient Fine-Tuning: A Novel Method for Improved Language-Queried Audio Source Separation},
  author={Zhang, Yinkai and Zhang, Dingbang and Wang, Tao and Rakhimova, Diana and Wang, Kai and Huang, Hao},
  booktitle={ICASSP 2026-2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={15637--15641},
  year={2026},
  organization={IEEE}
}

```

---

## 🔗 Acknowledgements

This repository builds upon the official [AudioSep](https://github.com/Audio-AGI/AudioSep) project. We thank the authors for releasing their code, pre-trained models, and evaluation datasets.
```
