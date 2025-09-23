# DP-LASS



Official implementation of Domain Partitioning Meets Parameter-Efficient Fine-Tuning: A Novel Method for Improved Language-Queried Audio Source Separation


## 🚀 Environment Setup Guide

### ✅ Step 1: Create the Conda Environment

Place the `environment.yml` file in your project root directory and run:

```bash
conda env create -f environment.yml
```


---

### ✅ Step 2: Activate the Environment

```bash
conda activate DP-LASS
```

---

### ✅ Step 3: Verify Installation

Check that PyTorch and CUDA are correctly installed:

```bash
python -c "import sys, torch; print(f'Python: {sys.version.split()[0]} | PyTorch: {torch.__version__} | CUDA: {torch.cuda.is_available()}')"
```

You should see output similar to:

```
Python: 3.10.16 | PyTorch: 1.13.1 | CUDA: True
```

Of course. Here is the "Model Training" section for your README, written in English and incorporating the information you provided.

---

## 💻 Model Training

This section provides a comprehensive guide to training the DP-LASS model on AudioSet or your own datasets.

### ✅ Step 1: Prepare Data(JSON Format)

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

---

### ✅ Step 1.5: Downloading AudioSet 

To train on AudioSet:

1.  Visit the official download page: [https://research.google.com/audioset/download.html ](https://research.google.com/audioset/download.html )
2.  Download the segment CSV files (`balanced_train_segments.csv`, etc.) and ` class_labels_indices.csv`.
3.  Use a tool like `yt-dlp` to download audio clips from YouTube using the provided video IDs and timestamps.
4.  Convert label IDs (e.g., `/m/07rwj`) to human-readable captions using the ` class_labels_indices.csv`, then format your data into the required JSON structure (Step 1).

> ⚠️ Note: AudioSet does not provide direct audio downloads — you must retrieve clips from YouTube, and availability is not guaranteed.

---


### ✅ Step 2: Prepare the Configuration File

All hyperparameters and path settings for the training process are managed through a YAML configuration file.

1.  **Locate the Config File**: An example configuration file is provided in the repository, such as `config/audiosep_base.yaml`.
2.  **Customize Your Configuration**: Create a copy of the example file (e.g., `my_training_config.yaml`) and modify the key parameters to match your setup:
    *   **`data`**: Data-related settings, including `sampling_rate` and `segment_seconds`.
    *   **`model`**: Model-specific configurations like `model_type`.
    *   **`train`**: Training parameters, including `batch_size_per_device`, `num_workers` for the data loader, and the optimizer's `learning_rate`.

### ✅ Step 3: Prepare Pre-trained Base Models

Our training methodology fine-tunes a pre-trained AudioSep model. You must download the necessary base models before starting.

1.  **Download Models**: The pretrained AudioSep and CLAP models are available at the official AudioSep Hugging Face repository.
    *   **Download URL**: [Hugging Face](https://huggingface.co/spaces/Audio-AGI/AudioSep/tree/main/checkpoint)
2.  **Model Origin**: These models are provided by the official AudioSep project. For more details, you can visit their GitHub repository: [https://github.com/Audio-AGI/AudioSep](https://github.com/Audio-AGI/AudioSep).
3.  **Place Models**: After downloading the checkpoint file (e.g., `audiosep_base_4M_steps.ckpt`), place it in a convenient location, such as a `pretrained_models/` directory in your project root. You will need to provide the path to this file when running the training script.

### ✅ Step 4: Run the Training Script

Once your data and configuration are ready, you can start the training process by executing the `train.py` script from your terminal.

Below is a complete example of the training command:

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

#### Parameter Explanation:

*   `--workspace`: **(Required)** The directory where all training artifacts (model checkpoints, logs, etc.) will be saved.
*   `--config_yaml`: **(Required)** The path to the YAML configuration file you prepared in Step 2.
*   `--checkpoint_path`: **(Required)** The path to the pre-trained AudioSep model checkpoint you downloaded in Step 3.
*   `--max_epochs`: The total number of epochs to train for. Defaults to `150`.
*   `--num_gpus`: The number of GPUs to use for training. Defaults to auto-detection if not specified.
*   `--strategy`: The distributed training strategy. `ddp` is recommended for multi-GPU training.
*   `--resume`: An optional flag. If included, the script will automatically search for the latest checkpoint (`last.ckpt`) in the `--workspace` directory and resume training from that point.

### ✅ Step 5: Monitor Training and Find Results

After launching the script, you can monitor its progress and find the results in the directory specified by `--workspace`:

*   `checkpoints/`: Contains all saved model checkpoints.
*   `logs/`: Contains detailed text log files that record loss values and other important training information.
*   `tf_logs/`: Contains TensorBoard log files. You can visualize the training progress in real-time by running:
    ```bash
    tensorboard --logdir /path/to/your/workspace/tf_logs
    ```


## 📊 Evaluation

### ✅ Step 1: Prepare Data and Pre-trained Models

Before running the evaluation, you need to download all the necessary datasets and model weights.

#### 1.1 Evaluation Datasets

We provide all the datasets used for evaluation, including AudioSet, AudioCaps, Clotho, ESC50, VGGSound, and Music.

*   **Download Link**: [Google Drive](https://drive.google.com/drive/folders/1PbCsuvdrzwAZZ_fwIzF0PeVGZkTk0-kL?usp=sharing)
*   **Data Source**: All evaluation datasets are provided by the official AudioSep project. For more information, please refer to their repository: [https://github.com/Audio-AGI/AudioSep](https://github.com/Audio-AGI/AudioSep).
*   **Action**: After downloading and extracting the files, we recommend placing them in a unified directory, such as `data/`.

#### 1.2 Pre-trained Models

The evaluation process requires two types of pre-trained models:

1.  **Base Models**: The original AudioSep and CLAP models.
    *   **Download Link**: [Hugging Face](https://huggingface.co/spaces/Audio-AGI/AudioSep/tree/main/checkpoint)

2.  **Fine-tuned DP-LASS (Reconv-Adapter) Models**: These are the fine-tuned models proposed in our paper.
    *   **Download Link**: [Google Drive](https://drive.google.com/drive/folders/1AdSQNIwIoV78RcHGQTTF7RlUQte21aZO?usp=sharing)

### ✅ Step 2: Recommended Directory Structure

To ensure the scripts run smoothly, we suggest organizing your downloaded files using the following structure:

```
your_project_root/
├── data/
│   ├── AudioCaps_evaluation/
│   ├── AudioSet_evaluation/
│   ├── Clotho_evaluation/
│   └── ... (and other datasets)
│
├── pretrained_models/
│   ├── audiosep_base_4M_steps.ckpt  # Base model
│   └── DP-LASS/                     # Contains all 7 fine-tuned models
│       ├── cluster0_model.pt
│       ├── cluster1_model.pt
│       └── ... 
│
└── evaluation/
    ├── metadata/
    ├── evaluator_audiocaps_sdri.py
    ├── evaluator_audioset_sdri.py
    └── ... (and other evaluation scripts)
```

### ✅ Step 3: Run the Evaluation Scripts

Within the `evaluation/` directory, we provide a separate evaluation script for each dataset. These scripts are designed to automatically load all 7 fine-tuned DP-LASS models. For each audio sample, the script intelligently selects the best-performing model for separation and then calculates the average performance metrics across the entire dataset.

#### 3.1 General Command Format

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

*   `--metadata_csv`: Path to the metadata file required for evaluation (located in `evaluation/metadata/`).
*   `--audio_dir`: Path to the directory where you stored the audio files for the evaluation dataset.
*   `--base_checkpoint`: Path to the original AudioSep **base model** checkpoint file.
*   `--dora_checkpoints`: **A list of paths** to **all seven** fine-tuned DP-LASS (Reconv-Adapter) models.
*   `--config_yaml`: Path to the project's YAML configuration file.

#### 3.3 Example: Evaluating on the AudioCaps Dataset

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

