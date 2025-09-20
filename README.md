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

This section provides a comprehensive guide to training the DP-LASS model on your own datasets.

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



## Pretrained model

The pretrained Reconv-Adapter models are  available at this URL:https://drive.google.com/drive/folders/1AdSQNIwIoV78RcHGQTTF7RlUQte21aZO?usp=sharing

The pretrained AudioSep and CLAP models are available at https://huggingface.co/spaces/Audio-AGI/AudioSep/tree/main/checkpoint

##  Evaluation
Download the evaluation data at (https://drive.google.com/drive/folders/1PbCsuvdrzwAZZ_fwIzF0PeVGZkTk0-kL?usp=sharing) 

The pretrained AudioSep and CLAP models, along with the evaluation datasets (AudioSet, AudioCaps, Clotho, ESC50, VGGSound, and Music), are all provided by the official AudioSep project repository: https://github.com/Audio-AGI/AudioSep 

## Training 
The AudioSet dataset can be accessed via the following link:
https://research.google.com/audioset/
