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

## Pretrained model

 The pretrained Reconv-Adapter models are  available at this URL:https://drive.google.com/drive/folders/1AdSQNIwIoV78RcHGQTTF7RlUQte21aZO?usp=sharing

The pretrained AudioSep and CLAP models are available at https://huggingface.co/spaces/Audio-AGI/AudioSep/tree/main/checkpoint

##  Evaluation
Download the evaluation data at (https://drive.google.com/drive/folders/1PbCsuvdrzwAZZ_fwIzF0PeVGZkTk0-kL?usp=sharing) 

The pretrained AudioSep and CLAP models, along with the evaluation datasets (AudioSet, AudioCaps, Clotho, ESC50, VGGSound, and Music), are all provided by the official AudioSep project repository: https://github.com/Audio-AGI/AudioSep 

## Training 
The AudioSet dataset can be accessed via the following link:
https://research.google.com/audioset/
