# HLIP
> Official PyTorch implementation of the following paper:\
> Towards Scalable Language-Image Pre-training for 3D Medical Imaging\
> University of Michigan

## Overview
<p align="center"><img src="https://github.com/Zch0414/hlip/blob/master/docs/github.png" width=96% height=96% class="center"></p>

We propose **H**ierarchical attention for **L**anguage-**I**mage **P**re-training (**HLIP**), inspired by the natural hierarchy of radiology data: slice, scan, and study. With this lightweight attention mechanism, HLIP can be trained directly on uncurated clinical datasets, enabling scalable language-image pre-training in 3D medical imaging. For real-world clinical use, HLIP can be applied to studies containing either a single scan (e.g., chest CT) or multiple scans (e.g., brain MRI).

## Updates
- **(Todo)** Release training code.
- **(Todo)** Release evaluation code.
- **(2025-06)** Release data process pipeline.
- **(2025-05)** Release HLIP models trained on chest CT and brain MRI, feel free to try our demos.

## Getting Started

### Install 
[open-clip](https://github.com/mlfoundations/open_clip/tree/main)
```bash
python3 -m venv env
source env/bin/activate
pip install -U pip
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
git clone git@github.com:mlfoundations/open_clip.git
cd open_clip
make install
make install-training
```

### Demo
Chest CT
```bash
python inference_rad_chestct.py \
  --model vit_base_singlescan_h2_token1176 \
  --resume /path/to/vit_base_singlescan_h2_token1176.pt \
  --data /docs/tst32751/tst32751.pt \
```
Brain MRI
```bash
python inference_pub_brain_5.py \
  --model vit_base_multiscan_h2_token1176 \
  --resume /path/to/vit_base_multiscan_h2_token1176.pt \
  --patch-size 8 16 16 \
  --num-slices 72 \
  --data /docs/BraTS-GLI-00459-000/ \
```
Visualizing the activation with:
```bash
--interpret
```
