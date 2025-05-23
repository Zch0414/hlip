# HLIP
> Official PyTorch implementation of the following paper:\
> Towards Scalable Language-Image Pre-training for 3D Medical Imaging\
> University of Michigan

## Overview
<p align="center"><img src="https://github.com/Zch0414/hlip/blob/master/docs/github.png" width=96% height=96% class="center"></p>

We propose **H**ierarchical attention for **L**anguage-**I**mage **P**re-training (**HLIP**), inspired by the natural hierarchy of radiology data: slice, scan, and study. With this lightweight attention mechanism, HLIP can be trained directly on uncurated clinical datasets, enabling scalable language-image pre-training in 3D medical imaging. For real-world clinical use, HLIP can be applied to studies containing either a single scan (e.g., chest CT) or multiple scans (e.g., brain MRI).

## Updates
- **(2025-05)** Release test demo and HLIP models trained on chest CT and brain MRI.

## Getting Started

### Install open-clip
[open-clip]([https://github.com/facebookresearch/deit/tree/main](https://github.com/mlfoundations/open_clip/tree/main))
```
- python3 -m venv env
- source .env/bin/activate
- pip install -U pip
- git clone 
```
