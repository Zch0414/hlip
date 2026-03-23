# HLIP
> Official PyTorch implementation of the following paper:\
> Towards Scalable Language-Image Pre-training for 3D Medical Imaging\
> University of Michigan\
> [![TMLR](https://img.shields.io/badge/TMLR-accepted-success.svg)](https://openreview.net/forum?id=WxHf4EcBWA)&nbsp; [![arXiv](https://img.shields.io/badge/arXiv%20paper-2505.21862-b31b1b.svg)](https://arxiv.org/abs/2505.21862)&nbsp; [![huggingface weights](https://img.shields.io/badge/%F0%9F%A4%97%20Weights-yellow)](https://huggingface.co/collections/zch0414/hlip)&nbsp;



## Overview
<p align="center">
  <img
    src="docs/github.png"
    alt="HLIP overview"
    style="width:96%; max-width:800px; height:auto;"
  />
</p>

Directly leveraging uncurated clinical studies enables scalable language-image pre-training in 3D medical imaging, as the scale is no longer constrained by the manual effort required from clinicians to select a single representative scan or slice from each study. This paradigm could be more effective when equipped with a hierarchical attention mechanism inspired by the natural structure of the data: slice, scan, and study. We name this framework **H**ierarchical attention for **L**anguage-**I**mage **P**re-training (**HLIP**). For real-world clinical use, HLIP can be applied to studies containing either a single scan (e.g., chest CT) or multiple scans (e.g., brain MRI).

## Updates
- **(2026-03)** Check out our new [paper](https://arxiv.org/abs/2512.11141), accepted at CVPR 2026, which introduces a new strategy, beyond the dual-loss approach presented in the HLIP blog, for handling itemized text supervision in language-image pre-training. The code and model weights are available [here](https://github.com/MLNeurosurg/ItemizedCLIP).
- **(2026-02)** Assets in **2025-11 (departured)** have been finalized and updated. We apologize for any inconvenience to researchers actively using this repository. This should be our last incremental update to HLIP. We have released four HLIP variants in the Hugging Face collection: [![huggingface weights](https://img.shields.io/badge/%F0%9F%A4%97%20Weights-yellow)](https://huggingface.co/collections/zch0414/hlip). The model released in **2025-11** is also included in this collection and is listed as [hlip-2025_10_08](https://huggingface.co/zch0414/hlip-2025_10_08). Technical details are provided in this [blog](https://zch0414.github.io/hlip-ablation/), and the implementation is based on this [code branch](https://github.com/Zch0414/hlip/tree/hlip-ablation). 
- **(2026-02)** HLIP is accepted by TMLR!
- ~~**(2025-11)** We release our updated model [![huggingface weights](https://img.shields.io/badge/%F0%9F%A4%97%20Weights-yellow)](https://huggingface.co/Zch0414/hlip-2025-10-08), along with a new [code branch](https://github.com/Zch0414/hlip/tree/hlip-ablation) focused on uncurated 3D medical datasets. The technical details are described in this [blog](https://zch0414.github.io/hlip-ablation/).~~
- **(2025-06)** Complete the initiation of HLIP repository.
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

### Models
| Data | Objective | Patch Size | Attention | Model |
| :--------: | :--------: | :--------: | :--------: | :--------: |
| CT-RATE (20K) | CLIP | <code>8, 24, 24</code> | <code>slice</code> + <code>scan</code> | [ViT Base](https://drive.google.com/file/d/1muu7L9H3KaL3nq3fNtN8kKF1eDK3R5Z4/view?usp=drive_link) |
| BrainMRI (220K) | CLIP |  <code>8, 16, 16</code> |<code>scan</code> + <code>study</code> | [ViT Base](https://drive.google.com/file/d/12BwJvd6IEZynXM8jkled0ND7t11iuySj/view?usp=drive_link) |
| HeadCT (240K) | CLIP | <code>8, 16, 16</code> | <code>scan</code> + <code>study</code> | [ViT Base](https://drive.google.com/file/d/1rfoz-kzF0iwaMQ-4MuR7F4NlTjtPIZa7/view?usp=drive_link) |

Future models will be released on HuggingFace and announced in the **Updates** section.

### Demo
Chest CT: an example from the external Rad-ChestCT dataset.
```bash
python inference_radchestct.py \
  --model clip_vit_base_slice_scan_token2744 \
  --use-cxr-bert \
  --resume /path/to/clip_vit_base_slice_scan_token2744.pt \
  --data ../../docs/tst32751/tst32751.pt
```

Brain MRI: an example from the external BraTS23 dataset.
```bash
python inference_pubbrain5.py \
  --model clip_vit_base_scan_study_token1176 \
  --resume /path/to/clip_vit_base_scan_study_token1176.pt \
  --patch-size 8 16 16 \
  --num-slices 48 \
  --data ../../docs/BraTS-GLI-00459-000
```
Visualizing the activation with <code>--interpret</code>. Increasing <code>--num-slcies</code> for better visualization quality.

### Evaluation
CT-RATE
```bash
torchrun --rdzv_endpoint=localhost:29500 --nproc_per_node 4 zeroshot_ctrate.py \
  --model clip_vit_base_slice_scan_token2744 \
  --resume /path/to/clip_vit_base_slice_scan_token2744.pt \
  --data-root /data/ct_rate/valid/ \
  --input-file ../../data/ct_rate/metafiles/valid_labels.csv \
  --zeroshot-template volume
```

Rad-ChestCT
```bash
torchrun --rdzv_endpoint=localhost:29500 --nproc_per_node 4 zeroshot_radchestct.py \
  --model clip_vit_base_slice_scan_token2744 \
  --resume /path/to/clip_vit_base_slice_scan_token2744.pt \
  --data-root /data/rad_chestct/ \
  --input-file ../../data/rad_chestct/files/rad_chestct_labels.csv \
  --zeroshot-template volume
```

Pub-Brain-5
```bash
torchrun --rdzv_endpoint=localhost:29500 --nproc_per_node 8 zeroshot_pubbrain5.py \
  --model clip_vit_base_scan_study_token1176 \
  --resume /path/to/vit_base_scan_study_token1176.pt \
  --data-root /data/pub_brain_5/ \
  --input-file ../../data/pub_brain_5/pub_brain_5.csv
```

RSNA
```bash
torchrun --rdzv_endpoint=localhost:29500 --nproc_per_node 8 zeroshot_rsna.py \
  --model clip_vit_base_scan_study_token1176 \
  --resume /path/to/vit_base_scan_study_token1176.pt \
  --data-root /data/rsna/ \
  --input-file ../../data/rsna/rsna.csv
```

### Training

Below, we provide a training script to reproduce our results on CT-RATE using the original reports as supervision. Training for 20 epochs takes approximately 6 hours on a single node with 4 A40 GPUs.
```bash
torchrun --rdzv_endpoint=localhost:29500 --nproc_per_node 4 main.py \
  --benchmark-type ct-rate \
  --logs-dir /path/to/logs/ \
  --zeroshot-frequency 1 \
  --zeroshot-template volume \
  --save-frequency 1 \
  --train-data /path/to/ct_rate/train/ \
  --train-file ../../data/ct_rate/files/raw_annotation.json \
  --image-process-cfg -1150 350 crop \
  --text-process-cfg report \
  --ct-rate data_root='"/path/to/ct_rate/valid/"' input_file='"../../data/ct_rate/metafiles/valid_labels.csv"' \
  --rad-chestct data_root='"/path/to/rad_chestct/"' input_file='"../../data/rad_chestct/files/rad_chestct_labels.csv"' \
  --report-to wandb \
  --wandb-project-name hlip \
  --warmup 47 \
  --batch-size 32 \
  --accum-batch 4 \
  --lr=8e-5 \
  --wd=0.2 \
  --force-patch-dropout 0.0 \
  --epochs=20 \
  --precision amp \
  --workers 4 \
  --local-loss \
  --gather-with-grad \
  --grad-checkpointing \
  --model clip_vit_base_slice_scan_token2744 \
  --use-cxr-bert \
  --lock-text \
  --dist-url "env://localhost:29500"
```

Our training implementation is closely aligned with [open-clip](https://github.com/mlfoundations/open_clip/tree/main), allowing us to leverage features such as <code>patch dropout</code> and <code>siglip</code>.

For <code>patch dropout</code>, try the following commands:
```bash
  --force-patch-dropout 0.5 \
  --beta2 0.95
```

For <code>siglip</code>, you can try it using the following commands, but make sure to modify the model configuration beforehand:
```bash
  --beta2 0.95 \
  --siglip
```

## Citation
If you find this repository helpful, please consider citing:
```bib
@article{zhao2026towards,
  title={Towards Scalable Language-Image Pre-training for 3D Medical Imaging},
  author={Chenhui Zhao and Yiwei Lyu and Asadur Zaman Chowdury and Edward S Harake and Akhil Kondepudi and Akshay T Rao and Xinhai Hou and Honglak Lee and Todd C Hollon},
  journal={Transactions on Machine Learning Research},
  issn={2835-8856},
  year={2026},
  url={https://openreview.net/forum?id=WxHf4EcBWA}
}
```
