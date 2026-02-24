# HLIP Ablation

## Training Scripts

**Pre-training**
```bash
torchrun --rdzv_endpoint=localhost:29500 --nproc_per_node 8 main.py \
--logs-dir /path/to/logs/ \
--train-data-filelist /path/to/data/mri/train/ /path/to/data/ct/train/ \
--valid-data-filelist /path/to/data/mri/valid/ /path/to/data/ct/valid/ \
--train-scan-filedict /path/to/data/mri/train/='["/path/to/MR/scans/train.json"]' /path/to/data/ct/train/='["/path/to/CT/scans/train.json"]'  \
--valid-scan-filedict /path/to/data/mri/valid/='["/path/to/MR/scans/valid.json"]' /path/to/data/ct/valid/='["/path/to/CT/scans/valid.json"]' \
--train-report-filedict /path/to/data/mri/train/='["/path/to/MR/reports/train.json"]' /path/to/data/ct/train/='["/path/to/CT/reports/train.json"]' \
--valid-report-filedict /path/to/data/mri/valid/='["/path/to/MR/reports/valid.json"]' /path/to/data/ct/valid/='["/path/to/CT/reports/valid.json"]' \
--zeroshot-frequency 1 \
--save-frequency 1 \
--report-to wandb \
--wandb-project-name hlip-ablation \
--train-data train \
--valid-data valid \
--ct data_root='"/path/to/data/ct/test/"' input_file='"../../data/ct.csv"' \
--mri data_root='"/path/to/data/mri/test/"' input_file='"../../data/mri.csv"' \
--num-scans 8 \
--warmup 2400 \
--batch-size 8 \
--accum-batch 6 \
--accum-freq 2 \
--lr 5e-4 \
--wd 0.2 \
--force-patch-dropout 0.50 \
--beta2 0.95 \
--epochs 25 \
--precision amp \
--workers 8 \
--grad-checkpointing \
--model ablate_seqposemb_clip_vit_base_multiscan_h2_dualdinotxt1568 \
--dist-url "env://localhost:29500"
```


**Unmasked Fine-tuning**
```bash
torchrun --rdzv_endpoint=localhost:29500 --nproc_per_node 8 main.py \
--logs-dir /path/to/logs/ \
--train-data-filelist /path/to/data/mri/train/ /path/to/data/ct/train/ \
--valid-data-filelist /path/to/data/mri/valid/ /path/to/data/ct/valid/ \
--train-scan-filedict /path/to/data/mri/train/='["/path/to/MR/scans/train.json"]' /path/to/data/ct/train/='["/path/to/CT/scans/train.json"]'  \
--valid-scan-filedict /path/to/data/mri/valid/='["/path/to/MR/scans/valid.json"]' /path/to/data/ct/valid/='["/path/to/CT/scans/valid.json"]' \
--train-report-filedict /path/to/data/mri/train/='["/path/to/MR/reports/gpt3.5/train.json"]' /path/to/data/ct/train/='["/path/to/CT/reports/gpt3.5/train.json"]' \
--valid-report-filedict /path/to/data/mri/valid/='["/path/to/MR/reports/gpt3.5/valid.json"]' /path/to/data/ct/valid/='["/path/to/CT/reports/gpt3.5/valid.json"]' \
--zeroshot-frequency 1 \
--save-frequency 1 \
--report-to wandb \
--wandb-project-name hlip-ablation \
--train-data train \
--valid-data valid \
--ct data_root='"/path/to/data/ct/test/"' input_file='"../../data/ct.csv"' \
--mri data_root='"/path/to/data/mri/test/"' input_file='"../../data/mri.csv"' \
--num-scans 8 \
--warmup 400 \
--batch-size 4 \
--accum-batch 4 \
--accum-freq 6 \
--lr 5e-5 \
--wd 0.2 \
--force-patch-dropout 0.0 \
--beta2 0.95 \
--epochs 5 \
--precision amp \
--workers 8 \
--grad-checkpointing \
--seed 42 \
--finetune "/path/to/logs/model/checkpoints/epoch_25.pt" \
--model ablate_rope_clip_vit_base_multiscan_h2_dualdinotxt1568 \
--dist-url "env://localhost:29500"
```

## Evaluation Scripts

**Pub-Brain-5**
```bash
torchrun --rdzv_endpoint=localhost:29500 --nproc_per_node 8 zeroshot_pubbrain5.py \
--model ablate_seqposemb_clip_vit_base_multiscan_h2_dualdinotxt1568 \
--resume /path/to/logs/model/checkpoints/epoch.pt \
--data-root /path/to/data/pub_brain_5/ \
--workers 8
```

Use <code>--huggingface zch0414/clip-vit_base-scan_study-dualdinotxt1568</code> to try the newest model directly.


**RSNA**
```bash
torchrun --rdzv_endpoint=localhost:29500 --nproc_per_node 8 zeroshot_rsna.py \
--model ablate_seqposemb_clip_vit_base_multiscan_h2_dualdinotxt1568 \
--resume /path/to/logs/model/checkpoints/epoch.pt \
--data-root /path/to/data/RSNA \
--workers 8
```

Use <code>--huggingface zch0414/clip-vit_base-scan_study-dualdinotxt1568</code> to try the newest model directly.