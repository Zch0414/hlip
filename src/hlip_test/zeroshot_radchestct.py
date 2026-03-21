import os
import sys
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('..'))

import math
import json
import random
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix

from open_clip import create_model_and_transforms, get_tokenizer, get_input_dtype, build_zero_shot_classifier
from open_clip.factory import _MODEL_CONFIGS
from open_clip_train.file_utils import pt_load
from open_clip_train.precision import get_autocast
from open_clip_train.distributed import is_master, init_distributed_device, all_gather_object

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Normalize

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from hlip import visual_encoder
from hlip.zeroshot_metadata_radchestct import CLASSNAMES, ORGANS, TEMPLATES, PROMPTS


def get_args_parser():
    parser = argparse.ArgumentParser('Perform Zero-shot', add_help=False)
    parser.add_argument('--model', default='clip_vit_base_slice_scan_token2744', type=str)
    parser.add_argument('--use-cxr-bert', default=False, action='store_true')
    parser.add_argument('--lora-text', default=False, action='store_true')
    parser.add_argument('--lock-text-freeze-layer-norm', default=False, action='store_true')
    parser.add_argument('--resume', default='/pretrained/chestct_clip_vit_base_slice_scan_token2744.pt', type=str)

    parser.add_argument('--data-root', default='/data/rad_chestct/')
    parser.add_argument('--input-file', '--zeroshot-rad-chestct', dest='input_file', default='../../data/rad_chestct/files/rad_chestct_labels.csv', type=str)
    parser.add_argument('--process-cfg', '--input-info', dest='process_cfg', nargs='+', default=['-1150', '350', 'crop'])
    parser.add_argument('--zeroshot-template', default='volume', type=str)
    parser.add_argument('--workers', default=4, type=int)
    parser.add_argument('--save', default='', type=str)

    # hack argument
    parser.add_argument('--horovod', default=False, action='store_true')
    return parser


# random
def random_seed(seed=0, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


# data
class RadChestCTDataset(Dataset):
    def __init__(
        self,
        root,
        input_file,
        process_cfg,
    ):
        self.cts = []
        df = pd.read_csv(input_file)
        for _, row in df.iterrows():
            recon = row['NoteAcc_DEID']
            self.cts.append((os.path.join(root, recon + '.pt'), row[CLASSNAMES].astype(int).tolist()))

        self.process_cfg = (float(process_cfg[0]), float(process_cfg[1]), str(process_cfg[2]))
        self.normalizer = Normalize(torch.as_tensor(IMAGENET_DEFAULT_MEAN).mean(), torch.as_tensor(IMAGENET_DEFAULT_STD).mean())

    def __len__(self):
        return len(self.cts)

    def __getitem__(self, idx):
        recon, target = self.cts[idx]

        img = torch.load(recon, weights_only=True)
        img = (img.float() - self.process_cfg[0]) / (self.process_cfg[1] - self.process_cfg[0])
        img = torch.clip(img, 0., 1.)
        img = img[None, ...]

        if self.process_cfg[2] == 'crop':
            # padding
            _, d, h, w = img.shape
            pad_d = max(112 - d, 0)
            pad_h = max(336 - h, 0)
            pad_w = max(336 - w, 0)
            pad_d1, pad_d2 = pad_d // 2, pad_d - pad_d // 2
            pad_h1, pad_h2 = pad_h // 2, pad_h - pad_h // 2
            pad_w1, pad_w2 = pad_w // 2, pad_w - pad_w // 2
            img = torch.nn.functional.pad(
                img[None, ...], (pad_w1, pad_w2, pad_h1, pad_h2, pad_d1, pad_d2),
                mode='constant',
                value=0,
            ).squeeze(0)

            # cropping
            _, d, h, w = img.shape
            start_d = (d - 112) // 2
            start_h = (h - 336) // 2
            start_w = (w - 336) // 2
            img = img[
                :,
                start_d:start_d + 112,
                start_h:start_h + 336,
                start_w:start_w + 336,
            ]

        elif self.process_cfg[2] == "resize":
            # padding to the longest side. 
            _, _, h, w = img.shape               
            size = max(h, w)
            pad_h = size - h; pad_w = size - w
            left = pad_w // 2; right = pad_w - left; top = pad_h // 2; bottom = pad_h - top
            img = torch.nn.functional.pad(img, (left, right, top, bottom), mode="constant", value=0)

            # resize to 384, crop to 336
            img = torch.nn.functional.interpolate(img, size=(384, 384), mode='bilinear')
            img = torch.nn.functional.interpolate(img[None, ...], size=(112, 384, 384), mode='nearest-exact')[0]
            img = img[:, :, 24:360, 24:360]

        else:
            raise NotImplementedError

        # normalize
        img = self.normalizer(img)

        return {'image': img[None, ...], 'target': torch.as_tensor(target, dtype=torch.long)}


def get_data(data_root, input_file, process_cfg, workers, distributed):
    dataset = RadChestCTDataset(data_root, input_file, process_cfg)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if distributed else None
    dataloader = DataLoader(
        dataset,
        batch_size=1, # only support 1 during evaluation; the speed bottleneck is data loading
        shuffle=False,
        sampler=sampler,
        num_workers=workers,
        pin_memory=True,
        drop_last=False,
    )
    return dataloader


# metric
def find_threshold(y_true, y_score):
    """
    Copy from https://github.com/alibaba-damo-academy/fvlm/blob/d768ec1546fb825fcc9ea9b3e7b2754a69f870c1/calc_metrics.py#L8C1-L8C32
    Finds the optimal threshold for binary classification based on ROC curve.

    Args:
        y_true (numpy.ndarray): True labels.
        y_score (numpy.ndarray): Predicted probabilities.

    Returns:
        float: Optimal threshold.
    """

    best_threshold = 0
    best_roc = 10000

    thresholds = np.linspace(0, 1, 100)
    for threshold in thresholds:
        y_pred = (y_score > threshold).astype(int)
        confusion = confusion_matrix(y_true, y_pred)
        TP = confusion[1, 1]
        TN = confusion[0, 0]
        FP = confusion[0, 1]
        FN = confusion[1, 0]
        TP_r = TP / (TP + FN)
        FP_r = FP / (FP + TN)
        curr_roc = math.sqrt(((1 - TP_r) ** 2) + (FP_r ** 2))
        if curr_roc <= best_roc:
            best_roc = curr_roc
            best_threshold = threshold

    return best_threshold


def compute_radchestct_metrics(ground_truth, prediction):
    assert prediction.shape == ground_truth.shape and prediction.shape[1] == len(CLASSNAMES), (
        f'Expected [N, {len(CLASSNAMES)}] inputs.'
    )

    ground_truth = ground_truth.cpu()
    prediction = prediction.cpu()

    calcification_sources = {
        'Coronary artery wall calcification',
        'Arterial wall calcification',
    }
    coronary_idx = CLASSNAMES.index('Coronary artery wall calcification')
    arterial_idx = CLASSNAMES.index('Arterial wall calcification')
    eval_classnames = [key for key in CLASSNAMES if key not in calcification_sources] + ['Calcification']

    results = {}
    aucs, accs, f1ws, precisions, recalls = [], [], [], [], []

    for key in eval_classnames:
        if key == 'Calcification':
            y_true = ground_truth[:, coronary_idx].to(torch.int64).numpy()
            coronary_score = prediction[:, coronary_idx].to(torch.float32).numpy()
            arterial_score = prediction[:, arterial_idx].to(torch.float32).numpy()
            y_score = np.where(
                (coronary_score > 0.5) | (arterial_score > 0.5),
                np.maximum(coronary_score, arterial_score),
                np.minimum(coronary_score, arterial_score),
            )
        else:
            idx = CLASSNAMES.index(key)
            y_true = ground_truth[:, idx].to(torch.int64).numpy()
            y_score = prediction[:, idx].to(torch.float32).numpy()

        threshold = find_threshold(y_true, y_score)
        y_pred = (y_score > threshold).astype(int)

        acc = accuracy_score(y_true, y_pred)
        f1w = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        try:
            auc = roc_auc_score(y_true, y_score)
        except ValueError:
            auc = np.nan

        results[f'auc (radchestct_{key})'] = float(auc)
        results[f'acc (radchestct_{key})'] = float(acc)
        results[f'weighted_f1 (radchestct_{key})'] = float(f1w)
        results[f'precision (radchestct_{key})'] = float(precision)
        results[f'recall (radchestct_{key})'] = float(recall)

        aucs.append(auc)
        accs.append(acc)
        f1ws.append(f1w)
        precisions.append(precision)
        recalls.append(recall)

    results.update({
        'auc (radchestct)': float(np.nanmean(aucs)),
        'acc (radchestct)': float(np.nanmean(accs)),
        'weighted_f1 (radchestct)': float(np.nanmean(f1ws)),
        'precision (radchestct)': float(np.nanmean(precisions)),
        'recall (radchestct)': float(np.nanmean(recalls)),
    })
    return results


# run
def run(model, tokenizer, dataloader, args):
    if args.zeroshot_template != 'organ':
        PROMPTS['Lung nodule'] = ('Not lung nodule', 'Lung nodule')
        PROMPTS['Lung opacity'] = ('Not lung opacity', 'Lung opacity')

    device = torch.device(args.device)
    autocast = get_autocast('amp', device_type=device.type)
    input_dtype = get_input_dtype('amp')

    with autocast():
        classifier = {}
        for key in CLASSNAMES:
            classifier[key] = build_zero_shot_classifier(
                model,
                tokenizer=tokenizer,
                classnames=PROMPTS[key],
                templates=TEMPLATES[ORGANS[key]] if args.zeroshot_template == 'organ' else TEMPLATES[args.zeroshot_template],
                num_classes_per_batch=None,
                device=device,
                use_tqdm=False,
            )

    prediction = []
    ground_truth = []
    with torch.inference_mode():
        for batch in tqdm(dataloader, total=len(dataloader), disable=not is_master(args)):
            image = batch['image'].to(device=device, dtype=input_dtype, non_blocking=True)
            ground_truth.append(batch['target'].cpu())

            with autocast():
                model_out = model(image=image)
                image_features = model_out['image_features']
                if image_features.ndim == 3:
                    image_features = image_features[:, 0, :]
                logit_scale = model_out['logit_scale']

                batch_prediction = []
                for key in CLASSNAMES:
                    logits_per_image = logit_scale * image_features @ classifier[key]
                    probs_per_image = logits_per_image.softmax(dim=-1)
                    batch_prediction.append(probs_per_image[:, 1].detach().cpu())

            prediction.append(torch.stack(batch_prediction, dim=1))

    return torch.cat(ground_truth, dim=0), torch.cat(prediction, dim=0)


# main
def main(args):
    if torch.cuda.is_available():
        # This enables tf32 on Ampere GPUs which is only 8% slower than
        # float16 and almost as accurate as float32
        # This was a default in pytorch until 1.12
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    device = init_distributed_device(args)
    if args.distributed:
        print(
            f'Running in distributed mode with multiple processes. Device: {args.device}.'
            f'Process (global: {args.rank}, local {args.local_rank}), total {args.world_size}.'
        )
    else:
        print(f'Running with a single process. Device {args.device}.')
    random_seed(0, 0)

    # create model
    for _c in os.listdir('../hlip/model_configs/'):
        _m, _e = os.path.splitext(_c)
        if _e.lower() == '.json':
            with open(os.path.join('../hlip/model_configs/', _c), 'r') as f:
                model_cfg = json.load(f)
            _MODEL_CONFIGS[_m] = model_cfg
    model, _, _ = create_model_and_transforms(args.model, device=args.device, precision='amp', output_dict=True)

    # replace with cxr_bert
    if args.use_cxr_bert:
        from transformers import AutoModel
        cxr_bert = AutoModel.from_pretrained('microsoft/BiomedVLP-CXR-BERT-specialized', trust_remote_code=True).bert
        if args.lora_text:
            from peft import LoraConfig, get_peft_model
            lora_config = LoraConfig(
                r=8,
                lora_alpha=8,
                target_modules=['query', 'value'],
                lora_dropout=0.0,
                bias='none',
            )
            cxr_bert = get_peft_model(cxr_bert, lora_config)
            for n, p in cxr_bert.named_parameters():
                p.requires_grad = (not args.lock_text_freeze_layer_norm) if 'LayerNorm' in n.split('.') else False
        cxr_bert.to(device=args.device)
        model.text.transformer = cxr_bert

    # load checkpoint
    checkpoint = pt_load(args.resume, map_location='cpu')
    sd = checkpoint['state_dict']
    sd = {k[len('module.'):]: v for k, v in sd.items()}
    model.load_state_dict(sd)
    tokenizer = get_tokenizer(args.model, trust_remote_code=True)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device], static_graph=False)

    # create dataset
    dataloader = get_data(
        data_root=args.data_root,
        input_file=args.input_file,
        process_cfg=args.process_cfg,
        workers=args.workers,
        distributed=args.distributed,
    )

    # run
    if args.distributed and not args.horovod:
        model = model.module

    model.eval()
    ground_truth, prediction = run(model, tokenizer, dataloader, args)
    if args.distributed:
        prediction = all_gather_object(args, prediction)
        ground_truth = all_gather_object(args, ground_truth)
    else:
        prediction = [prediction]
        ground_truth = [ground_truth]

    if is_master(args):
        prediction = torch.cat(prediction, dim=0)
        ground_truth = torch.cat(ground_truth, dim=0)

        print(f'Compute metrics on {prediction.shape[0]} cases.')
        results = compute_radchestct_metrics(ground_truth, prediction)
        for k, v in results.items():
            print(f'{k}: {v}')
        if args.save:
            p = Path(args.save)
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open('a', encoding='utf-8') as f:
                f.write(json.dumps(results, ensure_ascii=False))
                f.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Perform Zero-shot', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
