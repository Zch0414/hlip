import os
import sys
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('..'))

import torch
from open_clip_train.distributed import is_master, all_gather_object

from hlip_test.zeroshot_ctrate import run as run_ct_rate
from hlip_test.zeroshot_ctrate import compute_ctrate_metrics
from hlip_test.zeroshot_radchestct import run as run_rad_chestct
from hlip_test.zeroshot_radchestct import compute_radchestct_metrics


def zero_shot_eval(model, data, epoch, args, tokenizer):
    if 'ct-rate' not in data and 'rad-chestct' not in data:
        return {}
    if args.zeroshot_frequency == 0:
        return {}
    if (epoch % args.zeroshot_frequency) != 0 and epoch != args.epochs:
        return {}
    if args.distributed and not args.horovod:
        model = model.module

    if 'ct-rate' in data:
        ground_truth, prediction = run_ct_rate(model, tokenizer, data['ct-rate'], args)
        ct_rate_ground_truth = all_gather_object(args, ground_truth)
        ct_rate_prediction = all_gather_object(args, prediction)
    if 'rad-chestct' in data:
        ground_truth, prediction = run_rad_chestct(model, tokenizer, data['rad-chestct'], args)
        rad_chestct_ground_truth = all_gather_object(args, ground_truth)
        rad_chestct_prediction = all_gather_object(args, prediction)

    if not is_master(args):
        return {}
    
    results = {}
    if 'ct-rate' in data:
        prediction = torch.cat(ct_rate_prediction, dim=0)
        ground_truth = torch.cat(ct_rate_ground_truth, dim=0)
        ct_rate_results = compute_ctrate_metrics(ground_truth, prediction)
        for key in ['auc (ctrate)', 'acc (ctrate)', 'weighted_f1 (ctrate)']:
            results.update(ct_rate_results[key])
    if 'rad-chestct' in data:
        prediction = torch.cat(rad_chestct_prediction, dim=0)
        ground_truth = torch.cat(rad_chestct_ground_truth, dim=0)
        rad_chestct_results = compute_radchestct_metrics(ground_truth, prediction)
        for key in ['auc (radchestct)', 'acc (radchestct)', 'weighted_f1 (radchestct)']:
            results.update(rad_chestct_results[key])
    return results