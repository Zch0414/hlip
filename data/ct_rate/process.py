import os
import ast
import json
import argparse
import pandas as pd
import SimpleITK as sitk
import multiprocessing as mp

import torch


def get_args_parser():
    parser = argparse.ArgumentParser('Process', add_help=False)
    # Device
    parser.add_argument('--num-cpus', default=1, type=int)
    # Data
    parser.add_argument('--data', default='train', type=str)
    parser.add_argument('--root-dir', default='/download/ct_rate/dataset/')
    parser.add_argument('--save-dir', default='/data/ct_rate/', type=str)
    parser.add_argument('--save-astype', default='float32', type=str)
    # Process
    parser.add_argument('--start-index', default=0, type=int)
    parser.add_argument('--end-index', default=50000, type=int)
    parser.add_argument('--spacing', nargs='+', default=[3, 1, 1], type=int)
    return parser


def load_nifti_file(data, args):
    name = data["VolumeName"]
    try:
        # load in img
        path = os.path.join(args.root_dir, args.data, name.rsplit('_', 2)[0], name.rsplit('_', 1)[0], name)
        img = sitk.ReadImage(path)

        # scale
        img = img * data["RescaleSlope"] + data["RescaleIntercept"]

        # compute spacing
        (x, y), z = map(float, ast.literal_eval(data["XYSpacing"])), data["ZSpacing"]

        # array
        img = sitk.GetArrayFromImage(img) # d, h, w (z, y, x)

        # resize
        target_size = (int(img.shape[0] * z / args.spacing[0]), int(img.shape[1] * y / args.spacing[1]), int(img.shape[2] * x / args.spacing[2]))
        img = torch.nn.functional.interpolate(torch.from_numpy(img).float()[None, None, ...], size=target_size, mode='trilinear').squeeze().numpy()

        # rescale (here we maintain the original HU value, will do rescale during training and testing.)
        return img, name.rsplit('.', 2)[0]
    
    except Exception:
        return None, name.rsplit('.', 2)[0]


def single_worker(rows, args):
    results = {}

    for row in rows:
        img, name = load_nifti_file(row, args)
        if img is None:
            results[name] = 'fail'
            continue
        if args.save_astype == 'float32':
            img = torch.from_numpy(img).to(torch.float32)
        elif args.save_astype == 'float16':
            img = torch.from_numpy(img).to(torch.float16)

        img_save_dir = os.path.join(args.save_dir, args.data, name.rsplit('_', 2)[0], name.rsplit('_', 1)[0])
        os.makedirs(img_save_dir, exist_ok=True)
        torch.save(img, os.path.join(img_save_dir, name+'.pt'))

        results[name] = {'shape': img.shape, 'spacing': args.spacing}
    return results


def main(args):
    csv_file = pd.read_csv(f'./metafiles/{args.data}_metadata.csv')
    rows = [row[1] for row in csv_file.iterrows()][args.start_index: args.end_index]
    num_rows = len(rows)
    num_chunks = args.num_cpus
    chunk_size = (num_rows + num_chunks - 1) // num_chunks 
    rows_chunks = [rows[i:i + chunk_size] for i in range(0, num_rows, chunk_size)]
    input_chunks = [(rows_chunk, args) for rows_chunk in rows_chunks]

    with mp.Pool(processes=args.num_cpus) as pool:
       results_multi_workers = pool.starmap(single_worker, input_chunks)

    results = {}
    for results_one_worker in results_multi_workers:
        results.update(results_one_worker)
    with open(f'./files/{args.data}_info_{args.start_index}_{args.end_index}.json', 'w') as f:
        json.dump(results, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Process', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
