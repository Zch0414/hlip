import os
import argparse
import numpy as np
import multiprocessing as mp
from pathlib import Path

import torch


def get_args_parser():
    parser = argparse.ArgumentParser('Process', add_help=False)
    # Device
    parser.add_argument('--num-cpus', default=1, type=int)
    # Data
    parser.add_argument('--data-dir', default='/download/rad_chestct/', type=str)
    parser.add_argument('--save-dir', default='/data/rad_chestct/', type=str)
    parser.add_argument('--save-astype', default='float32', type=str)
    # Process
    parser.add_argument('--spacing', nargs='+', default=[3, 1, 1], type=int)
    return parser


def load_npz_file(data_path, args):
    try:
        # load in img
        img = np.load(data_path, allow_pickle=True)["ct"]

        # resize
        target_size = (int(img.shape[0] * 0.8 / args.spacing[0]), int(img.shape[1] * 0.8 / args.spacing[1]), int(img.shape[2] * 0.8 / args.spacing[2]))
        img = torch.nn.functional.interpolate(torch.from_numpy(img).float()[None, None, ...], size=target_size, mode='trilinear').squeeze().numpy()

        # rescale (here we maintain the original HU value, will do rescale during training and testing.)
        return img
    except Exception:
        return None


def single_worker(data_pathes, args):
    for data_path in data_pathes:
        img = load_npz_file(data_path, args)

        if img is None:
            print(f'{data_path} failed.')
            continue
        if args.save_astype == 'float32':
            img = torch.from_numpy(img).to(torch.float32)
        elif args.save_astype == 'float16':
            img = torch.from_numpy(img).to(torch.float16)

        name = str(data_path).split('/')[-1].split('.')[0]
        img_save_dir = os.path.join(args.save_dir, name)
        os.makedirs(img_save_dir, exist_ok=True)
        torch.save(img, os.path.join(img_save_dir, name+'.pt'))


def main(args):
    data_pathes = [p for p in Path(args.data_dir).rglob('*.npz')]
    num_data = len(data_pathes)
    num_chunks = args.num_cpus
    chunk_size = (num_data + num_chunks - 1) // num_chunks 
    data_chunks = [data_pathes[i:i + chunk_size] for i in range(0, num_data, chunk_size)]
    input_chunks = [(data_chunk, args) for data_chunk in data_chunks]

    with mp.Pool(processes=args.num_cpus) as pool:
       pool.starmap(single_worker, input_chunks)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Process', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
