import os
import sys
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('..'))

import re
from torchvision.transforms import Normalize
from open_clip_train.data import *

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


CT_RATE_INVALID_DATA = ['train_1267_a_4', 'train_11755_a_3', 'train_11755_a_4']


class StudyInfo(object):
    def __init__(self, root, key, value):
        self.scans = []
        for scan in value['recons']:
            scan = scan.rsplit('.', 2)[0]
            if scan in CT_RATE_INVALID_DATA:
                continue
            else:
                self.scans.append(os.path.join(root, key.rsplit('_', 1)[0], key, scan + '.pt'))
        
        self.reports = value['report']

    def get_scans(self, shuffle):
        if shuffle:
            random.shuffle(self.scans)
        return self.scans

    def get_report(self, shuffle):
        if shuffle:
            random.shuffle(self.reports)
        return 'This study shows:' + ' '.join(self.reports)
    
    def get_sentence(self, shuffle):
        if shuffle:
            return f'This study shows: {random.choice(self.reports)}'
        return f'This study shows: {self.reports[0]}'


class StudyDataset(Dataset):
    def __init__(
        self, 
        data_root, input_file, 
        image_process_cfg, text_process_cfg,
        tokenizer=None, is_train=False
    ):
        with open(input_file, 'r') as file:
            studies = json.load(file)
        self.studies = [StudyInfo(data_root, key, value) for key, value in studies.items()]

        # debug
        # self.studies = self.studies[:1536]
        
        self.image_process_cfg = (float(image_process_cfg[0]), float(image_process_cfg[1]), str(image_process_cfg[2]))
        self.text_process_cfg = text_process_cfg
        
        self.tokenizer = tokenizer
        self.is_train = is_train
        self.normalizer = Normalize(torch.as_tensor(IMAGENET_DEFAULT_MEAN).mean(), torch.as_tensor(IMAGENET_DEFAULT_STD).mean())

    def __len__(self):
        return len(self.studies)
    
    def __getitem__(self, idx):
        study = self.studies[idx]

        # get image
        scan = study.get_scans(shuffle=self.is_train)[0] # CT-RATE is a curated dataset
        
        img = torch.load(scan, weights_only=True)
        img = (img.float() - self.image_process_cfg[0]) / (self.image_process_cfg[1] - self.image_process_cfg[0])
        img = torch.clip(img, 0., 1.)
        img = img[None, ...]

        if self.image_process_cfg[2] == 'crop':
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

        elif self.image_process_cfg[2] == "resize":
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

        # get text
        if self.text_process_cfg == 'report':
            report = study.get_report(shuffle=self.is_train)
            sentence = report
        elif self.text_process_cfg == 'sentence':
            sentence = study.get_sentence(shuffle=self.is_train)
            report = sentence
        elif self.text_process_cfg == 'sentence and report':
            sentence = study.get_sentence(shuffle=self.is_train)
            report = study.get_report(shuffle=self.is_train)
        
        sentence = self.tokenizer([sentence])[0]
        report = self.tokenizer([report])[0]
        
        return {'image': img[None, ...], 'sentence': sentence, 'report': report}


def get_dataset(args, tokenizer, is_train):
    dataset = StudyDataset(
        data_root=args.train_data if is_train else args.valid_data,
        input_file=args.train_file if is_train else args.valid_file,
        image_process_cfg=args.image_process_cfg,
        text_process_cfg=args.text_process_cfg,
        tokenizer=tokenizer,
        is_train=is_train
    )

    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size if is_train else 1, # avoid CPU memory issue
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)
    
    return DataInfo(dataloader, sampler)


def get_data(args, tokenizer=None):
    data = {}
    if args.train_data:
        data["train"] = get_dataset(args, tokenizer=tokenizer, is_train=True)
    if args.valid_data:
        raise NotImplementedError
    if args.ct_rate:
        from hlip_test.zeroshot_ctrate import get_data as get_ct_rate
        data["ct-rate"] = get_ct_rate(
            data_root=args.ct_rate['data_root'],
            input_file=args.ct_rate['input_file'],
            process_cfg=args.image_process_cfg,
            workers=args.workers,
            distributed=args.distributed
        )
    if args.rad_chestct:
        from hlip_test.zeroshot_radchestct import get_data as get_rad_chestct
        data["rad-chestct"] = get_rad_chestct(
            data_root=args.rad_chestct['data_root'],
            input_file=args.rad_chestct['input_file'],
            process_cfg=args.image_process_cfg,
            workers=args.workers,
            distributed=args.distributed
        )
    return data