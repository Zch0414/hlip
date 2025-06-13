import os
import sys
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('..'))

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
        self.scans = np.array(self.scans)
        self.report = np.array(value['report'])

    def get_report(self, shuffle):
        if shuffle: # this is for qwen_organ_annotation
            return ' '.join(np.random.permutation(self.report).tolist())
        else:
            return ' '.join(self.report.tolist())

    def get_scans(self, shuffle):
        if shuffle: # this is for training
            return np.random.permutation(self.scans).tolist()
        else:
            return self.scans.tolist()


class StudyDataset(Dataset):
    def __init__(
        self, 
        json_root, data_root, input_filename, input_info,
        transform=None,
        tokenizer=None,
    ):
        with open(os.path.join(json_root, input_filename + '.json'), 'r') as file:
            studies = json.load(file)
        self.studies = [StudyInfo(root=os.path.join(data_root, 'train'), key=key, value=value) for key, value in studies.items()]
        
        self.input_info = (float(input_info[0]), float(input_info[1]), str(input_info[2]))
        self.transform = transform
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.studies)
    
    def __getitem__(self, idx):
        study = self.studies[idx]

        # get report
        report = self.tokenizer([str(study.get_report(shuffle=True))])[0]

        # get scan
        scan = study.get_scans(shuffle=True)[0] # CT-RATE is a curated dataset
        
        # load in scan
        img = torch.load(scan, weights_only=True)
        img = (img - self.input_info[0]) / (self.input_info[1] - self.input_info[0])
        img = torch.clip(img, 0., 1.)
        img = img[None, ...].float() # [1, d, h, w]

        # transform
        if self.transform:
            img = self.transform(img)
            img = torch.as_tensor(img).float()
        else: 
            if self.input_info[2] == "crop":
                # pad
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
                    value=0
                ).squeeze(0)
                
                # crop [hard code]: tuning this is not interesting
                _, d, h, w = img.shape
                start_d = (d - 112) // 2
                start_h = (h - 336) // 2
                start_w = (w - 336) // 2
                img = img[
                    :, 
                    start_d:start_d + 112,
                    start_h:start_h + 336,
                    start_w:start_w + 336
                ]

            elif self.input_info[2] == "resize":
                img = torch.nn.functional.interpolate(img[None, ...], size=(112, 336, 336), mode='trilinear').squeeze(0)

            else:
                raise NotImplementedError

        # normalize
        normalizer = Normalize(torch.as_tensor(IMAGENET_DEFAULT_MEAN).mean(), torch.as_tensor(IMAGENET_DEFAULT_STD).mean())
        img = normalizer(img)

        return img[None, ...], report


def get_train_dataset(args, preprocess_fn, tokenizer=None):
    input_filename = args.train_data
    assert input_filename
    dataset = StudyDataset(
        args.json_root, args.data_root, input_filename,
        args.input_info,
        preprocess_fn,
        tokenizer,
    )
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed else None
    shuffle = sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=True,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


def get_zeroshot_ct_rate_dataset(args, preprocess_fn):
    from ..hlip_test.zeroshot_ct_rate import get_data
    dataloader = get_data(args, preprocess_fn)
    dataloader.num_samples = len(dataloader.dataset)
    dataloader.num_batches = len(dataloader)
    return DataInfo(dataloader, None)


def get_data(args, tokenizer=None):
    data = {}
    if args.train_data:
        data["train"] = get_train_dataset(args, None, tokenizer=tokenizer)
    if args.zeroshot_ct_rate:
        data["zeroshot-ct-rate"] = get_zeroshot_ct_rate_dataset(args, None)
    return data