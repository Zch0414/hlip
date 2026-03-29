import os
import sys
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('..'))

from torchvision.transforms import Normalize
from open_clip_train.data import *

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


class StudyInfo(object):
    def __init__(self, root, key, value):
        self.scans = [os.path.join(root, key, scans, 'img.pt') for scans in value['scans']]
        self.impressions = value['impressions']
        self.findings = value['findings']
    
    def get_scans(self, shuffle):
        if shuffle:
            random.shuffle(self.scans)
        return self.scans

    def get_impressions(self, shuffle):
        if shuffle:
            random.shuffle(self.impressions)
        return 'This study shows:' + ' '.join(self.impressions)
    
    def get_sentence(self, shuffle):
        if shuffle:
            sentence = random.choice(self.impressions)
        else:
            sentence = self.impressions[0]
        return f'This study shows: {sentence}'
    
    def get_findings(self, shuffle):
        if shuffle:
            random.shuffle(self.findings)
        return 'This study looks like:' + ' '.join(self.findings)


class StudyDataset(Dataset):
    def __init__(
        self, 
        data_root, input_file,
        text_process_cfg,
        num_scans=None,
        tokenizer=None, is_train=False,
    ):
        with open(input_file, 'r') as file:
            studies = json.load(file)
        self.studies = [StudyInfo(data_root, key, value) for key, value in studies.items()]

        # debug
        # self.studies = self.studies[:1536]

        self.text_process_cfg = text_process_cfg
        self.num_scans = num_scans
    
        self.tokenizer = tokenizer
        self.is_train = is_train
        self.normalizer = Normalize(torch.as_tensor(IMAGENET_DEFAULT_MEAN).mean(), torch.as_tensor(IMAGENET_DEFAULT_STD).mean())

    def __len__(self):
        return len(self.studies)
    
    def __getitem__(self, idx):
        study = self.studies[idx]

        # get image
        scans = study.get_scans(shuffle=self.is_train)
        if self.is_train:
            repeats = -(-self.num_scans // len(scans))
            scans *= repeats
            scans = scans[:self.num_scans]
        
        image = []
        for scan in scans:
            img = torch.load(scan, weights_only=True)
            img = img.float() / 255.0
            img = self.normalizer(img[None, ...])
            image.append(img)

        # get text
        if self.text_process_cfg == 'impressions':
            sentence = study.get_impressions(shuffle=self.is_train)
            report = sentence
        elif self.text_process_cfg == 'sentence and findings':
            sentence = study.get_sentence(shuffle=self.is_train)
            report = study.get_findings(shuffle=False)
        elif self.text_process_cfg == 'impressions and findings':
            sentence = study.get_impressions(shuffle=self.is_train)
            report = study.get_findings(shuffle=False)

        sentence = self.tokenizer([sentence])[0]
        report = self.tokenizer([report])[0]

        return {'image': torch.stack(image, dim=0), 'sentence': sentence, 'report': report}
    

def get_dataset(args, tokenizer, is_train):
    dataset = StudyDataset(
        data_root=args.train_data if is_train else args.valid_data,
        input_file=args.train_file if is_train else args.valid_file,
        text_process_cfg=args.text_process_cfg,
        num_scans=args.num_scans,
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
        data["valid"] = get_dataset(args, tokenizer=tokenizer, is_train=False)
    return data