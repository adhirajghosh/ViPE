from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pickle
import os
import glob
import torch
from torchvision.datasets.utils import download_url
import json
from PIL import Image

from randaugment import RandomAugment

class RetrievalDataset(Dataset):
    def __init__(self, id_file, dataset, transform=None):
        with open(id_file, 'rb') as handle:
            caption_ids = pickle.load(handle)
        self.ids = caption_ids
        self.dataset = dataset
        self.transform = transform

        # Load captions from the text file
        self.captions = list(caption_ids.keys())

        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}

        txt_id = 0

        for img_id,image_name in enumerate(dataset):
            self.image.append(image_name)
            self.img2txt[img_id] = []
            caption_id = image_name.split('/')[-1].split('_')[1]
            caption = [k for k, v in self.ids.items() if v == caption_id][0]
            self.text.append(caption)
            self.img2txt[img_id].append(txt_id)
            self.txt2img[txt_id] = img_id
            txt_id += 1

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):

        image_name = self.dataset[index]
        image = Image.open(image_name).convert('RGB')
        # Apply transformations if provided
        if self.transform is not None:
            image = self.transform(image)
        ds_id = image_name.split('/')[-1][0]
        caption_id = image_name.split('/')[-1].split('_')[1]

        caption = [k for k, v in self.ids.items() if v == caption_id][0]
        return image,  caption, index


def create_dataset(dataset_dir, id_file, config=None, min_scale=0.5):
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(config['image_size'], scale=(min_scale, 1.0),
                                     interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        RandomAugment(2, 5, isPIL=True, augs=['Identity', 'AutoContrast', 'Brightness', 'Sharpness', 'Equalize',
                                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
        transforms.ToTensor(),
        normalize,
    ])
    transform_test = transforms.Compose([
        transforms.Resize((400,400), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        normalize,
    ])


    train_dir = dataset_dir+ "/image_train/"
    train_split = glob.glob(os.path.join(train_dir, '**', '*.jpg'), recursive=True) + \
                  glob.glob(os.path.join(train_dir, '**', '*.jpeg'), recursive=True) + \
                  glob.glob(os.path.join(train_dir, '**', '*.png'), recursive=True)

    train_dataset = RetrievalDataset(id_file, train_split, transform_train)

    test_dir = dataset_dir+ "/image_test/"
    test_split = glob.glob(os.path.join(test_dir, '**', '*.jpg'), recursive=True) + \
                glob.glob(os.path.join(test_dir, '**', '*.jpeg'), recursive=True) + \
                glob.glob(os.path.join(test_dir, '**', '*.png'), recursive=True)
    test_dataset = RetrievalDataset(id_file, test_split, transform_test)


    return train_dataset, test_dataset



def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
    loaders = []
    for dataset,sampler,bs,n_worker,is_train,collate_fn in zip(datasets,samplers,batch_size,num_workers,is_trains,collate_fns):
        if is_train:
            shuffle = (sampler is None)
            drop_last = True
        else:
            shuffle = False
            drop_last = False
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )
        loaders.append(loader)
    return loaders

def create_sampler(datasets, shuffles, num_tasks, global_rank):
    samplers = []
    for dataset,shuffle in zip(datasets,shuffles):
        sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank, shuffle=shuffle)
        samplers.append(sampler)
    return samplers
