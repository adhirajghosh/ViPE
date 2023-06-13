import torch
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
from blis.cy import diag_t
from torchvision.datasets.folder import default_loader
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torch.utils.data import DataLoader
import clip
import torchvision
from tqdm import tqdm
import torch
from torchmetrics.functional.multimodal import clip_score
from utils import  save_s_json
import numpy as np
# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



do_sample=False
use_visual_data=False # my data or haivmet data


model_name='gpt2-medium'
checkpoint_name = '{}_context_ctx_7_lr_5e-05-v4'.format(model_name)
if use_visual_data:
    saving_dir='/graphics/scratch2/staff/Hassan/checkpoints/lyrics_to_prompts/vis_emotion/emotions/visual_{}/{}/'.format(use_visual_data,checkpoint_name)
    # saving_dir = '/graphics/scratch2/staff/Hassan/checkpoints/lyrics_to_prompts/vis_emotion/emotions/visual_{}_adh_got2-medium-continue/{}/'.format(
    #     use_visual_data, checkpoint_name)
else:
    saving_dir = '/graphics/scratch2/staff/Hassan/checkpoints/lyrics_to_prompts/vis_emotion/emotions/visual_{}/'.format(
        use_visual_data)
    checkpoint_name='textual'

clip_dir=saving_dir + 'clip_score/'
os.makedirs(clip_dir, exist_ok=True)


import json

if use_visual_data:
    with open(saving_dir + 'vis_emotion_train_sample_{}_{}'.format(do_sample, checkpoint_name)) as file:
        text_train = json.load(file)
    with open(saving_dir + 'vis_emotion_test_sample_{}_{}'.format(do_sample, checkpoint_name)) as file:
        text_valid = json.load(file)

    text_train.extend(text_valid)

else:
    from datasets import load_dataset
    dataset = load_dataset('dair-ai/emotion')
    # Split the dataset into train and validation sets
    text_train = dataset['train']['text']
    valid_dataset = dataset['test']['text']

    text_train.extend(valid_dataset)

captions_dict = {i: p for i, p in enumerate(text_train)}

import torch
from torch.utils.data import DataLoader
from torchvision.datasets.folder import default_loader
import torchvision.transforms as transforms
from PIL import Image
import requests
from transformers import AutoProcessor, CLIPModel
import torch.nn.functional as F

# Create a custom dataset for loading images and captions
class ImageCaptionDataset(torch.utils.data.Dataset):
    def __init__(self, image_folder, captions_dict):
        self.image_folder = image_folder
        self.captions_dict = captions_dict
        self.image_ids = list(captions_dict.keys())


    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        image_path = f"{self.image_folder}/{image_id}.png"
        image = default_loader(image_path)
        #image = self.image_transform(image)
        caption = self.captions_dict[image_id]

        return image, caption

class DataCollator:

    def __init__(self, processor):
        self.processor = processor
    def collator(self, batch):
        image_b = []
        caption_b=[]
        for image, caption in batch:
            image_b.append(image)
            caption_b.append(caption)

        inputs = processor(text=caption_b, images=image_b, return_tensors="pt", padding=True,max_length=77,truncation=True)

        return inputs

    def __call__(self, batch):
       return self.collator(batch)


clip_model='openai/clip-vit-large-patch14'
processor = AutoProcessor.from_pretrained(clip_model)

# Create the dataset and data loader
dataset = ImageCaptionDataset(saving_dir + 'images', captions_dict)
dataloader = DataLoader(dataset, batch_size=10, shuffle=False,collate_fn=DataCollator(processor))

# Load the CLIP model and processor
model = CLIPModel.from_pretrained(clip_model)
model.to(device)
all_similarities=[]
# Compute CLIP score for each image-caption pair
for num, inputs in enumerate(tqdm(dataloader)):
    # Preprocess captions and images
    if num < 975:
        continue

    # Move inputs to the device (GPU if available)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Compute CLIP scores
    outputs = model(**inputs)
    # Compute the cosine similarity between each image and text embedding
    cos_similarities = F.cosine_similarity(outputs.text_embeds, outputs.image_embeds, dim=1)
    logits_per_image = outputs.logits_per_image

    all_similarities.extend(cos_similarities.tolist())

save_s_json(clip_dir, 'clip_scores', all_similarities)
save_s_json(clip_dir, 'clip_scores_results', {'mean': np.mean(all_similarities), 'std':  np.mean(all_similarities)})
print("mean:", np.mean(all_similarities))
print("std:", np.std(all_similarities))