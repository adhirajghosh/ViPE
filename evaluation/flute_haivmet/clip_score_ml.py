import torch
from blis.cy import diag_t
from torchvision.datasets.folder import default_loader
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torch.utils.data import DataLoader
import clip
import os
import torchvision
from tqdm import tqdm
import torch
from torchmetrics.functional.multimodal import clip_score
from utils import  save_s_json
import numpy as np
# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="generation images")

    parser.add_argument(
        "--model_name", type=str, default='gpt2', help="which gpt2 version to use?"
    )

    parser.add_argument(
        "--batch_size", type=int, default=30
    )

    parser.add_argument(
        "--context_length", type=int, default=5, help='number of previous lines from lyrics as the context'
    )



    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    use_visual_data =True
    shuffle=False
    Use_HAIVMet_prompts=False
    # Define the folder path containing the images

    checkpoint_name = '{}_context_ctx_{}_lr_5e-05-v4'.format(args.model_name, args.context_length)
    if Use_HAIVMet_prompts:
        checkpoint_name='humans'
    if not use_visual_data:
        checkpoint_name='textual'

    saving_dir='/mnt/lustre/lensch/hshahmohammadi86/checkpoints/songanimator/vis_emotion/Vis_FLUTE/vis_{}_shuffle_{}_haivmet_{}/{}/'.format(use_visual_data,shuffle,Use_HAIVMet_prompts,checkpoint_name)
    clip_dir=saving_dir + 'clip_score/'
    os.makedirs(clip_dir, exist_ok=True)

    import json
    if use_visual_data:
        with open(saving_dir + 'vis_flute_train_sample_{}_{}'.format(False, checkpoint_name)) as file:
            vis_train = json.load(file)

        with open(saving_dir + 'vis_flute_valid_sample_{}_{}'.format(False, checkpoint_name)) as file:
            vis_valid = json.load(file)

        captions_dict = {i: p for i, p in zip(vis_train['ids'], vis_train['vis_text'])}

        for k, v in zip(vis_valid['ids'], vis_valid['vis_text']):
            captions_dict[k] = v
    else:
        from utils import get_vis_flute_samples
        from datasets import load_dataset
        all_samples = load_dataset("ColumbiaNLP/FLUTE")['train']
        HAIVMet_Dir = '/graphics/scratch2/staff/Hassan/datasets/HAIVMet/flute.zip'
        vis_samples = get_vis_flute_samples(HAIVMet_Dir)

        #all_samples.filter(samle:)
        filtered_dataset = all_samples.filter(lambda example: example['hypothesis'] in vis_samples)

        captions_dict = {k: v for k, v in zip(filtered_dataset['id'], filtered_dataset['hypothesis'])}

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
            try:
                image = default_loader(image_path)
            except:
                print('oops, bad image, make sure you dont have many of these')
                image_id = self.image_ids[index-1]
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

            inputs = processor(text=caption_b, images=image_b, return_tensors="pt", padding=True)

            return inputs

        def __call__(self, batch):
           return self.collator(batch)

    clip_model='openai/clip-vit-large-patch14'
    processor = AutoProcessor.from_pretrained(clip_model)

    # Create the dataset and data loader
    dataset = ImageCaptionDataset(saving_dir + 'images', captions_dict)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,collate_fn=DataCollator(processor),num_workers=16)

    # Load the CLIP model and processor
    model = CLIPModel.from_pretrained(clip_model)
    model.to(device)

    all_similarities=[]
    # Compute CLIP score for each image-caption pair
    for inputs in tqdm(dataloader):
        # Preprocess captions and images

        # Move inputs to the device (GPU if available)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Compute CLIP scores
        outputs = model(**inputs)
        # Compute the cosine similarity between each image and text embedding
        cos_similarities = F.cosine_similarity(outputs.text_embeds, outputs.image_embeds, dim=1)
        logits_per_image = outputs.logits_per_image

        all_similarities.extend( cos_similarities.tolist())


    save_s_json(clip_dir, 'clip_scores', all_similarities)

    save_s_json(clip_dir, 'clip_scores_results', {'mean': np.mean(all_similarities), 'std':  np.mean(all_similarities)})
    print("mean:", np.mean(all_similarities))
    print("std:", np.std(all_similarities))


if __name__ == "__main__":
    main()
