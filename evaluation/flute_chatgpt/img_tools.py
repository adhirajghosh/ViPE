import os


from diffusers import StableDiffusionPipeline
import torch

from diffusers import StableDiffusionPipeline
from transformers import pipeline

import random
import numpy as np

# Set the seed for random number generators
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
from tqdm import tqdm
# Set the desired torch data type

#model_id = "runwayml/stable-diffusion-v1-5"
model_id = "stabilityai/stable-diffusion-2"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)

#pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", revision="fp16", torch_dtype=torch.float16)
pipe = pipe.to("cuda")


def generate_images(prompt_dict, saving_path, batch_size, gpu):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)

    batch=[]
    ids=[]
    for num, (p_id, prompt) in tqdm(enumerate(prompt_dict.items())):

        if not os.path.isfile("{}{}.png".format(saving_path, p_id)):
            batch.append(prompt)
            ids.append(p_id)
            if len(batch) < batch_size and (num+1) < len(prompt_dict) :
                continue

            images = pipe(batch ).images
            for img_id, img in zip(ids, images):
                img.save("{}{}.png".format(saving_path, img_id))
            batch = []
            ids = []
