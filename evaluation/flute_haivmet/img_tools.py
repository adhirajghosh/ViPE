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

        if  not os.path.isfile("{}{}.png".format(saving_path, p_id)):
            batch.append(prompt)
            ids.append(p_id)
            if len(batch) < batch_size and (num+1) < len(prompt_dict) :
                continue

            images = pipe(batch ).images
            for img_id, img in zip(ids, images):
                img.save("{}{}.png".format(saving_path, img_id))
            batch = []
            ids = []

def generate_images_retrieval(ds, save_path, gpu):

    model_id = 'dreamlike-art/dreamlike-photoreal-2.0'
    added_prompt = "high quality, HD, 32K, high focus, dramatic lighting, ultra-realistic, high detailed photography, vivid, vibrant,intricate,trending on artstation"
    negative_prompt = 'nude, naked, text, digits, worst quality, blurry, morbid, poorly drawn face, bad anatomy,distorted face, disfiguired, bad hands, missing fingers,cropped, deformed body, bloated, ugly, unrealistic'
    pipe = StableDiffusionPipeline.from_pretrained(model_id).to(gpu)

    for sample in ds:
        prompt = sample['prompt']
        print(prompt)
        for i in range(5):
            print("generating... ", i)
            image = pipe(prompt=prompt + added_prompt, negative_prompt=negative_prompt).images[0]
            outdir = os.path.join(save_path, sample['text'])
            if not os.path.exists(outdir):
                os.makedirs(outdir)

            if prompt[-1] == '.':
                outpath = os.path.join(outdir, '{}_{}.jpg'.format(prompt[:-1],str(i+1)))
            else:
                outpath = os.path.join(outdir, '{}_{}.jpg'.format(prompt, str(i + 1)))
            image.save(outpath)


