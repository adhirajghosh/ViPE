"""
Generate images for a given lyrics using latent diffusion models
"""
import os

# whcih gpu to use? if more than one
os.environ['CUDA_VISIBLE_DEVICES'] = '5'

from transformers import DataCollatorWithPadding
from torch.utils.data import  DataLoader
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
from torch import nn
import torch
import numpy as np

dir='/graphics/scratch2/staff/Hassan/stable-diffusion-v1-5/'
save_dir='results/'

batch_size=1 # how many images per verse
#load the tokenizer
tokenizer = CLIPTokenizer.from_pretrained(dir+"tokenizer")


#bulding the model
class FeatureExtractor(nn.Module):
    def __init__(self, tokenizer,scheduler ):
        super(FeatureExtractor, self).__init__()

        self.vae = AutoencoderKL.from_pretrained(dir, subfolder="vae")
        self.tokenizer=tokenizer
        self.text_encoder = CLIPTextModel.from_pretrained(dir + "text_encoder")
        self.unet = UNet2DConditionModel.from_pretrained(dir, subfolder="unet")
        self.scheduler= scheduler

    def forward(self, x):


        text_embeddings = self.text_encoder(input_ids=x.input_ids)[0]
        uncond_embeddings = self.text_encoder(input_ids=x.input_ids_uncond)[0]


        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        latents = torch.randn(
            (uncond_embeddings.shape[0], self.unet.in_channels, height // 8, width // 8),
            #generator=torch.manual_seed(0)  ,
        )
        latents = latents.to("cuda")


        latents = latents * self.scheduler.init_noise_sigma


        for t in self.scheduler.timesteps:

            if t==0:
                d=2
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)

            latent_model_input = self.scheduler.scale_model_input(latent_model_input, timestep=t)

            with torch.no_grad():
                # predict the noise residual
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            # noise_pred = noise_pred_uncond

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        # scale and decode the image latents with vae
        latents= 1 / 0.18215 * latents
        images = self.vae.decode(latents).sample


        return latents, images



height = 512                        # default height of Stable Diffusion
width = 512                         # default width of Stable Diffusion
num_inference_steps = 50           # Number of denoising steps
guidance_scale = 7.5                # Scale for classifier-free guidance

# generator = torch.manual_seed(0)    # Seed generator to create the inital latent noise

from diffusers import LMSDiscreteScheduler
from PIL import Image

scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
scheduler.set_timesteps(num_inference_steps)


model =FeatureExtractor(tokenizer=tokenizer, scheduler=scheduler)
model.to("cuda")

text=[]
names=[]


# open the lyric file
with open('ring_of_fire', 'r') as f:
    file=f.readlines()

lyrics=[l.replace('\n',' ')  for l in file if l !='\n']

for c, cap in enumerate(lyrics):

        print(c, ' out of ', len(lyrics))
        names=[]
        for i in range(batch_size):
            names.append(str(c)+"_"+ str(i))
        print(cap)
        text=[cap] * batch_size

        text_input = tokenizer(text, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")

        max_length = text_input.input_ids.shape[-1]
        uncond_input = tokenizer(
            [""] * len(text) , padding="max_length", max_length=max_length, return_tensors="pt"
        )

        text_input['input_ids_uncond'] = uncond_input.input_ids
        text_input['attention_mask_uncond'] = uncond_input.attention_mask


        batch = text_input.to("cuda")
        with torch.no_grad():
            latents_f, image = model(batch)

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (image * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]

        #pil_images[0].save('man_2.png')

        for name,img in zip(names,pil_images):
            img.save(save_dir + 'all_star/'+ name +'.png')



