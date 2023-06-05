import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from diffusers import StableDiffusionPipeline
import torch
from diffusers import StableDiffusionPipeline
from transformers import pipeline

model_id = "runwayml/stable-diffusion-v1-5"
model_id = "stabilityai/stable-diffusion-2"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)


# pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", revision="fp16", torch_dtype=torch.float16)
pipe = pipe.to("cuda")

use_extend=True
prompts=open('test','r').readlines()
path='/graphics/scratch2/students/ghoshadh/SongAnimator/prompt-extend/'
text_pipe = pipeline('text-generation', model=path, device=0, max_new_tokens=20)

#from transformers import pipeline

text_pipe = pipeline('text-generation', model='daspartho/prompt-extend')



for prompt in prompts:
    prompt=prompt.replace('\n','')
    if use_extend:
        prompt = text_pipe(prompt, num_return_sequences=1)[0]["generated_text"]

    image = pipe(prompt ).images[0]
    if len(prompt) >=20:
        image.save("results/{}_ext{}.png".format(prompt[0:20],str(use_extend)))
    else:
        image.save("results/{}_ext{}.png".format(prompt,str(use_extend)))


