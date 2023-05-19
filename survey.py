import os
import random
import torch
from transformers import pipeline
from diffusers import StableDiffusionPipeline

def create_image(prompts=["man day phoned someone"],  # prompts to dream about
        seeds=100,
        gpu=1,  # id of the gpu to run on
        name='all-star-L1-3',  # name of this project, for the output directory
        rootdir='',
        num_steps=100,  # number of steps between each pair of sampled points
        frame_index = 0,
        # --------------------------------------
        # args you probably don't want to change
        num_inference_steps=100,
        guidance_scale=7.5,
        eta=0.0,
        width=512,
        height=512,
        # --------------------------------------
):
    # assert len(prompts) == len(seeds)
    assert torch.cuda.is_available()
    # assert height % 8 == 0 and width % 8 == 0

    outdir = rootdir
    os.makedirs(outdir, exist_ok=True)

    pipe = StableDiffusionPipeline.from_pretrained("./stable-diffusion-v1-5")
    pipe = pipe.to("cuda")

    prompt = prompts[0]
    image = pipe(prompt).images[0]

    # image.save("astronaut_rides_horse.png")
    outpath = os.path.join(outdir, '{}.jpg'.format(name))
    image.save(outpath)

path1 = './survey_v1.0/lyrics_v2.0/'
path2 = './survey_v1.0/lyrics_v3.0/'
text_pipe = pipeline('text-generation', model='./prompt-extend', device=0, max_new_tokens=20)

elements = [1] * 15
elements.extend([0] * 15)
random.shuffle(elements)

for i, x in enumerate(os.listdir(path1)):
    with open(os.path.join(path1,x)) as f:
        lines1 = f.readlines()[0].split('. ')[1][:-2]
    with open(os.path.join(path2,x)) as g:
        lines2 = g.readlines()[0].split('. ')[1][:-2]



    # if elements[int(x)-1] == 0:
    if int(x) == 9:
        lines1 = text_pipe(lines1 + ',', num_return_sequences=1)[0]["generated_text"]
        lines2 = text_pipe(lines2 + ',', num_return_sequences=1)[0]["generated_text"]
        print(lines1)
        print(lines2)
        print()
        create_image(prompts = [lines1],
                     rootdir = path1,
                     name = x+'_extend')
        # create_image(prompts = [lines2],
        #              rootdir = path2,
        #              name = x+'_extend')

    # elif elements[int(x)-1] == 1:
    #     create_image(prompts = [lines1],
    #                  rootdir = path1,
    #                  name = x)
    #     create_image(prompts = [lines2],
    #                  rootdir = path2,
    #                  name = x)

