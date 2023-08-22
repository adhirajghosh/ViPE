import json
import os

import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = "2"
from utils import *
import torch
import requests
import torch
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt

from diffusers import StableDiffusionImg2ImgPipeline

args=dotdict({})

song_path='./mp3/'
song_name='Skyfall'
args.mp3_file=song_path + '{}.mp3'.format(song_name)
args.transcription_file='{}/{}_transcription'.format(song_path,song_name)
args.context_size=1
args.do_sample = True # generate prompts using ViPE with sampling
args.music_gap_prompt=['song: sky fall by adele']
args.prompt_file='{}/{}_ctx_{}_sample_{}_lyric2prompt'.format(song_path,song_name,args.context_size,args.do_sample)

args.checkpoint='/graphics/scratch2/staff/Hassan/checkpoints/lyrics_to_prompts/saved_models_mine/gpt2-medium_context_ctx_7_lr_5e-05-v4.ckpt/'
args.device='cuda'


lyric2prompt = get_lyrtic2prompts(args)
torch.cuda.empty_cache()


import PIL
from PIL import Image

import os
import torch
from diffusers import StableDiffusionInpaintPipeline, DPMSolverMultistepScheduler
from datetime import datetime
from in_painting_utils import *


output_path = "/graphics/scratch2/staff/Hassan/vipe videos/out/"

#@markdown DOWNLOAD MODEL WEIGHTS AND SET UP DIFFUSION PIPELINE <br><br>
#@markdown Pick your favourite inpainting model:
model_id = 'runwayml/stable-diffusion-inpainting' #@param ["stabilityai/stable-diffusion-2-inpainting", "runwayml/stable-diffusion-inpainting", "ImNoOne/f222-inpainting-diffusers","parlance/dreamlike-diffusion-1.0-inpainting","ghunkins/stable-diffusion-liberty-inpainting"] {allow-input: true}
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")
def dummy(images, **kwargs):
    return images, False
pipe.safety_checker = dummy
pipe.enable_attention_slicing() #This is useful to save some memory in exchange for a small speed decrease.

g_cuda = torch.Generator(device='cuda')

model_id_or_path = "runwayml/stable-diffusion-v1-5"
pipe_img2img = StableDiffusionImg2ImgPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float16)
pipe_img2img.to(args.device)



prompts={}
added_prompt = ", high quality, HD, 32K, ultra HD, high focus, dramatic lighting, ultra-realistic, DSLR photography, trending on artstation"
for num, l2p in enumerate(lyric2prompt):
    start = int(l2p['start'])
    # if num ==1:
    #     start= 3

    prompts[start] =l2p['prompt'][0] + added_prompt



# prompts={
#     0: "A young female warrior picking up a gun and gradually gets up, trending, 32k, high quality, fantasy world",
#     10:'A young girl walking towards the skyscrapers of New York City,  trending, 32k, high quality, fantasy world',
#     20:'An enthusiastic poltician giving a speech  to a large crowd,  trending, 32k, high quality, fantasy world',
#     30: 'a man running to seek shelter, a lightning bolt strikes through a dark ominous sky'
#     #7: "Ultra realistic city of angels fantastically detailed, fantasy world, colorful, rainbow "
# }


prompt = prompts[0]
#negative_prompt = "montage, frame, text, ugly, blur" #@param {type:"string"}

negative_prompt = 'text, worst quality, blurry, morbid, longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, cropped, low quality, deformed body, bloated, ugly, unrealistic, nude, naked'

#@markdown Number of initial example images to generate:
num_init_images = 1 #@param
#@markdown Random seed (arbitrary input to make the initial image generation deterministic):
seed = 42 #@param
#@markdown  The number of denoising steps (Higher number usually lead to a higher quality image at the expense of slower inference):
num_inference_steps = 30#@param
#@markdown Guidance scale defines how closely generated images to be linked to the text prompt:
guidance_scale = 7 #@param
#@markdown Heigth (and width) of the images in pixels (= resolution of the video generated in the next block, has to be divisible with 8):
height = 512 #@param
width = height
#@markdown Since the model was trained on 512 images increasing the resolution to e.g. 1024 will
#@markdown drastically reduce its imagination, so the video will vary a lot less compared to 512


#@markdown Number of outpainting steps:
num_outpainting_steps = int(list(prompts.keys())[-1]+ 10 ) #@param
# num_outpainting_steps= int(list(prompts.keys())[4]+ 10 )
#@markdown Width of the border in pixels to be outpainted during each step:
#@markdown <br> (make sure: mask_width < image resolution / 2)
mask_width = 128 #@param
#@markdown Number of images to be interpolated between each outpainting step:
num_interpol_frames = 30 #@param
fps = 30  # play speed

current_image = PIL.Image.new(mode="RGBA", size=(height, width))
mask_image = np.array(current_image)[:,:,3]
mask_image = Image.fromarray(255-mask_image).convert("RGB")
current_image = current_image.convert("RGB")

init_images =  pipe(prompt=[prompt]*num_init_images,
                    negative_prompt=[negative_prompt]*num_init_images,
                    image=current_image,
                    guidance_scale = guidance_scale,
                    height = height,
                    width = width,
                    generator = g_cuda.manual_seed(seed),
                    mask_image=mask_image,
                    num_inference_steps=num_inference_steps)[0]


image_grid(init_images, rows=1, cols=num_init_images)


#@markdown GENERATE VIDEO:  <br> <br>

#@markdown Pick an initial image from the previous block for your video: <br> (This is only relevant if num_init_images > 1)
init_image_selected = 1 #@param
if num_init_images == 1:
  init_image_selected = 0
else:
  init_image_selected = init_image_selected - 1
custom_init_image = False #@param {type:"boolean"}
init_image_address = "./out/image.jpeg"#@param {type:"string"}


if(custom_init_image):
  current_image = load_img(init_image_address,(width,height))
else :
  current_image = init_images[init_image_selected]
all_frames = []
all_frames.append(current_image)

theme_change=list(prompts.keys())[1:]
img2img=False
current_guidance_scale=guidance_scale
time_interval=-10
mask_width = 32

start_indx_guidance=0
start_indx_strength=0
change_counter=0
for i in range(num_outpainting_steps):

  if i in theme_change:
      mask_width = 128
      change_counter =0

  if change_counter < 3:
      change_counter +=1
  else:
      mask_width = 64

  # if  i in theme_change:
  #       timer_max=min(k for k in prompts.keys() if k > i+1)
  #       time_interval = int((timer_max - i)/2)
  #       current_guidance_scale=0
  #       start_indx_guidance = 0
  #       start_indx_strength = 0
  #
  #
  #       #img2img = True
  #
  # if i < i + time_interval:
  #     img2img = True
  #
  #     current_guidance_scale=list(np.linspace(4,7,time_interval))[ start_indx_guidance]
  #     current_strength_scale = list(np.linspace(0.2, .75, time_interval))[start_indx_strength]
  #
  #     start_indx_guidance +=1
  #     start_indx_strength +=1
  #
  #     time_interval +=1
  #
  # else:
  #     img2img = False



  print('Generating image: ' + str(i+1) + ' / ' + str(num_outpainting_steps))

  if img2img:
      input_prompt = prompts[max(k for k in prompts.keys() if k <= i + 1)]
      images = pipe_img2img(prompt=input_prompt, image=current_image,
                            strength=current_strength_scale, guidance_scale=current_guidance_scale).images
      current_image = images[0]

  prev_image_fix = current_image

  prev_image = shrink_and_paste_on_blank(current_image, mask_width)

  current_image = prev_image

  #create mask (black image with white mask_width width edges)
  mask_image = np.array(current_image)[:,:,3]
  mask_image = Image.fromarray(255-mask_image).convert("RGB")

  #inpainting step
  current_image = current_image.convert("RGB")
  input_prompt=prompts[max(k for k in prompts.keys() if k <= i)]




  images = pipe(prompt=input_prompt,
                negative_prompt=negative_prompt,
                image=current_image,
                guidance_scale = guidance_scale,
                height = height,
                width = width,
                #this can make the whole thing deterministic but the output less exciting
                #generator = g_cuda.manual_seed(seed),
                mask_image=mask_image,
                num_inference_steps=num_inference_steps)[0]


      # url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"
      #
      # response = requests.get(url)
      # init_image = Image.open(BytesIO(response.content)).convert("RGB")
      # init_image = init_image.resize((768, 512))




  current_image = images[0]
  current_image.paste(prev_image, mask=prev_image)

  #interpolation steps bewteen 2 inpainted images (=sequential zoom and crop)
  for j in range(num_interpol_frames - 1):
    interpol_image = current_image
    interpol_width = round(
        (1- ( 1-2*mask_width/height )**( 1-(j+1)/num_interpol_frames ) )*height/2
        )
    interpol_image = interpol_image.crop((interpol_width,
                                          interpol_width,
                                          width - interpol_width,
                                          height - interpol_width))

    interpol_image = interpol_image.resize((height, width))

    #paste the higher resolution previous image in the middle to avoid drop in quality caused by zooming
    interpol_width2 = round(
        ( 1 - (height-2*mask_width) / (height-2*interpol_width) ) / 2*height
        )
    prev_image_fix_crop = shrink_and_paste_on_blank(prev_image_fix, interpol_width2)
    interpol_image.paste(prev_image_fix_crop, mask = prev_image_fix_crop)

    all_frames.append(interpol_image)

  all_frames.append(current_image)

  #interpol_image.show()

print(len(all_frames), ' over all number of frames')
# @markdown RENDER THE GENERATED FRAMES INTO AN MP4 VIDEO.
video_file_name = "infinite_zoom"  # @param {type:"string"}
# @markdown frames per second:

now = datetime.now()
date_time = now.strftime("%m-%d-%Y_%H-%M-%S")
# @markdown Duplicates the first and last frames, use to add a delay before animation based on playback fps (15 = 0.5 seconds @ 30fps)
start_frame_dupe_amount = 0  # @param
last_frame_dupe_amount = 0  # @param
write_video(os.path.join(output_path,  "_{}_{}_.mp4".format(song_name, args.context_size)), all_frames, fps,args.mp3_file, False,
          start_frame_dupe_amount, last_frame_dupe_amount)
# write_video(os.path.join(output_path, video_file_name + "_{}_{}_.mp4"), all_frames, fps,args.mp3_file, True,
#           start_frame_dupe_amount, last_frame_dupe_amount)
# @markdown Once this block is finished, download your video from the "Files" folder menu on the left (output_path).



#@markdown CHECK SOME (equally spaced) FRAMES OF THE VIDEO:
# num_of_frames_to_chk = 4 #@param
# num_of_frames_to_chk = min(num_of_frames_to_chk, len(all_frames))
# idx = np.round(np.linspace(0, len(all_frames) - 1, num_of_frames_to_chk)).astype(int)
# image_grid(list(all_frames[i] for i in idx), rows = 1, cols = num_of_frames_to_chk)
#@markdown (This is relatively slow but still faster in some cases then to download the complete video in the 