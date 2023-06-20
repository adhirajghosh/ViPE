from pytube import YouTube
import os
import fire
from pathlib import Path
import numpy as np
import argparse
import whisper
import torch
import ffmpeg
import json
import ast
from PIL import Image
import time
import math
import torchvision.transforms as transforms
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from diffusers import (
    StableDiffusionImg2ImgPipeline,
    StableDiffusionPipeline,
    DPMSolverMultistepScheduler)
from moviepy.editor import ImageSequenceClip
from moviepy.editor import concatenate_videoclips
from moviepy.editor import AudioFileClip
from music_video_generation.modeling import GPT2Convertor
from music_video_generation.utils import dotdict, generate_from_sentences, get_close_variations_from_prompt
from vid import *

def parse_args():
    parser = argparse.ArgumentParser(description='Audio to Lyric Alignment')

    parser.add_argument('--song_name', help='Name of audio file', type=str, default='thunder.mp3')
    parser.add_argument("--model_name", type=str, default='gpt2-medium', help="which gpt2 version to use?")
    parser.add_argument("--diffusion_model", type=str, default='dreamlike-art/dreamlike-photoreal-2.0', help="which stable diffusion checkpoint to use?")
    parser.add_argument("--data_set_dir", type=str, default='/graphics/scratch2/staff/Hassan/genius_crawl/lyrics_to_prompts_v2.0.csv',help='path to the training data')
    parser.add_argument("--checkpoint", type=str, default='/graphics/scratch2/staff/Hassan/checkpoints/lyrics_to_prompts/saved_models_mine/gpt2-medium_context_ctx_7_lr_5e-05-v4.ckpt/', help="path to save the model")
    parser.add_argument("--img_size", type=int, default=640)
    parser.add_argument("--warmup_steps", type=int, default=1e3)
    parser.add_argument("--batch_size", type=int, default=30)
    parser.add_argument("--context_length", type=int, default=5, help='number of previous lines from lyrics as the context')
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument('--chunk', help='chunk interpolation. Type --chunk or --no-chunk', default=False,action=argparse.BooleanOptionalAction)
    parser.add_argument("--fps", type=float, default=10)
    parser.add_argument("--strength", type=float, default=0.6)
    parser.add_argument("--inf_steps", type=int, default=100)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--eta", type=int, default=0.05)
    parser.add_argument("--device_list", default=[1,2], nargs='+', type=int, help='GPU as a list')
    parser.add_argument('--device', help='gpu', type=str, default='0')
    parser.add_argument('--url', help='youtube url if song needs to be downloaded', type=str, default='https://www.youtube.com/watch?v=PSg7Zs5vlBQ&ab_channel=Audioandlyrics')
    parser.add_argument('--songdir', help='Where songs are kept', type=str, default='./mp3/')
    parser.add_argument('--outdir', help='Where results are kept', type=str, default='./results/vids/Thunder/9/')
    parser.add_argument('--timestamps', help='where timesteps are kept', type=str, default='./timestamps/')


    args = parser.parse_args()
    return args


def youtube2mp3 (url,outdir):
    yt = YouTube(url)
    video = yt.streams.filter(abr='192kbps').last()

    out_file = video.download(output_path=outdir)
    base, ext = os.path.splitext(out_file)
    song_name = f'{yt.title}.mp3'
    new_file = Path(f'{base}.mp3')
    os.rename(out_file, new_file)

    if new_file.exists():
        print(f'{yt.title} has been successfully downloaded.')
    else:
        print(f'ERROR: {yt.title}could not be downloaded!')

    return song_name

def whisper_transcribe(
        audio_fpath="audio.mp3", device='cuda'):
    whispers = {
        'tiny': None,  # 5.83 s
        'large': None  # 3.73 s
    }
    # accelerated runtime required for whisper
    # to do: pypi package for whisper

    for k in whispers.keys():
        options = whisper.DecodingOptions(
            task='translate',
            language='en',
        )
        # to do: be more proactive about cleaning up these models when we're done with them
        model = whisper.load_model(k).to(device)

        # start = time.time()
        print(f"Transcribing audio with whisper-{k}")

        # to do: calling transcribe like this unnecessarily re-processes audio each time.
        whispers[k] = model.transcribe(audio_fpath, task='translate')  # re-processes audio each time, ~10s overhead?
        # print(f"elapsed: {time.time() - start}")
    return whispers


def create_video_from_images(images, audio_file, output_file, fps):
    # Create an ImageSequenceClip from the list of images
    clip = ImageSequenceClip(images, fps=fps)

    # Set the duration of the clip based on the number of images and the desired FPS
    duration = len(images) / fps
    clip = clip.set_duration(duration)

    # Load the audio file
    audio = AudioFileClip(audio_file)

    # Set the audio of the clip
    clip = clip.set_audio(audio)

    clip.write_videofile(output_file, codec='libx264', audio_codec='aac', temp_audiofile='temp_audio.m4a',
                         remove_temp=True)

    print(f"Video created: {output_file}")

def img2emb(pipe, image, seed, device):
    # transform = transforms.Compose([
    #     transforms.PILToTensor()
    # ])
    generator = torch.Generator(device=device).manual_seed(seed)
    # image = transform(image).to(device, dtype=torch.float32)
    if isinstance(image, Image.Image):
        image = [image]

    if isinstance(image[0], Image.Image):
        w, h = image[0].size
        w, h = (x - x % 8 for x in (w, h))  # resize to integer multiple of 8

        image = [np.array(i.resize((w, h), resample= Image.Resampling.BICUBIC))[None, :] for i in image]
        image = np.concatenate(image, axis=0)
        image = np.array(image).astype(np.float32) / 255.0
        image = image.transpose(0, 3, 1, 2)
        image = 2.0 * image - 1.0
        image = torch.from_numpy(image).to(device)
    init_latents = pipe.vae.encode(image).latent_dist.sample(generator)
    return init_latents.to(device)



def main():
    args = parse_args()
    assert args.img_size % 8 == 0
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    if not os.path.exists(args.timestamps):
        os.makedirs(args.timestamps)

    if isinstance(args.device, list):
        device = f"cuda:{args.device[0]}"
        device1 = f"cuda:{args.device[1]}"
    else:
        device = f"cuda:{args.device}"
    print(device)
    #Download the song
    if not os.path.exists(os.path.join(args.songdir,args.song_name)):

        song_name = youtube2mp3(args.url, args.songdir)
    else:
        song_name = args.song_name


    song_path = os.path.join(args.songdir, song_name)
    print(song_path)

    song_length = float(ffmpeg.probe(song_path)['format']['duration'])

    # Load the transcription from whisper. We will use the large model

    # whispers = whisper_transcribe(song_path,device)
    # print(whispers)

    # with open('./timestamps/whispers.txt') as f:
    #     data = f.read()
    #
    # # reconstructing the data as a dictionary
    # whispers = ast.literal_eval(data)
    # model_name = args.model_name
    # dataset_dir = args.data_set_dir
    #
    # context_length = args.context_length
    #
    # #Load the hparams dict for the GPT2 model
    # hparams = dotdict({})
    # hparams.data_dir = dataset_dir
    # hparams.model_name = model_name
    # hparams.context_length = context_length
    # hparams.batch_size = args.batch_size
    # hparams.learning_rate = args.learning_rate
    # if isinstance(args.device, list):
    #     hparams.device = device1
    # else:
    #     hparams.device = device
    # hparams.warmup_steps = args.warmup_steps
    #
    # # Model initialisation and checkpoint loading
    # # model = GPT2Convertor(hparams)
    # # check_point_name = '{}_context_ctx_{}_lr_{}-v2.ckpt'.format(model_name, context_length, args.learning_rate)
    # # print(check_path+check_point_name)
    # # checkpoint = torch.load(check_path + check_point_name, map_location=lambda storage, loc: storage)
    # # model.load_state_dict(checkpoint['state_dict'])
    # # print('checkpoint loaded')
    #
    # model = GPT2LMHeadModel.from_pretrained(args.checkpoint)
    # model.to(device)
    # tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
    # tokenizer.pad_token = tokenizer.eos_token

    # tokenizer = model.tokenizer
    # model = model.model

    # if isinstance(args.device, list):
    #     model.to(device1)
    # else:
    #     model.to(device)
    #
    # #Add the prompts to the trancriptions
    # do_sample = True
    # for i, lines in enumerate(whispers['large']['segments']):
    #     lyric = lines['text']
    #     prompt = generate_from_sentences([lyric], model, tokenizer, hparams.device, do_sample)[
    #         0].replace(lyric, '')
    #     print(prompt)
    #     lines['prompt'] = prompt
    #
    # #Easier later on
    # whispers = whispers['large']['segments']
    # whisper_copy = whispers
    #
    # #Add start of the song if the lyrics don't start at the beginning
    # x = {}
    # if whispers[0]['start'] != 0.0:
    #     x['start'] = 0.0
    #     x['end'] = whispers[0]['start']
    #     x['text'], x['prompt'] = " "," "
    #     whispers.insert(0,x)
    #
    # #In case bugs exist towards the end of the transcriptions
    # if whispers[-1]['end'] > song_length:
    #     whispers[-1]['start'] = whispers[-2]['end']
    #     whispers[-1]['end'] = song_length
    #
    # torch.cuda.empty_cache()
    #
    # if not os.path.exists(args.timestamps):
    #     os.makedirs(args.timestamps)
    # print(os.path.join(args.timestamps,song_name.split('.')[0]+'.txt'))
    # with open(os.path.join(args.timestamps,song_name.split('.')[0]+'.txt'), "w") as output:
    #     output.write(str(whispers))


    # whispers_save = []
    # for i, lines in enumerate(whispers):
    #     x = {}
    #     x['start'] = lines['start']
    #     x['end'] = lines['end']
    #     x['text'] = lines['text']
    #     x['prompt'] = lines['prompt']
    #     whispers_save.append(x)
    #     print(lines["text"], " ", lines["prompt"])
    #
    #
    # if not os.path.exists(args.timestamps):
    #     os.makedirs(args.timestamps)
    # print(os.path.join(args.timestamps,song_name.split('.')[0]+'.txt'))
    # with open(os.path.join(args.timestamps,song_name.split('.')[0]+'.txt'), "w") as output:
    #     output.write(str(whispers_save))


    with open('./timestamps/thunder.txt') as f:
        data = f.read()

        # reconstructing the data as a dictionary
    whispers = ast.literal_eval(data)

    #Load the stable diffusion img2img and text2img models
    model_id = args.diffusion_model
    download = True

