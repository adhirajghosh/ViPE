import json
import os
from utils import *
import torch

import matplotlib.pyplot as plt
import numpy as np

x=list(range(300 *1))
y=[ np.sin(2*3.14*t/10) for t in x]
plt.plot(x,y)
plt.show()

args=dotdict({})

song_path='./mp3/'
song_name='Skyfall'
args.mp3_file=song_path + '{}.mp3'.format(song_name)
args.transcription_file='{}/{}_transcription'.format(song_path,song_name)
args.context_size=3
args.do_sample = False # generate prompts using ViPE with sampling
args.music_gap_prompt=['music, fantasy, exciting, colorful']
args.prompt_file='{}/{}_ctx_{}_sample_{}_lyric2prompt'.format(song_path,song_name,args.context_size,args.do_sample)

args.checkpoint='/graphics/scratch2/staff/Hassan/checkpoints/lyrics_to_prompts/saved_models_mine/gpt2-medium_context_ctx_7_lr_5e-05-v4.ckpt/'
args.device='cuda:1'


lyric2prompt = get_lyrtic2prompts(args)
torch.cuda.empty_cache()







