import zipfile
import torch
from diffusers import StableDiffusionPipeline
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import numpy as np
import argparse
import os
import regex as re
from utils import *
import pickle


dataset_dict = {
    'ad_slogans':1,
    'bizzoni':2,
    'copoet':3,
    'figqa':4,
    'flute':5,
    'tsvetkov':6
}

with open('./datasets/retrieval/metaphor_id.pickle', 'rb') as handle:
    metaphor_id = pickle.load(handle)

def parse_args():
    parser = argparse.ArgumentParser(description='Audio to Lyric Alignment')

    parser.add_argument('--datadir', help='where are the datasets stored?', type=str, default='/graphics/scratch2/staff/Hassan/datasets/HAIVMet/')
    parser.add_argument('--model', help='which model to use', type=str, default='haivmet')
    parser.add_argument('--dataset', help='which dataset to use', type=str, default='ad_slogans')
    parser.add_argument('--savedir', help='where to save the dataset', type=str, default='/graphics/scratch2/students/ghoshadh/SongAnimator/datasets/retrieval2/')
    parser.add_argument("--img_size", type=int, default=400)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument('--device', help='which gpu to use', type=str, default='cuda')
    parser.add_argument('--sd_model', help='which Stable Diffusion Checkpoint to use', type=str, default='dreamlike-art/dreamlike-photoreal-2.0')
    parser.add_argument('--sample', help='Sampling or no', type=bool, default=False)

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    model_id = args.sd_model
    batch_size = args.batch_size
    device = args.device
    pipe = StableDiffusionPipeline.from_pretrained(model_id).to(device)
    ds_id = dataset_dict[args.dataset]

    if args.model == 'haivmet':
        id_file = args.savedir + 'prompt_dict_haivmet2.pickle'
    elif args.model == 'vipe':
        id_file = args.savedir + 'prompt_dict_vipe2.pickle'
    else:
        id_file = args.savedir + 'prompt_dict_chatgpt.pickle'

    with open(id_file, 'rb') as handle:
        prompt_dict = pickle.load(handle)