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

with open('/mnt/lustre/lensch/lhr027/projects/SongAnimator/metaphor_id.pickle', 'rb') as handle:
    metaphor_id = pickle.load(handle)

def parse_args():
    parser = argparse.ArgumentParser(description='Audio to Lyric Alignment')

    parser.add_argument('--datadir', help='where are the datasets stored?', type=str, default='/mnt/lustre/lensch/lhr027/datasets/HAIVMet')
    parser.add_argument('--dataset', help='Name of dataset', type=str, default='ad_slogans')
    parser.add_argument('--model', help='which model to use', type=str, default='vipe')
    parser.add_argument('--checkpoint', help='path to the GPT to use', type=str, default='/mnt/lustre/lensch/lhr027/checkpoints/songanimator/best_model/gpt2-medium_context_ctx_7_lr_5e-05-v4.ckpt')
    parser.add_argument('--savedir', help='where to save the dataset', type=str, default='/mnt/lustre/lensch/lhr027/datasets/retrieval/')
    parser.add_argument("--img_size", type=int, default=400)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument('--sample', help='Sampling or no', type=bool, default=False)

    args = parser.parse_args()
    return args

def visualizer(text, model, tokenizer,device,do_sample,epsilon_cutoff=.0001,temperature=1):

    text=[tokenizer.eos_token +  i + tokenizer.eos_token  for i in text]
    batch=tokenizer(text, padding=True, return_tensors="pt")

    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    max_prompt_length=50
    #max_length=input_ids.shape[1] + max_prompt_length
    generated_ids =model.generate(input_ids=input_ids,attention_mask=attention_mask, max_new_tokens=max_prompt_length ,do_sample=do_sample,epsilon_cutoff=epsilon_cutoff,temperature=temperature)
    pred_caps = tokenizer.batch_decode(generated_ids[:,-(generated_ids.shape[1] -input_ids.shape[1]):], skip_special_tokens=True)
    return pred_caps

def main():
    args = parse_args()
    ds_path = os.path.join(args.savedir, args.model)
    if not os.path.exists(ds_path):
        os.makedirs(ds_path)

    device = 'cuda'

    print("The dataset being used is ", args.dataset)
    folder_dict = zip_process(os.path.join(args.datadir, args.dataset+'.zip'))

    model = GPT2LMHeadModel.from_pretrained(args.checkpoint)
    model.to(device)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
    tokenizer.pad_token = tokenizer.eos_token

    model_id = 'dreamlike-art/dreamlike-photoreal-2.0'
    pipe = StableDiffusionPipeline.from_pretrained(model_id).to(device)
    ds_id = dataset_dict[args.dataset]
    batch_size = args.batch_size

    ds_daiv = []
    for i, metaphor in enumerate(folder_dict.keys()):
        x = {}
        x['text'] = metaphor
        x['prompt'] = folder_dict[metaphor]
        ds_daiv.append(x)

    for sample_daiv in ds_daiv:
        sample_daiv['prompt'] = re.sub(r'[0-9]+', '', sample_daiv['prompt'])
        sample_daiv['prompt'] = re.sub("ALL·E -- .. - ", "", sample_daiv['prompt'])
        sample_daiv['prompt'] = re.sub("An A", "An", sample_daiv['prompt'])

    prompt_dict_daiv = {}
    for i in ds_daiv:
        prompt_dict_daiv[metaphor_id[i['text']]] = i['prompt']

    with open(os.path.join(args.savedir,'prompt_dict_daivmet.pickle'), 'wb') as handle:
        pickle.dump(prompt_dict_daiv, handle, protocol=pickle.HIGHEST_PROTOCOL)

    daiv_path = os.path.join(args.savedir, 'daivmet')
    if not os.path.exists(daiv_path):
        os.makedirs(daiv_path)
    generate_images(pipe, prompt_dict_daiv, ds_id, daiv_path, batch_size, args.img_size, device)

    ds_ours = []
    for i, metaphor in enumerate(folder_dict.keys()):
        y = {}
        y['text'] = metaphor
        y['prompt'] = visualizer([metaphor], model, tokenizer, device, args.sample,
                                 epsilon_cutoff=.0005, temperature=1.1)[0]
        ds_ours.append(y)

    prompt_dict_ours = {}
    for i in ds_ours:
        prompt_dict_ours[metaphor_id[i['text']]] = i['prompt']

    with open(os.path.join(args.savedir,'prompt_dict_vipe.pickle'), 'wb') as handle:
        pickle.dump(prompt_dict_ours, handle, protocol=pickle.HIGHEST_PROTOCOL)

    ours_path = os.path.join(args.savedir, 'vipe')
    if not os.path.exists(ours_path):
        os.makedirs(ours_path)
    generate_images(pipe, prompt_dict_ours, ds_id, ours_path, batch_size, args.img_size, device)

if __name__ == '__main__':
    main()







