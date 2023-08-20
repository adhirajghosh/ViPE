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
    parser = argparse.ArgumentParser(description='Generate datasets for image-text retrieval')

    parser.add_argument('--datadir', help='where are the datasets stored?', type=str, default='./datasets/HAIVMet/')
    parser.add_argument('--model', help='which model to use', type=str, default='vipe')
    parser.add_argument('--dataset', help='which dataset to use', type=str, default='ad_slogans')
    parser.add_argument('--savedir', help='where to save the dataset', type=str, default='./datasets/retrieval3/')
    parser.add_argument("--img_size", type=int, default=400)
    parser.add_argument("--num_images", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument('--device', help='which gpu to use', type=str, default='cuda')
    parser.add_argument('--pkl', help='Create pickle file?', type=bool, default=True)
    parser.add_argument('--sd_model', help='which Stable Diffusion Checkpoint to use', type=str, default='dreamlike-art/dreamlike-photoreal-2.0')
    parser.add_argument('--checkpoint', help='which ViPE Checkpoint to use', type=str, default='./models/vipe/')
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

def create_dataset_pickle(args):
    prompt_dict = {}
    for i in dataset_dict.keys():

        folder_dict = zip_process(os.path.join(args.datadir, i + '.zip'))
        dataset = []
        if args.model == 'haivmet':

            for i, metaphor in enumerate(folder_dict.keys()):
                info = {}
                info['text'] = metaphor
                info['prompt'] = folder_dict[metaphor]
                dataset.append(info)

            for sample in dataset:
                sample['prompt'] = re.sub(r'[0-9]+', '', sample['prompt'])
                sample['prompt'] = re.sub("ALL·E -- .. - ", "", sample['prompt'])
                sample['prompt'] = re.sub("An A", "An", sample['prompt'])

        elif args.model == 'chatgpt':

            with open('./datasets/retrieval/chatgpt_metaphor_id.pickle', 'rb') as handle:
                gpt_prompts = pickle.load(handle)
            for i, metaphor in enumerate(folder_dict.keys()):
                info = {}
                info['text'] = metaphor
                idx = metaphor_id[metaphor]
                info['prompt'] = gpt_prompts[idx]
                dataset.append(info)

        else:
            vipe_model = GPT2LMHeadModel.from_pretrained(args.checkpoint)
            vipe_model.to(args.device)

            tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
            tokenizer.pad_token = tokenizer.eos_token
            for i, metaphor in enumerate(folder_dict.keys()):
                info = {}
                info['text'] = metaphor
                info['prompt'] = visualizer([metaphor], vipe_model, tokenizer, args.device, args.sample,
                                            epsilon_cutoff=.0005, temperature=1.1)[0]
                dataset.append(info)

        for j in dataset:
            prompt_dict[j['text']] = (metaphor_id[j['text']], j['prompt'])

    with open(os.path.join(args.savedir, 'prompt_dict_{}.pickle'.format(args.model)), 'wb') as handle:
         pickle.dump(prompt_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


def main():
    args = parse_args()
    model_id = args.sd_model
    batch_size = args.batch_size
    device = args.device
    pipe = StableDiffusionPipeline.from_pretrained(model_id).to(device)

    folder_dict = zip_process(os.path.join(args.datadir, args.dataset + '.zip'))
    ds_id = dataset_dict[args.dataset]

    if not os.path.exists(args.savedir + 'prompt_dict_{}.pickle'.format(args.model)):
        create_dataset_pickle(args)

    id_file = args.savedir + 'prompt_dict_{}.pickle'.format(args.model)

    with open(id_file, 'rb') as handle:
        prompt_dict = pickle.load(handle)

    image_path = os.path.join(args.savedir, args.model)
    if not os.path.exists(image_path):
        os.makedirs(image_path)

    subset = {}
    for j in folder_dict.keys():
        info = prompt_dict[j]
        subset[j] = info

    generate_images(pipe, subset, ds_id, image_path, batch_size, args.img_size, args.num_images, device)

if __name__ == '__main__':
    main()
