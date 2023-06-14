import os

import torch
from transformers import BertTokenizer, BertForSequenceClassification, GPT2LMHeadModel, GPT2Tokenizer
from torch.utils.data import DataLoader
from datasets import load_dataset
from datasets import Dataset
from utils import visualizer, get_vis_batch, save_s_json, get_vis_flute_samples,update_dataset,get_haivment_prompts, SingleCollator, DoubleCollator
import json
import random

import zipfile
import random
from datasets import load_dataset, DatasetDict
from img_tools import generate_images_retrieval
import regex as re


device='cuda:1'
HAIVMet_Dir='/graphics/scratch2/staff/Hassan/datasets/HAIVMet/flute.zip'
vis_samples=get_vis_flute_samples(HAIVMet_Dir)

print(' found ', len(vis_samples), ' to be replaced in the dataset')
do_sample=False
generate_new_data=True # my model
use_visual_data=True # my data or haivmet data

generate_haivment_data=False
Use_HAIVMet_prompts=True # set to false to use your generated prompt


illus=False # does not do much !
shuffle= False # shuffle the prompt to see what happens!


model_name='gpt2'
checkpoint_name = '{}_context_ctx_7_lr_5e-05-v4'.format(model_name)

saving_dir='./models/lyrics_to_prompts/vis_emotion/Vis_FLUTE/vis_{}_shuffle_{}_haivmet_{}/{}/'.format(use_visual_data,shuffle,Use_HAIVMet_prompts,checkpoint_name)

# Create the directory if it does not exist
if not os.path.exists(saving_dir):
    os.makedirs(saving_dir)


#checkpoint_name = 'gpt2-large_context_ctx_1_lr_5e-05-v3.ckpt'


with zipfile.ZipFile(HAIVMet_Dir, 'r') as zip_file:
    # Get a list of all file names in the zip file
    file_names = zip_file.namelist()

    # Create an empty dictionary to store the folder names and first file names
    folder_dict = {}

    file_names = [file for file in file_names if len(file.split('/')) == 4 and file.split('/')[-1] != ''
                  and file.split('/')[-1] != 'gpt_prompt.txt' and file.split('/')[-1] != 'dalle_prompt.txt']
    # Iterate over the file names
    for file_name in file_names:

        # Extract the folder name and the first file name
        folder_name = file_name.split('/')[2]
        first_file_name = file_name.split('/')[3]

        # Check if the folder name is already in the dictionary
        if folder_name not in folder_dict:

            label='An ' +first_file_name.split('An')[-1][1:-4]

            if label[-1] !='.':
                label = label +'.'
            folder_dict[folder_name] = label

check_point_path='/graphics/scratch2/staff/Hassan/checkpoints/lyrics_to_prompts/saved_models_mine/{}.ckpt/'.format(checkpoint_name)


model = GPT2LMHeadModel.from_pretrained(check_point_path)
model.to(device)
tokenizer =  GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token=tokenizer.eos_token

ds_daiv = []
ds_ours = []
check = 0
for i, metaphor in enumerate(vis_samples):
    x, y = {}, {}
    x['text'], y['text'] = metaphor, metaphor
    x['prompt'] = folder_dict[metaphor]
    y['prompt'] = visualizer([metaphor], model, tokenizer, device, do_sample, epsilon_cutoff=.0005, temperature=1.1)[0]

    ds_daiv.append(x)
    ds_ours.append(y)


for sample_daiv in ds_daiv:
    sample_daiv['prompt'] = re.sub(r'[0-9]+', '', sample_daiv['prompt'])
    sample_daiv['prompt'] = re.sub("ALL·E -- .. - ", "", sample_daiv['prompt'])
    sample_daiv['prompt'] = re.sub("An A", "An", sample_daiv['prompt'])

torch.cuda.empty_cache()
print("Generate 5 images per prompt")
save_dir = '/graphics/scratch2/students/ghoshadh/SongAnimator/datasets/retrieval/ours/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
generate_images_retrieval(ds_ours,save_dir, device)



