import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
from datasets import load_dataset
import torch
from transformers import BertTokenizer, BertForSequenceClassification, GPT2LMHeadModel, GPT2Tokenizer
from torch.utils.data import DataLoader
from utils import visualizer, get_batch, save_s_json
import json
# Load the dataset
dataset = load_dataset('dair-ai/emotion')
# Split the dataset into train and validation sets
train_dataset = dataset['train']
valid_dataset = dataset['test']

device='cuda'


do_sample=False
use_visual_data=True # my data or haivmet data


model_name='gpt2-medium'
checkpoint_name = '{}_context_ctx_7_lr_5e-05-v4'.format(model_name)
if use_visual_data:
    saving_dir='/graphics/scratch2/staff/Hassan/checkpoints/lyrics_to_prompts/vis_emotion/emotions/visual_{}/{}/'.format(use_visual_data,checkpoint_name)
    # saving_dir = '/graphics/scratch2/staff/Hassan/checkpoints/lyrics_to_prompts/vis_emotion/emotions/visual_{}_adh_got2-medium-continue/{}/'.format(
    #     use_visual_data, checkpoint_name)
else:
    saving_dir = '/graphics/scratch2/staff/Hassan/checkpoints/lyrics_to_prompts/vis_emotion/emotions/visual_{}/'.format(
        use_visual_data)


# Create the directory if it does not exist
if not os.path.exists(saving_dir):
    os.makedirs(saving_dir)


os.makedirs(saving_dir + 'images/',exist_ok=True)
#checkpoint_name = 'gpt2-large_context_ctx_1_lr_5e-05-v3.ckpt'


from img_tools import generate_images

if use_visual_data:
    print('loading the visual version of the data')
    with open(saving_dir + 'vis_emotion_train_sample_{}_{}'.format(do_sample, checkpoint_name)) as file:
        vis_train = json.load(file)
    with open(saving_dir + 'vis_emotion_test_sample_{}_{}'.format(do_sample, checkpoint_name)) as file:
        text_valid = json.load(file)

    vis_train.extend(text_valid)

    prompt_dict={i:p for i,p in enumerate(vis_train)}

    print('generating  images')
    generate_images(prompt_dict=prompt_dict, saving_path=saving_dir + 'images/', batch_size=8, gpu=0)


else :

    vis_train = list(dataset['train']['text'])
    valid_dataset = list(dataset['test']['text'])
    vis_train.extend(valid_dataset)

    prompt_dict = {i: p for i, p in enumerate(vis_train)}
    generate_images(prompt_dict=prompt_dict, saving_path=saving_dir + 'images/', batch_size=8, gpu=1)