import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
from datasets import load_dataset
import torch
from transformers import BertTokenizer, BertForSequenceClassification, GPT2LMHeadModel, GPT2Tokenizer
from torch.utils.data import DataLoader
from utils import  get_batch, save_s_json
import json
# Load the dataset
dataset = load_dataset('dair-ai/emotion')
# Split the dataset into train and validation sets
train_dataset = dataset['train']
valid_dataset = dataset['test']

device='cuda'


do_sample=True
checkpoint_name='t5'
saving_dir='/graphics/scratch2/staff/Hassan/checkpoints/lyrics_to_prompts/vis_emotion/emotions_{}/'.format(checkpoint_name)

import json



# Create the directory if it does not exist
if not os.path.exists(saving_dir):
    os.makedirs(saving_dir)


os.makedirs(saving_dir + 'images/',exist_ok=True)
#checkpoint_name = 'gpt2-large_context_ctx_1_lr_5e-05-v3.ckpt'


from img_tools import generate_images

with open(saving_dir + 'emotion_train_sample_{}_{}'.format(do_sample, checkpoint_name)) as file:
    text_train = json.load(file)
with open(saving_dir + 'emotion_test_sample_{}_{}'.format(do_sample, checkpoint_name)) as file:
    text_valid = json.load(file)

text_train.extend(text_valid)

prompt_dict={i:p for i,p in enumerate(text_train)}

prompt_dict={k:prompt_dict[k] for k in list(prompt_dict.keys())[11000:] }
print('generating  images')
generate_images(prompt_dict=prompt_dict, saving_path=saving_dir + 'images/', batch_size=8, gpu=0)
