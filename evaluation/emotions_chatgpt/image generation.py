import os

os.environ['CUDA_VISIBLE_DEVICES']='2'
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
generate_new_data=False
illus=False

#checkpoint_name = 'gpt2-large_context_ctx_1_lr_5e-05-v3.ckpt'
use_visual_data=True
use_chatgpt=True
chat_gpt_random=False
shuffle=False

path_to_jsons = '/home/shahmoha/PycharmProjects/chatgpt/visual_emotions'
if chat_gpt_random:
    path_to_jsons='/home/shahmoha/PycharmProjects/chatgpt/visual_emotions_random'

model_name='gpt2-medium'
device='cuda'
checkpoint_name = '{}_context_ctx_3_lr_5e-05-v4'.format(model_name)
if use_chatgpt:
    checkpoint_name='chat_gpt'

saving_dir = '/graphics/scratch2/staff/Hassan/checkpoints/lyrics_to_prompts/vis_emotion/Vis_emotions_chatgpt/vis_{}_shuffle_{}_chatgpt_{}_random_{}/{}/'.format(
    use_visual_data, shuffle, use_chatgpt, chat_gpt_random, checkpoint_name)

# Create the directory if it does not exist
if not os.path.exists(saving_dir):
    os.makedirs(saving_dir)

os.makedirs(saving_dir + 'images/',exist_ok=True)
#checkpoint_name = 'gpt2-large_context_ctx_1_lr_5e-05-v3.ckpt'



from img_tools import generate_images

with open(saving_dir + 'vis_emotion_train_sample_{}_{}'.format(do_sample, checkpoint_name)) as file:
    text_train = json.load(file)
with open(saving_dir + 'vis_emotion_test_sample_{}_{}'.format(do_sample, checkpoint_name)) as file:
    text_valid = json.load(file)


text_train['text'].extend(text_valid['text'])

prompt_dict = {i: p for i, p in enumerate(text_train['text'])}
generate_images(prompt_dict=prompt_dict, saving_path=saving_dir + 'images/', batch_size=8, gpu=0)