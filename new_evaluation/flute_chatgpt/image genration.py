import os

import torch
from transformers import BertTokenizer, BertForSequenceClassification, GPT2LMHeadModel, GPT2Tokenizer
from torch.utils.data import DataLoader
from datasets import load_dataset
from datasets import Dataset
from utils import visualizer, get_vis_batch, save_s_json, get_vis_flute_samples,get_haivment_prompts
import json
import random
# Load the dataset
dataset = load_dataset("ColumbiaNLP/FLUTE")

import random
from datasets import load_dataset, DatasetDict

# Define the type frequencies for validation
type_frequencies = {
    'Idiom': 250,
    'Metaphor': 250,
    'Simile': 250,
    'Sarcasm,CreativeParaphrase': 750
}

# Set the random seed for reproducibility
seed = 42
random.seed(seed)

# Create an empty list to store the sampled examples (validation set)
sampled_examples = []

# Sample from each type according to the frequencies
for type_name, freq in type_frequencies.items():

    if len(type_name.split(',')) < 2:
        # Filter the dataset for the current type
        filtered_dataset = dataset['train'].filter(lambda example: example['type'] == type_name)
        filtered_dataset = list(filtered_dataset)
        # Randomly sample 'freq' number of examples from the filtered dataset
        sampled_examples.extend(random.sample(filtered_dataset, freq))
    else:
        t1, t2 = type_name.split(',')
        # Filter the dataset for the current type
        filtered_dataset = dataset['train'].filter(lambda example: example['type'] == t1 or example['type'] == t2  )
        # Randomly sample 'freq' number of examples from the filtered dataset
        filtered_dataset = list(filtered_dataset)
        sampled_examples.extend(random.sample(filtered_dataset, freq))

# Shuffle the sampled examples
random.shuffle(sampled_examples)

# Get the IDs of the sampled validation examples
validation_ids = [example['id'] for example in sampled_examples]
# Create the training set by removing the validation examples from the original dataset
train_examples = dataset['train'].filter(lambda example: example['id'] not in validation_ids)

# create the valid dataset
# Create an empty dictionary to store the dataset columns
valid_dataset = {}

# Iterate over each dictionary in the list
for example in sampled_examples:
    # Iterate over each key-value pair in the dictionary
    for key, value in example.items():
        # If the key is not present in the dataset dictionary, initialize it as an empty list
        if key not in valid_dataset:
            valid_dataset[key] = []
        # Append the value to the corresponding key in the dataset dictionary
        valid_dataset[key].append(value)

# Create the dataset from the dictionary
valid_dataset = Dataset.from_dict(valid_dataset)


# Create a new dataset with the updated train and validation sets
dataset = DatasetDict({
    'train': train_examples,
    'validation': valid_dataset,
})

# Print the number of examples in the train and validation sets
print("Number of examples in the train set:", len(dataset['train']))
print("Number of examples in the validation set:", len(dataset['validation']))



# Split the dataset into train and validation sets
train_dataset = dataset['train']
valid_dataset = dataset['validation']




device='cuda'
os.environ['CUDA_VISIBLE_DEVICES']='2'
use_chatgpt=True
chat_gpt_random=False # set false to use sampled obtained from the deterministic  chatgpt (temperature =0)

do_sample=False
generate_new_data=False # my model
use_visual_data=True # my data or chatgpt data

illus=False # does not do much !
shuffle= False # shuffle the prompt to see what happens!


model_name='gpt2-medium'
device='cuda'
checkpoint_name = '{}_context_ctx_7_lr_5e-05-v4'.format(model_name)
if use_chatgpt:
    checkpoint_name='chat_gpt'


saving_dir='/graphics/scratch2/staff/Hassan/checkpoints/lyrics_to_prompts/vis_emotion/Vis_FLUTE_chatgpt/vis_{}_shuffle_{}_chatgpt_{}_random_{}/{}/'.format(use_visual_data,shuffle,use_chatgpt,chat_gpt_random,checkpoint_name)

# Create the directory if it does not exist
if not os.path.exists(saving_dir):
    os.makedirs(saving_dir)

os.makedirs(saving_dir + 'images/',exist_ok=True)
#checkpoint_name = 'gpt2-large_context_ctx_1_lr_5e-05-v3.ckpt'



from img_tools import generate_images

if use_visual_data and use_chatgpt:

    # if generate_haivment_data:
    #     HAIVMet_Dir = '/graphics/scratch2/staff/Hassan/datasets/HAIVMet/flute.zip'
    #     vis_valid=get_haivment_prompts(HAIVMet_Dir,vis_samples,valid_dataset )
    #     vis_train = get_haivment_prompts(HAIVMet_Dir, vis_samples, train_dataset)

        # save_s_json(saving_dir, 'vis_flute_train_sample_{}_{}'.format(do_sample, checkpoint_name), vis_train)
        # save_s_json(saving_dir, 'vis_flute_valid_sample_{}_{}'.format(do_sample, checkpoint_name), vis_valid)

    with open(saving_dir + 'vis_flute_sample_{}_{}'.format(do_sample, checkpoint_name)) as file:
        vis_train = json.load(file)

    print('generating  images')
    prompt_dict = {i: p for i, p in zip(vis_train['id'], vis_train['hypothesis'])}
    # prompt_dict={k:prompt_dict[k] for k in list(prompt_dict.keys())[500:1000]}
    generate_images(prompt_dict=prompt_dict, saving_path=saving_dir + 'images/', batch_size=3, gpu=0)

