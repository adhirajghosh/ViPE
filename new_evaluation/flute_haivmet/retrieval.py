import os

import torch
from transformers import BertTokenizer, BertForSequenceClassification, GPT2LMHeadModel, GPT2Tokenizer
from torch.utils.data import DataLoader
from datasets import load_dataset
from datasets import Dataset
from new_evaluation.flute_haivmet.utils import visualizer, get_vis_batch, save_s_json, get_vis_flute_samples,update_dataset,get_haivment_prompts, SingleCollator, DoubleCollator
import json
import random
# Load the dataset
dataset = load_dataset("ColumbiaNLP/FLUTE")

import zipfile
import random
from datasets import load_dataset, DatasetDict

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

    file_names=[file for file in file_names if len(file.split('/')) ==4 and file.split('/')[-1] !='']
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
model.to("cuda")
tokenizer =  GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token=tokenizer.eos_token

ds_daiv = []
ds_ours = []
check = 0
for i, metaphor in enumerate(vis_samples):
    x, y = {}, {}
    x['text'], y['text'] = metaphor, metaphor
    x['prompt'] = folder_dict[metaphor]
    y['prompt'] = visualizer([metaphor], model, tokenizer, device, do_sample, epsilon_cutoff=.0005, temperature=1.1)

    ds_daiv.append(x)
    ds_ours.append(y)

