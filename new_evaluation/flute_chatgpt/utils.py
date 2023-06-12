
import torch
from torch.utils.data import Dataset, DataLoader
import os

from datasets import load_dataset
from bert_score import score
import json
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from tqdm import tqdm
import evaluate
from torch.nn.functional import cosine_similarity
import torch
from torch.utils.data import Dataset, DataLoader
import json
from transformers import DataCollatorWithPadding
from datasets import Dataset
import re

pattern = r'^([1-9]|[1-4][0-9]|50)\.\s'  # Regular expression pattern
def check_prompt_format(current_line):

    return bool(re.match(pattern, current_line))

#some prompts are reallt long like 60 or so, but only a few
def truncate(line, max_len):
    words=line.split(' ')
    if len(words) > max_len:
        words = words[0:max_len]
        return ' '.join(words)
    return  line


def get_chatgpt_hypothesis(path_to_jsons):
    hypothesis=[]

    for fine_index, filename in enumerate(os.listdir(path_to_jsons)):
        if filename.endswith('.json'):
            file_path = os.path.join(path_to_jsons, filename)
            with open(file_path) as file:
                json_data = json.load(file)

            #gpt_id = json_data['id'].split('chatcmpl-')[1]
            prompts = json_data['choices'][0]['message']['content'].split('\n')

            #hoepfuly we get 50 prompts, if not, its chatgpt's fault, so we just add noise
            if filename=='start_7500.json':
                number_of_prompts=34
            else:
                number_of_prompts=50

            for line_number in range(number_of_prompts):
                if len(prompts)> line_number and len(prompts[line_number].split('.')) > 1:
                    hypothesis.append(prompts[line_number].split('.')[1])

                elif  len(prompts)> line_number:
                    hypothesis.append(prompts[line_number])

                elif len(prompts)<= line_number:
                    hypothesis.append('noise')
                else:
                    print('yooo')

    return  hypothesis

from datasets import DatasetDict
def  update_dataset_chatgpt(dataset, prompt_list):

    new_dataset={k:[] for k in dataset['train'].features.keys()}
    # Iterate over the dataset examples
    for idx, example in enumerate(dataset['train']):
        example_copy=example.copy()
        example_copy['hypothesis'] = prompt_list[idx]

        for k, v in example_copy.items():
            new_dataset[k].append(v)
    new_dataset =  Dataset.from_dict(new_dataset)
    new_dataset= DatasetDict({'train': new_dataset})
    return new_dataset

def save_chat_gpt_prompt(dataset, prompt_list, haivmet_list, path):
    count = 0
    new_dataset = {k: [] for k in dataset['train'].features.keys()}
    # Iterate over the dataset examples
    for idx, example in enumerate(dataset['train']):
        example_copy = example.copy()
        if example['hypothesis'] in haivmet_list:
            example_copy['hypothesis'] = prompt_list[idx]
            count += 1
            for k, v in example_copy.items():
                new_dataset[k].append(v)

    save_s_json(path[0],path[1], data=new_dataset)
    print('saved the visual prompts')

def update_dataset_chatgpt_haivmet(dataset, prompt_list, haivmet_list):

    count=0
    new_dataset = {k: [] for k in dataset['train'].features.keys()}
    # Iterate over the dataset examples
    for idx, example in enumerate(dataset['train']):
        example_copy = example.copy()
        if example['hypothesis'] in haivmet_list:
            example_copy['hypothesis'] = prompt_list[idx]
            count +=1
        for k, v in example_copy.items():
            new_dataset[k].append(v)
    new_dataset = Dataset.from_dict(new_dataset)
    new_dataset = DatasetDict({'train': new_dataset})

    print('replaced ', count, ' samples based on haivment')
    return new_dataset

    # for id in ids:
    # d=3
    # vis_valid_dataset = Dataset.from_dict({'text': text_valid, 'label': valid_dataset['label']})
    # vis_train_dataset = Dataset.from_dict({'text': text_train, 'label': train_dataset['label']})

def get_batch(data, batch_size):

    #get a chunk of the data based on the batch_size
    for num, index in enumerate(range(0, len(data), batch_size)):
        if (num + 1) * batch_size < len(data):
            sub_data = data[index:(num + 1) * batch_size]
        else:
            sub_data = data[index:]

        yield sub_data

def get_vis_batch(data, batch_size, vis_samples):

    #get a chunk of the data based on the batch_size
    for num, index in enumerate(range(0, len(data), batch_size)):
        if (num + 1) * batch_size < len(data):
            sub_data = data[index:(num + 1) * batch_size]
        else:
            sub_data = data[index:]

        text = []
        text_types = []
        ids = []
        # only return those who exist in vis_samples
        for id, (premise, hypothesis) in zip(sub_data['id'],zip(sub_data['premise'],sub_data['hypothesis'])):
            #for flute_text in vis_samples:
            if premise in vis_samples:
                ids.append(id)
                text_types.append('premise')
                text.append(premise)

            if hypothesis  in vis_samples:
                    ids.append(id)
                    text_types.append('hypothesis')
                    text.append(hypothesis)

        yield (ids, text_types, text)

import zipfile

#from haiemet
def get_vis_flute_samples(zip_file_path):

    # Open the zip file
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        # Get a list of all the file and directory names in the zip file
        file_names = zip_ref.namelist()
        file_names =[file.split('/')[2] for file in file_names if len(file.split('/')) == 4 and file.split('/')[-1] != '']
        # Extract the directory names
        #subdirectories = [name for name in file_list if zip_ref.getinfo(name).is_dir()]
    return list(set(file_names))

def get_haivment_prompts(zip_file_path,vis_samples,dataset ):
    # to be replaced by our visual elaboration
    vis_data = {'ids': [], 'text_type': [], 'vis_text': [], 'text': []}

    # Open the zip file
    with zipfile.ZipFile(zip_file_path, 'r') as zip_file:
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

    for idx, example in enumerate(dataset):
        premise=example['premise']
        hypothesis=example['hypothesis']

        if premise in vis_samples:
            vis_data['vis_text'].append(folder_dict[premise])
            vis_data['ids'].append(example['id'])
            vis_data['text_type'].append('premise')
            vis_data['text'].append(premise)

        if hypothesis in vis_samples:
            vis_data['vis_text'].append(folder_dict[hypothesis])
            vis_data['ids'].append(example['id'])
            vis_data['text_type'].append('hypothesis')
            vis_data['text'].append(hypothesis)

    return vis_data

def save_s_json(path,name, data):
    with open(path + name,'w') as file:
        json.dump(data, file, indent=4)

def prepare_prompt(text):
    #text = text
    return text
def visualizer(text, model, tokenizer,device,do_sample,epsilon_cutoff=.0001,temperature=1):

    text=[tokenizer.eos_token +  prepare_prompt(i) + tokenizer.eos_token  for i in text]
    batch=tokenizer(text, padding=True, return_tensors="pt")

    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    max_prompt_length=50
    #max_length=input_ids.shape[1] + max_prompt_length
    generated_ids =model.generate(input_ids=input_ids,attention_mask=attention_mask, max_new_tokens=max_prompt_length ,do_sample=do_sample,epsilon_cutoff=epsilon_cutoff,temperature=temperature)
    pred_caps = tokenizer.batch_decode(generated_ids[:,-(generated_ids.shape[1] -input_ids.shape[1]):], skip_special_tokens=True)


    return pred_caps


class SingleCollator:

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    def collator(self, batch):
        text_b = []
        labels_b=[]
        for sample in batch:
            text = sample['text']
            label = sample['label']
            labels_b.append(label)
            text_b.append(text)

        tokens=self.tokenizer(text_b, padding=True, return_token_type_ids=False, return_tensors="pt")
        tokens['labels']= torch.Tensor(labels_b)
        return tokens

    def __call__(self, batch):
       return self.collator(batch)

class DoubleCollator:

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.label_to_id= {'Entailment':0,'Contradiction':1}

    def collator(self, batch):
        hypothesis_b = []
        labels_b=[]
        premise_b = []
        for sample in batch:
            label = sample['label']
            labels_b.append(self.label_to_id[label])
            premise_b.append(sample['premise'])
            hypothesis_b.append( sample['hypothesis'])

        tokens=self.tokenizer(premise_b,hypothesis_b, padding=True, return_token_type_ids=True, return_tensors="pt")
        tokens['labels']= torch.Tensor(labels_b)
        return tokens

    def __call__(self, batch):
       return self.collator(batch)