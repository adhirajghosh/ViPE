
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


def get_chatgpt_text(path_to_jsons):
    hypothesis=[]

    for fine_index, filename in enumerate(os.listdir(path_to_jsons)):
        if filename.endswith('.json'):
            file_path = os.path.join(path_to_jsons, filename)
            with open(file_path) as file:
                json_data = json.load(file)

            #gpt_id = json_data['id'].split('chatcmpl-')[1]
            prompts = json_data['choices'][0]['message']['content'].split('\n')

            #hoepfuly we get 50 prompts, if not, its chatgpt's fault, so we just add noise
            # if filename=='start_7500.json':
            #     number_of_prompts=34
            # else:
            #     number_of_prompts=50

            number_of_prompts = 50

            for line_number in range(number_of_prompts):
                if len(prompts)> line_number and len(prompts[line_number].split('.')) > 1:
                    hypothesis.append(prompts[line_number].split('.')[1])

                elif  len(prompts)> line_number:
                    hypothesis.append(prompts[line_number])

                elif len(prompts)<= line_number:
                    hypothesis.append('noise')
                    print('oops!')
                else:
                    print('yooo')

    return  hypothesis

from datasets import DatasetDict

def  update_dataset_chatgpt(dataset, prompt_list, portion):

    new_dataset={k:[] for k in dataset[portion].features.keys()}
    # Iterate over the dataset examples
    for idx, example in enumerate(dataset[portion]):
        example_copy=example.copy()
        example_copy['text'] = prompt_list[idx]

        for k, v in example_copy.items():
            new_dataset[k].append(v)
    #new_dataset =  Dataset.from_dict(new_dataset)
    #new_dataset= DatasetDict({portion: new_dataset})
    return new_dataset

def get_batch(data, batch_size):
    for num, index in enumerate(range(0, len(data), batch_size)):
        if (num + 1) * batch_size < len(data):
            samples = data[index:(num + 1) * batch_size]
        else:
            samples = data[index:]

        yield samples

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