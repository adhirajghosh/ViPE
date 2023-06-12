
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

class FigQADataset(Dataset):
    def __init__(self, dataset):
        self.start_phrases = dataset['startphrase']
        self.endings1 = dataset['ending1']
        self.endings2 = dataset['ending2']
        self.labels = dataset['labels']

    def __len__(self):
        return len(self.start_phrases)

    def __getitem__(self, index):
        start_phrase = self.start_phrases[index]
        ending1 = self.endings1[index]
        ending2 = self.endings2[index]
        label = self.labels[index]

        return start_phrase, ending1, ending2, label
def prepare_fig_qa(batch_size):
    # Load the Fig-Questions dataset
    dataset = load_dataset("nightingal3/fig-qa")

    # Create train, validation, and test data loaders
    train_dataset = FigQADataset(dataset['train'])
    valid_dataset = FigQADataset(dataset['validation'])
    test_dataset = FigQADataset(dataset['test'])

    batch_size = batch_size

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return  train_loader, valid_loader, test_loader



class VisFluteDataset(Dataset):
    def __init__(self, path, version, real,do_sample=True, replace_hyp=False, replace_prem=False, training=True):

        if not real:
            with open(path +'vis_flute_train_h_{}_p_{}_s_{}_v{}.0'.format(
                    replace_hyp, replace_prem,do_sample,version)) as file:
                train = json.load(file)

            with open(path + 'vis_flute_valid_h_{}_p_{}_s_{}_v{}.0'.format(
                    replace_hyp, replace_prem, do_sample, version )) as file:
                valid = json.load(file)
        else:
            with open(path + 'normal_flute_train') as file:
                train = json.load(file)
            with open(path + 'normal_flute_valid') as file:
                valid = json.load(file)

        if training:
            self.data =train
        else:
            self.data= valid

        self.label_2_index={'Contradiction':0,'Entailment':1}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        premise=sample['premise']
        hypothesis=sample['hypothesis']
        label = self.label_2_index[sample['label']]

        return {'premise':premise, 'hypothesis':hypothesis,'label':label }
class FluteCollator:

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.eos_token=self.tokenizer.eos_token

    def collator(self, batch):
        premise_b = []
        hypothesis_b = []
        label_b=[]
        for sample in batch:
            premise = sample['premise']
            hypothesis = sample['hypothesis']
            label = sample['label']
            hypothesis_b.append(hypothesis)
            premise_b.append(premise)
            label_b.append(label)

        tokens=self.tokenizer(premise_b,hypothesis_b , padding=True, return_token_type_ids=True, return_tensors="pt")
        tokens['labels']= torch.Tensor(label_b)
        return tokens

    def __call__(self, batch):
       return self.collator(batch)

def prepare_flute(path, version, real, batch_size,do_sample, replace_hyp, replace_prem, tokenizer):
    # Load the Fig-Questions dataset

    # Create train, validation, and test data loaders
    train_dataset = VisFluteDataset(path, version,real,do_sample, replace_hyp=replace_hyp, replace_prem=replace_prem, training=True)
    valid_dataset = VisFluteDataset(path, version, real,do_sample, replace_hyp=replace_hyp, replace_prem=replace_prem, training=False)
    data_collator =FluteCollator(tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,collate_fn=data_collator,num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size,collate_fn=data_collator,num_workers=4)
    return  train_loader, valid_loader


def generate_from_sentences(text, model, tokenizer,device,do_sample,epsilon_cutoff=.0001,temperature=1):
    text=[tokenizer.eos_token +  i + tokenizer.eos_token for i in text]
    batch=tokenizer(text, padding=True, return_tensors="pt")

    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    max_prompt_length=50
    #max_length=input_ids.shape[1] + max_prompt_length
    generated_ids =model.generate(input_ids=input_ids,attention_mask=attention_mask, max_new_tokens=max_prompt_length ,do_sample=do_sample,epsilon_cutoff=epsilon_cutoff,temperature=temperature)
    pred_caps = tokenizer.batch_decode(generated_ids[:,-(generated_ids.shape[1] -input_ids.shape[1]):], skip_special_tokens=True)

    # for prompt in pred_caps:
    #     print(prompt)


    return pred_caps
