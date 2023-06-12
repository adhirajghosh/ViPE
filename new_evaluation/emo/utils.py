
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
   # text = text
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