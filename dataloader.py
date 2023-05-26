import pandas as pd
import numpy as np
import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import T5Tokenizer, T5ForConditionalGeneration

raw_dataset = pd.read_csv('lyrics_to_prompts.csv', encoding='latin-1')

class MyDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file,encoding='latin-1')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Return a dictionary containing the desired columns
        sample = {
            'id': self.data['id'][idx],
            'gpt_id': self.data['gpt_id'][idx],
            'artist': self.data['artist'][idx],
            'song': self.data['song'][idx],
            'lyric': self.data['lyric'][idx],
            'prompt': self.data['prompt'][idx]
        }
        return sample

# Load the dataset
dataset = MyDataset('lyrics_to_prompts.csv')

# Split the dataset into train and test sets
train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)

# Create DataLoaders for train and test sets
batch_size = 32
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")
max_input_length = 128
max_target_length = 128
prefix = ""

def preprocess_function(examples):
    inputs = [prefix + ex['lyric'] for ex in examples]
    targets = [ex['prompt'] for ex in examples]
    print(targets)
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

print(preprocess_function(train_data[:2]))