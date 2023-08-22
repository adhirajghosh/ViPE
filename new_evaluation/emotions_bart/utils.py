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

    # for id in ids:
    # d=3
    # vis_valid_dataset = Dataset.from_dict({'text': text_valid, 'label': valid_dataset['label']})
    # vis_train_dataset = Dataset.from_dict({'text': text_train, 'label': train_dataset['label']})


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

def bart_paraphrase(text, model, tokenizer,device,do_sample,epsilon_cutoff=.0001,temperature=1.0):

    text=["paraphrase: " + i + " </s>" for i in text]
    batch=tokenizer(text, padding=True, return_tensors="pt", truncation=True)

    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    max_prompt_length=50

    #generated_ids =model.generate(input_ids=input_ids,attention_mask=attention_mask, max_new_tokens=max_prompt_length ,do_sample=do_sample,epsilon_cutoff=epsilon_cutoff,temperature=temperature)
    #pred_caps = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    generated_ids = model.generate(
        input_ids=input_ids, attention_mask=attention_mask,
        max_new_tokens=max_prompt_length,
        do_sample=do_sample,
        early_stopping=True,
        temperature=temperature,
        num_return_sequences=1
    )

    pred_caps = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    return pred_caps

def blue1(reference_sentences, hypothesis_sentences):
    from nltk.translate.bleu_score import corpus_bleu
    from nltk.tokenize import word_tokenize


    # Tokenize the reference sentences
    reference_sentences_tokenized = [[word_tokenize(sent) for sent in refs] for refs in reference_sentences]

    # Tokenize the hypothesis sentences
    hypothesis_sentences_tokenized = [word_tokenize(sent) for sent in hypothesis_sentences]

    # Calculate BLEU-1 score
    bleu_score = corpus_bleu(reference_sentences_tokenized, hypothesis_sentences_tokenized,
                             weights=(1, 0, 0, 0))


    return bleu_score



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