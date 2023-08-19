
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




def calculate_bert_score(predictions, references):

    # Compute BERTScore
    P, R, F1 = score(predictions, references, lang='en', verbose=True,device='cuda',batch_size=1500,nthreads=8)

    # Print the BERTScore
    pred=f"BERTScore: P={P.mean().item():.4f}, R={R.mean().item():.4f}, F1={F1.mean().item():.4f}"
    print(pred)
    return F1


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def compute_cosine_similarity(embeddings_1, embeddings_2):
    # Compute cosine similarity between embeddings_1 and embeddings_2
    similarities = cosine_similarity(embeddings_1, embeddings_2)

    return similarities


class SentenceDataset(Dataset):
    def __init__(self, sentences):
        self.sentences = sentences

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        return self.sentences[index]

def collate_fn(batch):
    # Tokenize sentences
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
    encoded_inputs = tokenizer(batch, padding=True, truncation=True, return_tensors='pt')

    return encoded_inputs

def get_mpnet_embed_batch(predictions, ground_truth, batch_size=10000):

    sentences_1=predictions
    sentences_2=ground_truth
    # Create data loaders
    dataset_1 = SentenceDataset(sentences_1)
    dataset_2 = SentenceDataset(sentences_2)

    dataloader_1 = DataLoader(dataset_1, batch_size=batch_size, collate_fn=collate_fn,num_workers=4)
    dataloader_2 = DataLoader(dataset_2, batch_size=batch_size, collate_fn=collate_fn,num_workers=4)

    # Load model from HuggingFace Hub
    model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Compute token embeddings
    embeddings_1 = []
    embeddings_2 = []

    with torch.no_grad():
        for count, (batch_1, batch_2) in enumerate(zip(dataloader_1, dataloader_2)):
            if count % 5 ==0:
                print(count,' out of ',len(dataloader_2))
            batch_1 = {key: value.to(device) for key, value in batch_1.items()}
            batch_2 = {key: value.to(device) for key, value in batch_2.items()}

            model_output_1 = model(**batch_1)
            model_output_2 = model(**batch_2)

            sentence_embeddings_1 = mean_pooling(model_output_1, batch_1['attention_mask'])
            sentence_embeddings_2 = mean_pooling(model_output_2, batch_2['attention_mask'])

            embeddings_1.append(sentence_embeddings_1)
            embeddings_2.append(sentence_embeddings_2)

    # Concatenate embeddings
    embeddings_1 = torch.cat(embeddings_1)
    embeddings_2 = torch.cat(embeddings_2)

    # Normalize embeddings
    embeddings_1 = torch.nn.functional.normalize(embeddings_1, p=2, dim=1)
    embeddings_2 = torch.nn.functional.normalize(embeddings_2, p=2, dim=1)

    # Compute cosine similarity
    similarities = compute_cosine_similarity(embeddings_1, embeddings_2)


    return similarities

