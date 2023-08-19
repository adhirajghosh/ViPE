import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from datasets import load_dataset
#train_loader, valid_loader, test_loader = prepare_fig_qa(batch_size=32)
from tqdm import tqdm

from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
import numpy as np
import json
from utils import prepare_flute



import torch
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification, BertTokenizer, AdamW
from transformers import get_linear_schedule_with_warmup
import torch.nn.functional as F

# Set device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define hyperparameters
epochs = 10
batch_size = 32
learning_rate = 5e-5
epsilon = 1e-8

version=1
Real=False
do_sample=False

replace_prem, replace_hyp=True, False

#TRUe, False: 75
#true, true: 60
# False True, 63

#real data 86

checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
train_dataloader, valid_dataloader=prepare_flute('data/vis_flute/',version,Real,32,do_sample,replace_hyp, replace_prem,tokenizer)
# Load the pre-trained BERT model and tokenizer
#load_pred='/graphics/scratch2/staff/Hassan/checkpoints/lyrics_to_prompts/FLUTE_BERT/synthetic_flute_v1.0/'
model = BertForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
#model = BertForSequenceClassification.from_pretrained(load_pred, num_labels=2)

# Move the model to the device
model = model.to(device)

# Create an optimizer and learning rate scheduler
optimizer = AdamW(model.parameters(), lr=learning_rate, eps=epsilon)
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

check_path='/graphics/scratch2/staff/Hassan/checkpoints/lyrics_to_prompts/FLUTE_BERT/'
# Training loop

best_loss=1e3
for epoch in range(epochs):
    total_loss = 0
    model.train()
    for batch in tqdm(train_dataloader):
        # Get input tokens and labels
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels']
        labels = F.one_hot(labels.to(torch.int64), num_classes=2).float().to(device)
        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        # Backward pass and optimization
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    # Calculate average loss for the epoch
    avg_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch + 1} - Average Loss: {avg_loss:.4f}")

    # Evaluation loop
    model.eval()
    total_correct = 0
    total_samples = 0

    eval_loss=0
    with torch.no_grad():
        for batch in tqdm(valid_dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            labels = F.one_hot(labels.to(torch.int64), num_classes=2).float().to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            eval_loss +=outputs.loss
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1)

            total_correct += (predictions == labels.argmax(dim=1)).sum().item()
            total_samples += labels.size(0)

    accuracy = total_correct / total_samples
    print(f"Eval Accuracy: {accuracy:.4f}")

    if (eval_loss/len(valid_dataloader) < best_loss):
        best_loss = eval_loss
        print('better val loss', best_loss)
        # if Real:
        #     model.save_pretrained(check_path + 'real_flute')
        # else:
        #     model.save_pretrained(check_path + 'synthetic_flute_h_{}_p_{}_s{}_v{}.0'.format(replace_hyp, replace_prem, do_sample, version))

