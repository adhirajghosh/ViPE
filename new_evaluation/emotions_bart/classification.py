import os

import torch
from transformers import BertTokenizer, BertForSequenceClassification, GPT2LMHeadModel, GPT2Tokenizer
from torch.utils.data import DataLoader
from datasets import load_dataset
from datasets import Dataset
import torch
from transformers import BartForConditionalGeneration, BartTokenizer



from utils import bart_paraphrase, get_batch, save_s_json
import json
# Load the dataset
dataset = load_dataset('dair-ai/emotion')
# Split the dataset into train and validation sets
train_dataset = dataset['train']
valid_dataset = dataset['test']
os.environ['CUDA_VISIBLE_DEVICES']='1'

device='cuda'
model_name='gpt2-medium'
do_sample=True
generate_new_data=True
checkpoint_name='t5'
saving_dir='/graphics/scratch2/staff/Hassan/checkpoints/lyrics_to_prompts/vis_emotion/emotions_{}/'.format(checkpoint_name)


# Create the directory if it does not exist
if not os.path.exists(saving_dir):
    os.makedirs(saving_dir)


from utils import SingleCollator,blue1

vipe_check='{}_context_ctx_7_lr_5e-05-v4'.format('gpt2-medium')
vipe_model_results_dir='/graphics/scratch2/staff/Hassan/checkpoints/lyrics_to_prompts/vis_emotion/Vis_FLUTE/vis_{}_shuffle_{}_haivmet_{}/{}/'.format(True,False,False,vipe_check)
with open(vipe_model_results_dir + 'vis_flute_train_sample_{}_{}'.format(False, vipe_check)) as file:
    vipe_vis_train = json.load(file)
with open(vipe_model_results_dir + 'vis_flute_valid_sample_{}_{}'.format(False, vipe_check)) as file:
    vipe_vis_valid = json.load(file)

print('blue1 score between my version and the original dataset valid: ', blue1(reference_sentences=vipe_vis_valid['text'],hypothesis_sentences=vipe_vis_valid['vis_text']))
print('blue1 score between my version and the original dataset valid: ', blue1(reference_sentences=vipe_vis_train['text'],hypothesis_sentences=vipe_vis_train['vis_text']))


if generate_new_data:
    # model = BartForConditionalGeneration.from_pretrained('eugenesiow/bart-paraphrase')
    # tokenizer = BartTokenizer.from_pretrained('eugenesiow/bart-paraphrase')
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    tokenizer = AutoTokenizer.from_pretrained("Vamsi/T5_Paraphrase_Paws")
    model = AutoModelForSeq2SeqLM.from_pretrained("Vamsi/T5_Paraphrase_Paws").to('cuda')

    model.to(device)

    batch_size = 600
    text_train=[]
    for num, batch in enumerate(get_batch(train_dataset['text'],batch_size)):
        if num %50:
            print(num, 'out of ', len(train_dataset['text'])/batch_size)
        text_train.extend(bart_paraphrase(batch, model,tokenizer, device, do_sample, epsilon_cutoff=None, temperature=1))

    save_s_json(saving_dir, 'emotion_train_sample_{}_{}'.format(do_sample,checkpoint_name),text_train)
    print('saved training data')

    text_valid=[]
    for batch in get_batch(valid_dataset['text'], batch_size):
        text_valid.extend(bart_paraphrase(batch, model,tokenizer, device, do_sample, epsilon_cutoff=.0005, temperature=1))
    save_s_json(saving_dir, 'emotion_test_sample_{}_{}'.format(do_sample,checkpoint_name),text_valid)
    print('saved valid data')


# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# # Tokenize the input text and convert to input tensors
# def tokenize_data(example):
#     encoded_inputs = tokenizer(example['text'], padding=True, truncation=True,return_tensors='pt')
#     return {'input_ids': encoded_inputs['input_ids'], 'attention_mask': encoded_inputs['attention_mask'], 'labels': example['label']}
#
# valid_dataset = valid_dataset.map(tokenize_data, batched=True)
# train_dataset = train_dataset.map(tokenize_data, batched=True)



print('loading the paraphrased version of the data')

with open(saving_dir + 'emotion_train_sample_{}_{}'.format(do_sample, checkpoint_name)) as file:
    text_train = json.load(file)
with open(saving_dir + 'emotion_test_sample_{}_{}'.format(do_sample, checkpoint_name)) as file:
    text_valid = json.load(file)

print('blue1 score between the paraphrase version and the original dataset train: ', blue1(train_dataset['text'],text_train))
print('blue1 score between the paraphrase version and the original dataset valid: ', blue1(reference_sentences=valid_dataset['text'],hypothesis_sentences=text_valid))

# Create a new dataset with new data
vis_valid_dataset = Dataset.from_dict({'text': text_valid, 'label': valid_dataset['label']})
vis_train_dataset = Dataset.from_dict({'text': text_train, 'label': train_dataset['label']})

train_dataloader = DataLoader(vis_train_dataset, batch_size=128, shuffle=True, collate_fn=SingleCollator(tokenizer))
valid_dataloader = DataLoader(vis_valid_dataset, batch_size=128, shuffle=False, collate_fn=SingleCollator(tokenizer))

# Load the BERT model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=6)

# Set up GPU training if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Set up the optimizer and loss function
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
loss_fn = torch.nn.CrossEntropyLoss()
import torch.nn.functional as F
# Training loop
epochs = 5
best_acc=0
for epoch in range(epochs):
    model.train()
    train_loss = 0.0

    for batch in train_dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        labels = F.one_hot(labels.to(torch.int64), num_classes=6).float().to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        train_loss += loss.item()

        loss.backward()
        optimizer.step()

    train_loss /= len(train_dataloader)

    # Validation loop
    model.eval()
    valid_loss = 0.0
    correct = 0
    total = 0

    all_pred=[]
    with torch.no_grad():
        for batch in valid_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            labels_onehot = F.one_hot(labels.to(torch.int64), num_classes=6).float().to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels_onehot)
            loss = outputs.loss
            valid_loss += loss.item()

            _, predicted = torch.max(outputs.logits, dim=1)

            all_pred.extend(list(predicted.cpu().numpy()))
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    valid_loss /= len(valid_dataloader)
    accuracy = correct / total

    print(f"Epoch {epoch+1}/{epochs} - "
          f"Train Loss: {train_loss:.4f} - "
          f"Valid Loss: {valid_loss:.4f} - "
          f"Accuracy: {accuracy:.4f}")

    if accuracy > best_acc:

        with open(saving_dir + 'results_sample_{}_{}'.format(do_sample, checkpoint_name), 'a') as file:
            file.write(f"Epoch {epoch+1}/{epochs} -\n "
                  f"Train Loss: {train_loss:.4f} -\n "
                  f"Valid Loss: {valid_loss:.4f} -\n "
                  f"Accuracy: {accuracy:.4f}\n")


        best_acc=accuracy

        with open(saving_dir + 'pred_test_best_model_pred_{}'.format(checkpoint_name), 'w') as file:
             json.dump(list(map(int, all_pred)), file, indent=4)

