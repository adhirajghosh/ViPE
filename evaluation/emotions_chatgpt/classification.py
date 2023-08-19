import os

import torch
from transformers import BertTokenizer, BertForSequenceClassification, GPT2LMHeadModel, GPT2Tokenizer
from torch.utils.data import DataLoader
from datasets import load_dataset
from datasets import Dataset
from utils import visualizer, get_batch, save_s_json,get_chatgpt_text,update_dataset_chatgpt
import json
# Load the dataset
dataset = load_dataset('dair-ai/emotion')
# Split the dataset into train and validation sets
train_dataset = dataset['train']
valid_dataset = dataset['test']

device='cuda'

do_sample=False
generate_new_data=False
illus=False

#checkpoint_name = 'gpt2-large_context_ctx_1_lr_5e-05-v3.ckpt'
use_visual_data=True
use_chatgpt=True
chat_gpt_random=False
shuffle=False
os.environ['CUDA_VISIBLE_DEVICES']='1'

path_to_jsons = '/home/shahmoha/PycharmProjects/chatgpt/visual_emotions'
if chat_gpt_random:
    path_to_jsons='/home/shahmoha/PycharmProjects/chatgpt/visual_emotions_random'

model_name='gpt2-medium'
device='cuda'
checkpoint_name = '{}_context_ctx_3_lr_5e-05-v4'.format(model_name)
if use_chatgpt:
    checkpoint_name='chat_gpt'

saving_dir = '/graphics/scratch2/staff/Hassan/checkpoints/lyrics_to_prompts/vis_emotion/Vis_emotions_chatgpt/vis_{}_shuffle_{}_chatgpt_{}_random_{}/{}/'.format(
    use_visual_data, shuffle, use_chatgpt, chat_gpt_random, checkpoint_name)

# Create the directory if it does not exist
if not os.path.exists(saving_dir):
    os.makedirs(saving_dir)


from utils import SingleCollator

#
# if generate_new_data:
#
#     check_point_path='/graphics/scratch2/staff/Hassan/checkpoints/lyrics_to_prompts/saved_models_mine/{}.ckpt/'.format(checkpoint_name)
#
#     model = GPT2LMHeadModel.from_pretrained(check_point_path)
#     model.to(device)
#     tokenizer =  GPT2Tokenizer.from_pretrained(model_name)
#     tokenizer.pad_token=tokenizer.eos_token
#     batch_size=256
#
#     text_train=[]
#     for num, batch in enumerate(get_batch(train_dataset['text'],batch_size)):
#         if num %50:
#             print(num, 'out of ', len(train_dataset['text'])/batch_size)
#         text_train.extend(visualizer(batch, model,tokenizer, device, do_sample, epsilon_cutoff=.0005, temperature=1.1))
#
#     save_s_json(saving_dir, 'vis_emotion_train_sample_{}_{}'.format(do_sample,checkpoint_name),text_train)
#     print('saved training data')
#
#     text_valid=[]
#     for batch in get_batch(valid_dataset['text'], batch_size):
#         text_valid.extend(visualizer(batch, model,tokenizer, device, do_sample, epsilon_cutoff=.0005, temperature=1.1))
#     save_s_json(saving_dir, 'vis_emotion_test_sample_{}_{}'.format(do_sample,checkpoint_name),text_valid)
#     print('saved valid data')

if  use_chatgpt:
    # get chat gpt prompts
    prompt_list_valid = get_chatgpt_text(path_to_jsons + '_valid/')
    # update the dataset
    dataset_valid = update_dataset_chatgpt(dataset, prompt_list_valid, 'test')

    save_s_json(saving_dir, 'vis_emotion_test_sample_{}_{}'.format(do_sample, checkpoint_name), dataset_valid)
    print('saved valid data')


   #get chat gpt prompts
    prompt_list_train=get_chatgpt_text(path_to_jsons + '_train')
    #update the dataset
    dataset_train=update_dataset_chatgpt(dataset, prompt_list_train, 'train')

    save_s_json(saving_dir, 'vis_emotion_train_sample_{}_{}'.format(do_sample, checkpoint_name), dataset_train)
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

if use_visual_data:
    print('loading the visual version of the data')

    with open(saving_dir + 'vis_emotion_train_sample_{}_{}'.format(do_sample, checkpoint_name)) as file:
        text_train = json.load(file)
    with open(saving_dir + 'vis_emotion_test_sample_{}_{}'.format(do_sample, checkpoint_name)) as file:
        text_valid = json.load(file)

    # Create a new dataset with new data
    vis_valid_dataset = Dataset.from_dict(text_valid)
    vis_train_dataset = Dataset.from_dict(text_train)

    train_dataloader = DataLoader(vis_train_dataset, batch_size=128, shuffle=True, collate_fn=SingleCollator(tokenizer))
    valid_dataloader = DataLoader(vis_valid_dataset, batch_size=128, shuffle=False, collate_fn=SingleCollator(tokenizer))
else:
    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True,collate_fn=SingleCollator(tokenizer))
    valid_dataloader = DataLoader(valid_dataset, batch_size=128, shuffle=False, collate_fn=SingleCollator(tokenizer))

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
        if use_visual_data:
            with open(saving_dir + 'results_illus_{}_results_sample_{}_{}'.format(illus,do_sample, checkpoint_name), 'a') as file:
                file.write(f"Epoch {epoch+1}/{epochs} -\n "
                      f"Train Loss: {train_loss:.4f} -\n "
                      f"Valid Loss: {valid_loss:.4f} -\n "
                      f"Accuracy: {accuracy:.4f}\n")
        else:
            with open(saving_dir + 'results_normal_results_sample_{}_{}'.format(do_sample, checkpoint_name), 'a') as file:
                file.write(f"Epoch {epoch+1}/{epochs} -\n "
                      f"Train Loss: {train_loss:.4f} -\n "
                      f"Valid Loss: {valid_loss:.4f} -\n "
                      f"Accuracy: {accuracy:.4f}\n")

        best_acc=accuracy

        if use_visual_data:
            with open(saving_dir + 'pred_test_visual_best_model_pred_{}'.format(checkpoint_name), 'w') as file:
                 json.dump(list(map(int, all_pred)), file, indent=4)
        else:
            with open(saving_dir + 'pred_test_normal_best_model_pred_{}'.format(checkpoint_name), 'w') as file:
                json.dump(list(map(int, all_pred)), file, indent=4)
