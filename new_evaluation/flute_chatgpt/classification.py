import os

import datasets
import torch
from transformers import BertTokenizer, BertForSequenceClassification, GPT2LMHeadModel, GPT2Tokenizer
from torch.utils.data import DataLoader
from datasets import load_dataset
from datasets import Dataset
from utils import visualizer, get_vis_batch, save_s_json, get_batch,update_dataset_chatgpt,get_haivment_prompts,get_chatgpt_hypothesis
import json
import random
from tqdm import  tqdm

# Load the dataset
dataset = load_dataset("ColumbiaNLP/FLUTE")

os.environ['CUDA_VISIBLE_DEVICES']='1'
use_chatgpt=True
chat_gpt_random=False # set false to use sampled obtained from the deterministic  chatgpt (temperature =0)

do_sample=False
generate_new_data=False # my model
use_visual_data=True # my data or chatgpt data

illus=False # does not do much !
shuffle= False # shuffle the prompt to see what happens!

path_to_jsons = '/home/shahmoha/PycharmProjects/chatgpt/visual_flute/'
if chat_gpt_random:
    path_to_jsons='/home/shahmoha/PycharmProjects/chatgpt/visual_flute_random/'

model_name='gpt2-medium'
device='cuda'
checkpoint_name = '{}_context_ctx_3_lr_5e-05-v4'.format(model_name)
if use_chatgpt:
    checkpoint_name='chat_gpt'


saving_dir='/graphics/scratch2/staff/Hassan/checkpoints/lyrics_to_prompts/vis_emotion/Vis_FLUTE_chatgpt/vis_{}_shuffle_{}_chatgpt_{}_random_{}/{}/'.format(use_visual_data,shuffle,use_chatgpt,chat_gpt_random,checkpoint_name)

# Create the directory if it does not exist
if not os.path.exists(saving_dir):
    os.makedirs(saving_dir)

from utils import get_vis_flute_samples, update_dataset_chatgpt_haivmet
HAIVMet_Dir='/graphics/scratch2/staff/Hassan/datasets/HAIVMet/flute.zip'
vis_samples=get_vis_flute_samples(HAIVMet_Dir)

from utils import save_chat_gpt_prompt
if  use_chatgpt:

   #get chat gpt prompts
    prompt_list=get_chatgpt_hypothesis(path_to_jsons)

    save_chat_gpt_prompt(dataset, prompt_list, vis_samples, [saving_dir, 'vis_flute_sample_{}_{}'.format(do_sample, checkpoint_name)])
    #update the dataset
    #dataset=update_dataset_chatgpt(dataset, prompt_list)
    if shuffle:
       random.shuffle(prompt_list)


    dataset =update_dataset_chatgpt_haivmet(dataset, prompt_list, vis_samples)

if use_visual_data and not use_chatgpt:

        # use your generated prompts
        if generate_new_data:

            from datasets import DatasetDict
            check_point_path = '/graphics/scratch2/staff/Hassan/checkpoints/lyrics_to_prompts/saved_models_mine/{}.ckpt/'.format(
                checkpoint_name)

            model = GPT2LMHeadModel.from_pretrained(check_point_path)
            model.to(device)
            tokenizer = GPT2Tokenizer.from_pretrained(model_name)
            tokenizer.pad_token = tokenizer.eos_token
            batch_size = 256

            # to be replaced by our visual elaboration
            text_train = {}
            # ids: id of the sample,
            # text type: eaiher premise or hypothesis
            # sample, the samplle to be transformed
            count = 0

            new_dataset = {k: [] for k in dataset['train'].features.keys()}
            # Iterate over the dataset examples
            for batch in tqdm(get_batch(dataset['train'], batch_size)):

                prompt_list = visualizer(batch['hypothesis'], model, tokenizer, device, do_sample,
                                         epsilon_cutoff=.0005, temperature=1.1)

                batch['hypothesis'] = prompt_list
                for k in batch.keys():
                    if not k in text_train:
                        text_train[k] = []

                    text_train[k].extend(batch[k])

            dataset = DatasetDict({'train': Dataset.from_dict(text_train)})

            save_s_json(saving_dir, 'vis_flute_sample_{}_{}'.format(do_sample, checkpoint_name), text_train)
            print('saved training data')

        with open(saving_dir + 'vis_flute_sample_{}_{}'.format(do_sample, checkpoint_name)) as file:
            vis_train = json.load(file)

        if shuffle:
            random.shuffle(vis_train['vis_text'])

        # for num, (m, ch) in enumerate(zip(vis_train['hypothesis'], prompt_list)):
        #     print('truth: ', dataset['train']['hypothesis'][num], '  mine: ', m, '  chatgpt: ', ch)
        dataset = DatasetDict({'train': Dataset.from_dict(vis_train)})


import random
from datasets import  DatasetDict

# Define the type frequencies for validation
type_frequencies = {
    'Idiom': 250,
    'Metaphor': 250,
    'Simile': 250,
    'Sarcasm,CreativeParaphrase': 750
}

# Set the random seed for reproducibility
seed = 42
random.seed(seed)

# Create an empty list to store the sampled examples (validation set)
sampled_examples = []

# Sample from each type according to the frequencies
for type_name, freq in type_frequencies.items():

    if len(type_name.split(',')) < 2:
        # Filter the dataset for the current type
        filtered_dataset = dataset['train'].filter(lambda example: example['type'] == type_name)
        filtered_dataset = list(filtered_dataset)
        # Randomly sample 'freq' number of examples from the filtered dataset
        sampled_examples.extend(random.sample(filtered_dataset, freq))
    else:
        t1, t2 = type_name.split(',')
        # Filter the dataset for the current type
        filtered_dataset = dataset['train'].filter(lambda example: example['type'] == t1 or example['type'] == t2  )
        # Randomly sample 'freq' number of examples from the filtered dataset
        filtered_dataset = list(filtered_dataset)
        sampled_examples.extend(random.sample(filtered_dataset, freq))

# Shuffle the sampled examples
random.shuffle(sampled_examples)

# Get the IDs of the sampled validation examples
validation_ids = [example['id'] for example in sampled_examples]
# Create the training set by removing the validation examples from the original dataset
train_examples = dataset['train'].filter(lambda example: example['id'] not in validation_ids)

# create the valid dataset
# Create an empty dictionary to store the dataset columns
valid_dataset = {}

# Iterate over each dictionary in the list
for example in sampled_examples:
    # Iterate over each key-value pair in the dictionary
    for key, value in example.items():
        # If the key is not present in the dataset dictionary, initialize it as an empty list
        if key not in valid_dataset:
            valid_dataset[key] = []
        # Append the value to the corresponding key in the dataset dictionary
        valid_dataset[key].append(value)

# Create the dataset from the dictionary
valid_dataset = Dataset.from_dict(valid_dataset)


# Create a new dataset with the updated train and validation sets
dataset = DatasetDict({
    'train': train_examples,
    'validation': valid_dataset,
})

# Print the number of examples in the train and validation sets
print("Number of examples in the train set:", len(dataset['train']))
print("Number of examples in the validation set:", len(dataset['validation']))



# Split the dataset into train and validation sets
train_dataset = dataset['train']
valid_dataset = dataset['validation']



from utils import  DoubleCollator

# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True,collate_fn=DoubleCollator(tokenizer))
valid_dataloader = DataLoader(valid_dataset, batch_size=128, shuffle=False, collate_fn=DoubleCollator(tokenizer))


num_labels=2
# Load the BERT model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)

# Set up GPU training if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Set up the optimizer and loss function
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
loss_fn = torch.nn.CrossEntropyLoss()
import torch.nn.functional as F
# Training loop
epochs = 10

best_acc=0
for epoch in range(epochs):
    model.train()
    train_loss = 0.0

    for batch in train_dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        labels = F.one_hot(labels.to(torch.int64), num_classes=num_labels).float().to(device)
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

            labels_onehot = F.one_hot(labels.to(torch.int64), num_classes=num_labels).float().to(device)
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
            with open(saving_dir + 'test_illus_{}_results_sample_{}_{}'.format(illus,do_sample, checkpoint_name), 'a') as file:
                file.write(f"Epoch {epoch+1}/{epochs} -\n "
                      f"Train Loss: {train_loss:.4f} -\n "
                      f"Valid Loss: {valid_loss:.4f} -\n "
                      f"Accuracy: {accuracy:.4f}\n")
        else:
            with open(saving_dir + 'test_normal_illus_{}_results_sample_{}_{}'.format(illus,do_sample, checkpoint_name), 'a') as file:
                file.write(f"Epoch {epoch+1}/{epochs} -\n "
                      f"Train Loss: {train_loss:.4f} -\n "
                      f"Valid Loss: {valid_loss:.4f} -\n "
                      f"Accuracy: {accuracy:.4f}\n")

        best_acc=accuracy

        if use_visual_data:
            with open(saving_dir + 'test_visual_best_model_pred_{}'.format(checkpoint_name), 'w') as file:
                 json.dump(list(map(int, all_pred)), file, indent=4)
        else:
            with open(saving_dir + 'test_normal_best_model_pred_{}'.format(checkpoint_name), 'w') as file:
                json.dump(list(map(int, all_pred)), file, indent=4)
