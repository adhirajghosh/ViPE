import os

import torch
from transformers import BertTokenizer, BertForSequenceClassification, GPT2LMHeadModel, GPT2Tokenizer
from torch.utils.data import DataLoader
from datasets import load_dataset
from datasets import Dataset
from utils import visualizer, get_vis_batch, save_s_json, get_vis_flute_samples,update_dataset,get_haivment_prompts
import json
import random
# Load the dataset
dataset = load_dataset("ColumbiaNLP/FLUTE")

import random
from datasets import load_dataset, DatasetDict

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



device='cuda'
os.environ['CUDA_VISIBLE_DEVICES']='1'
HAIVMet_Dir='/graphics/scratch2/staff/Hassan/datasets/HAIVMet/flute.zip'
vis_samples=get_vis_flute_samples(HAIVMet_Dir)

print(' found ', len(vis_samples), ' to be replaced in the dataset')
do_sample=False # do sampling while generating prompt with our model?
generate_new_data=True # my model
use_visual_data=True # my data or haivmet data

generate_haivment_data=False # calculate which samples are haivment data
Use_HAIVMet_prompts=False # set to false to use your generated prompt


illus=False # does not do much !
shuffle= False # shuffle the prompt to see what happens!


model_name='gpt2-medium'
checkpoint_name = '{}_context_ctx_7_lr_5e-05-v4'.format(model_name)

if Use_HAIVMet_prompts:
    checkpoint_name='humans'

saving_dir='/graphics/scratch2/staff/Hassan/checkpoints/lyrics_to_prompts/vis_emotion/Vis_FLUTE/vis_{}_shuffle_{}_haivmet_{}/{}/'.format(use_visual_data,shuffle,Use_HAIVMet_prompts,checkpoint_name)

# Create the directory if it does not exist
if not os.path.exists(saving_dir):
    os.makedirs(saving_dir)


#checkpoint_name = 'gpt2-large_context_ctx_1_lr_5e-05-v3.ckpt'


from utils import SingleCollator, DoubleCollator

if generate_new_data:

    check_point_path='/graphics/scratch2/staff/Hassan/checkpoints/lyrics_to_prompts/saved_models_mine/{}.ckpt/'.format(checkpoint_name)


    model = GPT2LMHeadModel.from_pretrained(check_point_path)
    model.to(device)
    tokenizer =  GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token=tokenizer.eos_token
    batch_size=256

    # to be replaced by our visual elaboration
    text_train={'ids':[], 'text_type':[], 'vis_text':[], 'text':[] }
    #ids: id of the sample,
    # text type: eaiher premise or hypothesis
    # sample, the samplle to be transformed
    count=0
    for num, ( ids, text_type, samples) in enumerate(get_vis_batch(train_dataset,batch_size, vis_samples)):
        if num %50:
            print(num, 'out of ', int(len(train_dataset)/batch_size))
        if len(ids) > 0:
            count += len(ids)
            text_train['vis_text'].extend(visualizer(samples, model,tokenizer, device, do_sample, epsilon_cutoff=.0005, temperature=1.1))
            text_train['ids'].extend(ids)
            text_train['text_type'].extend(text_type)
            text_train['text'].extend(samples)

    save_s_json(saving_dir, 'vis_flute_train_sample_{}_{}'.format(do_sample,checkpoint_name),text_train)
    print('saved training data')

    # to be replaced by our visual elaboration
    text_valid = {'ids': [], 'text_type': [], 'vis_text': [], 'text':[]}
    # ids: id of the sample,
    # text type: eaiher premise or hypothesis
    # sample, the samplle to be transformed
    for num, (ids, text_type, samples) in enumerate(get_vis_batch(valid_dataset, batch_size, vis_samples)):
        if num % 50:
            print(num, 'out of ', int(len(valid_dataset) / batch_size))
        if len(ids)>0:
            count += len(ids)
            text_valid['vis_text'].extend(
                visualizer(samples, model, tokenizer, device, do_sample, epsilon_cutoff=.0005, temperature=1.1))
            text_valid['ids'].extend(ids)
            text_valid['text_type'].extend(text_type)
            text_valid['text'].extend(samples)

    save_s_json(saving_dir, 'vis_flute_valid_sample_{}_{}'.format(do_sample, checkpoint_name), text_valid)
    print('saved validation data')

    print(' converted ', count, ' number of samples')

# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

if use_visual_data and not Use_HAIVMet_prompts:
    print('loading the visual version of the data')
    with open(saving_dir + 'vis_flute_train_sample_{}_{}'.format(do_sample, checkpoint_name)) as file:
        vis_train = json.load(file)
    with open(saving_dir + 'vis_flute_valid_sample_{}_{}'.format(do_sample, checkpoint_name)) as file:
        vis_valid = json.load(file)

    if shuffle:
        random.shuffle(vis_valid['vis_text'])
        random.shuffle(vis_train['vis_text'])

    print('number of unique sentences that got replaced: ',
          len(set(set(vis_train['text']).union(set(vis_valid['text'])))))
    vis_train_dataset = update_dataset(train_dataset, vis_train)
    vis_valid_dataset = update_dataset(valid_dataset, vis_valid)
    """
        replaced  1355  samples
        replaced  345  samples

    """
    train_dataloader = DataLoader(vis_train_dataset, batch_size=128, shuffle=True, collate_fn=DoubleCollator(tokenizer))
    valid_dataloader = DataLoader(vis_valid_dataset, batch_size=128, shuffle=False, collate_fn=DoubleCollator(tokenizer))
elif use_visual_data and Use_HAIVMet_prompts:

    """
    number of unique sentences that got replaced 782
    replaced  1355  samples
    replaced  345  samples

    """
    if generate_haivment_data:
        HAIVMet_Dir = '/graphics/scratch2/staff/Hassan/datasets/HAIVMet/flute.zip'
        vis_valid=get_haivment_prompts(HAIVMet_Dir,vis_samples,valid_dataset )
        vis_train = get_haivment_prompts(HAIVMet_Dir, vis_samples, train_dataset)

        save_s_json(saving_dir, 'vis_flute_train_sample_{}_{}'.format(do_sample, checkpoint_name), vis_train)
        save_s_json(saving_dir, 'vis_flute_valid_sample_{}_{}'.format(do_sample, checkpoint_name), vis_valid)

    with open(saving_dir + 'vis_flute_train_sample_{}_{}'.format(do_sample, checkpoint_name)) as file:
        vis_train = json.load(file)
    with open(saving_dir + 'vis_flute_valid_sample_{}_{}'.format(do_sample, checkpoint_name)) as file:
        vis_valid = json.load(file)


    print('number of unique sentences that got replaced',
          len(set(set(vis_train['text']).union(set(vis_valid['text'])))))

    if shuffle:
        random.shuffle(vis_valid['vis_text'])
        random.shuffle(vis_train['vis_text'])

    vis_train_dataset = update_dataset(train_dataset, vis_train)
    vis_valid_dataset = update_dataset(valid_dataset, vis_valid)
    """
    found  1148  to be replaced in the dataset
    number of unique sentences that got replaced 782
    replaced  1355  samples
    replaced  345  samples

    """
    train_dataloader = DataLoader(vis_train_dataset, batch_size=128, shuffle=True, collate_fn=DoubleCollator(tokenizer))
    valid_dataloader = DataLoader(vis_valid_dataset, batch_size=128, shuffle=False,
                                  collate_fn=DoubleCollator(tokenizer))
else:

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
