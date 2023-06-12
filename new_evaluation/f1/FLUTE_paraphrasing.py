import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import warnings
warnings.simplefilter("ignore", category=UserWarning)

from tqdm import tqdm
from datasets import load_dataset
import random

HAIVMet_Dir='/graphics/scratch2/staff/Hassan/datasets/HAIVMet/'

# Load the dataset
dataset = load_dataset("ColumbiaNLP/FLUTE")

type2sample={}
# Set the seed for reproducibility
seed = 42  # Choose any seed value you prefer
random.seed(seed)
train_x=[]
valid_x=[]
for type_i, freq in {'Idiom':250, 'Metaphor':250, 'Simile':250, 'Sarcasm,CreativeParaphrase':750}.items():

    if len(type_i.split(','))<2:
        portion=[sample for sample in dataset['train'] if sample['type']==type_i]
        random.shuffle(portion)
        valid_x.extend(portion[0:freq])
        train_x.extend(portion[freq:])
    else:
        t1, t2=type_i.split(',')
        portion = [sample for sample in dataset['train'] if sample['type'] == t1 or sample['type'] == t2 ]
        random.shuffle(portion)
        valid_x.extend(portion[0:freq])
        train_x.extend(portion[freq:])

assert len(dataset['train'])==(len(train_x) + len(valid_x))

#__________________________________________________________________

from transformers import GPT2Tokenizer, GPT2Model,GPT2LMHeadModel
from utils import generate_from_sentences

model_name='gpt2-medium'
device='cuda'
do_sample=True

replace_prem, replace_hyp=False, True


check_point_path='/graphics/scratch2/staff/Hassan/checkpoints/lyrics_to_prompts/saved_models_mine/gpt2-medium_context_ctx_5_lr_5e-05-v4.ckpt/'
#check_point_path='/graphics/scratch2/staff/Hassan/checkpoints/lyrics_to_prompts/saved_models_mine/gpt2-medium_context_ctx_0_lr_5e-05-v3.ckpt'

model = GPT2LMHeadModel.from_pretrained(check_point_path)
model.to(device)
tokenizer =  GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token=tokenizer.eos_token
batch_size=256
import json

def get_batch(data, batch_size):
    for num, index in enumerate(range(0, len(data), batch_size)):
        if (num + 1) * batch_size < len(data):
            samples = data[index:(num + 1) * batch_size]
        else:
            samples = data[index:]

        yield samples

train_x_vis=[]
valid_x_vis=[]

with open('data/vis_flute/normal_flute_train', 'w') as file:
    json.dump(train_x, file, indent=4)
with open('data/vis_flute/normal_flute_valid', 'w') as file:
    json.dump(valid_x, file, indent=4)

#3 times the original datat
size=1
for version in range(1):
    print('creating version number ', version +1)
    for portion, data in {'train':train_x, 'valid':valid_x}.items():

            for batch in tqdm(get_batch(data,batch_size)):
                premise_list=[s['premise'].replace(',',';') for s in batch]
                hypothesis_list = [s['hypothesis'] for s in batch]

                for _ in range(size):

                    if replace_prem:
                        premise_list_vis=generate_from_sentences(premise_list, model, tokenizer,device,do_sample,
                                                           epsilon_cutoff=.0005,temperature=1.1)
                    else:
                        premise_list_vis=premise_list


                    if replace_hyp:
                        hypothesis_list_vis = generate_from_sentences(hypothesis_list, model, tokenizer, device, do_sample,
                                                                   epsilon_cutoff=.0005, temperature=1.1)
                    else:
                        hypothesis_list_vis=hypothesis_list

                    #replace both and add them to the original premises and hypothesis
                    if not replace_prem and not replace_hyp:
                        premise_list_vis = generate_from_sentences(premise_list, model, tokenizer, device, do_sample,
                                                                   epsilon_cutoff=.0005, temperature=1.1)
                        hypothesis_list_vis = generate_from_sentences(hypothesis_list, model, tokenizer, device,
                                                                      do_sample,
                                                                      epsilon_cutoff=.0005, temperature=1.1)

                    for num , sample in enumerate(batch):

                        if not replace_prem and not replace_hyp:
                            sample['premise']= sample['premise'] + ' ' + premise_list_vis[num][1:]
                            sample['hypothesis'] =  sample['hypothesis'] + ' ' + hypothesis_list_vis[num][1:]
                        else:
                            sample['premise'] = premise_list_vis[num][1:]
                            sample['hypothesis'] = hypothesis_list_vis[num][1:]

                    if portion=='train':
                        train_x_vis.extend(batch)
                    else:
                        valid_x_vis.extend(batch)

    with open('data/vis_flute/vis_flute_train_h_{}_p_{}_s_{}_v{}.0'.format(replace_hyp, replace_prem,do_sample,version+1), 'w') as file:
        json.dump(train_x_vis, file, indent=4)
    with open('data/vis_flute/vis_flute_valid_h_{}_p_{}_s_{}_v{}.0'.format(replace_hyp, replace_prem,do_sample,version + 1), 'w') as file:
        json.dump(valid_x_vis, file, indent=4)