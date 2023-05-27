import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def generate_from_loader(valid_gen, model, tokenizer,device):
    name2cap={}
    for cb, (batch, (contexts, prompts)) in enumerate(valid_gen):
        if cb % 2 == 0:
            print(cb, ' out of ', len(valid_gen))

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch['token_type_ids'].to(device)

        # Set token type IDs for the prompts
        max_prompt_length=35
        # token_type_ids[attention_mask == 0]=1
        #prompts_token_type_ids = torch.ones(token_type_ids.shape[0],max_prompt_length, dtype=torch.long).to(device)
        ##Extend token_type_ids to cover the prompt segment
        #token_type_ids = torch.cat((token_type_ids, prompts_token_type_ids),dim=-1)

        # token_type_ids = torch.tensor(
        #     [[0] * input_ids.shape[1] + [1] * max_prompt_length for _ in range(input_ids.shape[0])]).to(device)
        # labels = input_ids.clone()
        #pred_caps_1=gen(model, batch,tokenizer)
        max_length=input_ids.shape[1] + max_prompt_length
        generated_ids = model.generate(input_ids=input_ids,attention_mask=attention_mask, max_length=max_length)
        #generated_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                      # token_type_ids=token_type_ids, max_length=max_length)
        pred_caps = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        for c, (context, pred_cap) in enumerate(zip(contexts,pred_caps)):

            #name2cap[name] = cap
            print('context: ', context)
            if ';' in context:
                print('pred: ', pred_cap.split(context.split(';')[-1])[1])
            else:
                print('pred: ', pred_cap.split(context.split('<|endoftext|>')[-1])[1])
            print('true:',prompts[c] )
            print('\n\n')

            #print('GT : ', tokenizer.sequences_to_texts(batch['caption'])[c])
            # img = io.imread(os.path.join(BASE_DATASET_DIR, 'valid_images', name[0:-3]+'png'))
            #     # img = io.imread(os.path.join(BASE_DATASET_DIR, 'val_images', name))
            #     #
            #     # #img = image.load_img(os.path.join(BASE_DATASET_DIR, 'valid_images', name[0:-3]+'png'))
            #     # # img.show()
            # plt.imshow(img)
            #     #
            # plt.show()
    return name2cap


def generate_from_sentences(text, model, tokenizer,device):
    text=[tokenizer.eos_token +  i + tokenizer.eos_token for i in text]
    batch=tokenizer(text, padding=True, return_tensors="pt")

    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    #token_type_ids = batch['token_type_ids'].to(device)

    # Set token type IDs for the prompts
    max_prompt_length=35
    # token_type_ids[attention_mask == 0]=1
    #prompts_token_type_ids = torch.ones(token_type_ids.shape[0],max_prompt_length, dtype=torch.long).to(device)
    ##Extend token_type_ids to cover the prompt segment
    #token_type_ids = torch.cat((token_type_ids, prompts_token_type_ids),dim=-1)

    # token_type_ids = torch.tensor(
    #     [[0] * input_ids.shape[1] + [1] * max_prompt_length for _ in range(input_ids.shape[0])]).to(device)
    # labels = input_ids.clone()
    #pred_caps_1=gen(model, batch,tokenizer)
    max_length=input_ids.shape[1] + max_prompt_length
    generated_ids = model.generate(input_ids=input_ids,attention_mask=attention_mask, max_length=max_length, do_sample=True)
    #generated_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                  # token_type_ids=token_type_ids, max_length=max_length)
    pred_caps = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    for prompt in pred_caps:
        print(prompt)

    #return pred_caps


class Dataset(Dataset):

    def __init__(self, data_dir, context_size, training=True):

        data=pd.read_csv(data_dir)
        data = data.sample(frac=1,random_state=0).reset_index(drop=True)

        self.context_size = context_size
        self.ids_2_sample={}
        self.keys=[str(i)+':'+str(j) for i,j in zip(data['ids'],data['gpt_ids'])]
        values=zip(data['lyrics'], data['prompts'])
        self.ids_2_sample={k:v for k,v in zip(self.keys,values)}

        if not training:
            valid_index=int(.10 * len(self.ids_2_sample))
            self.keys = list(self.ids_2_sample.keys())[0:valid_index]

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):

        key= self.keys[idx]
        context,prompt = self.ids_2_sample[key]
        context = str(context) # a couple of nan cases exist but that should not matter give the size of the dataset

        #extend the context by context size
        for c in range(self.context_size,0,-1):
            key_id, key_gpt_id=key.split(':')
            key_id=int(key_id)
            key_id -= c
            potential_key='{}:{}'.format(key_id,key_gpt_id)

            if potential_key in self.ids_2_sample:
                context = str(self.ids_2_sample[potential_key][0]) + ' ; ' + context

        return context, prompt


class ContextAwareDataCollator:

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.eos_token=self.tokenizer.eos_token

    def collator(self, batch):
        prompts = []
        contexts = []
        for context, prompt in batch:
            prompts.append( self.eos_token + prompt + self.eos_token )
            contexts.append(self.eos_token + context )

        tokens=self.tokenizer(contexts,prompts , padding=True, return_token_type_ids=True, return_tensors="pt")

        return tokens

    def __call__(self, batch):
       return self.collator(batch)


class ContextAwareDataCollatorForGeneration:

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.eos_token=self.tokenizer.eos_token

    def collator(self, batch):
        prompts = []
        start_tokens=[]
        contexts = []

        for context, prompt in batch:
            start_tokens.append( self.eos_token )
            contexts.append(self.eos_token + context )
            prompts.append(prompt)


        tokens=self.tokenizer(contexts,start_tokens,padding=True, return_token_type_ids=True, return_tensors="pt")
        return tokens,(contexts,prompts)

    def __call__(self, batch):
       return self.collator(batch)

# test
from tqdm import tqdm
# train_dataset =Dataset('/graphics/scratch2/staff/Hassan/genius_crawl/lyrics_to_prompts.csv',context_size=5,training=True)
# for i, j in tqdm(train_dataset):
#     pass
    # try:
    #    d= '' + j
    # except:
    #     print(i)
    #     print(j)
    #     print('\n\n')

# from transformers import GPT2Tokenizer
# tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# tokenizer.pad_token = tokenizer.eos_token
# data_collator = ContextAwareDataCollator(tokenizer)
#
# train_dataloader = DataLoader(train_dataset, batch_size=12,
#                               shuffle=True, num_workers=2, collate_fn=data_collator)
#
# for batch in tqdm(train_dataloader):
#     pass
#     print(batch)
# "You and I will never be the same ; I thought you have learned your lesson by now ; You can hide, and say you're not to blame ; You say you try, but you never change ; You and I will never be the same ; You say you try, but you never change"
# ' A person tied up in chains, struggling to break free'