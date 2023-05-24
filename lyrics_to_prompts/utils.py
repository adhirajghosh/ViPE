import pandas as pd
from torch.utils.data import Dataset, DataLoader


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__



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
            contexts.append(self.eos_token + context  )

        tokens=self.tokenizer(contexts,prompts , padding=True, return_token_type_ids=True, return_tensors="pt")

        return tokens

    def __call__(self, batch):
       return self.collator(batch)


# test
# train_dataset =Dataset('../',context_size=3,training=True)
# from transformers import GPT2Tokenizer
# tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# tokenizer.pad_token = tokenizer.eos_token
# data_collator = ContextAwareDataCollator(tokenizer)
#
#
# train_dataloader = DataLoader(train_dataset, batch_size=16,
#                               shuffle=True, num_workers=2, collate_fn=data_collator)
#
# for batch in train_dataloader:
#     print(batch)
# "You and I will never be the same ; I thought you have learned your lesson by now ; You can hide, and say you're not to blame ; You say you try, but you never change ; You and I will never be the same ; You say you try, but you never change"
# ' A person tied up in chains, struggling to break free'