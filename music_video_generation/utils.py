import pandas as pd
from torch.utils.data import Dataset, DataLoader

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def generate_from_sentences(text, model, tokenizer,device,do_sample):
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
    generated_ids = model.generate(input_ids=input_ids,attention_mask=attention_mask, max_length=max_length, do_sample=do_sample)
    #generated_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                  # token_type_ids=token_type_ids, max_length=max_length)
    pred_caps = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    for prompt in pred_caps:
        print(prompt)
        print('\n')

    return pred_caps