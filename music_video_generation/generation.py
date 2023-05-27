from modeling import GPT2Convertor

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from torch.utils.data import Dataset, DataLoader
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from utils import Dataset, ContextAwareDataCollator,ContextAwareDataCollatorForGeneration
from pytorch_lightning import  Trainer
from modeling import GPT2Convertor
from utils import dotdict, generate_from_loader,generate_from_sentences
import json
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="train hehe?")

    parser.add_argument(
        "--model_name", type=str, default='gpt2-medium', help="which gpt2 version to use?"
    )

    parser.add_argument(
        "--data_set_dir", type=str, default='/graphics/scratch2/staff/Hassan/genius_crawl/lyrics_to_prompts_v2.0.csv', help='path to the trainign data'
    )
    parser.add_argument(
        "--check_path", type=str, default='/graphics/scratch2/staff/Hassan/checkpoints/lyrics_to_prompts/', help="path to save the model"
    )
    parser.add_argument(
        "--batch_size", type=int, default=30
    )

    parser.add_argument(
        "--epochs", type=int, default=5
    )

    parser.add_argument(
        "--learning_rate", type=float, default=5e-5
    )
    parser.add_argument(
        "--warmup_steps", type=int, default=1e3
    )
    parser.add_argument(
        "--context_length", type=int, default=5, help='number of previous lines from lyrics as the context'
    )

    parser.add_argument(
        "--device", type=str, default='cuda', help='cuda or cpu?'
    )

    args = parser.parse_args()
    return args

def main():

    args = parse_args()

    hparams = dotdict({})
    hparams.data_dir = args.data_set_dir
    hparams.model_name = args.model_name
    hparams.context_length = args.context_length
    hparams.batch_size = args.batch_size
    hparams.learning_rate =args.learning_rate
    hparams.device=args.device
    hparams.warmup_steps=args.warmup_steps
    check_path=args.check_path
    check_path = check_path + '{}_v1.0/'.format(args.model_name)

    model = GPT2Convertor(hparams)

    checkpoint = torch.load(check_path+hparams.model_name +"-v3.ckpt", map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(args.device)
    print('checkpoint loaded')


    tokenizer = model.tokenizer
    tokenizer.padding_side='left'
    model = model.model

    train_dataset =Dataset(args.data_set_dir,context_size=5,training=False)
    data_collator = ContextAwareDataCollatorForGeneration(tokenizer)

    dataloader = DataLoader(train_dataset, batch_size=8,
                                  shuffle=False, num_workers=2, collate_fn=data_collator)

    text=[' feels like the weight of the world; like god in heaven gave me a turn ;', 'dont cling to me i swear i cant fix you ;']
    generate_from_sentences(text,model, tokenizer,hparams.device)
    #name2cap = generate_from_loader(dataloader, model, tokenizer,hparams.device)


if __name__ == "__main__":
    main()


# results=[]
#
# for name, cap in name2cap.items():
#     id=int(name[6:-4])
#     results.append({'image_id': id, 'caption': cap})
#
# jsonString = json.dumps(results)
#
# if real:
#     jsonFile = open(PATH +"results_real_{}.json".format(model_name), "w")
# else:
#     jsonFile = open(PATH + "results_syn_{}.json".format(model_name), "w")
#
# jsonFile.write(jsonString)
# jsonFile.close()

