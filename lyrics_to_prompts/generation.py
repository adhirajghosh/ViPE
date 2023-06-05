from modeling import GPT2Convertor

import os

from torch.utils.data import Dataset, DataLoader
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from utils import Dataset, ContextAwareDataCollator,ContextAwareDataCollatorForGeneration
from pytorch_lightning import  Trainer
from modeling import GPT2Convertor
from utils import dotdict, generate_from_loader,generate_from_sentences,to_coco_format,DatasetTest
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
        "--context_length", type=int, default=3, help='number of previous lines from lyrics as the context'
    )

    parser.add_argument(
        "--device", type=str, default='cuda', help='cuda or cpu?'
    )

    parser.add_argument(
        "--gpu", type=int, default=1, help='which gpu?'
    )
    parser.add_argument(
        "--ml", type=int, default=1
    )

    parser.add_argument(
        "--random", type=int, default=0, help='set 1 to generate with sampling'
    )

    args = parser.parse_args()
    return args

def main():

    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] =str(args.gpu)

    hparams = dotdict({})
    hparams.data_dir = args.data_set_dir
    hparams.model_name = args.model_name
    hparams.context_length = args.context_length
    hparams.batch_size = args.batch_size
    hparams.learning_rate =args.learning_rate
    hparams.device=args.device
    hparams.warmup_steps=args.warmup_steps

    if args.ml == 0:
        check_path = args.check_path
        check_path = check_path + '{}_v2.0/'.format(args.model_name)
        hparams.data_dir = args.data_set_dir
    else:
        check_path = args.check_path
        check_path = check_path + 'ml_logs_checkpoints/{}/'.format(args.model_name)

    model = GPT2Convertor(hparams)

    check_point_name='gpt2.ckpt'
    check_point_name='gpt2_context_ctx_7_lr_5e-05-v4.ckpt'
    #check_point_name = 'gpt2_context_ctx_7_lr_5e-05-v4.ckpt'
    #check_point_name='gpt2_context_ctx_0_lr_5e-05-v2.ckpt'
    #check_point_name = 'gpt2-medium_context_ctx_0_lr_5e-05-v1.ckpt'
    # check_point_name='gpt2-medium_context_ctx_0_lr_5e-05-v2.ckpt'
    check_point_name='gpt2-medium_context_ctx_3_lr_5e-05-v3.ckpt'
    check_point_name='{}_context_ctx_{}_lr_5e-05-v4.ckpt'.format(args.model_name,hparams.context_length )


    #check_point_name='gpt2-medium-v4.ckpt'

    # check_point_name='gpt2_token_type_ids_context_ctx_5_lr_5e-05.ckpt'
    #check_path='/graphics/scratch2/staff/Hassan/checkpoints/lyrics_to_prompts/ml_logs_checkpoints/gpt2_old/'
    checkpoint = torch.load(check_path+check_point_name, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])
    print('checkpoint loaded')
    tokenizer = model.tokenizer
    model=model.model
    model.to(args.device)


    text=[' feels like the weight of the world; like god in heaven gave me a turn ', 'dont cling to me i swear i cant fix you ;']
    generate_from_sentences(text,model, tokenizer,hparams.device,do_sample=True,top_k=0,num_beams=0)

   # use the following code for generation in the coco format for evaluation
    valid_dataset =DatasetTest(args.data_set_dir,context_size=hparams.context_length,training=False)
    data_collator = ContextAwareDataCollatorForGeneration(tokenizer)

    dataloader = DataLoader(valid_dataset, batch_size= hparams.batch_size,
                                  shuffle=False, num_workers=16, collate_fn=data_collator,prefetch_factor=3)
    id2cap, id2ground_truth = generate_from_loader(dataloader, model, tokenizer,hparams.device, args.random)

    results=[]
    for id, cap in id2cap.items():
        results.append({'image_id': id, 'caption': cap})

    jsonString = json.dumps(results)
    saving_dir=check_path+'evaluation/'
    os.makedirs(os.path.dirname(saving_dir), exist_ok=True)
    if args.random > 0:
        jsonFile = open(saving_dir + "random_generation_{}_.json".format(check_point_name), "w")
    else:
        jsonFile = open(saving_dir + "generation_{}_.json".format(check_point_name), "w")
    jsonFile.write(jsonString)
    jsonFile.close()

    if not os.path.exists(saving_dir+'/ground_truth.json'):
        output_json = saving_dir + "ground_truth.json"
        to_coco_format(id2ground_truth, output_json)

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

