from lyrics_to_prompts.modeling import GPT2Convertor
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
from lyrics_to_prompts.utils import dotdict,generate_from_sentences
import json
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="train hehe?")

    parser.add_argument(
        "--model_name", type=str, default='gpt2-medium', help="which gpt2 version to use?"
    )

    parser.add_argument(
        "--check_path", type=str, default='/graphics/scratch2/staff/Hassan/checkpoints/lyrics_to_prompts/', help="path to save the model"
    )

    parser.add_argument(
        "--epochs", type=int, default=5
    )

    parser.add_argument(
        "--learning_rate", type=float, default=5e-5
    )
    parser.add_argument(
        "--lyrics", type=str, default='sample_song'
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
    hparams.model_name = args.model_name
    hparams.context_length = args.context_length
    hparams.batch_size = args.batch_size
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

    lyrics=open(args.lyrics, 'r').readlines().split('\n')
    text=[' feels like the weight of the world; like god in heaven gave me a turn ;', 'dont cling to me i swear i cant fix you ;']
    generate_from_sentences(text,model, tokenizer,hparams.device)
    #name2cap = generate_from_loader(dataloader, model, tokenizer,hparams.device)


if __name__ == "__main__":
    main()

