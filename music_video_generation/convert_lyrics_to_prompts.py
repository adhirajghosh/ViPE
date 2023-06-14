import os
import torch
from modeling import GPT2Convertor
from utils import dotdict,generate_from_sentences
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="train hehe?")

    parser.add_argument(
        "--model_name", type=str, default='gpt2-medium', help="which gpt2 version to use?"
    )

    parser.add_argument(
        "--data_set_dir", type=str, default='/graphics/scratch2/staff/Hassan/genius_crawl/lyrics_to_prompts_v2.0.csv', help='path to the training data'
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
        "--context_length", type=int, default=7, help='number of previous lines from lyrics as the context'
    )

    parser.add_argument(
        "--device", type=str, default='cpu', help='cuda or cpu?'
    )

    parser.add_argument(
        "--gpu", type=int, default=1, help='which gpu?'
    )
    parser.add_argument(
        "--ml", type=int, default=1, help='use ml could checkpoints?'
    )

    parser.add_argument(
        "--do_sample", type=int, default=1, help='set 1 to generate with sampling'
    )

    parser.add_argument(
        "--lyrics", type=str, default='thunder', help='path to the lyric file'
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
    do_sample =True if args.do_sample >0 else False

    if args.ml == 0:
        check_path = args.check_path
        check_path = check_path + '{}_v2.0/'.format(args.model_name)
        hparams.data_dir = args.data_set_dir
    else:
        check_path = args.check_path
        check_path = check_path + 'ml_logs_checkpoints/{}/'.format(args.model_name)

    model = GPT2Convertor(hparams)

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

    with open(args.lyrics, 'r') as file:
        lyrics = file.read()
    lyrics=[line for line in lyrics.split('\n') if len(line.split(' ')) > 1]

    #modify this to change the influence  of the context size
    hparams.context_length=0

    for _ in range(hparams.context_length+1):
        lyrics = ['null'] + lyrics
    lyrics = [lyrics[i:i + hparams.context_length+1] for i in range(len(lyrics) - hparams.context_length )]
    for c, text in enumerate(lyrics):
        text='; '.join([t for t in text if t != 'null'])
        lyrics[c] = 'visualize ;' + text

    lyrics.pop(0)
    prompts=generate_from_sentences(lyrics, model, tokenizer, hparams.device,False,  top_k=100, epsilon_cutoff=.00005, temperature=1)

    with open(args.lyrics+'_prompts', 'w') as file:

        for line, prompt in zip(lyrics, prompts):
            #print(prompt.split(line)[1])
            # file.write(line + ' : '+ prompt.split(line)[1] + '\n')
            file.write(line.split(';')[-1] + ' : ' + prompt.split(line)[1] + '\n')

    #name2cap = generate_from_loader(dataloader, model, tokenizer,hparams.device)

if __name__ == "__main__":
    main()

