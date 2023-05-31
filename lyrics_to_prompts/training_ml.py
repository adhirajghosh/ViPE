import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from pytorch_lightning import  Trainer
from modeling import GPT2Convertor
from utils import dotdict

import json
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="train hehe?")

    parser.add_argument(
        "--model_name", type=str, default='gpt2', help="which gpt2 version to use?"
    )

    parser.add_argument(
        "--data_set_dir", type=str, default='/graphics/scratch2/staff/Hassan/genius_crawl/lyrics_to_prompts.csv', help='path to the trainign data'
    )

    parser.add_argument(
        "--data_set_dir_ml", type=str, default='/mnt/lustre/lensch/hshahmohammadi86/datasets/genuis_chatgpt/lyrics_to_prompts.csv',
        help='path to the training data'
    )

    parser.add_argument(
        "--check_path", type=str, default='/graphics/scratch2/staff/Hassan/checkpoints/lyrics_to_prompts/', help="path to save the model"
    )

    parser.add_argument(
        "--check_path_ml", type=str, default='/mnt/lustre/lensch/hshahmohammadi86/checkpoints/songanimator/',
        help="path to save the model"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32
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

    parser.add_argument(
        "--ml", type=int, default=1, help='set to 1 to use ml paths'
    )

    parser.add_argument(
        "--load_checkpoint", type=int, default=3, help='which checkpoint version to load if resume training'
    )

    args = parser.parse_args()
    return args

def main():

    #print('job is running')
    args = parse_args()

    hparams = dotdict({})

    hparams.model_name = args.model_name
    hparams.context_length = args.context_length
    hparams.batch_size = args.batch_size
    hparams.learning_rate =args.learning_rate
    hparams.device=args.device
    hparams.warmup_steps=args.warmup_steps
    max_epochs=args.epochs
    load_checkpoint=args.load_checkpoint


    if args.ml ==0:
        check_path = args.check_path
        check_path = check_path +'{}_v1.0/'.format(args.model_name)
        hparams.data_dir = args.data_set_dir
    else:
        check_path = args.check_path_ml
        check_path = check_path + '{}/'.format(args.model_name)
        hparams.data_dir = args.data_set_dir_ml

    model_name='{}_context_ctx_{}_lr_{}'.format(args.model_name, args.context_length,args.learning_rate)

    # Specify the directory path and file name
    file_path = check_path + 'hparams_'+model_name
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Open the file for writing
    with open(file_path, 'w') as file:
        file.write(json.dumps(hparams))

    tb_logger = pl_loggers.TensorBoardLogger(save_dir=check_path+"logs/", name=model_name)
    checkpoint_callback = ModelCheckpoint(dirpath=check_path, save_top_k=5, monitor="val_loss",save_weights_only=True,filename=model_name)
    early_stop = EarlyStopping(monitor="val_loss", mode="min",patience=3)
    model = GPT2Convertor(hparams)
    #model.to(args.device)

    if load_checkpoint > -1:
        if load_checkpoint==0:
            check_name=check_path+model_name + '.ckpt'
        else:
            check_name = check_path + model_name +'-v{}.ckpt'.format(load_checkpoint)

        checkpoint = torch.load(check_name, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'])
        print('checkpoint loaded: ',model_name +'-v{}.ckpt'.format(load_checkpoint) )

    #trainer=Trainer(accelerator='gpu', devices=8, callbacks=[checkpoint_callback, early_stop], logger=tb_logger,max_epochs=max_epochs,strategy='ddp')
    trainer = Trainer(accelerator='gpu', devices=8, callbacks=[checkpoint_callback, early_stop], logger=tb_logger,  max_epochs=max_epochs,strategy='ddp')
    trainer.fit(model)

if __name__ == "__main__":
    main()
