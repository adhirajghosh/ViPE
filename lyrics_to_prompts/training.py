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
    parser = argparse.ArgumentParser(description="train hehe")

    parser.add_argument(
        "--model_name", type=str, default='gpt2', help="which gpt2 version to use?"
    )

    parser.add_argument(
        "--data_set_dir", type=str, default='/graphics/scratch2/staff/Hassan/genius_crawl/lyrics_to_prompts.csv', help='path to the trainign data'
    )
    parser.add_argument(
        "--check_path", type=str, default='/graphics/scratch2/staff/Hassan/checkpoints/lyrics_to_prompts/gpt2_v1.0/', help="path to save the model"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32
    )

    parser.add_argument(
        "--epochs", type=int, default=5
    )

    parser.add_argument(
        "--learning_rate", type=float, default=5e-4
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
    hparams.model_name = 'gpt2'
    hparams.context_length = args.context_length
    hparams.batch_size = args.batch_size
    hparams.learning_rate =args.learning_rate
    hparams.device=args.device
    check_path=args.check_path
    max_epochs=args.epochs

    with open(check_path +'hparams.txt', 'w') as file:
        file.write(json.dumps(hparams))

    tb_logger = pl_loggers.TensorBoardLogger(save_dir=check_path+"logs/", name="lightning_logs")
    checkpoint_callback = ModelCheckpoint(dirpath=check_path, save_top_k=1, monitor="v_loss",save_weights_only=True,filename=args.model_name)
    early_stop = EarlyStopping(monitor="v_loss", mode="min",patience=3)
    model = GPT2Convertor(hparams)
    #model.to(args.device)

    # checkpoint = torch.load(check_path+"correct_bert_first_layer_frozen_vit.ckpt", map_location=lambda storage, loc: storage)
    #
    # model.load_state_dict(checkpoint['state_dict'])
    # print('checkpoint loaded')

    #trainer=Trainer(accelerator='gpu', devices='0,1,2', callbacks=[checkpoint_callback, early_stop], logger=tb_logger,max_epochs=max_epochs,strategy='ddp')
    trainer = Trainer(accelerator='gpu', devices=1, callbacks=[checkpoint_callback, early_stop], logger=tb_logger,    max_epochs=max_epochs)

    trainer.fit(model)


    # synthetic 1 lr:  0.0007585775750291836
    # syns 2  0.0005248074602497723
    #syns 3 Learning rate set to 0.0005248074602497723
    #real lr : earning rate set to 0.0005248074602497723

    #for fine tuning all
    # batch_size = 64, learning_rate = 5e-5,


if __name__ == "__main__":
    main()
