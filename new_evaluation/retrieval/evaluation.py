import argparse
import os
import yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path
import torch.multiprocessing as mp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader
from models.blip_retrieval import blip_retrieval
import utils
from dataset import create_train_dataset, create_test_dataset, create_loader, create_sampler

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json
import os



def train(model, data_loader, optimizer, epoch, device, config):
    # train
    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_itm', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_ita', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50

    for i, (image, caption, idx) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        image = image.to(device, non_blocking=True)
        idx = idx.to(device, non_blocking=True)

        if epoch > 0:
            alpha = config['alpha']
        else:
            alpha = config['alpha'] * min(1, i / len(data_loader))

        loss_ita, loss_itm = model(image, caption, alpha=alpha, idx=idx)
        loss = loss_ita + loss_itm

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metric_logger.update(loss_itm=loss_itm.item())
        metric_logger.update(loss_ita=loss_ita.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluation(model, data_loader, device, config):
    # test
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Evaluation:'

    print('Computing features for evaluation...')
    start_time = time.time()

    texts = data_loader.dataset.text
    num_text = len(texts)
    text_bs = 256
    text_ids = []
    text_embeds = []
    text_atts = []
    for i in range(0, num_text, text_bs):
        text = texts[i: min(num_text, i + text_bs)]
        text_input = model.tokenizer(text, padding='max_length', truncation=True, max_length=35,
                                     return_tensors="pt").to(device)
        text_output = model.text_encoder(text_input.input_ids, attention_mask=text_input.attention_mask, mode='text')
        text_embed = F.normalize(model.text_proj(text_output.last_hidden_state[:, 0, :]))
        text_embeds.append(text_embed)
        text_ids.append(text_input.input_ids)
        text_atts.append(text_input.attention_mask)

    text_embeds = torch.cat(text_embeds, dim=0)
    text_ids = torch.cat(text_ids, dim=0)
    text_atts = torch.cat(text_atts, dim=0)
    text_ids[:, 0] = model.tokenizer.enc_token_id

    image_feats = []
    image_embeds = []
    for image, img_caption, _ in data_loader:
        image = image.to(device)
        image_feat = model.visual_encoder(image)
        image_embed = model.vision_proj(image_feat[:, 0, :])
        image_embed = F.normalize(image_embed, dim=-1)

        image_feats.append(image_feat)
        image_embeds.append(image_embed)

    image_feats = torch.cat(image_feats, dim=0)
    image_embeds = torch.cat(image_embeds, dim=0)

    sims_matrix = image_embeds @ text_embeds.t()
    score_matrix_i2t = torch.full((len(data_loader.dataset.image), len(texts)), -100.0).to(device)

    num_tasks = utils.get_world_size()
    rank = utils.get_rank()
    step = sims_matrix.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix.size(0), start + step)

    for i, sims in enumerate(metric_logger.log_every(sims_matrix[start:end], 50, header)):
        topk_sim, topk_idx = sims.topk(k=config['k_test'], dim=0)

        encoder_output = image_feats[start + i].repeat(config['k_test'], 1, 1).to(device)
        encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(device)
        output = model.text_encoder(text_ids[topk_idx],
                                    attention_mask=text_atts[topk_idx],
                                    encoder_hidden_states=encoder_output,
                                    encoder_attention_mask=encoder_att,
                                    return_dict=True,
                                    )
        score = model.itm_head(output.last_hidden_state[:, 0, :])[:, 1]
        score_matrix_i2t[start + i, topk_idx] = score + topk_sim


    sims_matrix = sims_matrix.t()
    score_matrix_t2i = torch.full((len(texts), len(data_loader.dataset.image)), -100.0).to(device)

    step = sims_matrix.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix.size(0), start + step)

    for i, sims in enumerate(metric_logger.log_every(sims_matrix[start:end], 50, header)):
        topk_sim, topk_idx = sims.topk(k=config['k_test'], dim=0)
        encoder_output = image_feats[topk_idx].to(device)
        encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(device)
        output = model.text_encoder(text_ids[start + i].repeat(config['k_test'], 1),
                                    attention_mask=text_atts[start + i].repeat(config['k_test'], 1),
                                    encoder_hidden_states=encoder_output,
                                    encoder_attention_mask=encoder_att,
                                    return_dict=True,
                                    )
        score = model.itm_head(output.last_hidden_state[:, 0, :])[:, 1]
        score_matrix_t2i[start + i, topk_idx] = score + topk_sim

    if args.distributed:
        dist.barrier()
        torch.distributed.all_reduce(score_matrix_i2t, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(score_matrix_t2i, op=torch.distributed.ReduceOp.SUM)


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluation time {}'.format(total_time_str))

    return score_matrix_i2t.cpu().numpy(), score_matrix_t2i.cpu().numpy()


@torch.no_grad()
def itm_eval(scores_i2t, scores_t2i, txt2img, img2txt):
    # Images->Text
    ranks = np.zeros(scores_i2t.shape[0])
    print(scores_i2t.shape)
    print(scores_t2i.shape)
    for index, score in enumerate(scores_i2t):
        inds = np.argsort(score)[::-1]
        # Score
        ranks[index] = np.where(inds == img2txt[index])[0][0]
        # rank = 1e20
        # for i in img2txt[index]:
        #     tmp = np.where(inds == i)[0][0]
        #     # print(tmp)
        #     if tmp < rank:
        #         rank = tmp
        # ranks[index] = rank

    # Compute metrics
    tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    # Text->Images
    ranks = np.zeros(scores_t2i.shape[0])

    for index, score in enumerate(scores_t2i):
        inds = np.argsort(score)[::-1]
        # ranks[index] = np.where(inds == txt2img[index])[0][0]
        rank = 1e20
        for i in txt2img[index]:
            tmp = np.where(inds == i)[0][0]
            # print(tmp)
            if tmp < rank:
                rank = tmp
        ranks[index] = rank

    # Compute metrics
    ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    tr_mean = (tr1 + tr5 + tr10) / 3
    ir_mean = (ir1 + ir5 + ir10) / 3
    r_mean = (tr_mean + ir_mean) / 2

    eval_result = {'txt_r1': tr1,
                   'txt_r5': tr5,
                   'txt_r10': tr10,
                   'txt_r_mean': tr_mean,
                   'img_r1': ir1,
                   'img_r5': ir5,
                   'img_r10': ir10,
                   'img_r_mean': ir_mean,
                   'r_mean': r_mean}
    return eval_result


def main(args, config):
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = '0'
    if 'WORLD_SIZE' not in os.environ:
        os.environ['WORLD_SIZE'] = '1'
    #
    utils.init_distributed_mode(args)

    device = torch.device(args.device)


    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    #### Dataset ####
    if args.id_type == 'metaphor':
        id_file = args.data_dir+"metaphor_id.pickle"
    else:
        if args.dataset == 'haivmet':
            id_file = args.data_dir + "prompt_dict_haivmet.pickle"
        elif args.dataset ==  'vipe':
            id_file = args.data_dir + "prompt_dict_vipe.pickle"
        elif args.dataset == 'chatgpt':
            id_file = args.data_dir + "prompt_dict_chatgpt.pickle"

    print("Creating retrieval dataset on ", args.dataset)

    if args.evaluate:
        test_dataset = create_test_dataset(os.path.join(args.data_dir,args.dataset, 'eval'), id_file, config)
        samplers = [None]

        test_loader = create_loader([test_dataset], samplers,
                                    batch_size=[config['batch_size_test']],
                                    num_workers=[ 8],
                                    is_trains=[ False],
                                    collate_fns=[ None])[0]
        print(len(test_dataset))
    else:
        train_dataset = create_train_dataset(os.path.join(args.data_dir, args.dataset, 'train'), id_file, config)
        test_dataset = create_test_dataset(os.path.join(args.data_dir, args.dataset, 'train'), id_file, config)
        samplers = [None, None]
        train_loader, test_loader = create_loader([train_dataset, test_dataset], samplers,
                                                  batch_size=[config['batch_size_train'], config['batch_size_test']],
                                                  num_workers=[8, 8],
                                                  is_trains=[True, False],
                                                  collate_fns=[None, None])
        print(len(train_dataset), len(test_dataset))


    #### Model ####
    print("Creating model")
    model = blip_retrieval(pretrained=config['pretrained'], image_size=config['image_size'], vit=config['vit'],
                           vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'],
                           queue_size=config['queue_size'], negative_all_rank=config['negative_all_rank'])
    model = model.to(device)


    print("Doing DDP", sum(p.numel() for p in model.parameters()))
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=1e-4, weight_decay=0.05)

    best = 0
    best_epoch = 0

    print("Start training")
    start_time = time.time()

    for epoch in range(0, config['max_epoch']):
        if not args.evaluate:

            utils.cosine_lr_schedule(optimizer, epoch, int(config['max_epoch']), float(1e-4), float(config['min_lr']))

            train_stats = train(model, train_loader, optimizer, epoch, device, config)


        score_test_i2t, score_test_t2i = evaluation(model_without_ddp, test_loader, device, config)

        if utils.is_main_process():

            test_result = itm_eval(score_test_i2t, score_test_t2i, test_loader.dataset.txt2img,
                                   test_loader.dataset.img2txt)
            print(test_result)
            if test_result['r_mean'] > best:
                save_obj = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'config': config,
                    'epoch': epoch,
                }
                torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_best.pth'))
                best = test_result['r_mean']
                best_epoch = epoch
            if args.evaluate:
                log_stats = {
                             **{f'test_{k}': v for k, v in test_result.items()}}
                with open(os.path.join(args.output_dir, "evaluate.txt"), "a") as f:
                    f.write(json.dumps(log_stats) + "\n")
            else:
                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                            **{f'test_{k}': v for k, v in test_result.items()},
                            'epoch': epoch,
                            'best_epoch': best_epoch,
                            }

                with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                    f.write(json.dumps(log_stats) + "\n")

        if args.evaluate:
            break

        dist.barrier()
        torch.cuda.empty_cache()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    dist.destroy_process_group()

if __name__ == '__main__':
    # torch.multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./new_evaluation/retrieval/configs/config_base.yml')
    parser.add_argument('--data_dir', default='/graphics/scratch2/students/ghoshadh/SongAnimator/datasets/retrieval/')
    parser.add_argument('--dataset', default='haivmet')
    parser.add_argument('--id_type', default='metaphor', help='prompt or metaphor')
    parser.add_argument('--output_dir', default='output/new/haivmet_train_lr1e-4/')
    parser.add_argument('--evaluate', default=False, type=bool)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--gpu', default='0,1')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))

    # mp.spawn(main, args=(args, config), nprocs=args.world_size)
    main(args, config)


