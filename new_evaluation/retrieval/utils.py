import os
import zipfile
import re
import tqdm
import torch
import torch.distributed as dist
import time
from collections import defaultdict, deque
import datetime
import math
import numpy as np
import io
import os
import time
from collections import defaultdict, deque
import datetime

import torch
import torch.distributed as dist

def zip_process(file):
    with zipfile.ZipFile(file, 'r') as zip_file:
        # Get a list of all file names in the zip file
        file_names = sorted(zip_file.namelist())[4:]
        #     print(file_names)
        # Create an empty dictionary to store the folder names and first file names
        folder_dict = {}
        max_index=4
        folder_index=2
        prompt_index=3

        if file.split('/')[-1]=='tsvetkov.zip' or file.split('/')[-1]=='bizzoni.zip':
            max_index = 3
            folder_index = 1
            prompt_index = 2

        file_names = [file for file in file_names if len(file.split('/')) == max_index and file.split('/')[-1] != ''
                      and file.split('/')[-1] != 'gpt_prompt.txt' and file.split('/')[-1] != 'dalle_prompt.txt'
                      and file.split('/')[0] != '__MACOSX']

        # Iterate over the file names
        for file_name in sorted(file_names):

            # Extract the folder name and the first file name
            folder_name = file_name.split('/')[folder_index]
            first_file_name = file_name.split('/')[prompt_index]
            # Check if the folder name is already in the dictionary
            if folder_name not in folder_dict:
                label = 'An ' + first_file_name.split('An')[-1][1:-4]

                if label[-1] != '.':
                    label = label + '.'
                folder_dict[folder_name] = label

    return folder_dict

def generate_images(pipe, prompt_dict, ds_id, saving_path, batch_size, size, gpu):
    # added_prompt = "high quality, HD, 32K, high focus, dramatic lighting, ultra-realistic, high detailed photography, vivid, vibrant,intricate,trending on artstation"
    # negative_prompt = 'nude, naked, text, digits, worst quality, blurry, morbid, poorly drawn face, bad anatomy,distorted face, disfiguired, bad hands, missing fingers,cropped, deformed body, bloated, ugly, unrealistic'
    for i in range(4):

        generator = torch.Generator(gpu).manual_seed(i)
        batch = []
        ids = []
        for num, (p_id, prompt) in enumerate(prompt_dict.items()):
            if not os.path.isfile("{}/{}_{}_{}.png".format(saving_path, str(ds_id), p_id, str(i+1))):
                # batch.append(prompt+added_prompt)
                batch.append(prompt)
                ids.append(p_id)
                if len(batch) < batch_size and (num + 1) < len(prompt_dict):
                    continue
                print(prompt)

                #the last batch might not have the same size as batch_size so i used len(batch) instead of len(batch_size)
                # images = pipe(batch, generator=generator, negative_prompt=[negative_prompt]*len(batch), height = size, width = size).images
                images = pipe(batch, generator=generator, height = size, width = size).images

                for num, (img_id, img) in enumerate(zip(ids, images)):
                    img.save("{}/{}_{}_{}.png".format(saving_path, str(ds_id), ids[num], str(i+1)))
                batch = []
                ids = []

def generate_images_retrieval(pipe, ds, save_path, gpu):

    added_prompt = "high quality, HD, 32K, high focus, dramatic lighting, ultra-realistic, high detailed photography, vivid, vibrant,intricate,trending on artstation"
    negative_prompt = 'nude, naked, text, digits, worst quality, blurry, morbid, poorly drawn face, bad anatomy,distorted face, disfiguired, bad hands, missing fingers,cropped, deformed body, bloated, ugly, unrealistic'

    for sample in ds:
        prompt = sample['prompt']
        print(prompt)
        for i in range(5,11):
            print("generating... ", i)
            image = pipe(prompt=prompt + added_prompt, negative_prompt=negative_prompt).images[0]
            outdir = os.path.join(save_path, sample['text'])
            if not os.path.exists(outdir):
                os.makedirs(outdir)

            if prompt[-1] == '.':
                outpath = os.path.join(outdir, '{}_{}.jpg'.format(prompt[:-1],str(i)))
            else:
                outpath = os.path.join(outdir, '{}_{}.jpg'.format(prompt, str(i)))
            image.save(outpath)





def cosine_lr_schedule(optimizer, epoch, max_epoch, init_lr, min_lr):
    """Decay the learning rate"""
    lr = (init_lr - min_lr) * 0.5 * (1. + math.cos(math.pi * epoch / max_epoch)) + min_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_lr_schedule(optimizer, step, max_step, init_lr, max_lr):
    """Warmup the learning rate"""
    lr = min(max_lr, init_lr + (max_lr - init_lr) * step / max_step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def step_lr_schedule(optimizer, epoch, init_lr, min_lr, decay_rate):
    """Decay the learning rate"""
    lr = max(min_lr, init_lr * (decay_rate ** epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr




class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def global_avg(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {:.4f}".format(name, meter.global_avg)
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def compute_acc(logits, label, reduction='mean'):
    ret = (torch.argmax(logits, dim=1) == label).float()
    if reduction == 'none':
        return ret.detach()
    elif reduction == 'mean':
        return ret.mean().item()


def compute_n_params(model, return_str=True):
    tot = 0
    for p in model.parameters():
        w = 1
        for x in p.shape:
            w *= x
        tot += w
    if return_str:
        if tot >= 1e6:
            return '{:.1f}M'.format(tot / 1e6)
        else:
            return '{:.1f}K'.format(tot / 1e3)
    else:
        return tot


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}, world {}): {}'.format(
        args.rank, args.world_size, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


def pre_caption(caption, max_words=50):
    caption = re.sub(
        r"([.!\"()*#:;~])",
        ' ',
        caption.lower(),
    )
    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n')
    caption = caption.strip(' ')

    # truncate caption
    caption_words = caption.split(' ')
    if len(caption_words) > max_words:
        caption = ' '.join(caption_words[:max_words])

    return caption