import torch
from model import Model
import argparse
import fire
import numpy as np
import os
from utils import *
from transformers import pipeline
from torch.nn.parallel import DistributedDataParallel as DDP

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--gpu', help='GPU id', type=int, default=0)
    parser.add_argument('--fps', help='frame rate of gif', type=float, default=12.5)
    parser.add_argument('--song_path', help='song file path', default = './all_star_mod2')
    parser.add_argument('--result_dir', help='name of folder for outputs',
                        default='./results')
    parser.add_argument('--name', help='name of song folder/gif/log',
                        default='song1_extend')
    parser.add_argument('--extend', help='use prompt extend or not. Type --extend or --no-extend', default=False,
                        action=argparse.BooleanOptionalAction)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    song_name = args.song_path.split('/')[-1]
    output_dir = os.path.join(args.result_dir, song_name,args.name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    video_path = os.path.join(args.result_dir, song_name, args.name + ".mp4")
    print("Loading full song")
    full_song = song_to_list(args.song_path)

    model_name = ['dreamlike-art/dreamlike-photoreal-2.0', 'runwayml/stable-diffusion-v1-5']
    model = Model(device=args.gpu, dtype=torch.float32)
    text_pipe = pipeline('text-generation', model='./prompt-extend', device=args.gpu, max_new_tokens=30)

    params = {"t0": 41, "t1": 47, "motion_field_strength_x": 20, "motion_field_strength_y": 20, "smooth_bg": True, "video_length": 30}

    line_no = 1
    for stanza in full_song:
        for line in stanza:
            if args.extend:
                line = text_pipe(line + ',', num_return_sequences=1)[0]["generated_text"]
            print(line)
            out_path = os.path.join(output_dir, str(line_no)+".mp4")
            print("Saving to ", out_path)
            model.process_text2video(line, model_name = model_name[0], fps = args.fps, path = out_path, seed = np.random.randint(100,999),**params)
            line_no = line_no+1

    print("Merging video")
    merge_mp4(folder_path = output_dir, output_path=video_path)



if __name__ == '__main__':
    fire.Fire(main)

