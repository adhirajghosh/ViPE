import argparse
import fire
import sys
import numpy as np
import os
from transformers import pipeline
from utils import *
from vid import create_video
from logger import Logger


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--gpu', help='GPU id', type=int, default=0)
    parser.add_argument('--fps', help='frame rate of gif', type=float, default=12.5)
    parser.add_argument('--steps', help='number of images between two lines', type=int, default=1)
    parser.add_argument('--song_path', help='song file path', default = './all_star_mod2')
    parser.add_argument('--concreteness_path', help='file path for concreteness score', default='/graphics/scratch2/students/ghoshadh/datasets/ac_EN_ratings/AC_ratings_google3m_koeper_SiW_fix.csv')
    parser.add_argument(
        '--embedding', help='file path for embeddings', default = '/graphics/scratch/shahmoha/checkpoints/final models/fast_text/normal_fasttext_gensim')
    parser.add_argument('--result_dir', help='name of folder for outputs',
                        default='./results')
    parser.add_argument('--name', help='name of song folder/gif/log',
                        default='song1_extend')
    # parser.add_argument('--log', help='name of log file',
    #                     default='./results/song-2/all-star-fasttext-extend_old.txt')
    parser.add_argument('--extend', help='use prompt extend or not. Type --extend or --no-extend',default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('--lemma', help='root word for all words. Type --lemma or --no-lemma',default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('--extend_model', help='path to model extender',
                        default='./prompt-extend')

    args = parser.parse_args()
    return args

def main():
    mod = True
    csv.field_size_limit(sys.maxsize)
    args = parse_args()
    song_name = args.song_path.split('/')[-1]
    output_dir = os.path.join(args.result_dir, song_name,args.name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    log_dir = os.path.join(args.result_dir,song_name, args.name+".txt")
    gif_dir = os.path.join(args.result_dir,song_name, args.name+".gif")
    sys.stdout = Logger(log_dir)
    print("Loading full song")
    full_song = song_to_list(args.song_path, args.lemma)
    if mod == False:
        print("Loading Concreteness index")
        word_score = concrete_score(args.concreteness_path)

    if args.extend == True:
        print("Using prompt extender")
        text_pipe = pipeline('text-generation', model=args.extend_model, device=args.gpu, max_new_tokens=20)

    #change the list full_song
    frame_index = 0
    for i, stanza in enumerate(full_song):
        for j, line in enumerate(stanza):
            if  mod is False:
                new_lyric = change_lyric(args.embedding, line, word_score)
                if args.extend:
                    new_lyric = text_pipe(new_lyric+',', num_return_sequences=1)[0]["generated_text"]
                full_song[i][j] = new_lyric
            else:
                if args.extend:
                    new_lyric = text_pipe(line + ',', num_return_sequences=1)[0]["generated_text"]
                    full_song[i][j] = new_lyric

        print(full_song[i])
        new_frame_index = create_video(prompts = full_song[i],
                     seeds = np.random.randint(100,999, size=(len(full_song[i]))),
                     gpu = args.gpu,
                     rootdir = output_dir,
                     num_steps = args.steps,
                     frame_index = frame_index
                     )
        frame_index = new_frame_index

    gif(output_dir, gif_dir, args.fps)

if __name__ == '__main__':
    fire.Fire(main)

