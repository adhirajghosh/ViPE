from pytube import YouTube
import os
import fire
from pathlib import Path
import argparse
from music_video_generation.whisper import *

def parse_args():
    parser = argparse.ArgumentParser(description='Audio to Lyric Alignment')
    parser.add_argument('--song_name', help='Name of audio file', type=str, default='placeholder.mp3')
    parser.add_argument('--url', help='youtube url if song needs to be downloaded', type=str, default='https://www.youtube.com/watch?v=XbGs_qK2PQA&ab_channel=EminemVEVO')
    parser.add_argument('--outdir', help='Where songs are kept', type=str, default='./mp3/')
    parser.add_argument('--timestamps', help='where timesteps are kept', type=str, default='./timestamps/')
    args = parser.parse_args()
    return args

def youtube2mp3 (url,outdir):
    yt = YouTube(url)

    video = yt.streams.filter(abr='192kbps').last()

    out_file = video.download(output_path=outdir)
    base, ext = os.path.splitext(out_file)
    base = base.split('(')[0]
    song_name = f'{base}.mp3'
    new_file = Path(song_name)
    os.rename(out_file, new_file)

    if new_file.exists():
        print(f'{yt.title} has been successfully downloaded.')
    else:
        print(f'ERROR: {yt.title}could not be downloaded!')

    return song_name

def main():
    args = parse_args()
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    if not os.path.exists(args.timestamps):
        os.makedirs(args.timestamps)

    if not os.path.exists(os.path.join(args.outdir,args.song_name)):
        song_name = youtube2mp3(args.url, args.outdir)
    else:
        song_name = args.song_name

    song_path = os.path.join(args.outdir, song_name)
    x = whisper_lyrics(song_path)
    if not os.path.exists(args.timestamps):
        os.makedirs(args.timestamps)
    with open(os.path.join(args.timestamps,song_name.split('.')[0]+'.txt'), "w") as output:
        output.write(str(x))

if __name__ == '__main__':
    fire.Fire(main)
