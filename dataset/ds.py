import pickle
import os
import csv
import numpy as np
import json
import yaml
import re

def starts_with_1_to_50(string):
    regex = r"^(?:[1-9]|[1-4]\d|50)"
    return bool(re.match(regex, string))

lyric_path='/graphics/scratch2/staff/Hassan/genius_crawl/'
prompt_path = '/graphics/scratch2/staff/Hassan/chatgpt_data/'
ds_path = "../SongAnimator.csv"
data={}
file = lyric_path+"dataset_50.pickle"
with open(file, 'rb') as handle:
    file = pickle.load(handle)

lyric_prompt = {}
my_id = []
my_gpt_id = []
my_artist = []
my_song = []
my_lyric = []
my_prompt = []
idx = 1
for artist in file.keys():
    for songs in range(len(file[artist])):

        full_song = file[artist][songs]['title']
        song = full_song.split(' by\xa0')[0]
        # print(song)
        # print(artist)
        lyric = file[artist][songs]['lyrics']

        # check if prompts exist
        if os.path.exists(os.path.join(prompt_path, artist, full_song)) == False or os.stat(
                os.path.join(prompt_path, artist, full_song)).st_size == 0:
            continue
        else:
            with open(os.path.join(prompt_path, artist, full_song), 'r') as f:
                gpt = yaml.load(f.readlines()[0])

        gpt_id = gpt['id'].split('chatcmpl-')[1]
        prompts = gpt['choices'][0]['message']['content'].split('\n')

        # Section for debugging
        # 1. Removing cases where prompts are nonsensical. I chose 3 because by definition, the shortest line prompt has a number from 0-9 followed by '. ', so 3 characters.
        # 2. Fix case where there are more prompts than lyrics
        # 3. Exclude songs that were not processed
        # 4. If there exist lines that aren't part of the song at the end

        prompts = [x for x in prompts if starts_with_1_to_50(x) == True]

        if not prompts:
            continue

        for i in range(min(len(prompts), len(lyric))):
            line = lyric[i]
            my_id.append(idx)
            my_gpt_id.append(gpt_id)
            my_artist.append(artist)
            my_song.append(song)
            my_lyric.append(line)
            my_prompt.append(prompts[i].split('.')[1])
            idx = idx + 1

lyric_prompt['id'] = my_id
lyric_prompt['gpt_id'] = my_gpt_id
lyric_prompt['artist'] = my_artist
lyric_prompt['song'] = my_song
lyric_prompt['lyric'] = my_lyric
lyric_prompt['prompt'] = my_prompt

keys = lyric_prompt.keys()
with open(ds_path, "w") as outfile:
    writer = csv.writer(outfile)
    writer.writerow(keys)
    writer.writerows(zip(*lyric_prompt.values()))