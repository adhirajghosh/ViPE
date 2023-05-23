import pickle
import os
import csv
import numpy as np
import json
import yaml
import re
from tqdm import  tqdm

pattern = r'^([1-9]|[1-4][0-9]|50)\.\s'  # Regular expression pattern
def check_prompt_format(current_line):
    return bool(re.match(pattern, current_line))

# def starts_with_1_to_50(string):
#     regex = r"^(?:[1-9]|[1-4]\d|50)"
#     return bool(re.match(regex, string))

lyric_path='/graphics/scratch2/staff/Hassan/genius_crawl/'
prompt_path = '/graphics/scratch2/staff/Hassan/chatgpt_data_v2.0/'
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
size_threshold=1000 # bytes

for artist,songs  in tqdm(file.items()):

    #check if the artist exist
    if os.path.exists(os.path.join(prompt_path, artist)):

        for song in songs:

            full_title = song['title']
            lyric = song['lyrics']

            # check if prompts exist
            if os.path.exists(os.path.join(prompt_path, artist, full_title)) == False or os.stat(
                    os.path.join(prompt_path, artist, full_title)).st_size < size_threshold:
                continue
            else:
                with open(os.path.join(prompt_path, artist, full_title), 'r') as f:
                    gpt = yaml.load(f.readlines()[0],Loader=yaml.Loader)

            gpt_id = gpt['id'].split('chatcmpl-')[1]
            prompts = gpt['choices'][0]['message']['content'].split('\n')

            # Section for debugging
            # 1. Removing cases where prompts are nonsensical. I chose 3 because by definition, the shortest line prompt has a number from 0-9 followed by '. ', so 3 characters.
            # 2. Fix case where there are more prompts than lyrics
            # 3. Exclude songs that were not processed
            # 4. If there exist lines that aren't part of the song at the end

            prompts = [x for x in prompts if check_prompt_format(x)]

            if not prompts:
                continue

            for i in range(min(len(prompts), len(lyric))):
                line = lyric[i]
                my_id.append(idx)
                my_gpt_id.append(gpt_id)
                my_artist.append(artist)
                my_song.append(full_title.split('by\xa0')[0])
                my_lyric.append(line)
                my_prompt.append(prompts[i].split('.')[1])
                idx = idx + 1

lyric_prompt['ids'] = my_id
lyric_prompt['gpt_ids'] = my_gpt_id
lyric_prompt['artists'] = my_artist
lyric_prompt['titles'] = my_song
lyric_prompt['lyrics'] = my_lyric
lyric_prompt['prompts'] = my_prompt

keys = lyric_prompt.keys()
with open(ds_path, "w") as outfile:
    writer = csv.writer(outfile)
    writer.writerow(keys)
    writer.writerows(zip(*lyric_prompt.values()))