import os
import json
from tqdm import tqdm
from utils import save_pkl,load_pkl
import numpy as np

dataset=load_pkl('dataset')
path='/graphics/scratch2/staff/Hassan/genius_crawl/genius_data/'
prepared=[]

#min number of words per line
min_word_count=2
# max number of lines per lyrics
max_lines=45

#some artist have very few english songs
min_en_lyrics=10

data={}
# data['songs'][0]['lyrics']
# data['songs'][0]['language']

all_lines_count=0
all_songs_count=0
for c, name in enumerate(tqdm(os.listdir(path))):

    # if (c+1) % 1000==0:
    #     #print(all_songs_count)
    #     print(c+1)

    with open('{}{}'.format(path,name)) as f:
       artist_collection = json.load(f)
    name=name[:-5]

    if len(artist_collection['songs'])>=50:
        data[name] = []

        for song in artist_collection['songs'][0:50]:
            if song['language']=='en':
                song_data = {'title': 0, 'lyrics': 0}

                lyrics=[i for i in song['lyrics'].split('\n')[1:]  if len(np.unique(i.split(' '))) >= min_word_count]
                #truncate the long lyrics because chatgpt gets confused
                if len(lyrics)> max_lines:
                    lyrics=lyrics[0:max_lines]

                all_songs_count +=1
                all_lines_count += len(lyrics)
                song_data['title']=song['full_title']
                song_data['lyrics'] =lyrics

                #add the lyric to the artist list
                data[name].append(song_data)

        #remove artist if no/or not enough english lyrics were added
        if len(data[name]) < min_en_lyrics:
            all_songs_count = all_songs_count -  len(data[name])
            del data[name]

print('processed probably less than {} number of lines from exactly {} songs :'.format(all_lines_count,all_songs_count))

save_pkl(data,'dataset')
