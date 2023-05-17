import os
import json
import random

from tqdm import tqdm
from utils import load_pkl

data=load_pkl('/graphics/scratch2/staff/Hassan/genius_crawl/dataset_50')

for i in range(50):

    paths = [ '/graphics/scratch2/staff/Hassan/chatgpt_data_v2.0/',
             '/graphics/scratch2/staff/Hassan/chatgpt_data_v3.0/']
    artists = list(os.listdir(paths[-1]))
    name=random.choice(artists)
    artist_lyrics=data[name]

    paths=[pp + name for pp in paths]

    tracks_list=[list(os.listdir(p)) for p in paths]

    random_track=random.choice(tracks_list[-1])

    random_track_lyrics=[l['lyrics'] for l in artist_lyrics if l['title'].replace('/','-')==random_track ][0]

    responses={}
    for num, p in enumerate(paths):
        with open('{}/{}'.format(p, random_track)) as f:
            response = json.load(f)

            responses[num]=response['choices'][0]['message']['content'].split('\n')

    valid=True
    for k, v in responses.items():
        if len(v) !=len(random_track_lyrics):
            valid =False


    if valid:

        index=random.choice(range(len(responses[0])-4))

        for k, v in responses.items():

            prompt=v[index + 3]+ '\n\n\n'
            context='\n'.join(random_track_lyrics[index:index + 4])
            scrap=prompt+ context
            if k==0:
                f=open('lyrics_v{}.0/'.format(),'w')
                f.write(scrap)
                f.close()

        #     print('v_', k+2, ': [',random_track_lyrics[index],']: [', v[index+3],']' )
        # print(' ')
