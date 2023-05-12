import os
import json
from tqdm import tqdm
from utils import save_pkl,load_pkl

dataset=load_pkl('dataset')
path='/graphics/scratch2/staff/Hassan/genius_crawl/genius_data/'
prepared=[]

#min number of words per line
min_word_count=2
# max number of lines per lyrics
max_lines=45

data={}
# data['songs'][0]['lyrics']
# data['songs'][0]['language']

all_lines_count=0
all_songs_count=0
for c, name in enumerate(tqdm(os.listdir(path))):

    if all_songs_count>250000:
        break
    if (c+1) % 1000==0:
        print(all_songs_count)
        print(c+1)

    with open('{}{}'.format(path,name)) as f:
       artist_collection = json.load(f)
    name=name[:-5]

    if len(artist_collection['songs'])>=50:
        data[name] = []

        for song in artist_collection['songs'][0:50]:
            if song['language']=='en':
                song_data = {'title': 0, 'lyrics': 0}
                song['lyrics'] = song['lyrics'].replace('\n\n', '\n \n')
                lyrics=[i for i in song['lyrics'].split('\n')[1:] if len(i.split(' ')) >= min_word_count]
                #truncate the long lyrics because chatgpt gets confused
                if len(lyrics)> max_lines:
                    lyrics=lyrics[0:max_lines]

                all_songs_count +=1
                all_lines_count += len(lyrics)
                song_data['title']=song['full_title']
                song_data['lyrics'] =lyrics

                #add the lyric to the artist list
                data[name].append(song_data)

        #remove artist if no lyrics were added
        if len(data[name]) < 1:
            del data[name]

print('processed {} number of lines from {} songs :'.format(all_lines_count,all_songs_count))

save_pkl(data,'dataset')



