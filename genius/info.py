from utils import load_pkl
import numpy as np
import random

max_lines=50
path_out='/graphics/scratch2/staff/Hassan/genius_crawl/'
data=load_pkl(path_out+'dataset_'+str(max_lines))

print('number of artists: ',len(data))

songs_count=0
lines_count=0
lengths=[]

for name, songs in data.items():
    if len(songs)>=15:
        choice=random.randint(1, 5)
        # if choice==1:
        #     print('[ ', songs[1]['lyrics'][0] + "--" +  songs[1]['lyrics'][1] + "--" +  songs[1]['lyrics'][2], ' ]')
        songs_count+=len(songs)
        for song in songs:
            max_l=max([len(l.split(' ')) for l in song['lyrics']])
            #min_l = min([len(np.unique(l.split(' '))) for l in song['lyrics']])
            ll = len(song['lyrics'])
            lengths.append(ll)
            if ll>max_lines:
                print('oops1')
            if ll < 14:
                print('oops2')
            if max_l>20:
                print('oops3')
            # if min_l < 2:
            #     singl_words +=1
    else:
        print('oops')

print('total songs: ', songs_count)
print('total lines: ', sum(lengths))
print('mean , std  lines per track: ', np.mean(lengths), ' ', np.std(lengths))
print('min , max  lines per track: ', np.min(lengths), ' ', np.max(lengths))
# print(singl_words)

long_name_count=0
for name in data.keys():

    if len(name.split(' '))>15:
        long_name_count +=1

    for song in data[name]:
        if len(song['title'].split(' ')) > 15:
            long_name_count +=1
        if '/' in song['title']:
            long_name_count +=1

    if '/' in name:
        long_name_count += 1

print('number of bad name or title', long_name_count)

