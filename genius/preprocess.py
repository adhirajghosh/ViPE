import os
import json
from tqdm import tqdm
from utils import save_pkl, dotdict
from utils import preprocess_lyrics
from string import digits

remove_digits = str.maketrans('', '', digits)
path='/graphics/scratch2/staff/Hassan/genius_crawl/genius_data/'


config=dotdict({})
#min number of unique words per line, otherwise omit
config.min_unique_word_per_line=2
#max number of words per line, otherwise omit
config.max_line_length=20

#some artist have very few english songs, so lets set a min limit for this
config.min_en_lyrics=20
#min number of tacks an artist must have, also number of tracks we use from each artist
config.min_tracks=50



#lyrics should contain at least 15 lines with at least 4 unique words
config.min_line_per_track=15
config.min_unique_word_per_line_in_track=4

# max  number of lines per lyric
config.max_lines=50

data={}
all_lines_count=0
all_songs_count=0

corrupted_files=['Lit genius']
skipped_tracks=0

for c, name in enumerate(tqdm(os.listdir(path))):

    with open('{}{}'.format(path,name)) as f:
       artist_collection = json.load(f)
    name=name[:-5] # remove .json

    #some files contain garbage
    if name not in corrupted_files:

        if len(artist_collection['songs']) >= config.min_tracks:
            data[name] = []

            for song in artist_collection['songs'][0:config.min_tracks]:

                #first check the language and make sure it contains something
                lyrics=song['lyrics'].split('\n')[1:]
                if song['language']=='en' and len(lyrics) > 1:
                    song_data = {'title': 0, 'lyrics': 0}

                    # preprocess the lyrics
                    lyrics=preprocess_lyrics(lyrics,config)

                    if lyrics:
                        all_songs_count +=1
                        all_lines_count += len(lyrics)
                        song_data['title']=song['full_title']
                        song_data['lyrics'] =lyrics

                        #add the lyric to the artist list
                        data[name].append(song_data)
                    else:
                        skipped_tracks +=1

            #remove artist if no/or not enough english lyrics were added
            if len(data[name]) < config.min_en_lyrics:
                all_songs_count = all_songs_count -  len(data[name])
                del data[name]

print('processed probably less than {} number of lines from exactly {} songs :'.format(all_lines_count,all_songs_count))
print('skipped {} number of tracks(lyrics)'.format(skipped_tracks))

save_pkl(data,'dataset_'+str(config.max_lines))

