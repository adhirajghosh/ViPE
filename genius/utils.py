import pickle
import numpy as np
import string
from string import digits
remove_digits = str.maketrans('', '', digits)

def save_pkl(file,name):
    with open(name+'.pickle', 'wb') as handle:
        pickle.dump(file, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_pkl(name):
    with open(name + '.pickle', 'rb') as handle:
        file = pickle.load(handle)
    return file

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__




def remove_non_ascii(a_str):
    ascii_chars = set(string.printable)

    return ''.join(
        filter(lambda x: x in ascii_chars, a_str)
    )

def preprocess_lyrics(lyrics,config):

    if 'You might also like' in lyrics:
        lyrics.remove('You might also like')

    # sanity check: lyrics should contain at least 10 lines with more than 3 unique words
    unique_pass_len = len([i for i in lyrics if len(np.unique(i.split(' '))) >= config.min_unique_word_per_line_in_track])
    if unique_pass_len < config.min_line_per_track:
      return False

    new_lyrics=[]
    for line in lyrics:

        # lets remove non ascii characters
        line=remove_non_ascii(line)

        line_len_unique = len(np.unique(line.split(' ')))
        line_len = len(line.split(' '))

        #check the number of word counts
        if line_len_unique >= config.min_unique_word_per_line and  line_len <= config.max_line_length:
            new_lyrics.append(line)
    lyrics=new_lyrics


   #still contain 'min_line_per_track' number of lines?
    if len(lyrics) <config.min_line_per_track:
        return False

    # some lyrics contain '(Verse 1)', lets remove it
    lyrics[0] = lyrics[0].replace('(Verse 1)', ' ')

    # truncate the long lyrics because chatgpt gets confused
    if len(lyrics) > config.max_lines:
        lyrics = lyrics[0:config.max_lines]

    #remove 'Embed' that is attached to the last word in the last line
    elif 'Embed' in lyrics[-1]:
        # remove  23Embed or 1Embed from the last line
        last_line = lyrics[-1].split(' ')
        last_word = last_line[-1].replace('Embed', '').translate(remove_digits)
        last_line[-1] = last_word
        lyrics[-1] = ' '.join(last_line)



    return lyrics

