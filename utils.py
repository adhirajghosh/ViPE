import csv
import gensim
import sys
import imageio
import os
import spacy

def song_to_list(path, lemma):
    # with open(path) as f:
    #     lines = f.readlines()
    # lines.append('\n')
    # full_song = []
    # stanza = []
    # for i in lines:
    #     if i == '\n':
    #         full_song.append(stanza)
    #         stanza = []
    #     else:
    #         stanza.append(i[:-1])
    # return full_song

    nlp = spacy.load("en_core_web_sm")
    with open(path) as f:
        lines = f.readlines()
    lines.append('\n')
    full_song = []
    stanza = []

    for i in lines:
        if i == '\n':
            full_song.append(stanza)
            stanza = []
        else:
            if lemma:
                doc = nlp(i[:-1])
                lemmatised_text = ' '.join([token.lemma_ for token in doc])
                stanza.append(lemmatised_text)
            else:
                new_line = []
                for j in i[:-1].split(' '):
                    if "'" in j:
                        doc = nlp(j)
                        lemmatised_text = ' '.join([token.lemma_ for token in doc])
                        new_line.append(lemmatised_text)
                    else:
                        new_line.append(j)
                stanza.append(" ".join(new_line))

    return full_song

def concrete_score(path):
    csv.field_size_limit(sys.maxsize)
    file = open(path,"r")
    data = list(csv.reader(file, delimiter=","))
    list_x = []
    for i in data[1:]:
        key = i[0]
        value = float(i[1])
        list_x.append([key, value])
    word_score = dict(list_x)
    file.close()
    return word_score

def change_lyric(emb_path, lyric, word_score):
    embeddings_300d = gensim.models.KeyedVectors.load_word2vec_format(
        emb_path, binary=True)

    new_lyric = ''
    for root_word in lyric.split(" "):
        if root_word not in word_score.keys():
            print(root_word)
            new_lyric = " ".join([new_lyric, root_word])
            continue

        replaced_word = ''
        i = 0.0
        bracket = 0
        topn = 100
        max_score = 8.0
        similar = embeddings_300d.most_similar(root_word, topn=topn)
        while i < max_score:
            sim_words = similar[bracket:bracket + 20]
            for word, _ in sim_words:
                if word not in word_score.keys():
                    continue
                score = word_score[word]
                if score > i:
                    i = score
                    if word_score[word] > word_score[root_word]:
                        replaced_word = word
                    else:
                        replaced_word = root_word

            bracket = bracket + 20
            if bracket > topn:
                max_score = max_score - 0.2
                similar = embeddings_300d.most_similar(word, topn=bracket)
                topn = bracket
        new_lyric = " ".join([new_lyric, replaced_word])

    new_lyric = new_lyric[1:]
    return new_lyric

def gif(result_path, output_path, fps):
    images = []
    for filename in os.listdir(result_path):
        images.append(imageio.imread(result_path + filename))
    imageio.mimsave(output_path, images, fps = fps)

