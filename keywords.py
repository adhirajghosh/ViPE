import gensim

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import csv
import sys


csv.field_size_limit(sys.maxsize)
file = open("/graphics/scratch2/students/ghoshadh/datasets/ac_EN_ratings/AC_ratings_google3m_koeper_SiW_fix.csv", "r")
data = list(csv.reader(file, delimiter=","))
list_x = []
for i in data[1:]:
    key = i[0]
    value = float(i[1])
    list_x.append([key,value])
word_score = dict(list_x)

file.close()
embeddings_300d = gensim.models.KeyedVectors.load_word2vec_format(
    '/graphics/scratch/shahmoha/checkpoints/final models/fast_text/normal_fasttext_gensim' , binary=True)

lyric = 'i am not the sharpest tool in the shed'
lyric = lyric.split(' ')
new_lyric = ''
for root_word in lyric:
    if root_word not in word_score.keys():
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
print(new_lyric)