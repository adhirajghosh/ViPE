import pandas as pd
from tqdm import tqdm
data=pd.read_csv('/graphics/scratch2/staff/hassan/genuis_chatgpt/lyric_canvas.csv')

ll=[]
pl=[]
for l, p in tqdm(zip(data['lyrics'],data['prompts'])):
    ll.append(len(str(l).split(' ')))
    pl.append(len(str(p).split(' ')))
    if pl[-1] > 25:
        print(p)
print('total samples :', len(data))
print('wrong file in lyrics ', sum([1 for i in ll if i ==1]))
print('max len in liyrics ', max(ll))
print('max len in prompts ', max(pl))
print('min len in liyrics ', min(ll))
print('min len in prompts ', min(pl))

d=2