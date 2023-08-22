sp=[.8379, .8027, .8173,.8187,.5410,.60,.77]
vp=[.2227, .2778, .2634, .2418,.2270,.2511,.10]
keys=['FLUTE','Humans','ViPE-M','chatgpt','emotion+_chatgpt','emotion_mine','t5']

import numpy as np


def score(ss):
    return (2/(1+np.exp(- 8 * ss))) -1

for num, (v,s) in enumerate(zip(vp,sp)):
    angular_sim = 1 - np.arccos(v) / np.pi
    #f1=1 * angular_sim
    refined_ang=score(v)
    #print(refined_ang)

    f1=refined_ang * s * 2
    #f1 = (2*np.exp(angular_sim))/(2 * np.e)
    f1 = f1/(refined_ang + s)
    print(keys[num] , ': ', round(f1,4))