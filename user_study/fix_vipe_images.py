import pickle
with open('/graphics/scratch2/students/ghoshadh/SongAnimator/datasets/retrieval/metaphor_id.pickle', 'rb') as handle:
    metaphor_id = pickle.load(handle)

with open('/graphics/scratch2/students/ghoshadh/SongAnimator/datasets/retrieval/prompt_dict_vipe.pickle', 'rb') as handle:
    vipe = pickle.load(handle)

vipe={k.strip().replace('\n', '').replace('.','').replace('\xa0', ' ') :v for k,v in vipe.items()}

with open('/graphics/scratch2/students/ghoshadh/SongAnimator/datasets/retrieval/user_study/user_study.txt', 'r') as file:
    lines=file.readlines()

lines=[i.strip().replace('\n', '').replace('.','') for i in lines ]

vip_m2p = {}

# Iterate through lines
for c, line in enumerate(lines):

    if line.isdigit():
        vip_m2p[lines[c+1]] = lines[c]
import shutil
import os

def get_img(paths, img_id, out_path, out_name):
    for path in paths:
        for filename in os.listdir(path):
            if img_id in filename and filename.endswith('.png'):
                src_path = os.path.join(path, filename)
                dest_path = os.path.join(out_path, out_name + '.png')
                shutil.copy(src_path, dest_path)
                print(f"Image {img_id} found and copied to {out_path}")
                return  # If you want to stop after finding the first image


path='/graphics/scratch2/students/ghoshadh/SongAnimator/datasets/retrieval/vipe/train/image_{}/'
out_path='/graphics/scratch2/staff/hassan/user_study/vipe/'
paths=[path.format('train'), path.format('test')]


#get the image number
for c, metaphor in enumerate(vip_m2p.keys()):
    image_info=vipe[metaphor]
    get_img(paths, image_info[0], out_path, vip_m2p[metaphor])



