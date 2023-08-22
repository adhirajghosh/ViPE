"""
using coco API to evaluate the generated captions (prompts)
"""
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
import numpy as np
import json
from json import encoder
import pandas as pd
from utils import load_pkl
from utils import to_coco_format
import os

saving_dir='/graphics/scratch2/staff/Hassan/checkpoints/lyrics_to_prompts/ml_logs_checkpoints/gpt2-medium/evaluation/'

#saving_dir='/graphics/scratch2/staff/Hassan/checkpoints/lyrics_to_prompts/gpt2-medium_v2.0/evaluation/'

model_name='gpt2_context_ctx_0_lr_5e-05-v4.ckpt_.json'
model_name='gpt2-medium_context_ctx_5_lr_5e-05-v4.ckpt_.json'
#model_name='gpt2-medium_context_ctx_3_lr_5e-05-v3.ckpt_.json'

do_sample=False

GT = '{}ground_truth.json'.format(saving_dir)
# create coco object and cocoRes object
coco = COCO(GT)

#testing the evluation with GT captions
# id2captions = {}
# for name, cap in zip(name2cap['names'], name2cap['captions']):
#     id = int(name[6:-4])
#     if id in id2captions:
#         id2captions[id].append(cap)
#     else:
#         id2captions[id] = [cap]
#
#
# results=[]
# for id, cap in id2captions.items():
#     results.append({'image_id': id, 'caption': id2captions[id][1]})
# jsonString = json.dumps(results)
# jsonFile = open(SAVING_DIR + "testing.json", "w")
# jsonFile.write(jsonString)
# jsonFile.close()
################################################

file_name=saving_dir +'generation_{}'.format(model_name)

if do_sample:
    file_name = saving_dir + 'random_generation_{}'.format(model_name)

cocoRes = coco.loadRes(file_name)
#cocoRes = coco.loadRes(SAVING_DIR + "testing.json")

# create cocoEval object by taking coco and cocoRes
cocoEval = COCOEvalCap(coco, cocoRes)

# evaluate on a subset of images by setting
# cocoEval.params['image_id'] = cocoRes.getImgIds()
# please remove this line when evaluating the full validation seint(mt
#cocoEval.params['image_id'] = cocoRes.getImgIds()

# evaluate results
# SPICE will take a few minutes the first time, but speeds up due to caching
cocoEval.evaluate()

# print output evaluation scores
evaluation_result={}
for metric, score in cocoEval.eval.items():
    print ('%s: %.3f'%(metric, score))
    evaluation_result[metric]=score

jsonString = json.dumps(evaluation_result)
if do_sample:
    jsonFile = open(saving_dir +"random_results_{}.json".format(model_name), "w")
else:
    jsonFile = open(saving_dir + "results_{}.json".format(model_name), "w")
jsonFile.write(jsonString)
jsonFile.close()