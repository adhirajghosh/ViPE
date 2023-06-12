import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'

from utils import prepare_fig_qa, get_mpnet_embed_batch,calculate_bert_score
from datasets import load_dataset
#train_loader, valid_loader, test_loader = prepare_fig_qa(batch_size=32)
from tqdm import tqdm

# Load the Fig-Questions dataset
dataset = load_dataset("nightingal3/fig-qa")

# Get the train, validation, and test datasets
train_dataset = dataset['train']
valid_dataset = dataset['validation']
test_dataset = dataset['test']
all_data=[train_dataset,valid_dataset]

start_sentences=[]
options_1=[]
options_2=[]
labels=[]
data={'start_sentences':start_sentences,'options_1':options_1,'options_2':options_2, 'labels':labels}

for cur_data in all_data:
    for ct, batch in tqdm(enumerate(cur_data)):
        for k,v in zip(data.keys(),batch.values()):
            data[k].append(v)

similarities_1=get_mpnet_embed_batch(data['start_sentences'],data['options_1'],batch_size=500).cpu().numpy()
similarities_2=get_mpnet_embed_batch(data['start_sentences'],data['options_2'],batch_size=500).cpu().numpy()
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
import numpy as np

# similarities_1=calculate_bert_score(data['start_sentences'],data['options_1'])
# similarities_2=calculate_bert_score(data['start_sentences'],data['options_2'])

x=np.stack((similarities_1,similarities_2),axis=1)
model.fit(x,data['labels'])
pred_lables=model.predict(x)

#pred_lables=[1 if i>j else 0 for i,j in zip(similarities_1,similarities_2) ]

from sklearn.metrics import accuracy_score, f1_score
accuracy = accuracy_score(data['labels'], pred_lables)
f1_result=f1_score(data['labels'], pred_lables)

print(accuracy)
print(f1_result)
#_________________
# average_similarity =  compute_google_bleu(predictions, ground_truth)
# jsonString = json.dumps(str(average_similarity))
# jsonFile = open(saving_dir +"google_bleu_{}.json".format(model_name), "w")
# jsonFile.write(jsonString)
# jsonFile.close()

# _________________
# average_similarity =  get_mpnet_embed_batch(predictions, ground_truth, batch_size=512)
#
# print(f"Average Cosine Similarity: {average_similarity.item():.4f}")
# jsonString = json.dumps(f"Average Cosine Similarity: {average_similarity.item():.4f}")
# if do_sample:
#     jsonFile = open(saving_dir + "mpnet_score_{}.json".format(model_name), "w")
# else:
#     jsonFile = open(saving_dir + "random_mpnet_score_{}.json".format(model_name), "w")
#
# jsonFile.write(jsonString)
# jsonFile.close()
#
# #___________________
# evaluation_result=calculate_bert_score(predictions,ground_truth)
# jsonString = json.dumps(evaluation_result)
# if do_sample:
#     jsonFile = open(saving_dir +"random_bertScore_{}.json".format(model_name), "w")
# else:
#     jsonFile = open(saving_dir + "bertScore_{}.json".format(model_name), "w")
# jsonFile.write(jsonString)
# jsonFile.close()