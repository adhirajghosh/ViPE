import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from bert_score import score
import json
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from tqdm import tqdm
import evaluate
import pickle


# from evaluate import load
# bertscore = load("bertscore")
# predictions = ["hello there", "general kenobi"]
# references = ["hello there", "general kenobi"]
# results = bertscore.compute(predictions=predictions, references=references, lang="en")
# bertscore.compute(predictions=['a group of cows are grazing in a grassy field'], references=['a statue of a cow is placed on top of a building'], lang="en")
# {'precision': [0.9035313129425049], 'recall': [0.8925918936729431], 'f1': [0.8980282545089722], 'hashcode': 'roberta-large_L17_no-idf_version=0.3.12(hug_trans=4.29.2)'}



def calculate_bert_score(predictions, references):
    # Convert the dictionaries to lists of captions
    pred_captions = [item['caption'] for item in predictions]
    ref_captions = [item['caption'] for item in references['annotations']]

    # Compute BERTScore
    P, R, F1 = score(pred_captions, ref_captions, lang='en', verbose=True,device='cuda',batch_size=3072,nthreads=8)

    # Print the BERTScore
    pred=f"BERTScore: P={P.mean().item():.4f}, R={R.mean().item():.4f}, F1={F1.mean().item():.4f}"
    print(pred)
    return pred



def compute_google_bleu(predictions, references):
    pred_captions = [item['caption'] for item in predictions]
    ref_captions = [item['caption'] for item in references['annotations']]

    google_bleu = evaluate.load("google_bleu")
    result = google_bleu.compute(predictions=pred_captions, references=ref_captions)
    print(result)

    return result

# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


from torch.nn.functional import cosine_similarity

def compute_cosine_similarity(embeddings_1, embeddings_2):
    # Compute cosine similarity between embeddings_1 and embeddings_2
    similarities = cosine_similarity(embeddings_1, embeddings_2)

    return similarities


import torch
from torch.utils.data import Dataset, DataLoader

nli=True
if nli:
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/nli-mpnet-base-v2')
else:
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')



class SentenceDataset(Dataset):
    def __init__(self, sentences):
        self.sentences = sentences

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        return self.sentences[index]

def collate_fn(batch):
    # Tokenize sentences
    encoded_inputs = tokenizer(batch, padding=True, truncation=True, return_tensors='pt')

    return encoded_inputs

def get_mpnet_embed_batch(predictions, ground_truth, batch_size=10000):
    # Convert the dictionaries to lists of captions
    # sentences_1 = [item['caption'] for item in predictions]
    # sentences_2 = [item['caption'] for item in ground_truth['annotations']]
    #
    # sentences_1=predictions
    # sentences_2=ground_truth
    # Create data loaders

    sentences_1 = predictions
    sentences_2 = ground_truth
    dataset_1 = SentenceDataset(sentences_1)
    dataset_2 = SentenceDataset(sentences_2)

    dataloader_1 = DataLoader(dataset_1, batch_size=batch_size, collate_fn=collate_fn,num_workers=4)
    dataloader_2 = DataLoader(dataset_2, batch_size=batch_size, collate_fn=collate_fn,num_workers=4)

    # Load model from HuggingFace Hub
    if nli:
        model = AutoModel.from_pretrained('sentence-transformers/nli-mpnet-base-v2')
    else:
        model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Compute token embeddings
    embeddings_1 = []
    embeddings_2 = []

    with torch.no_grad():
        for count, (batch_1, batch_2) in enumerate(zip(dataloader_1, dataloader_2)):
            if count % 50 ==0:
                print(count,' out of ',len(dataloader_2))
            batch_1 = {key: value.to(device) for key, value in batch_1.items()}
            batch_2 = {key: value.to(device) for key, value in batch_2.items()}

            model_output_1 = model(**batch_1)
            model_output_2 = model(**batch_2)

            sentence_embeddings_1 = mean_pooling(model_output_1, batch_1['attention_mask'])
            sentence_embeddings_2 = mean_pooling(model_output_2, batch_2['attention_mask'])

            embeddings_1.append(sentence_embeddings_1)
            embeddings_2.append(sentence_embeddings_2)

    # Concatenate embeddings
    embeddings_1 = torch.cat(embeddings_1)
    embeddings_2 = torch.cat(embeddings_2)

    # Normalize embeddings
    embeddings_1 = torch.nn.functional.normalize(embeddings_1, p=2, dim=1)
    embeddings_2 = torch.nn.functional.normalize(embeddings_2, p=2, dim=1)

    # Compute cosine similarity
    similarities = compute_cosine_similarity(embeddings_1, embeddings_2)

    # Average cosine similarity
    average_similarity = torch.mean(similarities)

    return average_similarity



saving_dir='/graphics/scratch2/staff/Hassan/checkpoints/lyrics_to_prompts/ml_logs_checkpoints/gpt2/evaluation/'
#saving_dir='/graphics/scratch2/staff/Hassan/checkpoints/lyrics_to_prompts/gpt2_v2.0/evaluation/'


model_name='gpt2_context_ctx_7_lr_5e-05-v4'
#model_name='gpt2_context_ctx_0_lr_5e-05-v4'
model_name='gpt2_context_ctx_0_lr_5e-05-v4'
model_name='gpt2_context_ctx_7_lr_5e-05-v4'


# GT = '{}ground_truth.json'.format(saving_dir)
# pred='{}generation_{}.ckpt_.json'.format(saving_dir,model_name)
#
#
# with open(pred, 'r') as file:
#     predictions=json.load(file)
#
# with open(GT, 'r') as file:
#     ground_truth=json.load(file)


#_________________
# average_similarity =  compute_google_bleu(predictions, ground_truth)
# jsonString = json.dumps(str(average_similarity))
# jsonFile = open(saving_dir +"google_bleu_{}.json".format(model_name), "w")
# jsonFile.write(jsonString)
# jsonFile.close()

# _________________

# get_mpnet_embed_batch(['an astronaut is on a space mission'], ['a group of people looking at spaceship in the sky'], batch_size=512)
# # 30.37
# get_mpnet_embed_batch(['a group of cows are grazing in a grassy field'],['a statue of a cow is placed on top of a building'])
# 0  out of  1
# tensor(0.2469, device='cuda:0')
# get_mpnet_embed_batch(['a group of cows are grazing in a grassy field'],['a statue of a cow'])
# 0  out of  1
# tensor(0.2766, device='cuda:0')
# get_mpnet_embed_batch(['a group of cows are grazing in a grassy field'],['a statue of a man'])
# 0  out of  1
# tensor(-0.0756, device='cuda:0')
# get_mpnet_embed_batch(['a group of cows are grazing in a grassy field'],['a group of cows are eating'])
# 0  out of  1
# tensor(0.7336, device='cuda:0')

with open('./datasets/retrieval/prompt_dict_chatgpt.pickle', 'rb') as handle1:
    chatgpt_id = pickle.load(handle1)
with open('./datasets/retrieval/prompt_dict_haivmet.pickle', 'rb') as handle2:
    haivmet_id = pickle.load(handle2)
with open('./datasets/retrieval/prompt_dict_vipe.pickle', 'rb') as handle3:
    vipe_id = pickle.load(handle3)

for i in range(3):
    if i ==0:
        dataset = 'haivmet'
        corpus = haivmet_id
    elif i == 1:
        dataset = 'chatgpt'
        corpus = chatgpt_id
    elif i == 2:
        dataset = 'vipe'
        corpus = vipe_id

    ground_truth = [k for k, _ in corpus.items()]
    predictions = [w for _,(v,w )in corpus.items()]
    average_similarity =  get_mpnet_embed_batch(predictions, ground_truth, batch_size=512)
    print(dataset)
    print(f"Average Cosine Similarity: {average_similarity.item():.4f}")
    # jsonString = json.dumps(f"Average Cosine Similarity: {average_similarity.item():.4f}")
    # if nli:
    #     jsonFile = open(saving_dir + "nli_mpnet_score_{}_{}.json".format(dataset,model_name), "w")
    # else:
    #     jsonFile = open(saving_dir + "mpnet_score_{}_{}.json".format(dataset, model_name), "w")
    #
    # jsonFile.write(jsonString)
    # jsonFile.close()

#___________________
# evaluation_result=calculate_bert_score(predictions,ground_truth)
# jsonString = json.dumps(evaluation_result)
# if do_sample:
#     jsonFile = open(saving_dir +"random_bertScore_{}.json".format(model_name), "w")
# else:
#     jsonFile = open(saving_dir + "bertScore_{}.json".format(model_name), "w")
# jsonFile.write(jsonString)
# jsonFile.close()