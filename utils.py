import csv
import gensim
import sys
import imageio
import torch
import os
import spacy
from moviepy.editor import VideoFileClip, concatenate_videoclips
import clip
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer, CLIPModel


def song_to_list(path, lemma=False):
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
        images.append(imageio.imread(os.path.join(result_path,filename)))
    imageio.mimsave(output_path, images, fps = fps)

def merge_mp4(folder_path, output_path):

    # Get the list of MP4 files in the folder
    file_list = os.listdir(folder_path)
    file_list = [file for file in file_list if file.endswith('.mp4')]

    # Create a list to store the video clips
    video_clips = []

    # Iterate over the MP4 files and load them as video clips
    for file in file_list:
        file_path = os.path.join(folder_path, file)
        video_clip = VideoFileClip(file_path)
        video_clips.append(video_clip)

    # Concatenate the video clips into a single video
    final_video = concatenate_videoclips(video_clips)

    # Write the final merged video to the output file
    final_video.write_videofile(output_path, codec='libx264')


def cond_clip(embeds, images, gpu="cuda"):
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")

    clip_score = 0.0
    for i, combo in enumerate(zip(embeds, images)):
    # for i in range(len(embeds)):
        cond_embed, image = combo
        # cond_embed = embeds[i]
        # image = images[i]
        # print(cond_embed.shape)
        image_input = processor(images = image,return_tensors="pt")
        image_input = image_input.to(gpu)
        cond_embed = cond_embed.to(gpu)
        model = model.to(gpu)

        with torch.no_grad():
            image_features = model.get_image_features(**image_input)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        cond_embed = cond_embed / cond_embed.norm(dim=-1, keepdim=True)

        print(image_features.shape)
        print(cond_embed.shape)
        clip_score = clip_score + torch.matmul(cond_embed, image_features.T).softmax(dim=-1).squeeze()


    return clip_score/i

@torch.no_grad()
def latent_to_image(latents, SD):
    image = SD.decode_latents(latents)
    image = SD.numpy_to_pil(image)
    return image

def load_gif(path):
    gif = Image.open(path)
    frames = []
    for i in range(1, gif.n_frames):
        gif.seek(i)
        frame = np.array(gif)
        frames.append(torch.tensor(frame).unsqueeze(0))
        frames[-1] = frames[-1].permute(0, 3, 1, 2)
    return torch.cat(frames, dim=0)