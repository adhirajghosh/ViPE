import os

os.environ['CUDA_VISIBLE_DEVICES'] = "2"

from tqdm import tqdm
import torch
import numpy as np
import argparse
from utils import prepare_ViPE, generate_from_tokens, SentenceDataset, collator_cl, write_list_to_json
from profanity_check import predict_prob
import pandas as pd
from torch.utils.data import DataLoader
import json
import matplotlib.pylab as plt


def parse_args():
    parser = argparse.ArgumentParser(description="general information about the lyrics")

    parser.add_argument(
        "--lyric_canvas_path", type=str, default='/graphics/scratch2/staff/hassan/genuis_chatgpt/lyric_canvas.csv',
        help='path to lyric canvas file'
    )
    parser.add_argument(
        "--vipe_path", type=str,
        default='/graphics/scratch2/staff/hassan/saved_models_mine/gpt2-medium_context_ctx_7_lr_5e-05-v4.ckpt/',
        help='vipe checkpoint directory'
    )
    parser.add_argument(
        "--batch_size", type=int,
        default=400,
        help='batch size for vipe prompt generation'
    )

    return parser.parse_args()


def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.device = device
    model_types = ['vipe', 'gpt2']

    batch_size = args.batch_size

    lyric_canvas = pd.read_csv(args.lyric_canvas_path)
    lyric_canvas = lyric_canvas.sample(frac=1, random_state=0).reset_index(drop=True)
    valid_index = int(.10 * len(lyric_canvas))
    lyric_canvas = lyric_canvas[0:valid_index]
    print('using ', len(lyric_canvas), ' validation samples')

    lyrics = lyric_canvas['lyrics'].tolist()
    chatgpt_prompts = lyric_canvas['prompts'].tolist()

    # Create a list of indices to remove
    indices_to_remove = [c for c, i in enumerate(lyrics) if str(i) == 'nan']

    # Create a new list containing non-"nan" values
    new_lyrics = [lyrics[i] for i in range(len(lyrics)) if i not in indices_to_remove]
    new_chatgpt_prompts = [chatgpt_prompts[i] for i in range(len(chatgpt_prompts)) if i not in indices_to_remove]
    print('Removed', len(indices_to_remove), 'NAN samples')
    # Update the lists with the new ones without "nan" values
    lyrics = new_lyrics
    chatgpt_prompts = new_chatgpt_prompts

    for model_type in model_types:

        saving_path = 'results/{}_list'.format(model_type)
        if not os.path.exists(saving_path):

            model, tokenizer = prepare_ViPE(args, model_type)
            dataset = SentenceDataset(lyrics)

            if model_type != 'vipe':
                collate_fn = collator_cl(tokenizer, False)
            else:
                collate_fn = collator_cl(tokenizer, True)

            lyrics_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=1)

            vipe_prompts = []
            for batch in tqdm(lyrics_loader):
                with torch.no_grad():
                    prompts = generate_from_tokens(batch, model, tokenizer, device=args.device,
                                                   do_sample=False, top_k=None,
                                                   epsilon_cutoff=None, temperature=1)

                    vipe_prompts.extend(prompts)

            write_list_to_json(vipe_prompts, saving_path)
            print('done generating prompts using ' + model_type)

    vipe_list = 'results/{}_list'.format(model_types[0])
    gpt2_list = 'results/{}_list'.format(model_types[1])

    with open(vipe_list, 'r') as json_file:
        vipe_prompts = json.load(json_file)

    with open(gpt2_list, 'r') as json_file:
        gpt2_prompts = json.load(json_file)

    vipe_scores = predict_prob(vipe_prompts)
    lyric_scores = predict_prob(lyrics)
    chatgpt_scores = predict_prob(chatgpt_prompts)
    gpt2_scores = predict_prob(gpt2_prompts)

    # Concatenate all scores
    results = {'Lyrics': lyric_scores, 'GPT3.5': chatgpt_scores, 'ViPE-M': vipe_scores, 'GPT2-M': gpt2_scores}

    # Define bins
    bins = np.linspace(0, 1, 6)

    line_styles = ['-', '--',':', '-.']

    # Calculate means for other lists based on the indices from the first list
    for idx, (method, scores) in enumerate(results.items()):
            bin_means = []
            for i in range(len(bins) - 1):
                indices = (lyric_scores > bins[i]) & (lyric_scores <= bins[i + 1])
                mean_value = np.mean(scores[indices])
                bin_means.append(mean_value)
            plt.plot(bin_means, label=method, linestyle=line_styles[idx % len(line_styles)])

    # for method, scores in results.items():
    #         bin_means = []
    #         for i in range(len(bins) - 1):
    #             indices = (lyric_scores > bins[i]) & (lyric_scores <= bins[i + 1])
    #             mean_value = np.mean(scores[indices])
    #             bin_means.append(mean_value)
    #         plt.plot(bin_means, label=method)

    # Add labels and title
    plt.xlabel('Profanity Intervals')
    plt.ylabel('Mean Value')
    #plt.title('Mean Values in Bins for Different Methods (Based on Lyrics)')
    plt.xticks(range(5), ['(0, 0.2]', '(0.2, 0.4]', '(0.4, 0.6]', '(0.6, 0.8]', '(0.8, 1.0]'])
    plt.legend()
    plt.show()
    # data = np.vstack([v for _,v in results.items()]).T
    # sns.violinplot(data=data)
    # plt.xticks([0, 1, 2, 3], list(results.keys()))
    # plt.ylabel('Scores')
    # plt.title('Violin Plot of List Scores')
    # plt.show()

    # line_styles = ['solid', 'dashed', 'dashdot', 'dotted']
    # colors = ['blue', 'red', 'black', 'purple']
    # new_indices=0

    for c, (name, scores) in enumerate(results.items()):
        m = np.mean(scores)
        sig = np.std(scores)
        print('{} : mean: {}, std: {}'.format(name, round(m, 4), round(sig, 4)))
        # if c==0:# sort the the scores for the lyrics and then reorder the other scores accordingly
        #     indexed_values = list(zip(scores, range(len(scores))))
        #     # Sort the indexed_values list based on the values
        #     sorted_values = sorted(indexed_values, key=lambda x: x[0])
        #     # Extract the new order of indices after sorting
        #     new_indices = [index for value, index in sorted_values]
        #
        # # Reorder the other lists based on the new indices
        # scores = [scores[index] for index in new_indices]
        # scores.sort()
        # plt.hist(scores, bins=100,color=colors[c],label=name)
        # plt.plot(range(len(scores)),scores, color=colors[c],linestyle=line_styles[c], label=name)

    # plt.legend()
    # plt.show()

if __name__ == "__main__":
    main()
