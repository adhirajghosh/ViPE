import openai
from utils import load_pkl, save_pkl
import json
import os
from tqdm import tqdm
import numpy as np
import time
import argparse


def parse_args():
	parser = argparse.ArgumentParser(description="lets get rolling!")

	parser.add_argument(
	"--start", type=int, default=0
	)
	parser.add_argument(
	"--end", type=int, default=-1
	)
	parser.add_argument(
		"--path", type=str, default='/graphics/scratch2/staff/Hassan/chatgpt_data/'
	)

	parser.add_argument(
		"--api_key", type=str, default="..."
	)
	args = parser.parse_args()
	return args

def main():
    args=parse_args()

    openai.api_key = args.api_key
    path=args.path
    start_indx=args.start
    end_indx=args.end

    #system role
    f = open("system_role", "r")
    system_role=f.read()

    #all the songs
    data=load_pkl('../GeniusCrawl/dataset_50')

    failures=0
    total_tokens=0
    song_count=1
    miss_aligned_count=0
    response_time=[]

    names=list(data.keys())

    start_all = time.time()
    for _, name in enumerate(tqdm(names[start_indx:end_indx])):
        songs=data[name]

        #print((c+1), ' out of ', len(names[start_indx:end_indx]))

        for song in songs:
            lyrics=song['lyrics']
            title=song['title']

            #create directory
            file_name = os.path.join(path,name.replace('/','-'),title.replace('/','-'))
            os.makedirs(os.path.dirname(file_name), exist_ok=True)

            #check if the output is already there
            if not os.path.isfile(file_name):

                # enumerate the lines
                song = [str(c + 1) + ". " + i for c, i in enumerate(lyrics)]
                song = '\n'.join(song)

                #get a slow response from chatgpt
                start_i = time.time()

                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                            {"role": "user", "content": system_role},
                            #{"role": "user", "content": '\nPrioritize rule number 2 and don\'t use generic terms. Do you understand?'},
                            {"role": "assistant", "content": 'Yes, I understand. Let\'s get started!'},
                            {"role": "user", "content": song},
                        ]
                )


                end_i = time.time()
                time_elapsed_i=(end_i - start_i)
                response_time.append(time_elapsed_i)

                #save the response
                with open(file_name, 'w') as f:
                    json.dump(response, f)

                song_count+=1

                pred_len=len(response.choices[0].message.content.split('\n'))

                # hopefully we get back the same number of lines
                if pred_len !=len(song.split('\n')):
                    miss_aligned_count +=1

                #check the stop reason
                for choice in response.choices:
                    if choice.finish_reason != 'stop':
                        print(choice.finish_reason)
                        failures +=1

                total_tokens += response.usage['total_tokens']

                if song_count %10==0:
                    print('\naverage response time: {} seconds'.format(np.mean(response_time)))

                if song_count % 10 ==0:
                    print('start: {}, end: {}'.format(start_indx, end_indx))
                    print('total songs processed: {}'.format(song_count))
                    print('tokens usage : {} '.format(total_tokens))
                    print('cost : {} dollars '.format(total_tokens * 0.000002))
                    print('number of fail cases: {}'.format(failures))
                    print('number of miss aligned  cases: {}'.format(miss_aligned_count))
                    end_all = time.time()
                    time_elapsed = (end_all - start_all) / 60
                    print('time taken so far: {} mins'.format(time_elapsed))
                    print('________________________________________________')

if __name__ == "__main__":
    main()
