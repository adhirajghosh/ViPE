import os
import zipfile
import tqdm
import torch

def zip_process(file):
    with zipfile.ZipFile(file, 'r') as zip_file:
        # Get a list of all file names in the zip file
        file_names = sorted(zip_file.namelist())[4:]
        #     print(file_names)
        # Create an empty dictionary to store the folder names and first file names
        folder_dict = {}

        file_names = [file for file in file_names if len(file.split('/')) == 4 and file.split('/')[-1] != ''
                      and file.split('/')[-1] != 'gpt_prompt.txt' and file.split('/')[-1] != 'dalle_prompt.txt'
                      and file.split('/')[0] != '__MACOSX']

        # Iterate over the file names
        for file_name in sorted(file_names):

            # Extract the folder name and the first file name
            folder_name = file_name.split('/')[2]
            first_file_name = file_name.split('/')[3]
            # Check if the folder name is already in the dictionary
            if folder_name not in folder_dict:
                label = 'An ' + first_file_name.split('An')[-1][1:-4]

                if label[-1] != '.':
                    label = label + '.'
                folder_dict[folder_name] = label

    return folder_dict

def generate_images(pipe, prompt_dict, ds_id, saving_path, batch_size, size, gpu):
    added_prompt = "high quality, HD, 32K, high focus, dramatic lighting, ultra-realistic, high detailed photography, vivid, vibrant,intricate,trending on artstation"
    negative_prompt = 'nude, naked, text, digits, worst quality, blurry, morbid, poorly drawn face, bad anatomy,distorted face, disfiguired, bad hands, missing fingers,cropped, deformed body, bloated, ugly, unrealistic'
    for i in range(4):
        generator = torch.Generator(gpu).manual_seed(i)
        batch = []
        ids = []
        for num, (p_id, prompt) in enumerate(prompt_dict.items()):
            if not os.path.isfile("{}/{}_{}_{}.png".format(saving_path, str(ds_id), p_id, str(i+1))):
                batch.append(prompt+added_prompt)
                ids.append(p_id)
                if len(batch) < batch_size and (num + 1) < len(prompt_dict):
                    continue
                print(prompt)

                #the last batch might not have the same size as batch_size so i used len(batch) instead of len(batch_size)
                images = pipe(batch, generator=generator, negative_prompt=[negative_prompt]*len(batch), height = size, width = size).images
                for num, (img_id, img) in enumerate(zip(ids, images)):
                    img.save("{}/{}_{}_{}.png".format(saving_path, str(ds_id), ids[num], str(i+1)))
                batch = []
                ids = []

def generate_images_retrieval(pipe, ds, save_path, gpu):

    added_prompt = "high quality, HD, 32K, high focus, dramatic lighting, ultra-realistic, high detailed photography, vivid, vibrant,intricate,trending on artstation"
    negative_prompt = 'nude, naked, text, digits, worst quality, blurry, morbid, poorly drawn face, bad anatomy,distorted face, disfiguired, bad hands, missing fingers,cropped, deformed body, bloated, ugly, unrealistic'

    for sample in ds:
        prompt = sample['prompt']
        print(prompt)
        for i in range(5):
            print("generating... ", i)
            image = pipe(prompt=prompt + added_prompt, negative_prompt=negative_prompt).images[0]
            outdir = os.path.join(save_path, sample['text'])
            if not os.path.exists(outdir):
                os.makedirs(outdir)

            if prompt[-1] == '.':
                outpath = os.path.join(outdir, '{}_{}.jpg'.format(prompt[:-1],str(i+1)))
            else:
                outpath = os.path.join(outdir, '{}_{}.jpg'.format(prompt, str(i + 1)))
            image.save(outpath)


