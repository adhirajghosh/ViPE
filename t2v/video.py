import argparse
import ast
import math
# from music_video_generation.modeling import GPT2Convertor
from t2v.vid import *
from utils import *


def parse_args():
    parser = argparse.ArgumentParser(description='Audio to Lyric Alignment')

    parser.add_argument('--song_name', help='Name of audio file', type=str, default='city_of_angels.mp3')
    parser.add_argument("--model_name", type=str, default='gpt2-medium', help="which gpt2 version to use?")
    parser.add_argument("--diffusion_model", type=str, default='dreamlike-art/dreamlike-photoreal-2.0', help="which stable diffusion checkpoint to use?")
    parser.add_argument("--data_set_dir", type=str, default='/graphics/scratch2/staff/Hassan/genius_crawl/lyrics_to_prompts_v2.0.csv',help='path to the training data')
    parser.add_argument("--checkpoint", type=str, default='/graphics/scratch2/staff/Hassan/checkpoints/lyrics_to_prompts/saved_models_mine/gpt2-medium_context_ctx_7_lr_5e-05-v4.ckpt/', help="path to save the model")
    parser.add_argument("--img_size", type=int, default=600)
    parser.add_argument("--warmup_steps", type=int, default=1e3)
    parser.add_argument("--batch_size", type=int, default=30)
    parser.add_argument("--context_length", type=int, default=5, help='number of previous lines from lyrics as the context')
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument('--chunk', help='chunk interpolation. Type --chunk or --no-chunk', default=False,action=argparse.BooleanOptionalAction)
    parser.add_argument("--fps", type=float, default=10)
    parser.add_argument("--chunk_size", type=float, default=4)
    parser.add_argument("--inf_steps", type=int, default=100)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--eta", type=int, default=0.05)
    parser.add_argument("--device_list", default=[0,1], nargs='+', type=int, help='GPU as a list')
    parser.add_argument('--device', help='cpu', type=str, default='0')
    parser.add_argument('--url', help='youtube url if song needs to be downloaded', type=str, default='https://www.youtube.com/watch?v=xU0PhTs-v-8&ab_channel=BridgetoGrace-Topic')
    parser.add_argument('--songdir', help='Where songs are kept', type=str, default='./mp3/')
    parser.add_argument('--outdir', help='Where results are kept', type=str, default='./results/vids/Thunder/t2v1/')
    parser.add_argument('--timestamps', help='where timesteps are kept', type=str, default='./timestamps/')


    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    assert args.img_size % 8 == 0
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    if not os.path.exists(args.timestamps):
        os.makedirs(args.timestamps)

    if isinstance(args.device_list, list):
        device = f"cuda:{args.device_list[0]}"
        device1 = f"cuda:{args.device_list[1]}"
    else:
        device = f"cuda:{args.device}"

    # Download the song
    if not os.path.exists(os.path.join(args.songdir, args.song_name)):
        print(os.path.join(args.songdir, args.song_name))
        song_name = youtube2mp3(args.url, args.songdir)
    else:
        song_name = args.song_name

    song_path = os.path.join(args.songdir, song_name)
    print(song_path)

    # Load the transcription from whisper. We will use the large model

    # whispers = whisper_transcribe(song_path,device)
    # print(whispers)

    # with open('./timestamps/whispers.txt') as f:
    #     data = f.read()
    #
    # # reconstructing the data as a dictionary
    # whispers = ast.literal_eval(data)

    song_length = float(ffmpeg.probe(song_path)['format']['duration'])

    # model_name = args.model_name
    # dataset_dir = args.data_set_dir
    #
    # context_length = args.context_length
    #
    # #Load the hparams dict for the GPT2 model
    # hparams = dotdict({})
    # hparams.data_dir = dataset_dir
    # hparams.model_name = model_name
    # hparams.context_length = context_length
    # hparams.batch_size = args.batch_size
    # hparams.learning_rate = args.learning_rate
    # if isinstance(args.device, list):
    #     hparams.device = device1
    # else:
    #     hparams.device = device
    # hparams.warmup_steps = args.warmup_steps
    #
    #
    #
    # model = GPT2LMHeadModel.from_pretrained(args.checkpoint)
    # model.to(device)
    # tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
    # tokenizer.pad_token = tokenizer.eos_token
    #
    #
    # if isinstance(args.device, list):
    #     model.to(device1)
    # else:
    #     model.to(device)
    #
    # #Add the prompts to the trancriptions
    # do_sample = True
    # for i, lines in enumerate(whispers['large']['segments']):
    #     lyric = lines['text']
    #     prompt = generate_from_sentences([lyric], model, tokenizer, hparams.device, do_sample)[
    #         0].replace(lyric, '')
    #     print(prompt)
    #     lines['prompt'] = prompt
    #
    # #Easier later on
    # whispers = whispers['large']['segments']
    # whisper_copy = whispers
    #
    # #Add start of the song if the lyrics don't start at the beginning
    # x = {}
    # if whispers[0]['start'] != 0.0:
    #     x['start'] = 0.0
    #     x['end'] = whispers[0]['start']
    #     x['text'], x['prompt'] = " "," "
    #     whispers.insert(0,x)
    #
    # #In case bugs exist towards the end of the transcriptions
    # if whispers[-1]['end'] > song_length:
    #     whispers[-1]['start'] = whispers[-2]['end']
    #     whispers[-1]['end'] = song_length
    #
    # torch.cuda.empty_cache()
    #
    # # if not os.path.exists(args.timestamps):
    # #     os.makedirs(args.timestamps)
    # # print(os.path.join(args.timestamps,song_name.split('.')[0]+'.txt'))
    # # with open(os.path.join(args.timestamps,song_name.split('.')[0]+'.txt'), "w") as output:
    # #     output.write(str(whispers))
    #
    # whispers_save = []
    # for i, lines in enumerate(whispers):
    #     x = {}
    #     x['start'] = lines['start']
    #     x['end'] = lines['end']
    #     x['text'] = lines['text']
    #     x['prompt'] = lines['prompt']
    #     whispers_save.append(x)
    #     print(lines["text"], " ", lines["prompt"])
    #
    #
    # if not os.path.exists(args.timestamps):
    #     os.makedirs(args.timestamps)
    # print(os.path.join(args.timestamps,song_name.split('.')[0]+'.txt'))
    # with open(os.path.join(args.timestamps,song_name.split('.')[0]+'.txt'), "w") as output:
    #     output.write(str(whispers_save))

    with open(args.timestamps) as f:
        data = f.read()

        # reconstructing the data as a dictionary
    whispers = ast.literal_eval(data)

    # Load the stable diffusion img2img and text2img models
    model_id = args.diffusion_model
    download = True

    if isinstance(args.device_list, list):
        text2img = StableDiffusionPipeline.from_pretrained(model_id).to(device1)
    else:
        text2img = StableDiffusionPipeline.from_pretrained(model_id).to(device)

    # text2vid = Model(device=device, dtype=torch.float32)
    # text2img.scheduler = LMSDiscreteScheduler.from_config(text2img.scheduler.config)
    # img2img.scheduler = LMSDiscreteScheduler.from_config(img2img.scheduler.config)

    # Misc
    regenerate_all_init_images = False
    prompt_lag = True
    added_prompt = "extreme detail, high quality, HD, 32K, perfect face, high detailed photography, vivid, vibrant,intricate, dramatic lighting, ultra-realistic,trending on artstation"
    negative_prompt = 'bad face, blurry face, bad anatomy,distorted face, disfiguired, bad hands, missing fingers, nude, naked, text, digits, worst quality, blurry, morbid, poorly drawn face, cropped, deformed body, bloated, ugly, unrealistic'

    frame_index = 0
    uncond_input = text2img.tokenizer(negative_prompt, padding='max_length', max_length=60, truncation=True,
                                     return_tensors="pt")
    with torch.no_grad():
        uncond_embed = text2img.text_encoder(uncond_input.input_ids.to(text2img.device))[0]

    # seed = np.random.randint(1000, 9999)
    seed = np.random.randint(6666666)
    init_a = torch.randn(
        (1, text2img.unet.in_channels, args.img_size // 8, args.img_size // 8),
        generator=torch.Generator(device=text2img.device).manual_seed(seed),
        device=text2img.device
    )


    
    for i, lines in enumerate(whispers):
        start = lines['start']
        end = lines['end']
        print(lines['text'], start, " ", end)

    for i, lines in enumerate(whispers):
        start = lines['start']
        end = lines['end']
        print(lines['prompt'], start, " ", end)

        # Number of images
        num_steps = (int)((end - start) * args.fps)
        prompt_1 = lines['prompt']
        prompt_2 = whispers[i + 1]['prompt'] if i < (len(whispers) - 1) else lines['prompt']
        prompt1_tok = text2img.tokenizer(prompt_1 + added_prompt, padding='max_length', max_length=60, truncation=True,
                                        return_tensors="pt")
        with torch.no_grad():
            prompt1_embed = text2img.text_encoder(prompt1_tok.input_ids.to(text2img.device))[0]

        prompt2_tok = text2img.tokenizer(prompt_2 + added_prompt, padding='max_length', max_length=60, truncation=True,
                                        return_tensors="pt")
        with torch.no_grad():
            prompt2_embed = text2img.text_encoder(prompt2_tok.input_ids.to(text2img.device))[0]

        init_b = torch.randn(
            (1, text2img.unet.in_channels, args.img_size // 8, args.img_size // 8),
            generator=torch.Generator(device=text2img.device).manual_seed(seed),
            device=text2img.device
        )

        if i == 0:
            params = {"t0": 44, "t1": 48, "motion_field_strength_x": 5 , "motion_field_strength_y": 5, "smooth_bg": False,
                      "video_length": math.ceil(0.9*num_steps), "chunk_size":args.chunk_size, "resolution": args.img_size}
            #
            # frame_index, latent = text2vid.process_text2video(prompt_1+added_prompt, model_name=model_id, fps=args.fps, path=args.outdir,
            #                          seed=seed, frame_index = frame_index, **params)
            frame_index = params['video_length']
            for i, t in enumerate(np.linspace(0, 1, math.ceil(0.2*num_steps))):
                print("generating... ", frame_index)
                if args.chunk:
                    prompt2_embed_mod = new_embeds(prompt1_embed, prompt2_embed)
                    cond_embedding = slerp(float(t), prompt1_embed, prompt2_embed_mod)
                else:
                    cond_embedding = slerp(float(t), prompt1_embed, prompt2_embed)
                init = slerp(float(t), init_a.detach(), init_b.detach())

                with autocast("cuda"):
                    im = diffuse(text2img, cond_embedding, init.to(text2img.device), uncond_embed, args.inf_steps, args.guidance_scale,
                                 args.eta)

                image = Image.fromarray(im)
                outpath = os.path.join(args.outdir, 'frame%06d.jpg' % frame_index)
                image.save(outpath)
                frame_index += 1

            init_a = init_b
            bleed_over = math.ceil(0.1*num_steps)
            torch.cuda.empty_cache()

        elif i > 0 and i < len(whispers) - 1:
            init_b = torch.randn(
                (1, text2img.unet.in_channels, args.img_size // 8, args.img_size // 8),
                generator=torch.Generator(device=device).manual_seed(seed),
                device=device
            )
            # params = {"t0":44, "t1": 48, "motion_field_strength_x": 5, "motion_field_strength_y": 5,"resolution": args.img_size,
            #           "smooth_bg": False, "video_length": math.ceil(num_steps-bleed_over- 0.1*num_steps), "chunk_size": args.chunk_size}
            #
            # frame_index, latent = text2vid.process_text2video(prompt_1 + added_prompt, model_name=model_id, fps=args.fps,
            #                                           path=args.outdir,
            #                                           seed=seed, frame_index=frame_index,
            #                                           **params)
            frame_index+=math.ceil(num_steps - bleed_over - 0.1 * num_steps)
            for i, t in enumerate(np.linspace(0, 1, math.ceil(0.2 * num_steps))):
                print("generating... ", frame_index)
                if args.chunk:
                    prompt2_embed_mod = new_embeds(prompt1_embed, prompt2_embed)
                    cond_embedding = slerp(float(t), prompt1_embed, prompt2_embed_mod)
                else:
                    cond_embedding = slerp(float(t), prompt1_embed, prompt2_embed)
                init = slerp(float(t), init_a.detach(), init_b.detach())

                with autocast("cuda"):
                    im = diffuse(text2img, cond_embedding, init.to(text2img.device), uncond_embed, args.inf_steps, args.guidance_scale, args.eta)

                image = Image.fromarray(im)
                outpath = os.path.join(args.outdir, 'frame%06d.jpg' % frame_index)
                image.save(outpath)
                frame_index += 1
            init_a = init_b
            bleed_over = math.ceil(0.1 * num_steps)
            torch.cuda.empty_cache()
        
        elif i ==len(whispers)-1:
            init_b = torch.randn(
                (1, text2img.unet.in_channels, args.img_size // 8, args.img_size // 8),
                generator=torch.Generator(device=device).manual_seed(seed),
                device=device
            )
            for i, t in enumerate(np.linspace(0, 1, num_steps-bleed_over)):
                print("generating... ", frame_index)
                if args.chunk:
                    prompt2_embed_mod = new_embeds(prompt1_embed, prompt2_embed)
                    cond_embedding = slerp(float(t), prompt1_embed, prompt2_embed_mod)
                else:
                    cond_embedding = slerp(float(t), prompt1_embed, prompt2_embed)

                init = slerp(float(t), init_a.detach(), init_b.detach())

                with autocast("cuda"):
                    im = diffuse(text2img, cond_embedding, init.to(text2img.device), uncond_embed, args.inf_steps,
                                 args.guidance_scale, args.eta, t, -1)

                image = Image.fromarray(im)
                outpath = os.path.join(args.outdir, 'frame%06d.jpg' % frame_index)
                image.save(outpath)
                frame_index += 1

                if frame_index == int(song_length)*args.fps:
                    break
            torch.cuda.empty_cache()

    image_files = [os.path.join(args.outdir, i) for i in os.listdir(args.outdir)]
    video_path = args.outdir[:-1]+".mp4"
    create_video_from_images(image_files, song_path, video_path, args.fps)

if __name__ == '__main__':
    fire.Fire(main)