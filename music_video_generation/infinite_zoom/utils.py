import whisper
import ffmpeg
import json
import os
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from generation import generate_from_sentences

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def whisper_transcribe(  audio_fpath="audio.mp3", device='cuda'):
    model = whisper.load_model('large').to(device)
    whispers= model.transcribe(audio_fpath, task='translate')
    return whispers

def prepare_ViPE(args):
    model = GPT2LMHeadModel.from_pretrained(args.checkpoint)
    model.to(args.device)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def get_duration(mp3_file):
    return float(ffmpeg.probe(mp3_file)['format']['duration'])


#get a list of sentences and append them together based on the context size
def prepare_lyrics(lyrics, context_size):

    for _ in range(context_size+1):
        lyrics = ['null'] + lyrics
    lyrics = [lyrics[i:i + context_size+1] for i in range(len(lyrics) - context_size )]
    for c, text in enumerate(lyrics):
        text='; '.join([t for t in text if t != 'null'])
        lyrics[c] =  text

    lyrics.pop(0)

    return lyrics


def get_lyrtic2prompts(args):
    # if the prompt_file does not exist, generate it again
    if not os.path.isfile(args.prompt_file):
        #check if we have the transcription_file already
        if not os.path.isfile(args.transcription_file):
            transcription = whisper_transcribe(args.mp3_file,args.device)
            with open(args.transcription_file, 'w') as file:
                json.dump(transcription,file, indent = 6)

        # laod the transcription_file
        with open(args.transcription_file, 'r') as file:
            transcription= json.load(file)['segments']

        model, tokenizer=prepare_ViPE(args)

        #preapare_lyrics
        processed_lyrics = prepare_lyrics([line['text'] for line in transcription], args.context_size)
        #Add the prompts to the transcription
        for i, lines in enumerate(transcription):
            print('generating prompt using ViPE {} out of {}'.format(i+1, len(transcription)))
            lyric = processed_lyrics[i]
            # prompt = generate_from_sentences([lyric], model, tokenizer, hparams.device, do_sample)[
            #     0].replace(lyric, '')
            prompt=generate_from_sentences([lyric], model, tokenizer,device=args.device, do_sample=args.do_sample)
            #print(prompt)
            lines['prompt'] = prompt

        #Add start of the song if the lyrics don't start at the beginning
        x = {}
        if transcription[0]['start'] != 0.0:
            x['start'] = 0.0
            x['end'] = transcription[0]['start']
            x['text'], x['prompt'] = args.music_gap_prompt, generate_from_sentences(args.music_gap_prompt, model, tokenizer,device=args.device, do_sample=args.do_sample)
            transcription.insert(0,x)

        #get the duration of the song
        song_length = get_duration(args.mp3_file)
        #In case bugs exist towards the end of the transcriptions
        if transcription[-1]['end'] > song_length:
            transcription[-1]['start'] = transcription[-2]['end']
            transcription[-1]['end'] = song_length

        with open(args.prompt_file, 'w') as file:
            json.dump(transcription,file, indent = 6)

    with open(args.prompt_file, 'r') as file:
        lyric2prompt = json.load(file)

    return lyric2prompt