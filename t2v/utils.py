import os
from pytube import YouTube
from PIL import Image
import numpy as np
import torch
from pathlib import Path
import torchvision
from torchvision.transforms import Resize, InterpolationMode
import imageio
from einops import rearrange
import cv2
from PIL import Image
import decord
# from moviepy.editor import ImageSequenceClip, AudioFileClip
import whisper


def create_video(frames, fps, rescale=False, path=None, watermark=None):
    if path is None:
        dir = "temporal"
        os.makedirs(dir, exist_ok=True)
        path = os.path.join(dir, 'movie.mp4')

    outputs = []
    for i, x in enumerate(frames):
        x = torchvision.utils.make_grid(torch.Tensor(x), nrow=4)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = (x * 255).numpy().astype(np.uint8)

        outputs.append(x)
        # imageio.imsave(os.path.join(dir, os.path.splitext(name)[0] + f'_{i}.jpg'), x)

    imageio.mimsave(path, outputs, fps=fps)
    return path

def save_images(frames, path=None, frame_index=0):

    for i, x in enumerate(frames):
        print("Generating ", frame_index)
        x = torchvision.utils.make_grid(torch.Tensor(x), nrow=4)
        x = (x * 255).numpy().astype(np.uint8)

        im = Image.fromarray(x)
        # outpath = os.path.join(outdir, 'frame%06d.jpg' % 0)
        outpath = os.path.join(path, 'frame%06d.jpg' % frame_index)
        im.save(outpath)
        frame_index += 1
    return frame_index


def create_gif(frames, fps, rescale=False, path=None, watermark=None):
    if path is None:
        dir = "temporal"
        os.makedirs(dir, exist_ok=True)
        path = os.path.join(dir, 'canny_db.gif')

    outputs = []
    for i, x in enumerate(frames):
        x = torchvision.utils.make_grid(torch.Tensor(x), nrow=4)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = (x * 255).numpy().astype(np.uint8)
        outputs.append(x)
        # imageio.imsave(os.path.join(dir, os.path.splitext(name)[0] + f'_{i}.jpg'), x)

    imageio.mimsave(path, outputs, fps=fps)
    return path

def prepare_video(video_path:str, resolution:int, device, dtype, normalize=True, start_t:float=0, end_t:float=-1, output_fps:int=-1):
    vr = decord.VideoReader(video_path)
    initial_fps = vr.get_avg_fps()
    if output_fps == -1:
        output_fps = int(initial_fps)
    if end_t == -1:
        end_t = len(vr) / initial_fps
    else:
        end_t = min(len(vr) / initial_fps, end_t)
    assert 0 <= start_t < end_t
    assert output_fps > 0
    start_f_ind = int(start_t * initial_fps)
    end_f_ind = int(end_t * initial_fps)
    num_f = int((end_t - start_t) * output_fps)
    sample_idx = np.linspace(start_f_ind, end_f_ind, num_f, endpoint=False).astype(int)
    video = vr.get_batch(sample_idx)
    if torch.is_tensor(video):
        video = video.detach().cpu().numpy()
    else:
        video = video.asnumpy()
    _, h, w, _ = video.shape
    video = rearrange(video, "f h w c -> f c h w")
    video = torch.Tensor(video).to(device).to(dtype)

    # Use max if you want the larger side to be equal to resolution (e.g. 512)
    # k = float(resolution) / min(h, w)
    k = float(resolution) / max(h, w)
    h *= k
    w *= k
    h = int(np.round(h / 64.0)) * 64
    w = int(np.round(w / 64.0)) * 64

    video = Resize((h, w), interpolation=InterpolationMode.BILINEAR, antialias=True)(video)
    if normalize:
        video = video / 127.5 - 1.0
    return video, output_fps


def post_process_gif(list_of_results, image_resolution):
    output_file = "/tmp/ddxk.gif"
    imageio.mimsave(output_file, list_of_results, fps=4)
    return output_file



def song_to_list(path):
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
            stanza.append(i[:-1])
    return full_song

# def merge_mp4(folder_path, output_path):
#
#     # Get the list of MP4 files in the folder
#     file_list = os.listdir(folder_path)
#     file_list = [file for file in file_list if file.endswith('.mp4')]
#
#     # Create a list to store the video clips
#     video_clips = []
#
#     # Iterate over the MP4 files and load them as video clips
#     for file in file_list:
#         file_path = os.path.join(folder_path, file)
#         video_clip = VideoFileClip(file_path)
#         video_clips.append(video_clip)
#
#     # Concatenate the video clips into a single video
#     final_video = concatenate_videoclips(video_clips)
#
#     # Write the final merged video to the output file
#     final_video.write_videofile(output_path, codec='libx264')


def create_video_from_images(images, audio_file, output_file, fps):
    # Create an ImageSequenceClip from the list of images
    clip = ImageSequenceClip(images, fps=fps)

    # Set the duration of the clip based on the number of images and the desired FPS
    duration = len(images) / fps
    clip = clip.set_duration(duration)

    # Load the audio file
    audio = AudioFileClip(audio_file)

    # Set the audio of the clip
    clip = clip.set_audio(audio)

    clip.write_videofile(output_file, codec='libx264', audio_codec='aac', temp_audiofile='temp_audio.m4a',
                         remove_temp=True)

    print(f"Video created: {output_file}")

def generate_from_sentences(text, model, tokenizer,device,do_sample,top_k=100, epsilon_cutoff=.00005, temperature=1):
    text=[tokenizer.eos_token +  i + tokenizer.eos_token for i in text]
    batch=tokenizer(text, padding=True, return_tensors="pt")

    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)


    # Set token type IDs for the prompts
    max_prompt_length=50

    max_length=input_ids.shape[1] + max_prompt_length
    generated_ids = model.generate(input_ids=input_ids,attention_mask=attention_mask, max_length=max_length, do_sample=do_sample,top_k=top_k, epsilon_cutoff=epsilon_cutoff, temperature=temperature)

    pred_caps = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    return pred_caps


def youtube2mp3 (url,outdir):
    yt = YouTube(url)
    video = yt.streams.filter(abr='192kbps').last()

    out_file = video.download(output_path=outdir)
    base, ext = os.path.splitext(out_file)
    song_name = f'{yt.title}.mp3'
    new_file = Path(f'{base}.mp3')
    os.rename(out_file, new_file)

    if new_file.exists():
        print(f'{yt.title} has been successfully downloaded.')
    else:
        print(f'ERROR: {yt.title}could not be downloaded!')

    return song_name

def whisper_transcribe(
        audio_fpath="audio.mp3", device='cuda'):
    whispers = {
        'tiny': None,  # 5.83 s
        'large': None  # 3.73 s
    }
    # accelerated runtime required for whisper
    # to do: pypi package for whisper

    for k in whispers.keys():
        options = whisper.DecodingOptions(
            task='translate',
            language='en',
        )
        # to do: be more proactive about cleaning up these models when we're done with them
        model = whisper.load_model(k).to(device)

        # start = time.time()
        print(f"Transcribing audio with whisper-{k}")

        # to do: calling transcribe like this unnecessarily re-processes audio each time.
        whispers[k] = model.transcribe(audio_fpath, task='translate')  # re-processes audio each time, ~10s overhead?
        # print(f"elapsed: {time.time() - start}")
    return whispers

def img2emb(pipe, image, seed, device):
    # transform = transforms.Compose([
    #     transforms.PILToTensor()
    # ])
    generator = torch.Generator(device=device).manual_seed(seed)
    # image = transform(image).to(device, dtype=torch.float32)
    if isinstance(image, Image.Image):
        image = [image]

    if isinstance(image[0], Image.Image):
        w, h = image[0].size
        w, h = (x - x % 8 for x in (w, h))  # resize to integer multiple of 8

        image = [np.array(i.resize((w, h), resample= Image.Resampling.BICUBIC))[None, :] for i in image]
        image = np.concatenate(image, axis=0)
        image = np.array(image).astype(np.float32) / 255.0
        image = image.transpose(0, 3, 1, 2)
        image = 2.0 * image - 1.0
        image = torch.from_numpy(image).to(device)
    init_latents = pipe.vae.encode(image).latent_dist.sample(generator)
    return init_latents.to(device)

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class CrossFrameAttnProcessor:
    def __init__(self, unet_chunk_size=2):
        self.unet_chunk_size = unet_chunk_size

    def __call__(
            self,
            attn,
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        query = attn.to_q(hidden_states)

        is_cross_attention = encoder_hidden_states is not None
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.cross_attention_norm:
            encoder_hidden_states = attn.norm_cross(encoder_hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        # Sparse Attention
        if not is_cross_attention:
            video_length = key.size()[0] // self.unet_chunk_size
            # former_frame_index = torch.arange(video_length) - 1
            # former_frame_index[0] = 0
            former_frame_index = [0] * video_length
            key = rearrange(key, "(b f) d c -> b f d c", f=video_length)
            key = key[:, former_frame_index]
            key = rearrange(key, "b f d c -> (b f) d c")
            value = rearrange(value, "(b f) d c -> b f d c", f=video_length)
            value = value[:, former_frame_index]
            value = rearrange(value, "b f d c -> (b f) d c")

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states