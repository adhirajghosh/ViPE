from PIL import Image
import numpy as np
import moviepy.editor as mp
from PIL import Image
import requests
import numpy as np
import cv2

#
#
# def write_video(file_path, frames, fps, reversed=True, start_frame_dupe_amount=0, last_frame_dupe_amount=0):
#     """
#     Writes frames to an mp4 video file
#     :param file_path: Path to output video, must end with .mp4
#     :param frames: List of PIL.Image objects
#     :param fps: Desired frame rate
#     :param reversed: if order of images to be reversed (default = True)
#     """
#     if reversed == True:
#         frames.reverse()
#
#     w, h = frames[0].size
#     fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
#     # fourcc = cv2.VideoWriter_fourcc('h', '2', '6', '4')
#     # fourcc = cv2.VideoWriter_fourcc(*'avc1')
#     writer = cv2.VideoWriter(file_path, fourcc, fps, (w, h))
#
#     ## start frame duplicated
#     for x in range(start_frame_dupe_amount):
#         np_frame = np.array(frames[0].convert('RGB'))
#         cv_frame = cv2.cvtColor(np_frame, cv2.COLOR_RGB2BGR)
#         writer.write(cv_frame)
#
#     for frame in frames:
#         np_frame = np.array(frame.convert('RGB'))
#         cv_frame = cv2.cvtColor(np_frame, cv2.COLOR_RGB2BGR)
#         writer.write(cv_frame)
#
#     ## last frame duplicated
#     for x in range(last_frame_dupe_amount):
#         np_frame = np.array(frames[len(frames) - 1].convert('RGB'))
#         cv_frame = cv2.cvtColor(np_frame, cv2.COLOR_RGB2BGR)
#         writer.write(cv_frame)
#
#     writer.release()




def write_video(file_path, frames, fps, audio_path, reversed=True, start_frame_dupe_amount=0, last_frame_dupe_amount=0):
    """
    Writes frames to a video file with an MP3 audio file overlaid
    :param file_path: Path to output video, must end with .mp4
    :param frames: List of PIL.Image objects
    :param fps: Desired frame rate
    :param audio_path: Path to the MP3 audio file
    :param reversed: if order of images to be reversed (default = True)
    """
    if reversed:
        frames.reverse()

    w, h = frames[0].size

    # Generate the video without saving the frames
    video_clip = mp.ImageSequenceClip([np.array(frame.convert('RGB')) for frame in frames], fps=fps)

    # Set the duration of the video based on the number of frames
    video_duration = len(frames) / fps
    video_clip = video_clip.set_duration(video_duration)

    # Load the audio file
    audio_clip = mp.AudioFileClip(audio_path)

    # Set the audio duration to match the video duration
    audio_duration = audio_clip.duration
    if audio_duration > video_duration:
        audio_clip = audio_clip.subclip(0, video_duration)
    elif audio_duration < video_duration:
        audio_clip = mp.concatenate_audioclips([audio_clip] * int(video_duration / audio_duration))

    # Set the audio of the video clip
    video_clip = video_clip.set_audio(audio_clip)

    # Write the final video file
    video_clip.write_videofile(file_path, codec='libx264', audio_codec='aac', fps=fps)




def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def shrink_and_paste_on_blank(current_image, mask_width):
    """
    Decreases size of current_image by mask_width pixels from each side,
    then adds a mask_width width transparent frame,
    so that the image the function returns is the same size as the input.
    :param current_image: input image to transform
    :param mask_width: width in pixels to shrink from each side
    """

    height = current_image.height
    width = current_image.width

    # shrink down by mask_width
    prev_image = current_image.resize((height - 2 * mask_width, width - 2 * mask_width))
    prev_image = prev_image.convert("RGBA")
    prev_image = np.array(prev_image)

    # create blank non-transparent image
    blank_image = np.array(current_image.convert("RGBA")) * 0
    blank_image[:, :, 3] = 1

    # paste shrinked onto blank
    blank_image[mask_width:height - mask_width, mask_width:width - mask_width, :] = prev_image
    prev_image = Image.fromarray(blank_image)

    return prev_image


def load_img(address, res=(512, 512)):
    if address.startswith('http://') or address.startswith('https://'):
        image = Image.open(requests.get(address, stream=True).raw)
    else:
        image = Image.open(address)
    image = image.convert('RGB')
    image = image.resize(res, resample=Image.LANCZOS)
    return image
