import cv2
import os

# Path to the directory containing the input images
image_folder = './results/song/'

# Path to the output video file
video_name = './results/output.mp4'

# Fetch all image file names from the directory
images = [img for img in os.listdir(image_folder) if img.endswith(".png")]

# Sort the images by their filenames (assuming the filenames are in numerical order)
images.sort(key=lambda x: int(x.split(".")[0]))

# Determine the dimensions of the images (assuming all images have the same dimensions)
image_path = os.path.join(image_folder, images[0])
frame = cv2.imread(image_path)
height, width, layers = frame.shape

# Define the video codec, frames per second (FPS), and output video format
video_codec = cv2.VideoWriter_fourcc(*"mp4v")
fps = 24  # Adjust as needed
video = cv2.VideoWriter(video_name, video_codec, fps, (width, height))

# Iterate over each image and add it to the video
for image in images:
    image_path = os.path.join(image_folder, image)
    frame = cv2.imread(image_path)
    video.write(frame)

# Release the video writer and close any open windows
video.release()
cv2.destroyAllWindows()