"""
create a gif out of images! not sure if its useful
"""

import os
import imageio

images = []
dir='./results/all_star/'
for filename in os.listdir(dir):
    images.append(imageio.imread(dir+ filename))
imageio.mimsave('./results/all-star.gif', images)
