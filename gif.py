"""
create a gif out of images! not sure if its useful
"""

import os

import imageio
images = []
dir='results/bloody_mary/'
for filename in os.listdir(dir):
    images.append(imageio.imread(dir+ filename))
imageio.mimsave('results/bloody_mary.gif', images)
