#!/usr/bin/env python
import cv2, os, glob
from weahtr.utils import *
import matplotlib.pyplot as plt
import numpy as np

base = "/data/"

# list files to process
images =  glob.glob(os.path.join(base, "data/character_training_data/images/*.png"))
images =  glob.glob(os.path.join(base, "data/format_1/*.jpg"))
images = images[2]

image = cv2.imread(images)
image = remove_lines(image)
cv2.imwrite("demo.png", image)

# figure, ax = plt.subplots(nrows=len(images)//2, ncols=2, figsize=(12, 6))
# for i in range(len(images)):
#     image = images[i]
#     image = remove_lines(image)
#     ax.ravel()[i].imshow(image)
#     ax.ravel()[i].set_axis_off()
# plt.tight_layout()
# plt.savefig('demo.png')

