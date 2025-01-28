#!/usr/bin/env python
import cv2, os, glob
from weahtr.utils import *
import matplotlib.pyplot as plt
import numpy as np

base = "/data/"

# list files to process
#images =  glob.glob(os.path.join(base, "data/character_training_data/images/*.png"))
images =  glob.glob(os.path.join(base, "demo_input/format_1/images/*.jpg"))
images = images[3]
print(images)

image = cv2.imread(images)

# replace matte with average of the image
# content to properly binarize
_ , image , _ = replace_matte(image)

# binarize
image = binarize(
  image,
  51,
  10
)

# invert and dilate
image = cv2.bitwise_not(image)
image = cv2.dilate(
    image,
    None,
    iterations = 5
)

corners = find_contours(image)
print(corners)
#cv2.imwrite("demo.png", crop)


