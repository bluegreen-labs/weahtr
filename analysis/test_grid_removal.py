#!/usr/bin/env python
import cv2, os, glob
from weahtr.utils import *
import matplotlib.pyplot as plt
import numpy as np

base = "/data/"

# list files to process
images =  glob.glob(os.path.join(base, "data/character_training_data/images/*.png"))
#images =  glob.glob(os.path.join(base, "data/format_1/*.jpg"))
images = images[2]

image = cv2.imread(images)
crop = subset_cell(image)
crop = remove_lines(crop)
cv2.imwrite("demo.png", crop)
