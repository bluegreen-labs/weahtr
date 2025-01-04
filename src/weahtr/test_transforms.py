#!/usr/bin/env python
import cv2
from transform import *
import matplotlib.pyplot as plt

# read demo image
im = cv2.imread("../../output/format_1_month/preview/6317_033_format_1_month_1_1.jpg")

def visualize_augmentations(dataset, samples=10, cols=5):
    rows = samples // cols
    figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 6))
    for i in range(samples):
        image = dataset[i]
        ax.ravel()[i].imshow(image)
        ax.ravel()[i].set_axis_off()
    plt.tight_layout()
    plt.show()    

dataset = []

# grab 10 iterations
for i in range(10):
  tr = train_transform(image = im)['image']
  dataset.append(tr)

visualize_augmentations(dataset)
