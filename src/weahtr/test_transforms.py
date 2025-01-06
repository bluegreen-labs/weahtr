#!/usr/bin/env python
import cv2
from transform import *
import matplotlib.pyplot as plt
from model import *

# read demo image
im = cv2.imread("../../output/format_1_month/preview/6317_033_format_1_month_1_1.jpg")

m = model(
  model = "trocr",
  config = "../../demo_input/format_1/format_1.yml"
)

def visualize_augmentations(dataset, labels, samples=10, cols=5):
    rows = samples // cols
    figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 6))
    for i in range(samples):
        image = dataset[i]
        ax.ravel()[i].imshow(image)
        ax.ravel()[i].set_title(labels[i])
        ax.ravel()[i].set_axis_off()
    plt.tight_layout()
    plt.savefig('foo.png')
    #plt.show()

dataset = []
labels = []

# grab 10 iterations
for i in range(10):
  tr = train_transform(image = im)['image']
  label, conf = m.predict(tr)
  dataset.append(tr)
  labels.append(label + " / " + str(round(conf,2))) 

visualize_augmentations(dataset, labels)
