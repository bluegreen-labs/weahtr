#!/usr/bin/env python

# import libraries
import glob, os
from weahtr import * 
from weahtr.utils import *
from weahtr.transform import *

# set base directory
base = "/data/"

# list files to process
images =  glob.glob(
  os.path.join(
    base,
    "demo/random/images/*.png"
    )
  )

# define custom function
def custom_function(image):
  image = remove_lines(image, mean_fill = True)
  image = binarize(image, window_size = 101, C = 15)
  image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
  return(image)

# convert and write to disk
image = cv2.imread(images[2])

# custom function
image_custom = custom_function(image)
cv2.imwrite("custom_function.png", image_custom)

# use transformation function
image_transform = train_transform(image = image)['image']
cv2.imwrite("transform.png", image_transform)
