#!/usr/bin/env python

# import libraries
import glob, os
from weahtr import * 
from weahtr.utils import *

# set base directory
base = "/data/"

# list files to process
images =  glob.glob(
  os.path.join(
    base,
    "demo/ReData/images/*.jpg"
    )
  )

# initiate template matching setup
t = weahtr.template(
  images = images,
  template = os.path.join(base, "demo/ReData/format_redata.png"),
  config = os.path.join(base, "demo/ReData/format_redata.yml"),
  guides = os.path.join(base,"demo/ReData/format_redata_full.json"),
  method = "table"
)

# match all templates, write homography datat to file
# updates state of "t" with log files
#t.match(preview = True)

# run model without transcription
# only return the subsets of the
# rows and columns selected in the
# config file and a custom pre-processing
# function

# define custom function
def custom_function(image):
  image = remove_lines(image, mean_fill = True)
  image = binarize(image, window_size = 31, C = 8)
  image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
  return(image)

# define custom function to resize
# height (for pylaia)
#def custom_function(image, height = 128):
#  (h, w) = image.shape[:2]
#  r = height / float(h)
#  dim = (int(w * r), height)
#  image = cv2.resize(image, dim)
#  return(image)

# transcribe the first two columns
labels = t.process(
  model = "tesseract",
  slices = False,
  preview = True,
  f = custom_function
)
