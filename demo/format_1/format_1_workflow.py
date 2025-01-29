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
    "demo/format_1/images/*.jpg"
    )
  )

# initiate template matching setup
t = weahtr.template(
  images = images,
  template = os.path.join(base, "demo/format_1/format_1.jpg"),
  config = os.path.join(base, "demo/format_1/format_1.yml"),
  guides = os.path.join(base,"demo/format_1/format_1_full.json"),
  method = "table",
  model = "tesseract"
)

# match all templates, write homography datat to file
# updates state of "t" with log files
t.match(preview = True)

# run model without transcription
# only return the subsets of the
# rows and columsn selected in the
# config file
# t.process(
#   model = "tesseract",
#   slices = True, # only return "sliced" table output
#   f = remove_lines # apply the remove lines function on each cell
# )

def custom_function(image):
  image = remove_lines(image)
  image = binarize(image)
  return(image)

# transcribe the first two columns
labels = t.process(
  model = "tesseract",
  slices = False
)

print(labels)

