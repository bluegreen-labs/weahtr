#!/usr/bin/env python

# import libraries
import glob, os
from weahtr import *
from weahtr.utils import *

# set base directory
base = "/data/"

# # list files to process
# images =  glob.glob(
#   os.path.join(
#     base,
#     "demo_input/format_1/images/*.jpg"
#     )
#   )
# 
# # initiate template matching setup
# t = weahtr.template(
#   images = images,
#   template = os.path.join(base, "demo_input/format_1/format_1.jpg"),
#   config = os.path.join(base, "demo_input/format_1/format_1.yml"),
#   guides = os.path.join(base,"demo_input/format_1/format_1_full.json"),
#   method = "table"
# )

# list files to process
images =  glob.glob(
  os.path.join(
    base,
    "demo_input/format_6/images/*.jpg"
    )
  )

# initiate template matching setup
t = weahtr.template(
  images = images,
  template = os.path.join(base, "demo_input/format_6/format_6.jpg"),
  config = os.path.join(base, "demo_input/format_6/format_6.yml"),
  guides = os.path.join(base,"demo_input/format_6/format_6_full.json"),
  method = "table"
)

# match all templates, write homography datat to file
# updates state of "t" with log files
t.match(preview = True)

# run model
# labels = t.process(
#   model = "trocr",
#   slices = True, # only return "sliced" table output
#   f = remove_lines # apply the remove lines function on each cell
# )
