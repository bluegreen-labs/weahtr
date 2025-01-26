#!/usr/bin/env python

# import libraries
import glob, os
from weahtr import *

base = "/data/"

# list files to process
images =  glob.glob(os.path.join(base, "demo_input/format_1/images/*.jpg"))

#initiate template setup
t = weahtr.template(
  images = images,
  template = os.path.join(base, "demo_input/format_1/format_1.jpg"),
  config = os.path.join(base, "demo_input/format_1/format_1_testing.yml"),
  guides = os.path.join(base,"demo_input/format_1/format_1_full.json"),
  method = "table"
)

# match all templates, write homography datat to file
# updates state of "t" with log files
#t.match(preview = True)
#print(t.log)

# run model
labels = t.process(
  model = "trocr",
  slices = True
)
