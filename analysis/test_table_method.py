#!/usr/bin/env python

# import libraries
import glob, os
from weahtr.weahtr import *

base = "/data/"

# list files to process
images =  glob.glob(os.path.join(base, "demo_input/format_1/images/*.jpg"))

#initiate template setup
t = template(
  images = images,
  template = os.path.join(base, "demo_input/format_1/format_1.jpg"),
  config = os.path.join(base, "demo_input/format_1/format_1_outline.yml"),
  guides = os.path.join(base,"demo_input/format_1/format_1_outline.json"),
  method = "table"
)

# match all templates, write homography datat to file
# updates state of "t" with log files
t.match(preview = True)
 
# # write log file to disk
# t.save_log(path = '/docker_data_dir/output/')
