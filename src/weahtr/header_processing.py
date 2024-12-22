#!/usr/bin/env python

# import libraries
import glob
import weahtr

# list files to process
images =  glob.glob("/docker_data_dir/data/format_1/*.jpg")

# initiate template setup
t = weahtr.template(
  images = images,
  template = "/docker_data_dir/demo_input/format_1/format_1.jpg",
  config = "/docker_data_dir/demo_input/format_1/format_1.yml",
  guides = "/docker_data_dir/demo_input/format_1/format_1_month.json",
  method = "features",
  model = "tesseract"
)

# match all templates, write homography datat to file
# updates state of "t" with log files
t.match(preview = False)

# write log file to disk
t.save_log(path = '/docker_data_dir/output/')

# call the labelling ML routine
# takes a method argument to pick
# which routine to use
t.process(
  slices = True
)
