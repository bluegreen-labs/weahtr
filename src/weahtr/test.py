# The package API

# import libraries
import glob
import weahtr
import json

# list files to process
images =  glob.glob("/docker_data_dir/demo_input/format_1/images/*.jpg")

# initiate template setup
t = weahtr.template(
  images = images,
  template = "/docker_data_dir/demo_input/format_1/format_1.jpg",
  config = "/docker_data_dir/demo_input/format_1/format_1.yml",
  model = "tesseract",
  method = "fft",
  guides = "/docker_data_dir/demo_input/format_1/format_1_month.json"
)

# match all templates, write homography datat to file
# updates state of "t" with log files
t.match()

# get failed files if any
with open("test.json", "w") as out:
      json.dump(t.log, out)

# write log file to disk

# call the labelling ML routine
# takes a method argument to pick
# which routine to use
t.process(
  slices = True,
  preview = True
)


