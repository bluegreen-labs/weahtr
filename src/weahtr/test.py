# The package API

# import libraries
import glob
import weahtr

# list files to process
images =  glob.glob("/docker_data_dir/demo_input/format_1/images/*.jpg")

# initiate template setup
t = weahtr.template(
  images = images,
  template = "/docker_data_dir/demo_input/format_1/format_1.jpg",
  config = "/docker_data_dir/demo_input/format_1/format_1.yml"
  )

# match all templates, write homography datat to file
# updates state of "t" with log files
t.match(method = "fft", preview = True)

# get failed files if any
#failed_files = t.log

# call the labelling ML routine
# takes a method argument to pick
# which routine to use
t.label(
  guides = "/docker_data_dir/demo_input/format_1/format_1_small.json",
  model = "tesseract",
  preview = True
  )


