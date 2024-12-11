# The package API

# import libraries
import glob
import weahtr

# list files to process
images =  glob.glob("/docker_data_dir/demo_input/format_1/*.jpg", recursive=False)

# initiate setup
t = weahtr.template(
  images = images,
  template = "/docker_data_dir/demo_input/format_12.jpg",
  config = "/docker_data_dir/demo_input/format_config.yml"
  )

# match all templates, write homography datat to file
# updates state of "t" with log files
t.match(method = "fft")

# get failed files if any
#failed_files = t.log

# call the labelling ML routine
# takes a method argument to pick
# which routine to use
t.label(guides = "/docker_data_dir/demo_input/format_1_table_guides.txt")
