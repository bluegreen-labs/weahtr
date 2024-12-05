# The package API

# import libraries
import glob
import weahtr

# list files to process
images =  glob.glob("/docker_data_dir/demo_input/format_1/*.jpg", recursive=False)

# initiate setup
t = template.template(
  images = images,
  template = "/docker_data_dir/demo_input/format_1.jpg",
  config = "/docker_data_dir/demo_input/format_config.yml"
  )

# match all templates, write homography datat to file
t.match(method = "features")

# call the labelling ML routine
# takes a method argument to pick
# which routine to use
t.label()
