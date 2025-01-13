#!/usr/bin/env python

# import libraries
import glob, os
import weahtr
from model import *
import cv2

base = "/data/"

# list files to process
images =  glob.glob(os.path.join(base, "data/format_1/*.jpg"))

#initiate template setup
t = weahtr.setup(
  images = images,
  template = os.path.join(base, "demo_input/format_1/format_1.jpg"),
  config = os.path.join(base, "demo_input/format_1/format_1.yml"),
  guides = os.path.join(base,"demo_input/format_1/format_1_month.json"),
  method = "features",
  model = "trocr"
)

# run newly trained model 
# labels = t.process(
#   model = "trocr"
# )

# # match all templates, write homography datat to file
# # updates state of "t" with log files
# t.match(preview = False)
# 
# # write log file to disk
# t.save_log(path = '/docker_data_dir/output/')
# 
# # call the labelling ML routine
# # takes a method argument to pick
# # which routine to use
# t.process(
#   slices = True
# )
# 
# # annotate slices
# 
# # train a model
m.train(
  images = images,
  labels = "demo_input"
)



 
# 
# # read labels file
# df = pd.read_csv("/home/khufkens/BGlabs/git_repos/weaHTRpy/output/format_1_month/header_data_majority_vote.csv")
# df['file_name'] = df.agg(lambda x: f"{x['folder']}_{x['image']:03d}_format_1_month_1_1.jpg", axis=1)
# df = df[['file_name','final_month']]
# df = df.rename(columns={'final_month': 'text'})
# 
# # set root dir (where the images live as tallied in the label file)
# root_dir = '/home/khufkens/BGlabs/git_repos/weaHTRpy/output/format_1_month/preview/'
# 
# m = model(
#   model = "trocr", 
#   config = "/home/khufkens/BGlabs/git_repos/weaHTRpy/demo_input/format_1/format_1.yml"
#   #path = root_dir,
#   #labels = df
# )
# 
# img = cv2.imread("/home/khufkens/BGlabs/git_repos/weaHTRpy/output/format_1_month/preview/6118_004_format_1_month_1_1.jpg")
