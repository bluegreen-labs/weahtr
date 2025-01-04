#!/usr/bin/env python

# import libraries
import glob
import weahtr
from model import *

# list files to process
#images =  glob.glob("/docker_data_dir/data/format_1/*.jpg")

# initiate template setup
# t = weahtr.setup(
#   images = images,
#   template = "/docker_data_dir/demo_input/format_1/format_1.jpg",
#   config = "/docker_data_dir/demo_input/format_1/format_1.yml",
#   guides = "/docker_data_dir/demo_input/format_1/format_1_month.json",
#   method = "features",
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
# m.train(
#   labels = "..."
# )
# 
# # run newly trained model 
# t.process(
#   model = "trocr"
# )

# read labels file
df = pd.read_csv("/home/khufkens/BGlabs/git_repos/weaHTRpy/output/format_1_month/header_data_majority_vote.csv")
df['file_name'] = df.agg(lambda x: f"{x['folder']}_{x['image']:03d}_format_1_month_1_1.jpg", axis=1)
df = df[['file_name','final_month']]
df = df.rename(columns={'final_month': 'text'})

# set root dir (where the images live as tallied in the label file)
root_dir = '/home/khufkens/BGlabs/git_repos/weaHTRpy/output/format_1_month/preview/'

m = model(model = "trocr", config = "bla", path = root_dir, labels = df)
m.train()
