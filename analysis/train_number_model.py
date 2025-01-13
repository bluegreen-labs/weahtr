#!/usr/bin/env python

# import libraries
import glob, os
from weahtr import *
import cv2
import pandas as pd

base = "/data/"

# list files to process
images =  glob.glob(os.path.join(base, "data/character_training_data/images/*.png"))

df = pd.read_csv("../data/character_training_data/labels.csv")
df = df.rename(columns={'final_value': 'text'})
df = df.rename(columns={'filename': 'file_name'})
df['text'] = str(df['text'])

# setup model training
m = weahtr.model(
  model = "trocr", # model to use
  config = os.path.join(base,"demo_input/format_1/format_1.yml"), # config file
  labels = df, # the labels
  images = os.path.join(base, "data/character_training_data/images/")
)

# initiate a training run
m.train()
