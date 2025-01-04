#!/usr/bin/env python

import os, glob, math
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import VisionEncoderDecoderModel
from transformers import default_data_collator
from evaluate import load
import cv2
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import TrOCRProcessor
from transformers import AdamW
from tqdm import tqdm
from utils import *
from transform import *

root_dir = "../../output/format_1_month/traindata/test/Mai/"

# list all files
files = glob.glob(os.path.join(root_dir, "*.jpg"))

# load network
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# override device for testing
device = "cpu"

model = VisionEncoderDecoderModel.from_pretrained("models/TrOCR/", local_files_only = True)
model.to(device)

# set special tokens used for creating the decoder_input_ids from the labels
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id

# make sure vocab size is set correctly
model.config.vocab_size = model.config.decoder.vocab_size

# set beam search parameters
model.config.eos_token_id = processor.tokenizer.sep_token_id
model.config.max_length = 64
model.config.early_stopping = True
model.config.no_repeat_ngram_size = 3
model.config.length_penalty = 2.0
model.config.num_beams = 4

for f in files:
 image = cv2.imread(f)
 print(f)
 for i in range(10):
        tr = train_transform(image = image)['image']
        pixel_values = processor(tr, return_tensors="pt").pixel_values
        return_dict = model.generate(pixel_values, output_scores = True, return_dict_in_generate=True)
        ids, scores = return_dict['sequences'], return_dict['sequences_scores']
        generated_text = processor.batch_decode(ids, skip_special_tokens=True)[0]
        print(generated_text)
        print(math.exp(scores.item()))
 print("-----")
