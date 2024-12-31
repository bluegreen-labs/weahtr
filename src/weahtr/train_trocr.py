#!/usr/bin/env python

import os
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

# read header in fo
data = "/home/khufkens/BGlabs/git_repos/weaHTRpy/output/format_1_month/header_data_majority_vote.csv"
df = pd.read_csv(data)

# input output files
df['file_name'] = df.agg(lambda x: f"{x['folder']}_{x['image']:03d}_format_1_month_1_1.jpg", axis=1)
df = df[['file_name','final_month']]
df = df.rename(columns={'final_month': 'text'})

train_df, test_df = train_test_split(df, test_size=0.2)
# we reset the indices to start from zero
train_df.reset_index(drop=True, inplace=True)
test_df.reset_index(drop=True, inplace=True)

class COBECORE_dataset(Dataset):
    def __init__(self, root_dir, df, processor, max_target_length=128):
        self.root_dir = root_dir
        self.df = df
        self.processor = processor
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # get file name + text 
        file_name = self.df['file_name'][idx]
        text = self.df['text'][idx]
        
        # prepare image (i.e. resize + normalize)
        image = cv2.imread(os.path.join(self.root_dir, file_name))
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        
        # add labels (input_ids) by encoding the text
        labels = self.processor.tokenizer(
         text, 
         padding="max_length", 
         max_length=self.max_target_length).input_ids
         
        # important: make sure that PAD tokens are ignored by the loss function
        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]

        encoding = {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels)}
        return encoding

# loss function
cer_metric = load("cer")

# load network
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")

train_dataset = COBECORE_dataset(
	root_dir='/home/khufkens/BGlabs/git_repos/weaHTRpy/output/format_1_month/preview/',
        df=train_df,
        processor=processor
        )
        
eval_dataset = COBECORE_dataset(
	root_dir='/home/khufkens/BGlabs/git_repos/weaHTRpy/output/format_1_month/preview/',
        df=test_df,
        processor=processor
        )

encoding = train_dataset[0]
for k,v in encoding.items():
  print(k, v.shape)

labels = encoding['labels']
labels[labels == -100] = processor.tokenizer.pad_token_id
label_str = processor.decode(labels, skip_special_tokens=True)
print(label_str)

train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
eval_dataloader = DataLoader(eval_dataset, batch_size=4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# override device for testing
device = "cpu"

model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-stage1")
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

def compute_cer(pred_ids, label_ids):
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
    cer = cer_metric.compute(predictions=pred_str, references=label_str)

    return cer

optimizer = AdamW(model.parameters(), lr=5e-5)

# optimizer = torch.optimAdamW() # check syntax

for epoch in range(1):  # loop over the dataset multiple times
   # train
   model.train()
   train_loss = 0.0
   for batch in tqdm(train_dataloader):
      # get the inputs
      for k,v in batch.items():
        batch[k] = v.to(device)

      # forward + backward + optimize
      outputs = model(**batch)
      loss = outputs.loss
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()

      train_loss += loss.item()

   print(f"Loss after epoch {epoch}:", train_loss/len(train_dataloader))
    
   # evaluate
   model.eval()
   valid_cer = 0.0
   with torch.no_grad():
     for batch in tqdm(eval_dataloader):
       # run batch generation
       outputs = model.generate(batch["pixel_values"].to(device))
       # compute metrics
       cer = compute_cer(pred_ids=outputs, label_ids=batch["labels"])
       valid_cer += cer 

   print("Validation CER:", valid_cer / len(eval_dataloader))

model.save_pretrained(".")
