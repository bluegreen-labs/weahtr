# TrOCR model training and inference class
# model training routines depend on the
# underlying model and framework and are
# described in other files
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
from dataloader import *

def load_trocr(processor_model, encoder_model, device, train = True):
    print("preloading model")
    processor = TrOCRProcessor.from_pretrained(processor_model)
    
    if train:
      model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-stage1")
    else:
      model = VisionEncoderDecoderModel.from_pretrained(encoder_model, local_files_only = True)
    
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
    return model, processor

class model:
  def __init__(self, model, config, path, labels=None, **kwargs):
    self.model = model
    self.path = path
    self.labels = labels
    self.device = "cpu"
    self.batch_size = 4
    
    # preloading inference processor and model
    # load network to allow for faster itterations
    if self.model == "trocr":
      if labels is None:
        train = False
      else:
        train = True
        
      self.trocr_model, self.trocr_processor = load_trocr(
        "microsoft/trocr-base-stage1",
        "microsoft/trocr-base-stage1",
        self.device,
        train = train
      )

  #----- private functions ----

  def __train_trocr(self):
    # create training and validation dataset
    train_df, val_df = train_test_split(
      self.labels,
      test_size=0.2
    )
    
    # we reset the indices to start from zero
    train_df.reset_index(drop=True, inplace=True)
    val_df.reset_index(drop=True, inplace=True)
  
    train_dataset = weaHTR_dataset(
        root_dir = self.path,
        df = train_df,
        processor = self.trocr_processor,
        transform = train_transform
    )
            
    val_dataset = weaHTR_dataset(
        root_dir = self.path,
        df = val_df,
        processor = self.trocr_processor,
        transform = train_transform
    )
    
    # set the data loader
    train_dataloader = DataLoader(
      train_dataset,
      batch_size = self.batch_size,
      shuffle=True
    )
    
    val_dataloader = DataLoader(
      val_dataset,
      batch_size = self.batch_size
    )
  
    # loss function is the character error rate
    cer_metric = load("cer")
    def compute_cer(pred_ids, label_ids):
        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
        cer = cer_metric.compute(predictions=pred_str, references=label_str)
        return cer
    
    # define model
    model = self.trocr_model
    
    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    
    for epoch in range(1):  # loop over the dataset multiple times
       # train
       model.train()
       train_loss = 0.0
       for batch in tqdm(train_dataloader):
          # get the inputs
          for k,v in batch.items():
            batch[k] = v.to(self.device)
    
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
           outputs = model.generate(batch["pixel_values"].to(self.device))
           # compute metrics
           cer = compute_cer(pred_ids=outputs, label_ids=batch["labels"])
           valid_cer += cer 
    
       print("Validation CER:", valid_cer / len(eval_dataloader))
    
    model.save_pretrained(".")

  #----- public functions ----
  
  def train(self, transform = None):
    if self.model == "trocr":
      print("training")
      self.__train_trocr()
    else:
      print("bla")
    
  def predict(self, image):
    if self.model == "trocr":
      print("predicting")
