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
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import TrOCRProcessor
from transformers import AdamW
from tqdm import tqdm
import yaml
from transformers.utils import logging
logging.set_verbosity_error() 

from weahtr.utils import *
from weahtr.transform import *
from weahtr.dataloader import *
import pytesseract

class model_loader:
  def load_trocr(processor_model, encoder_model, device, train = True):
      processor = TrOCRProcessor.from_pretrained(processor_model)
      
      if train:
        model = VisionEncoderDecoderModel.from_pretrained(encoder_model)
      else:
        model = VisionEncoderDecoderModel.from_pretrained(
          encoder_model,
          local_files_only = True
        )
      
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
  def __init__(self, model, config, images=None, labels=None):
    self.model = model
    self.images = images
    self.labels = labels
    
    # read in template matching config
    # file including output directory etc
    with open(config, 'r') as file:
      try:
        self.config = yaml.safe_load(file)
      except:
        raise ValueError("No yaml config file, or badly formatted YML file (check all quotes) ...")
    
    print("preloading model")
    
    # preloading inference processor and model
    # load network to allow for faster itterations
    if self.model == "trocr":
      if labels is None:
        train = False
        encoder_model = self.config['trocr']['custom_path']
      else:
        train = True
        encoder_model = self.config['trocr']['model']
      
      self.trocr_model, self.trocr_processor = model_loader.load_trocr(
        processor_model = self.config['trocr']['processor'],
        encoder_model = encoder_model,
        device = self.config['device'],
        train = train
      )

  #----- private functions ----
  
  def __predict_tesseract(self, image):
    
    # set default path on Docker
    if os.path.exists("/.dockerenv"):
      bin_path = "/opt/conda/envs/weahtr/bin/"
    else:
      bin_path = self.config['tesseract']['bin_path']

    # set path binary
    pytesseract.pytesseract.tesseract_cmd = os.path.join(bin_path, "tesseract")

    # extract values
    ocr_result = pytesseract.image_to_data(
      image,
      lang = os.path.splitext(self.config['tesseract']['model'])[0],
      config = self.config['tesseract']['config'],
      output_type = 'data.frame'
    )

    # get results with maximum confidence
    # assume only one viable result per image
    # see psm 8 setting and conservative (tight) cropping
    ocr_result = ocr_result[ocr_result.conf == max(ocr_result.conf)]
    
    # split out the content
    text = ocr_result.text.iloc[0]
    confidence = ocr_result.conf.iloc[0]
    
    return text, confidence
  
  def __predict_trocr(self, image):
    
    pixel_values = self.trocr_processor(
      image,
      return_tensors="pt").pixel_values
    
    return_dict = self.trocr_model.generate(
      pixel_values,
      output_scores = True,
      return_dict_in_generate=True
    )
    
    # split out label and confidence levels
    text = self.trocr_processor.batch_decode(
      return_dict['sequences'],
      skip_special_tokens=True)[0]
    confidence = math.exp(return_dict['sequences_scores'].item())
    return text, confidence

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
        root_dir = self.images,
        df = train_df,
        processor = self.trocr_processor,
        transform = train_transform
    )
            
    val_dataset = weaHTR_dataset(
        root_dir = self.images,
        df = val_df,
        processor = self.trocr_processor,
        transform = train_transform
    )
    
    # set the data loader
    train_dataloader = DataLoader(
      train_dataset,
      batch_size = self.config['batch_size'],
      shuffle=True
    )
    
    val_dataloader = DataLoader(
      val_dataset,
      batch_size = self.config['batch_size']
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
    optimizer = torch.optim.AdamW(
      model.parameters(),
      lr = float(self.config['learning_rate'])
    )
    
    for epoch in range(self.config['epochs']):  # loop over the dataset multiple times
       # train
       model.train()
       train_loss = 0.0
       for batch in tqdm(train_dataloader):
          # get the inputs
          for k,v in batch.items():
            batch[k] = v.to(self.config['device'])
    
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
      print("No valid model specified")
      exit()
    
  def predict(self, image):
    if self.model == "trocr":
      text, confidence = self.__predict_trocr(image)
    
    if self.model == "tesseract":
      text, confidence = self.__predict_tesseract(image)
      if np.isnan(text):
        text = '//'
        confidence = 0
    
    return text, confidence
