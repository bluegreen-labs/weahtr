import os
import cv2
import torch
from torch.utils.data import Dataset

class weaHTR_dataset(Dataset):
    def __init__(self, root_dir, df,max_target_length=128, processor=None, transform=None):
        self.root_dir = root_dir
        self.df = df
        self.processor = processor
        self.max_target_length = max_target_length
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # get file name + text 
        file_name = self.df['file_name'][idx]
        text = self.df['text'][idx]
        
        # prepare image (i.e. resize + normalize)
        image = cv2.imread(os.path.join(self.root_dir, file_name))
        
        if self.transform is not None:
            image = self.transform(image=image)["image"]
        
        # convert to tensor
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

