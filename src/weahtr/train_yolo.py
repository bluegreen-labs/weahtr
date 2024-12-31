#!/usr/bin/env python
from ultralytics import YOLO

# load a pretrained model 
# medium sized model architecture
model = YOLO("yolo11m-cls.pt")  

# Train the model
results = model.train(
 data="../../output/format_1_month/traindata/",
 project = "../../output/ml_models/",
 # general settings
 epochs=20,
 imgsz=320,
 half=True,
 exist_ok=True,
 batch = -1,
 # augmentation settings
 # mostly disabling non-text ones
 mosaic = 0,
 fliplr = 0,
 erasing = 0,
 crop_fraction = 0.9,
 degrees = 10
)


