#!/usr/bin/env python

# import libraries
import glob, os
from weahtr import * 
from weahtr.utils import *
from weahtr.transform import *
import pytesseract

# set path binary Docker
bin_path = "/opt/conda/envs/weahtr/bin/"
pytesseract.pytesseract.tesseract_cmd = os.path.join(bin_path, "tesseract")

# set base directory
base = "/data/"

# list files to process
images =  glob.glob(
  os.path.join(
    base,
    "demo/random/images/*.png"
    )
  )

# define custom function
def custom_function(image):
  image = remove_lines(image, mean_fill = False)
  image = binarize(image, window_size = 101, C = 15)
  image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
  return(image)

for i in images:
  # convert and write to disk
  image = cv2.imread(i)

  # custom function
  image = custom_function(image)
  
  # use transformation function
  image = train_transform(image = image)['image']
  
  # extract values
  ocr_result = pytesseract.image_to_data(
    image,
    lang = "cobecore-V6",
    config = '--psm 8 -c tessedit_char_whitelist=0123456789.,',
    output_type = 'data.frame'
  )
  
  ocr_result = ocr_result[ocr_result.conf == max(ocr_result.conf)]
  text = ocr_result.text.iloc[0]
  print("{} {}".format(i, text))

