import os
import numpy as np
import pandas as pd
import cv2
import pytesseract

from utils import * 

def label_cells(
  locations,
  im,
  path,
  prefix,
  model
  ):

  # split out the locations
  # convert to integer
  x = np.asarray(locations[0][3], dtype=float)
  x = x.astype(int)
  x = np.sort(x)
  y = np.asarray(locations[0][2], dtype=float)
  y = y.astype(int)
  y = np.sort(y)
  
  # split prefix
  prefix_values = prefix.split("_")
  
  # initiate empty vectores
  text = []
  conf = []
  file_name = []
  row = []
  col = []

  # loop over all x values
  for j, y_value in enumerate(y[0:(len(y)-1)]):
   for i, x_value in enumerate(x[0:(len(x)-1)]):
    
    # generates cropped sections based upon
    # row and column locations
    try:
     # provide padding, disabled for better
     # HTR performance
     col_width = int(round((x[i+1] - x[i])/4))
     row_width = int(round((y[j+1] - y[j])/4))
     
     x_min = int(x[i] - col_width)
     x_max = int(x[i+1] + col_width)
     
     y_min = int(y[j] - row_width)
     y_max = int(y[j+1] + row_width)

     # trap end of table issues 
     # (when running out of space)
     if x_max > im.shape[1]:
       x_max = int(im.shape[1])
      
     if y_max > im.shape[0]:
       y_max = int(im.shape[0])
       
     # crop image to size
     crop_im = im[y_min:y_max, x_min:x_max]
     img = cv2.cvtColor(crop_im, cv2.COLOR_BGR2RGB)

     # TF pre-processing
     if model == "tesseract":
      try:
        ocr_result = pytesseract.image_to_data(
          img,
          lang='cobecore-V6',
          config='--psm 8 -c tessedit_char_whitelist=0123456789.,',
          output_type='data.frame'
          )
  
        # get results with maximum confidence
        # assume only one viable result per image
        # see psm 8 setting and conservative (tight) cropping
        ocr_result = ocr_result[ocr_result.conf == max(ocr_result.conf)]
        
        # split out the content
        new_label = ocr_result.text.iloc[0]
        confidence = ocr_result.conf.iloc[0]
        
        if np.isnan(new_label):
          conf.append(0)
          text.append('/')
        else:
          # write label output data to vectors
          text.append(new_label)
          conf.append(confidence)
      
      except:
        # write label output data to vectors
        conf.append(0)
        text.append('//')
      
     # add filename and row / col numbers
     file_name.append(prefix)
     col.append(i + 1)
     row.append(j + 1)
    
    except:
     # Continue to next iteration on fail
     # happens when index runs out
     continue
  
  # concat data into pandas data frame
  df = pd.DataFrame({'text':text,
                   'conf':conf,
                   'col':col,
                   'row':row,
                   'file':file_name}
                   )

  # construct path
  out_file = os.path.join(path + "/labels/", prefix + "_labels.csv")
  
  # write data to disk
  df.to_csv(out_file, sep=',', index = False)
  
  return df
