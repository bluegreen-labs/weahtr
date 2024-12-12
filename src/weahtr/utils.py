import os
import numpy as np
import pandas as pd
import cv2
import math
import scipy.ndimage.interpolation as ndii

#--- log-polar fft template matching utility functions ---

def match_size(image, template):
  
    # get max dimensions of both images
    max_rows = max(image.shape[0], template.shape[0])
    max_cols = max(image.shape[1], template.shape[1])
    
    # calculate optimal fft sizes (due to non-linear
    # processing times relative to size)
    max_rows = cv2.getOptimalDFTSize(max_rows)
    max_cols = cv2.getOptimalDFTSize(max_cols)
    
    # pad to max dimensions with a black border for the
    # image a white one for the template - pad to the right
    # and at the bottom, keeping the image top-left
    image = cv2.copyMakeBorder(
      image, 0, max_rows - image.shape[0], 0, max_cols - image.shape[1],
      cv2.BORDER_CONSTANT,
      value = 0
    )
    
    template = cv2.copyMakeBorder(
      template, 0, max_rows - template.shape[0], 0, max_cols - template.shape[1],
      cv2.BORDER_CONSTANT,
      value = 1
    )
    
    return (image, template)

# Central point for running FFT
def calculate_fft(img):
  fft = cv2.dft(np.float32(img),flags = cv2.DFT_COMPLEX_OUTPUT)
  fft_shifted = np.fft.fftshift(fft)
  magnitude = cv2.magnitude(fft_shifted[:,:,0], fft_shifted[:,:,1])
  return (fft_shifted, magnitude)

# converts image to its log polar representation
# returns the log polar representation and log base
def logpolar(img, centerTrans, angleStep, logBase):
  
  # check cartToPolar() in opencv to replace some of the logic
  anglesMap = np.zeros(img.shape, dtype=np.float64)
  anglesVector = -np.linspace(0, np.pi, img.shape[0], endpoint=False)
  anglesMap.T[:] = anglesVector
  radiusMap = np.zeros(img.shape, dtype=np.float64)
  radiusVector = np.power(logBase, np.arange(img.shape[1], dtype=np.float64)) - 1.0
  radiusMap[:] = radiusVector
  x = radiusMap * np.sin(anglesMap) + centerTrans[1]
  y = radiusMap * np.cos(anglesMap) + centerTrans[0]
  outputImg = np.zeros(img.shape)
  ndii.map_coordinates(img, [x, y], output=outputImg)
  outputImg = cv2.dft(np.float32(outputImg), flags = cv2.DFT_COMPLEX_OUTPUT)
  return outputImg

def phase_correlation(img_orig, img_transformed):
  orig_conj = np.copy(img_orig)
  orig_conj[:,:,1] = -orig_conj[:,:,1]
  orig_mags = cv2.magnitude(img_orig[:,:,0],img_orig[:,:,1])
  img_trans_mags = cv2.magnitude(img_transformed[:,:,0],img_transformed[:,:,1])
  realPart = (orig_conj[:,:,0] * img_transformed[:,:,0] - orig_conj[:,:,1] * img_transformed[:,:,1]) / (orig_mags * img_trans_mags)
  imaginaryPart = (orig_conj[:,:,0] * img_transformed[:,:,1] + orig_conj[:,:,1] * img_transformed[:,:,0]) / ( orig_mags * img_trans_mags)
  result = np.dstack((realPart, imaginaryPart))
  result_idft = cv2.idft(result)
  result_mags = cv2.magnitude(result_idft[:,:,0],result_idft[:,:,1])
  return np.unravel_index(np.argmax(result_mags), result_mags.shape)
  
#--- general utility functions ---

def preview(image, template, path):
  
  # split band and crop
  image = binarize(image)
  image = image[0:template.shape[0],0:template.shape[1]]
    
  # construct preview
  preview = np.full((image.shape[0],image.shape[1],3),255, dtype=np.uint8)
  preview[:,:,1] = image
  preview[:,:,2] = template
  
  # resize to fifth of the original
  preview = cv2.resize(preview, None, fx=0.3, fy = 0.3)
    
  # write to preview path
  cv2.imwrite(path,preview)
  
# binarize the image
def binarize(image):
  
  if len(image.shape) == 3:
    # convert images to grayscale
    # using red channel for max
    # constrast
    _,_,image = cv2.split(image)
  
  # Adaptive thresholding
  image = cv2.adaptiveThreshold(
      image,
      255,
      cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
      11,
      2
  )

  # Median blurring
  image = cv2.medianBlur(image, 5)
  
  return image

def load_guides(filename, mask_name):
   # check if the guides file can be read
   # if not return error
  try:
    guides = []
    file = open(u''+filename,'r')
    lines = file.readlines()
    for line in lines:
     if line.find("Guide:" + mask_name) > -1:
       data = line.split('|')
       data[0] = data[0].split(":")
       data[1] = data[1].split(",")
       data[2] = data[2].split(",")
       data[3] = data[3].split(",")
       guides.append(data)
    file.close()
  except:
    print("No subset location file found!")
    print("looking for: " + mask_name + ".csv")
    exit()
    
  # split out the locations
  # convert to integer
  x = np.asarray(guides[0][3], dtype=float)
  x = x.astype(int)
  x = np.sort(x)
  y = np.asarray(guides[0][2], dtype=float)
  y = y.astype(int)
  y = np.sort(y)
  
  # create empty cell registry
  row = []
  col = []
  x_min = []
  x_max = []
  y_min = []
  y_max = []
  
  # loop over all x/y values and construct
  # cropping data
  for i, x_value in enumerate(x[0:(len(x)-1)]):
    for j, y_value in enumerate(y[0:(len(y)-1)]):
    
     # pad the cell boundary values
     # set to 0 for no padding
     col_pad = int(round((x[i+1] - x[i])/4))
     row_pad = int(round((y[j+1] - y[j])/3))
     
     x_min.append(int(x[i] - col_pad))
     x_max.append(int(x[i+1] + col_pad))
     y_min.append(int(y[j] - row_pad))
     y_max.append(int(y[j+1] + row_pad))
     
     # create dictionary
     # remember 0 indexed matrix notation so +1
     # for clarity
     row.append(j + 1)
     col.append(i + 1)
  
  # concat data into pandas data frame
  df = pd.DataFrame(
    {'row': row,
     'col': col,
     'x_min': x_min,
     'x_max': x_max,
     'y_min': y_min,
     'y_max': y_max
    }
   )
   
  return df

def print_labels(im, locations, df):
  
  # split out the locations
  # convert to integer
  x = np.asarray(locations[0][3], dtype=float)
  x = x.astype(int)
  x = np.sort(x)
  y = np.asarray(locations[0][2], dtype=float)
  y = y.astype(int)
  y = np.sort(y)
  
  # loop over all rows
  for i, row in df.iterrows():
   y_value = int(row['row'])
   x_value = int(row['col'])
   label = row['text']
   conf = row['conf']
   
   center_x = x[x_value -1]
   center_y = y[y_value]
   
   try:
     if conf > 85:
      cv2.putText(im, str(label) ,(center_x, center_y),
        cv2.FONT_HERSHEY_SIMPLEX, 2,(255,255,255),9,cv2.LINE_AA)
     else:
      cv2.putText(im, str(label) ,(center_x, center_y),
        cv2.FONT_HERSHEY_SIMPLEX, 2,(255,0,0),9,cv2.LINE_AA) 
   except:
    continue
  return im
