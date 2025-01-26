import os, json
import numpy as np
import pandas as pd
import cv2
import scipy.ndimage.interpolation as ndii
from collections import Counter

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
      value = 0
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
  # logPolar(src,logPolarMat,Point(src.cols/2, src.rows/2),src.cols/8,INTER_CUBIC);
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

def preview_match(image, template, path):
  
  # split band and crop
  image = binarize(image)
  image = image[0:template.shape[0],0:template.shape[1]]
    
  # construct preview
  preview = np.full((template.shape[0],template.shape[1],3),255, dtype=np.uint8)
  preview[:,:,1] = image
  preview[:,:,2] = template
  
  # resize to fifth of the original
  preview = cv2.resize(preview, None, fx=0.3, fy = 0.3)
    
  # write to preview path
  cv2.imwrite(path, preview)
  
# binarize the image
def binarize(image, window_size = 91, C = 6):
  
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
      window_size,
      C
  )
  
  # erode (fat lines register poorly)
  # use dilate as using inverted image
  kernel = np.ones((3, 3), np.uint8)
  image = cv2.dilate(image, kernel)

  return image

def load_guides(filename):
   # check if the guides file can be read
   # if not return error
  try:
    with open(filename, "r") as file:
      guides = json.load(file)
  except:
    print("No subset location file found!")
    print("looking for: " + mask_name + ".csv")
    exit()
    
  # split out the locations
  # convert to integer
  x = guides['cols']
  x = np.sort(x)
  y = guides['rows']
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
    
     x_min.append(int(x[i]))
     x_max.append(int(x[i+1]))
     y_min.append(int(y[j]))
     y_max.append(int(y[j+1]))
     
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

def preview_labels(im, df, path):
    
  # loop over all rows
  for i, row in df.iterrows():
   x = int(row['x'])
   y = int(row['y'])
   conf = row['conf']
   label = row['text']
   
   try:
    if conf > 85:
      cv2.putText(im, str(label) ,(x, y),
        cv2.FONT_HERSHEY_SIMPLEX, 2,(255,255,255),4,cv2.LINE_AA)
    else:
      cv2.putText(im, str(label) ,(x, y),
        cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255),4,cv2.LINE_AA) 
   except:
    continue
  
  # rescale
  im = cv2.resize(
    im,
    None,
    fx = 0.25,
    fy = 0.25
  )

  # write to file
  cv2.imwrite(
    path,
    im
  )

def highpass(img, sigma):
    return img - cv2.GaussianBlur(img, (0,0), sigma) + 127

# replaces matte cutting out the central part with a page
# or replace the cut oute matte with the mean of the
# content of the page - to retain original image dimensions
# and thereby allowing for easier homography based
# transformations
def replace_matte(image):
  
  # retain original copy
  image_orig = image
  
  # black border to make it work
  # on white matte images
  image = cv2.copyMakeBorder(
    image, 500, 500, 500, 500,
    cv2.BORDER_CONSTANT,
    None,
    0
  )
  
  try:
    _,_,gray = cv2.split(image)
  except:
    gray = image

  # adaptive threshold the image
  ret, bw = cv2.threshold(
    gray, 0, 255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
  )
  
  # first define a working kernel
  kernel = np.ones((3,3),np.uint8)
  
  # find morphological edges using the kernel
  # and five (5) iterations of the routine
  edged = cv2.morphologyEx(
    bw,
    cv2.MORPH_CLOSE,
    kernel,
    iterations = 5
  )
  
  # find corners of the largest rectangle
  corners = find_contours(edged)
  
  # original coordinate references
  pts2 = np.float32(
    [[0,0],
    [image.shape[1],0],
    [image.shape[1],image.shape[0]],
    [0,image.shape[0]]]
  )
  
  # homography calculation based upon the approximate
  # location of the bounding box of the sheet and
  # the original dimensions
  h = cv2.getPerspectiveTransform(corners, pts2)
  
  # crop using cropping homography
  im_crop = cv2.warpPerspective(image, h, None)
  
  # ----- create new matte ----
  # reshuffle coordinates to a set
  # of points to use in the coordinate
  # transform mapping
  x = corners[:,0]
  y = corners[:,1]
  w = int(max(x) - min(x))
  h = int(max(y) - min(y))
  x = int(min(x))
  y = int(min(y))
  
  # create empty image
  colours = image_orig[y-500:(y-500+h),x-500:(x-500+w)].mean(axis=0).mean(axis=0)
  
  # create RGB layers with the mean of the inset image (table)
  r = np.full((image_orig.shape[0], image_orig.shape[1]), int(colours[0]))
  g = np.full((image_orig.shape[0], image_orig.shape[1]), int(colours[1]))
  b = np.full((image_orig.shape[0], image_orig.shape[1]), int(colours[2]))
  
  # create an RGB stack
  matte = np.stack((r, g, b), axis = 2).astype(np.uint8)
  
  # overwrite the central part with the original image
  matte[y-500:(y-500+h),x-500:(x-500+w)] = image_orig[y-500:(y-500+h),x-500:(x-500+w)]
  
  # return the destination results
  return (im_crop, matte, h)

# get most common element from a list of strings
def most_common(lst):
    data = Counter(lst)
    return data.most_common(1)[0][0]


def find_contours(image):
  # from this image which is all edges extract
  # the contours of this center area
  (contours, _) = cv2.findContours(
    image,
    cv2.RETR_LIST,
    cv2.CHAIN_APPROX_NONE
  )
  
  # order these by length and use the longest one
  # to create a bounding rectangle / polygon
  contours = sorted(
    contours,
    key = cv2.contourArea,
    reverse = True
  )
  
  # get approximate contour
  for c in contours:
      p = cv2.arcLength(c, True)
      
      # 0.02 is the epsilon factor which determines
      # the reduction in complexity
      # which stops when the polygon has only 4 points
      # should be increased a bit to 0.023
      corners = cv2.approxPolyDP(c, 0.03 * p, True)
      
      if len(corners) != 4:
          raise ValueError("Sheet is not defined by four corners.")
      else:
        break
  
  # reshuffle corner coordinates
  corners = corners.reshape((4,2))
  corners_new = np.zeros((4,2),dtype = np.float32)

  add = corners.sum(1)
  corners_new[0] = corners[np.argmin(add)]
  corners_new[2] = corners[np.argmax(add)]

  diff = np.diff(corners,axis = 1)
  corners_new[1] = corners[np.argmin(diff)]
  corners_new[3] = corners[np.argmax(diff)]
  
  return corners_new

# remove vertical and horizontal
# lines by replacing it with the 
# average colour
# NOTE: Ugly clean up
def remove_lines(image):
  image_bin = binarize(image)
  image_bin = cv2.bitwise_not(image_bin)
  
  # hough lines
  linesP = cv2.HoughLinesP(image_bin, 1, np.pi/180, 50, None, 50, 100)
  
  # mean colours of image
  colours = image.mean(axis=0).mean(axis=0)
  colours = (int(colours[0]),int(colours[1]),int(colours[2]))
  
  rows = []
  cols = []
  
  # Draw the lines
  if linesP is not None:
    for i in range(0, len(linesP)):
      l = linesP[i][0]
      
      angle = abs(np.arctan2(l[3] - l[1], l[2] - l[0]) * 180.0 / np.pi)
      
      if angle >= 88 and angle <= 92:
        cv2.line(image, (l[0], l[1]), (l[2], l[3]), colours, 3, cv2.LINE_AA)
        
      if angle >= 0 and angle <= 2:
        cv2.line(image, (l[0], l[1]), (l[2], l[3]), colours, 3, cv2.LINE_AA)
      
      if angle >= 178 and angle <= 182:
        cv2.line(image, (l[0], l[1]), (l[2], l[3]), colours, 3, cv2.LINE_AA)
      
  return image

# subsets cell to the largerst outline
# marked by column or row lines, if no
# bounding box is available the original
# image is returned

def subset_cell(image, distance = 10):
  image_bin = binarize(image)
  image_bin = cv2.bitwise_not(image_bin)
  
  # grab dimensions
  height, width = image_bin.shape
  min_dim = min(height, width)
  
  # Hough line detection
  linesP = cv2.HoughLinesP(
    image_bin, 1,
    np.pi/180, 50, None,
    min_dim * 0.5,
    min_dim * 0.2
  )
  
  # more complex merging of lines is possible but simple logic will
  # do in this case, more reading on this topic can be found here:
  # https://stackoverflow.com/questions/45531074/how-to-merge-lines-after-houghlinesp
  
  # filter the detected lines on their angle
  # use a two degree tolerance
  if linesP is not None:
    
    # empty placeholders
    rows = []
    cols = []
    
    for i in range(0, len(linesP)):
      l = linesP[i][0]
      
      angle = abs(np.arctan2(l[3] - l[1], l[2] - l[0]) * 180.0 / np.pi)
      
      if angle >= 88 and angle <= 92:
        cols.append(int(np.average([l[0], l[2]])))
      if angle >= 0 and angle <= 2:
        rows.append(int(np.average([l[1], l[3]])))
      if angle >= 178 and angle <= 182:
        rows.append(int(np.average([l[1], l[3]])))
  
    # find the extremes in the detected
    # rows and columns
    min_row = min(rows)
    max_row = max(rows)
    min_col = min(cols)
    max_col = max(cols)
    diff_row = abs(min_row - max_row)
    diff_col = abs(min_col - max_col)
    
    # defaults (whole image)
    if diff_row < distance:
      # decision logic
      if max_row < height * 0.66:
        tmp_max_row = height
        tmp_min_row = max_row
        
      if min_row > height * 0.66:
        tmp_max_row = min_row
        tmp_min_row = 0
    else:
      tmp_max_row = max_row
      tmp_min_row = min_row
    
    if diff_col < distance:
      if max_col < width * 0.33:
        tmp_max_col = width
        tmp_min_col = max_col
    
      if min_col > width * 0.66:
        tmp_max_col = min_col
        tmp_min_col = 0
    else:
      tmp_max_col = max_col
      tmp_min_col = min_col
  
    image = image[
      tmp_min_row:tmp_max_row,
      tmp_min_col:tmp_max_col
    ]
  
  return image
