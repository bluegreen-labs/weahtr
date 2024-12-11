import os
import numpy as np
import pandas as pd
import yaml
import cv2
import logging
from tqdm import tqdm
from utils import *

class template:
  
  # initiating instance, with unique elements
  # dynamically set
  def __init__(self, images, template, config):
    
    # read in template matching config
    # file including output directory etc
    with open(config, 'r') as file:
      try:
        self.config = yaml.safe_load(file)
      except:
        print("No yaml config file at this location...")
    
    # split out output directory
    out_dir = self.config['output']
    
    # check and create output directories
    if not os.path.exists(os.path.join(out_dir, 'homography')):
      os.makedirs(os.path.join(out_dir, 'homography'), exist_ok=True)
      
    if not os.path.exists(os.path.join(out_dir, 'preview')):
      os.makedirs(os.path.join(out_dir, 'preview'), exist_ok=True)  
    
    if not os.path.exists(os.path.join(out_dir, 'labels')):
      os.makedirs(os.path.join(out_dir, 'labels'), exist_ok=True)
    
    # create output paths
    self.homography_path = os.path.join(out_dir, 'homography')
    self.preview_path = os.path.join(out_dir, 'preview')
    self.label_path = os.path.join(out_dir, 'labels')
    
    # create log list
    self.log = []
    
    # create homography dictionary
    self.homography = {}
    self.method = []
    
    # set basic info such as the list
    # of images to consider and the template to use
    self.images = images
    self.template = template
    
  #--- private functions ----
    
  def __fft(self, image, template):
    
    # set scale ratio
    scale_ratio = self.config['scale_ratio']
    
    # binarize
    image = binarize(image)
    template = binarize(template)
    
    # scale images for speed
    image = cv2.resize(
      image,
      None,
      fx = scale_ratio, 
      fy = scale_ratio
      )
      
    template = cv2.resize(
      template,
      None,
      fx = scale_ratio, 
      fy = scale_ratio
    )
    
    # fft transforms
    fft_img, img_mag = calculate_fft(image)
    fft_template, template_mag = calculate_fft(template)

    # calculate fft parameters
    center = [math.floor((img_mag.shape[0] + 1) / 2), math.floor((img_mag.shape[1] + 1 ) / 2)]
    max_val = np.maximum(center, np.asarray(img_mag.shape) - center)
    max_distance = ((max_val[0] ** 2 + max_val[1] ** 2 ) ** 0.5)
    base = math.exp(math.log(max_distance) / img_mag.shape[1])
    angles = ( 1.0 * math.pi ) / img_mag.shape[0]
    
    # log-polar transform
    # check warpPolar in opencv
    image_lp = logpolar(img_mag, center, angles, base)
    template_lp = logpolar(template_mag, center, angles, base)
    
    # check order things
    # check phaseCorrelation in opencv
    angle, scale = phase_correlation(template_lp, image_lp)
    scale = base ** scale
    
    # conversion to degrees
    angle = -(float(angle) * 180.0 ) / image_lp.shape[0]
    
    # correct range
    if angle < - 45:
        angle += 180
    else:
        if angle > 90.0:
            angle -= 180
    
    # check scale and rotation values before proceeding
    if scale > self.config['fft']['scale_tolerance']:
      raise ValueError('Scale value out of range')
    
    if abs(angle) > self.config['fft']['rotation_tolerance']:
      raise ValueError('Rotation value out of range')
    
    RS = cv2.getRotationMatrix2D((center[0], center[1]), angle, scale)
    RS = np.vstack([RS, [0,0,1]])
    image_RS = cv2.warpPerspective(image, RS, (image.shape[1], image.shape[0]))
    
    fft_img_new, _ = calculate_fft(image_RS)
    y, x = phase_correlation(fft_img_new, fft_template)
    
    if x > image.shape[0] // 2:
        x -= image.shape[0]
    if y > image.shape[1] // 2:
        y -= image.shape[1]
    
    # validate translation values
    translation_threshold = self.config['fft']['translation_tolerance'] * scale_ratio
    
    if x > translation_threshold or y > translation_threshold:
      raise ValueError('Translation values out of range')

    # create translation matrix
    T = np.float32([[1,0,-x],[0,1,-y],[0,0,1]])
    
    # calculate combined homography
    h = np.matmul(T,RS)
    
    # correct for scale factor
    h = h * [[1,1,1/scale_ratio],
            [1,1,1/scale_ratio],
            [scale_ratio, scale_ratio,1]]
    
    return h

  def __features(self, image, template):
    
    # set scale ratio
    scale_ratio = self.config['scale_ratio']
    
    # binarize
    image = binarize(image)
    template = binarize(template)
    
    # scale images for speed
    # and general accuracy
    image = cv2.resize(
      image,
      None,
      fx = scale_ratio, 
      fy = scale_ratio
      )
      
    template = cv2.resize(
      template,
      None,
      fx = scale_ratio, 
      fy = scale_ratio
    )
    
    # Detect ORB features and compute descriptors
    orb = cv2.ORB_create(self.config['features']['max_features'])
    keypoints1, descriptors1 = orb.detectAndCompute(image, None)
    keypoints2, descriptors2 = orb.detectAndCompute(template, None)
    
    # create BFMatcher object
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
    matches = matcher.match(descriptors1, descriptors2, None)
    
    # Sort them in the order of their distance
    matches = sorted(matches, key = lambda x:x.distance)
  
    # Remove not so good matches
    numGoodMatches = int(len(matches) * self.config['features']['good_match'])
    matches = matches[:numGoodMatches]
    
    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype = np.float32)
    points2 = np.zeros((len(matches), 2), dtype = np.float32)
  
    for i, match in enumerate(matches):
      points1[i, :] = keypoints1[match.queryIdx].pt
      points2[i, :] = keypoints2[match.trainIdx].pt
    
    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
    
    # correct for scale factor
    h = h * [[1,1,1/scale_ratio],
             [1,1,1/scale_ratio],
             [scale_ratio, scale_ratio,1]]
    
    # return matched image and ancillary data
    return h
  
  def __transform(self, image, h):
    
    # read current image
    im = cv2.imread(image)
    # read template
    template = cv2.imread(self.template, cv2.IMREAD_GRAYSCALE)
    
    if self.method == "fft":
      im, _ = match_size(im, template)
      dst = cv2.warpPerspective(im, h, None)
    else:
      dst = cv2.warpPerspective(im, h, None)
    
    dst = dst[0:template.shape[0],0:template.shape[1]]
    
    return dst

  #--- public functions ----
  
  ##--- cropping function ----
    
  # static variables and private
  # internal functions
  def crop(self, images):

    for image in images:
      image = cv2.imread(images)
        
      # black border to make it work
      # on white matte images
      image = cv2.copyMakeBorder(
        image, 500, 500, 500, 500,
        cv2.BORDER_CONSTANT,
        None,
        0
      )
      
      # convert to grayscale (red channel)
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
      
      # from this image which is all edges extract
      # the contours of this center area
      (contours, _) = cv2.findContours(
        edged,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_NONE
      )
      
      # order these by length and use the longest one
      # to create a bounding rectangle / polygon
      contours = sorted(contours, key=cv2.contourArea, reverse=True)
      
      # get approximate contour
      for c in contours:
          p = cv2.arcLength(c, True)
          
          # 0.02 is the epsilon factor which determines
          # the reduction in complexity
          # which stops when the polygon has only 4 points
          corners = cv2.approxPolyDP(c, 0.02 * p, True)
          
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
      
      # reshuffle coordinates to a set
      # of points to use in the coordinate
      # transform mapping
      x = corners_new[:,0]
      y = corners_new[:,1]
      w = int(max(x) - min(x))
      h = int(max(y) - min(y))
      x = int(min(x))
      y = int(min(y))
  
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
      h = cv2.getPerspectiveTransform(corners_new, pts2)
      
      # set crop file
      h_crop_file = os.path.join(self.homography_path, pathname + '_crop.txt')
  
      # pad the image to account for
      # white matte setups
      im = cv2.copyMakeBorder(
        im, 500, 500, 500, 500,
        cv2.BORDER_CONSTANT,
        None,
        0
      )
  
      # crop using cropping homography
      im = cv2.warpPerspective(im, h_crop, None)
      
      # return the destination results
      return h
  
  # match templates and write homography
  # files to disk to speed up (repeated)
  # processing
  
  def match(self, method):
    
    # set method as state variable
    self.method = method
    
    # read template
    template = cv2.imread(self.template, cv2.IMREAD_GRAYSCALE)
    
    for image in tqdm(self.images):
      
      # extract basename image
      basename = os.path.basename(image)
      pathname, _ = os.path.splitext(basename)
      
      # read current image
      im = cv2.imread(image)
      
      # template match the cropped image to
      # a reference template using a method of choice
      try:
        if method == "features":
          h = self.__features(im, template)
        else:
          # match image sizes
          im, template_new = match_size(im, template)
          h = self.__fft(im, template_new)
      except:
        # always log state internally
        self.log.append(image)
        continue
      
      # add homography to dictionary
      self.homography[image] = h
      
      # save homography file
      h_file = os.path.join(self.homography_path, pathname + '_match.txt')
      np.savetxt(h_file, h, delimiter=",")
      
      # output preview on request
      if self.config['preview']:
        
        # process using homography
        dst = cv2.warpPerspective(im, h, None)
        
        # generate preview
        preview(
          dst,
          template,
          os.path.join(self.preview_path, pathname + ".jpg")
        )
        
  # label matched templates
  def label(self, guides):
    
    # load template guides
    guides = load_guides(guides, "format_1")
    
    # sideload homography dictionary if possible
    if len(self.homography) == 0:
      # extract basename image
      basename = os.path.basename(image)
      pathname, _ = os.path.splitext(basename)
    
      #h_file = os.path.join(self.homography_path, pathname + '_match.txt')
      #h = np.genfromtxt(h_file, delimiter=',')
      #self.homography = XXX
      print("bla")
    
    # loop over all homography files / images
    # and transform, slice and label the data
    for image, h in self.homography.items():
      
      # format the homography file to use for
      # image deformation
      #h_file = os.path.join(self.homography_path, pathname + '_match.txt')
      #h = np.genfromtxt(h_file, delimiter=',')
      
      # transform the image using the provided homography
      matched_image = self.__transform(image, h)
