import os
import numpy as np
import pandas as pd
import yaml
import cv2
from utils import *

class template:
  
  # initiating routine
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
    
    if not os.path.exists(os.path.join(out_dir, 'logs')):
      os.makedirs(os.path.join(out_dir, 'logs'), exist_ok=True)
    
    # create output paths
    self.homography_path = os.path.join(out_dir, 'homography')
    self.preview_path = os.path.join(out_dir, 'preview')
    self.label_path = os.path.join(out_dir, 'labels')
    self.log_path = os.path.join(out_dir, 'labels')
    
    # set basic info such as the list
    # of images to consider and the template to use
    self.images = images
    self.template = template
  
  #--- private functions ----
  
  def __match_preview(self, image, template, pathname):
    
    # split band and crop
    _,_,image = cv2.split(image)
    image = self.__binarize(image)
    image = image[0:template.shape[0],0:template.shape[1]]
      
    # construct preview
    preview = np.full((image.shape[0],image.shape[1],3),255, dtype=np.uint8)
    preview[:,:,1] = image
    preview[:,:,2] = template
    
    # resize to fifth of the original
    preview = cv2.resize(preview, None, fx=0.2, fy = 0.2)
      
    # write to preview path
    cv2.imwrite(os.path.join(
      self.preview_path,
      pathname + '_match.jpg'),
      preview
    )
  
  def __error_log(self, path, prefix, content):
    filename = os.path.join(path, prefix + "_error_log.txt")
    with open(filename, "a") as text_file:
       text_file.write(content + "\n")
  
  # binarize the image
  def __binarize(self, image):
    # Adaptive thresholding
    image = cv2.adaptiveThreshold(
        image,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
        11,
        2
    )
    
    # Median blurring
    image = cv2.medianBlur(image, 3)
    
    return image
  
  # static variables and private
  # internal functions
  def __inner_crop(self, image):
      
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
    
    # return the destination results
    return h
    
  #def __fft(self):
      # code goes here

  def __features(self, image, template):
    
    # convert images to grayscale
    # using red channel for max
    # constrast
    _,_,image = cv2.split(image)
    
    # scale images for speed
    # and general accuracy
    image = cv2.resize(
      image,
      None,
      fx = self.config['features']['scale_ratio'], 
      fy = self.config['features']['scale_ratio']
      )
      
    template = cv2.resize(
      template,
      None,
      fx = self.config['features']['scale_ratio'], 
      fy = self.config['features']['scale_ratio']
    )
    
    # binarize
    image = self.__binarize(image)
    template = self.__binarize(template)
    
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
  
    # Prune reference points based upon distance between
    # key points. This assumes a fairly good alignment to start with
    # due to the protocol used
    p1 = pd.DataFrame(data=points1)
    p2 = pd.DataFrame(data=points2)
    refdist = abs(p1 - p2)
    
    # Assumes that the size of input and reference data don't
    # differ more than 20% of their total axis size
    # check axis / order !!
    # USE QA/QC values
    reference_distance = (self.config['features']['reference_distance'] / 100)
    
    refdist.loc[:,0] = np.float32(refdist.loc[:,0] < (image.shape[1] * reference_distance))
    refdist.loc[:,1] = np.float32(refdist.loc[:,1] < (image.shape[0] * reference_distance))
    refdist = refdist.sum(axis = 1) == 2
    points1 = points1[refdist]
    points2 = points2[refdist]
    
    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
    
    # set scale ratio
    scale_ratio = self.config['features']['scale_ratio']
    
    # correct for scale factor, only works if both the template
    # and the matching image are of the same size
    h = h * [[1,1,1/scale_ratio],
             [1,1,1/scale_ratio],
             [scale_ratio, scale_ratio,1]]
    
    # return matched image and ancillary data
    return h
  
  #--- public functions ----
  
  # match templates and write homography
  # files to disk to speed up (repeated)
  # processing
  
  def match(self, method):
    
    # read template
    template = cv2.imread(self.template, cv2.IMREAD_GRAYSCALE)
    
    for image in self.images:
      
      # extract basename image
      basename = os.path.basename(image)
      pathname, _ = os.path.splitext(basename)
      
      # read current image
      im = cv2.imread(image)
      
      if self.config['crop']:
        
        # check if crop homography file exists
        # in output location
        if not os.path.exists(h_file):

          # calculate and save cropping homography
          try:
            h = self.__inner_crop(im)
          except:
            print("Cropping failed... skipping file")
            self.__error_log(self.log_path, "cropping", image)
            next
          
          #h_file = os.path.join(self.homography_path, pathname + '_crop.txt')
          #np.savetxt(h_file, h, delimiter=",")

        else:
          # read homography from file if available
          h = np.genfromtxt(h_file, delimiter=',')

        # pad the image to account for
        # white matte setups
        im = cv2.copyMakeBorder(
          im, 500, 500, 500, 500,
          cv2.BORDER_CONSTANT,
          None,
          0
        )

        # crop using cropping homography
        im = cv2.warpPerspective(im, h, None)
    
      # template match the cropped image to
      # a reference template using a method of choice
      try:
        if method == "features":
          h = self.__features(im, template)
        else:
          print("bla")
      except:
        self.__error_log(self.log_path, "matching", image)
        next
      
      h_file = os.path.join(self.homography_path, pathname + '_match.txt')
      np.savetxt(h_file, h, delimiter=",")
      
      # output preview on request
      if self.config['preview']:
        dst = cv2.warpPerspective(im, h, None)
        self.__match_preview(dst, template, pathname)
        
  # label matched templates
  def label(self, guides):
    
    # load template guides
    guides = load_guides(guides, "format_1")
    
    print(guides)
    
    # read template
    template = cv2.imread(self.template, cv2.IMREAD_GRAYSCALE)
    
    # loop over all images
    for image in self.images:
      
      # extract basename image
      basename = os.path.basename(image)
      pathname, _ = os.path.splitext(basename)
      
      # read current image
      im = cv2.imread(image)
    
      # format the homography file to use for
      # image deformation
      h_file = os.path.join(self.homography_path, pathname + '_match.txt')

      # check if crop homography file exists
      # in output location
      if os.path.exists(h_file):
        
        # read homography file
        h = np.genfromtxt(h_file, delimiter=',')
        
        # warp image
        dst = cv2.warpPerspective(im, h, None)
        dst = dst[0:template.shape[0],0:template.shape[1]]
        
      else:
        print("Skipping, no template match found for file: " + image)
        next
        
    print("Done")
