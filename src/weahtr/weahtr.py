# libraries
import os, yaml, json
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
import shutil
import requests

from weahtr.utils import *
from weahtr.model import *

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

# template processing class
class template():
  
  # initiating instance, with unique elements
  # dynamically set, this allows for dynamic
  # iterations in python scripts
  def __init__(self, images, template, config, **kwargs):
    
    # validate inputs
    if not os.path.exists(template):
      raise ValueError("Template image does not exist, check path ...")
      
    if not os.path.exists(config):
      raise ValueError("Config file does not exist, check path ...")
    
    # some feedback
    print("\n")
    print("Setting up output directories at:")
    
    # read in template matching config
    # file including output directory etc
    with open(config, 'r') as file:
      try:
        self.config = yaml.safe_load(file)
        self.config_file = config
      except:
        raise ValueError("No yaml config file, or badly formatted YML file (check all quotes) ...")
    
    # split out output directory
    out_dir = self.config['output']
    
    # set subdirectory based on profile name
    # this allows iterative processing or multiple
    # methods or parts of the image (think table + header)
    sub_dir = self.config['profile_name']
    
    # feedback continued
    print(os.path.join(out_dir, sub_dir))
    print("\n")
    
    # where the model should live
    model_path = os.path.join(
         self.config['tesseract']['src_path'],
         self.config['tesseract']['model']
    )
    
    # if on Docker set dst_path
    if os.path.exists("/.dockerenv"):
      dst_path = "/opt/conda/envs/weahtr/share/tessdata/"
    else:
      dst_path = self.config['tesseract']['dst_path']
      
    try:
      shutil.copy(
        model_path,
        dst_path
      )
    except:
      print("Tesseract model or destination path does not exist!")
    
    # check and create output directories
    if not os.path.exists(os.path.join(out_dir, sub_dir, 'homography')):
      os.makedirs(os.path.join(out_dir, sub_dir, 'homography'), exist_ok=True)
      
    if not os.path.exists(os.path.join(out_dir, sub_dir, 'preview')):
      os.makedirs(os.path.join(out_dir, sub_dir, 'preview'), exist_ok=True)  
    
    if not os.path.exists(os.path.join(out_dir, sub_dir, 'labels')):
      os.makedirs(os.path.join(out_dir, sub_dir, 'labels'), exist_ok=True)
    
    # create output paths
    self.homography_path = os.path.join(out_dir, sub_dir, 'homography')
    self.preview_path = os.path.join(out_dir, sub_dir, 'preview')
    self.label_path = os.path.join(out_dir, sub_dir, 'labels')
    
    # create log list
    self.log = []
    
    # create homography dictionary
    self.homography = {}
    
    # create methods placeholder
    if 'method' in kwargs:
      self.method = kwargs['method']
    
    # create guides placeholder
    if 'guides' in kwargs:
      self.guides = kwargs['guides']
    
    # create methods placeholder
    if 'model' in kwargs:
      self.model = kwargs['model']
    
    # set basic info such as the list
    # of images to consider and the template to use
    
    # this funkiness avoids a string being
    # read as a list, if a single image is
    # provided and directly assigned to self.images
    self.images = []
    for image in images:
      self.images.append(image)
    
    # the template path
    self.template = template
    
  #--- private functions ----
    
  def __fft(self, image, template):
    
    # set scale ratio
    scale_ratio = self.config['scale_ratio']

    # binarize
    image = binarize(
      image,
      self.config['threshold']['window_size'],
      self.config['threshold']['C']
    )
    template = binarize(
      template,
      self.config['threshold']['window_size'],
      self.config['threshold']['C']
    )

    # scale images for speed
    # and general accuracy
    image = cv2.resize(image, None, fx = scale_ratio, fy = scale_ratio)
    template = cv2.resize(template, None, fx = scale_ratio, fy = scale_ratio)
    
    # match image sizes
    image, template = match_size(image, template)

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
    
    # validate translation values (decrease scale as before correction)
    translation_threshold = self.config['fft']['translation_tolerance'] * scale_ratio
    
    if abs(x) > translation_threshold or abs(y) > translation_threshold:
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

  def __table(self, image, guides):
    
    # replace matte with average of the image
    # content to properly binarize
    _ , image , _ = replace_matte(image)
    
    # binarize
    image = binarize(
      image,
      self.config['threshold']['window_size'],
      self.config['threshold']['C']
    )
    
    # invert and dilate
    image = cv2.bitwise_not(image)
    image = cv2.dilate(
        image,
        None,
        iterations = 5
    )

    # find maximal contours by area
    pts1 = find_contours(image)
    pts1 = np.float32(pts1)
    
    # load template guides
    # NOTE: move outside function for speed (pre-load)
    # only pass pts2 variable
    cells = load_guides(guides)
    xmin = min(cells['x_min'])
    xmax = max(cells['x_max'])
    ymin = min(cells['y_min'])
    ymax = max(cells['y_max'])
    
    # destination of transform from template
    pts2 = np.float32([
        [xmin, ymin],
        [xmax, ymin],
        [xmax, ymax],
        [xmin, ymax]
        ])
        
    # calculate homography (transformation matrix)
    h = cv2.getPerspectiveTransform(pts1, pts2)
    
    return h
    
  def __features(self, image, template):
    
    # set scale ratio
    scale_ratio = self.config['scale_ratio']
    
    # binarize
    image = binarize(
      image,
      self.config['threshold']['window_size'],
      self.config['threshold']['C']
    )
    template = binarize(
      template,
      self.config['threshold']['window_size'],
      self.config['threshold']['C']
    )
    
    # scale images for speed
    # and general accuracy
    image = cv2.resize(image, None, fx = scale_ratio, fy = scale_ratio)
    template = cv2.resize(template, None, fx = scale_ratio, fy = scale_ratio)
    
    # Detect ORB features and compute descriptors
    orb = cv2.ORB_create(self.config['features']['max_features'])
    keypoints1, descriptors1 = orb.detectAndCompute(image, None)
    keypoints2, descriptors2 = orb.detectAndCompute(template, None)
    
    # create BFMatcher object
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
    matches = matcher.match(descriptors1, descriptors2, None)
    
    # Sort them in the order of their distance
    matches = sorted(matches, key = lambda x:x.distance)
  
    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype = np.float32)
    points2 = np.zeros((len(matches), 2), dtype = np.float32)
  
    for i, match in enumerate(matches):
      points1[i, :] = keypoints1[match.queryIdx].pt
      points2[i, :] = keypoints2[match.trainIdx].pt
    
    # Find homography
    h, _ = cv2.findHomography(points1, points2, cv2.RANSAC)
    
    # correct for scale factor
    h = h * [[1,1,1/scale_ratio],
             [1,1,1/scale_ratio],
             [scale_ratio, scale_ratio,1]]
    
    # validate translation values
    translation_threshold = self.config['fft']['translation_tolerance']
    x, y = abs(h[:2, 2])
    
    if x > translation_threshold or y > translation_threshold:
      raise ValueError('Translation values out of range')
    
    # return matched image and ancillary data
    return h
  
  def __transform(self, image, h):
    
    # read image and template
    im = cv2.imread(image)
    template = cv2.imread(self.template, cv2.IMREAD_GRAYSCALE)
    
    if self.method == "fft":
      im, _ = match_size(im, template)
      dst = cv2.warpPerspective(im, h, None)
    else:
      dst = cv2.warpPerspective(im, h, None)
    
    dst = dst[0:template.shape[0],0:template.shape[1]]
    
    return dst

  def __label_cells(self, im, cells, prefix, slices, m, f):
    
    # initiate empty vectores
    text = []
    conf = []
    file_name = []
    row = []
    col = []
    x = []
    y = []
    text_soft_val = []
    conf_soft_val = []
    majority_frac = []
    
    # generates cropped sections based upon
    # row and column locations
    t = tqdm(cells.iterrows(), total=cells.shape[0], leave = False)
    for index, cell in t:
      t.set_description(
        "Processing column %i, row %i " % (cell['col'], cell['row']),
        refresh = True
      )
      
      # add filename and row / col numbers
      file_name.append(prefix)
      col.append(cell['col'])
      row.append(cell['row'])
      x.append(cell['x_min'])
      y.append(cell['y_max'] - (cell['y_max'] - cell['y_min'])/4)
      
      try: # traps failures to crop properly
        
        # trap end of table issues 
        # (when running out of space)
        if cell['x_max'] > im.shape[1]:
          cell['x_max'] = int(im.shape[1])
          
        if cell['y_max'] > im.shape[0]:
          cell['y_max'] = int(im.shape[0])
        
        # apply padding factors
        pad_left = int((cell['x_max'] - cell['x_min']) * self.config['pad_left'])
        pad_right = int((cell['x_max'] - cell['x_min']) * self.config['pad_right'])
        pad_top = int((cell['y_max'] - cell['y_min']) * self.config['pad_top'])
        pad_bottom = int((cell['y_max'] - cell['y_min']) * self.config['pad_bottom'])
        
        x_min = cell['x_min'] - pad_left
        x_max = cell['x_max'] + pad_right
        y_min = cell['y_min'] - pad_top
        y_max = cell['y_max'] + pad_bottom
        
        # crop image to size
        crop_im = im[y_min:y_max, x_min:x_max]
        
        if f is not None:
          try:
            crop_im = f(crop_im)
          except:
            #logging
            continue
        
        if slices:
          # write data slices to file
          # for post processing such as citizen science or model training
          filename = os.path.join(
              self.preview_path,
              prefix + "_" + self.config['profile_name'] + 
              "_" + str(cell['col']) + "_" + str(cell['row']) + ".jpg"
          )
          cv2.imwrite(filename, crop_im)
          
        else:
          try:
            # transcription based upon model forwarded
            # and selected in config
            label, confidence = m.predict(crop_im)
            text.append(label)
            conf.append(round(confidence, 3))
          except:
            # write label output data to vectors
            conf.append(0)
            text.append('//')
            
          try:
            # soft validation runs
            if self.config['soft_val'] > 0:
              
              text_tmp = []
              conf_tmp = []
            
              i = 1
              while i <= self.config['soft_val']:
                # augment the original image
                # using the train transform
                crop_im_aug = train_transform(image = crop_im)['image']
                
                # classification
                label_aug, confidence_aug = m.predict(crop_im_aug)
                text_tmp.append(label_aug)
                conf_tmp.append(confidence_aug)
                i += 1
              
              # calculate unique largest fraction
              tmp_frac = round(1 - len(list(set(text_tmp)))/self.config['soft_val'], 3)
              
              # add the number of classes detected
              majority_frac.append(tmp_frac)
              
              label_aug = most_common(text_tmp)
              conf_aug = np.average(conf_tmp)
              text_soft_val.append(label_aug)
              conf_soft_val.append(round(conf_aug, 3))
          except:
            majority_frac.append(0)
            text_soft_val.append('//')
            conf_soft_val.append(0)
      except:
       # Continue to next iteration
       continue
    
    if not slices:
      
      # concat data into pandas data frame
      df = pd.DataFrame(
        {'text':text,
         'conf': conf,
         'text_val': text_soft_val,
         'conf_val': conf_soft_val,
         'majority_frac': majority_frac,
         'col':col,
         'row':row,
         'x': x,
         'y': y,
         'file':file_name
        }
      )
    
      # construct path
      filename = os.path.join(
        self.label_path,
        prefix + "_" + self.config['profile_name'] + "_labels.csv"
      )
      
      # write data to disk
      df.to_csv(filename, sep=',', index = False)
      
      return df

  #--- public functions ----
  def save_log(self, path):
    
    # set filename
    filename = os.path.join(
      path, self.config['profile_name'] + "_log.json" 
    )
    
    with open(filename, "w") as out:
      json.dump(self.log, out)
  
  # match templates and write homography
  # files to disk to speed up (repeated)
  # processing
  def match(self, preview = False, **kwargs):
    
    # set template if not inherited
    if not hasattr(self, 'template') and preview:
      # set method as state variable
      try:
        self.template = kwargs['template']
      except:
        raise ValueError("No template is provided, while a preview is requested.")
    
    # set method if not inherited
    if not hasattr(self, 'method'):
      # set method as state variable
      try:
        self.method = kwargs['method']
      except:
        raise ValueError("No method set in template or function call.")
    
    # set guides if not inherited for the table method
    # requires referencing of the largest table outline
    if self.method == "table":
      if not hasattr(self, 'guides'):
        # set method as state variable
        try:
          self.guides = kwargs['guides']
        except:
          raise ValueError("No guides set in template or function call.")
    
    # read template
    template = cv2.imread(self.template, cv2.IMREAD_GRAYSCALE)
    
    if self.method == "table":
      message = "Finding table    "
    else:
      message = "Matching template"
  
    for image in tqdm(self.images, desc = message):
      
      # extract basename image
      basename = os.path.basename(image)
      pathname, _ = os.path.splitext(basename)
      
      # read current image
      im = cv2.imread(image)
      
      # template match the cropped image to
      # a reference template using a method of choice
      try:
        if self.method == "features":
          h = self.__features(im, template)
        elif self.method == "fft":
          h = self.__fft(im, template)
        else:
          h  = self.__table(im, self.guides)
      except:
        # always log state internally
        self.log.append(image)
        continue

      # add homography to dictionary
      self.homography[image] = h.tolist()
      
      # output preview on request
      if preview:
        
        # process using homography
        dst = cv2.warpPerspective(
          im, h,
          (template.shape[1], template.shape[0])
        )
        
        # generate preview
        preview_match(
          dst,
          template,
          os.path.join(self.preview_path, pathname + ".jpg")
        )
    
    # save homography file
    h_file = os.path.join(
      self.homography_path, self.config['profile_name'] + '_homography.json'
      )
    
    # write homographies to json file
    with open(h_file, "w") as out:
      json.dump(self.homography, out)

  # process matched templates
  def process(self, preview = False, slices = False, **kwargs):
    
    # set method if not inherited from
    # first setup/template call
    if not hasattr(self, 'guides'):
      # set method as state variable
      try:
        self.guides = kwargs['guides']
      except:
        raise ValueError("No guides location set in template or function call.")
    
    if not hasattr(self, 'model'):
      # set method as state variable
      try:
        self.model = kwargs['model']
      except:
        raise ValueError("No model set in template or function call.")
    
    if 'f' in kwargs:
      f = kwargs['f']
    else:
      f = None
    
    # Validating all inputs
    if not os.path.exists(self.guides):
      raise ValueError("Template guides file does not exist, check path ...")
    
    # read template for previews
    template = cv2.imread(self.template, cv2.IMREAD_GRAYSCALE)
    
    # load template guides
    cells = load_guides(self.guides)
    
    # remove any cells that need to be skipped, such as rows to skip
    # and columns to process
    cells = cells[~cells["row"].isin(self.config['skip_rows'])]
    cells = cells[cells["col"].isin(self.config['select_cols'])]
    
    # create empty labels placeholder
    output = pd.DataFrame()
    
    # check if file exists, save homography file
    h_file = os.path.join(
      self.homography_path, self.config['profile_name'] + '_homography.json'
    )
    
    # Read JSON homography file
    if hasattr(self, 'homography'):
      try:
        with open(h_file, "r") as file:
          self.homography = json.load(file)
      except:
        raise ValueError("Failed to load homography file, check path!")
    
    # preload model, forwarded to labeller
    # shitty setup, should be fixed - read up on class
    # inheritance
    m = model(self.model, self.config_file)
    
    # loop over all homography files / images
    # and transform, slice and label the data
    for image, h in tqdm(self.homography.items(), desc="Processing data  "):
        
      # extract basename image
      basename = os.path.basename(image)
      pathname, _ = os.path.splitext(basename)
      
      # transform the image using the provided homography
      # to fit the template and derived guides
      matched_image = self.__transform(image, np.array(h))
      
      # label the cells in the table / sheet / header
      labels = self.__label_cells(
          matched_image,
          cells,
          pathname,
          slices,
          m = m, # forward the model
          f = f # forward a post-processing function
      )
      
      # only provide a preview when not processing slices (cells)
      if preview and not slices:
        preview_labels(
          matched_image,
          labels,
          os.path.join(self.label_path, pathname + ".jpg")
        )
      
      # concat stuff / this is probably
      # a bad idea for very large datasets as
      # not memory efficient / pre-allocation would
      # be better but not as flexible, data is alsw
      # written to file so output internally can be
      # scrapped
    #   output = pd.concat([output, labels], ignore_index=True)
    # return output
