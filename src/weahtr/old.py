#!/usr/bin/env python

# Import necessary libraries.
import os, argparse, glob, tempfile, shutil, warnings
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# HTR libraries
import pytesseract
import easyocr
from shutil import copyfile

# local functions
from inner_crop import *
from utils import *
from match import *
from label_cells import *

# ignore python warnings
warnings.simplefilter("ignore")

# argument parser
def getArgs():

   parser = argparse.ArgumentParser(
    description = '''Table alignment and subsetting script: 
                    Allows for the alignment of scanned tables relative to a
                    provided template. Subsequent cutouts can be made when 
                    providing a csv file with row and columns coordinates.''',
    epilog = '''post bug reports to the github repository''')
    
   parser.add_argument('-t',
                       '--template',
                       help = 'template to match to data',
                       default = '../data/templates/format_1.jpg'
                       )
                       
   parser.add_argument('-i',
                       '--input_directory',
                       default = '../data-raw/demo/format_1/',
                       help = 'input directory where to grab images from')

   parser.add_argument('-o',
                       '--output_directory',
                       help = 'location where to store the data',
                       default = '../data/output/data/')

   parser.add_argument('-s',
                       '--subsets',
                       default = True,
                       help = 'create subsets according to a coordinates file'
                       )

   parser.add_argument('-sr',
                       '--scale_ratio',
                       help = 'shrink data by factor x, for faster processing',
                       default = 1,
                       type = float
                       )
                       
   parser.add_argument('-gm',
                       '--good_match',
                       help = 'good match ratio',
                       default = 0.5,
                       type = float
                       )
                       
   parser.add_argument('-m',
                       '--model',
                       help='HTR model to use',
                       default = "tesseract"
                       )

   parser.add_argument('-cs',
                       '--classify_subset',
                       help='classify subset with a DNN',
                       default = False
                       )

   parser.add_argument('-gi',
                       '--guides',
                       help='name of file containing cell guides',
                       default = '../data/templates/guides.txt'
                       )

   parser.add_argument('-mf',
                       '--max_features',
                       help='max number of ORB features to use',
                       default = 25000,
                       type = int
                       )

   return parser.parse_args()

def error_log(path, prefix, content):
    filename = os.path.join(path, prefix + "_error_log.txt")
    with open(filename, "a") as text_file:
       text_file.write(content + "\n")

def setup_outdir(output_directory):
  
  if not os.path.exists(output_directory):
    os.makedirs(output_directory)
    
  if not os.path.exists(output_directory + "/headers/"):
    os.makedirs(output_directory + "/headers/")

  if not os.path.exists(output_directory + "/previews/"):
    os.makedirs(output_directory + "/previews/")
  
  if not os.path.exists(output_directory + "/labels/"):
    os.makedirs(output_directory + "/labels/")

if __name__ == '__main__':
  
  # loading tesseract  model
  # TODO error trapping on non-docker setup
  #copyfile(
  #  "./models/cobecore-V6.traineddata",
  #  "/usr/share/tesseract-ocr/5/tessdata/cobecore-V6.traineddata"
  #)
  
  # parse arguments
  args = getArgs()
  
  # get scale ratio
  scale_ratio = float(args.scale_ratio)

  # extract filename and extension of the mask
  mask_name, file_extension = os.path.splitext(args.template)
  mask_name = os.path.basename(mask_name)
  
  # load mask file
  try:
   template = cv2.imread(args.template)
  except:
   print("No valid mask image found at:")
   print(args.template)
   exit()

  # load guides
  try:
   guides = load_guides(args.guides, mask_name)
  except:
    print("No valid guides file found at:")
    print(args.guides)
    exit()

  # create a copy of the original
  template_original = template
  
  # OTSU thresholding + resizing
  template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
  template = cv2.GaussianBlur(template,(7, 7),0)
  ret, template = cv2.threshold(
    template, 0, 255,
    cv2.THRESH_BINARY+cv2.THRESH_OTSU
    )
  
  template = cv2.resize(
    template,
    (0,0),
    fx = scale_ratio,
    fy = scale_ratio
   )
  
  # list image files to be aligned and subdivided
  files = glob.glob(args.input_directory + "/*.jpg")
  #files = files[0:2]
  
  # Some verbose feedback before kicking off the processing
  print("\n")
  print("Reading reference image : " + str(args.template))
  print("Sourcing from : " + str(args.input_directory))
  print("Saving to : " + str(args.output_directory))
  print("\n")
  
  # loop over all files, align and subdivide and report progress
  #with tqdm(total = files.shape[0], dynamic_ncols=True) as pbar:
  for file in files:
                  
      # compile final output directory name
      archive_id = os.path.basename(file).split('.')[0].split('_')[0]
      output_directory = args.output_directory + "/" + archive_id + "/"
      prefix = mask_name + "_" + os.path.basename(file).split('.')[0]
      
      # read input data
      try:
        im = cv2.imread(file)
      except:
        error_log(args.output_directory, "read", file)
        print("Failed to read : " + file)
        continue
  
      # crop red channel, reproject original
      # using the same parameters (crop)
      #if args.crop:
      im = inner_crop(im)
      
      # create a "red" grayscale copy
      im_tmp = im
      #im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
      _,_,im = cv2.split(im)

      # flatten image (binarization)
      im = binarize(im)

      # resize the image to the size of the template
      im = cv2.resize(
        im,
        (template.shape[1], template.shape[0])
      )
 
      # resize the original data to the size of
      # the full original template (required)
      # to be able to rescale to full resolution
      # with simple homography conversion factors
      im_tmp = cv2.resize(
        im_tmp,
       (template_original.shape[1],template_original.shape[0])
      )
  
      # setup output directories if required
      setup_outdir(output_directory)
      
      # align images
      try:
        im_aligned, h = match_template(
          im,
          template,
          im_tmp,
          args.max_features,
          args.good_match,
          args.scale_ratio
        )
        
        # create an alignment preview
        im_preview = im_aligned
        im_preview[:,:,2] = template_original[:,:,1]
        
        # Write aligned image to disk, including markings of
        im_preview = cv2.resize(im_preview, (0,0), fx = 0.25, fy = 0.25)
        filename = os.path.join(output_directory + "/previews",
                                prefix + "_preview.jpg")
        cv2.imwrite(filename, im_preview, [cv2.IMWRITE_JPEG_QUALITY, 50])
        
        # feedback
        print("Aligned image:")
        print(filename)
          
      except:
        print("FAILED to align image:")
        print(file)
        error_log(args.output_directory, "alignment", file)
        continue
        
      # cutting things up into cookies    
      if args.classify_subset:
        
        print("labelling cells...")
        
        try:
          labels = label_cells(
           guides,
           im_aligned,
           output_directory,
           prefix,
           args.model
           )
           
          # which cells were ok or not overprint data
          label_preview = print_labels(im_aligned, guides, labels)
          label_preview = cv2.resize(label_preview, (0,0), fx = 0.25, fy = 0.25)
          filename = os.path.join(output_directory + "/previews",
                                prefix + "_label_preview.jpg")
          cv2.imwrite(filename, label_preview, [cv2.IMWRITE_JPEG_QUALITY, 50])

        except:
          print("error - labelling failed")
          error_log(args.output_directory, "label", file)
          continue
      
      # update progress
      #pbar.update()
