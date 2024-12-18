#!/usr/bin/env python
import os, argparse, glob
import cv2
import numpy as np
import weahtr

# argument parser
def getArgs():

  parser = argparse.ArgumentParser(
    description = '''weaHTR framework command line script''',
    epilog = '''post bug reports to the github repository'''
  )
    
  parser.add_argument(
    '-i',
    '--image_dir',
    help = 'a directory of jpg images to reference',
    required = True
  )

  parser.add_argument(
    '-t',
    '--template',
    help = 'empty reference image template to match',
    required = True
  )

  parser.add_argument(
    '-tg',
    '--template_guides',
    help = 'location of the template JSON guide (region of interest) file'
    required = True
  )

  parser.add_argument(
    '-c',
    '--config_file',
    help = 'location of the YAML configuration file'
    required = True
  )

  parser.add_argument(
    '-m',
    '--method',
    help = 'template matching method',
    default = 'fft'
  )

  parser.add_argument(
    '-mo',
    '--model',
    help = 'ML model to use (tesseract or custom)',
    default = 'tesseract'
  )

  parser.add_argument(
    '-p',
    '--preview',
    help = 'generate preview images for visual inspection (True/False)',
    default = True
  )

  parser.add_argument(
    '-o',
    '--output_directory',
    help = 'output directory where to store all results',
    default = "/tmp"
  )

  return parser.parse_args()

if __name__ == '__main__':
  
  # parse arguments
  args = getArgs()
  
  # initiate template setup
  t = weahtr.template(
    images = args.image_dir,
    template = args.template,
    config = args.config
  )
  
  # match template
  t.match(
    method = args.method,
    preview = args.preview
  )
  
  # call the labelling ML routine
  t.label(
    guides = args.template_guides,
    model = args.model,
    preview = args.preview
    )
