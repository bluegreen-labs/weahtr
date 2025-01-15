# Data generator class
import os, glob, random
import cv2
import numpy as np
import pandas as pd
import albumentations as A
import torch
from torchvision import datasets, transforms
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
from weahtr.transform import *

class generator:
  def __init__(self,
    data_path,
    background = None,
    values = 3,
    decimal = True,
    sign = True
    ):
    
    # initiate things
    self.data_path = data_path
    self.background = background
    self.decimal = decimal
    self.sign = sign
    self.values = values
    
    # TODO validate paths!!
  
  #---- Private functions
  def __digits(
    self,
    path
    ):
    
    # load the UNIPEN decimals/sign images
    comma = glob.glob(os.path.join(self.data_path, "UNIPEN/comma/*.png"))
    point = glob.glob(os.path.join(self.data_path, "UNIPEN/point/*.png"))
    minus = glob.glob(os.path.join(self.data_path, "UNIPEN/minus/*.png"))
    
    # read in the background grid image and apply
    # the random transform to the data
    if self.background is not None:
      background = cv2.imread(self.background)
      background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
      
      # random sample of 200x200 from background image
      background = transform_grid(image=background)['image']
    
    # fixed width and height, if you change these change
    # the transform_grid() function as well
    height = 200
    width = 200
    
    # create numpy arrays to hold the data
    blank_image = np.ones((height, width, 2), np.uint8) * 255
    numbers = np.ones((height, width), np.uint8) * 255
    
    # Define the transformation to convert images to PyTorch tensors
    transform = transforms.Compose([transforms.ToTensor()])
    
    # Load the MNIST dataset with the specified transformation
    # will download automatically if not available
    mnist = datasets.MNIST(
      root = os.path.join(self.data_path),
      train = True,
      download = True,
      transform = transform
      )
    
    # Create a DataLoader to load the dataset in batches
    # randomly with a batch size of 1
    train_loader_pytorch = torch.utils.data.DataLoader(
      mnist,
      batch_size = 1,
      shuffle = True
      )
    
    # build the number backwards from the last decimal to the decimal point
    # onward to the minus sign
    digits = []
    images = []
    
    # Print the first few images in a random batch
    for i, (image, label) in enumerate(train_loader_pytorch):
      if i <= (self.values - 1):  # Print the first 5 samples
        
        # grab a random image
        img = image[0].squeeze().numpy()
        img = 255 - (img * 255)
        
        # transform
        img = transform_number(image = img)['image']
        x, y = img.shape
        
        tmp = blank_image
        tmp[:,:, 0] = numbers
        start_x = (width - (i + 1) * 35)
        
        tmp[60:(60 + y), start_x - x:start_x, 1] = img
        numbers = np.min(tmp, axis=2)
        
        # save label item
        digits.insert(0, label.item())
        
        # insert decimal point, randomly
        if i == 0 and self.decimal and random.random() < 0.5:
          decimal_separator = random.choice([".", ","])
          digits.insert(0, decimal_separator)
          
          # pick a random UNIPEN decimal separator
          if decimal_separator == ".":
            decimal_sep_img = random.choice(point)
          else:
            decimal_sep_img = random.choice(comma)
          
          # grab random decimal character
          decimal_sep_img = cv2.imread(decimal_sep_img)
          decimal_sep_img = cv2.cvtColor(decimal_sep_img, cv2.COLOR_BGR2GRAY)
          kernel = np.ones((3, 3), np.uint8) 
          decimal_sep_img = cv2.dilate(decimal_sep_img, kernel, iterations=2) 
          decimal_sep_img = cv2.bitwise_not(decimal_sep_img)
          
          #cv2.imwrite("test_sep.png", decimal_sep_img)
          # transform
          decimal_sep_img = transform_number(image = decimal_sep_img)['image']
          
          # insert in numbers layer
          x, y = decimal_sep_img.shape
          tmp = blank_image
          
          tmp[:,:, 0] = numbers
          start_x = (width - (i + 1) * 35) - 18
          
          tmp[68:(68 + y), start_x - x:start_x, 1] = decimal_sep_img
          numbers = np.min(tmp, axis=2)
      else:
        break  # Exit the loop after printing 5 samples
    
    # combine digits
    digits = [str(x) for x in digits]
    
    # add minus sign if desired
    if self.sign and random.random() < 0.5:
      digits.insert(0, "-")
            
      # grab and convert the sign
      minus_img = random.choice(minus)
      minus_img = cv2.imread(minus_img)
      minus_img = cv2.cvtColor(minus_img, cv2.COLOR_BGR2GRAY)
      kernel = np.ones((3, 3), np.uint8) 
      minus_img = cv2.dilate(minus_img, kernel, iterations=2)
      minus_img = cv2.bitwise_not(minus_img)
      minus_img = transform_sign(image = minus_img)['image']
      
      # insert in numbers layer
      x, y = minus_img.shape
      tmp = blank_image
      tmp[:,:, 0] = numbers
      
      start_x = (width - (self.values + 1) * 40)
      if start_x < x:
        start_x = x + 1
      
      tmp[80:(80 + y), start_x - x:start_x, 1] = minus_img
      numbers = np.min(tmp, axis=2)
    
    background[:,:,1] = numbers
    dst = np.min(background, axis=2)
    dst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
    
    # final transformation
    dst = transform_image(image = dst)['image']
    
    # join digits
    value = ''.join(digits)
    
    # check conversion of formats
    filename = os.path.join(path, "images", value + '.png')
    cv2.imwrite(filename, dst)
    
    # join digits
    value = ''.join(digits)
    
    # return the generated string / digits
    # both in concatenated and full form
    return digits, value, filename
  
  #---- Public functions
  
  def generate(self, samples = 1, path = "."):
    
    # create empty lists
    text = []
    filename = []
    
    # check if paths exists, if not error
    # if path exists create sub-folder images
    # to store the images
    if not os.path.exists(os.path.join(path, 'images')):
      os.makedirs(os.path.join(path, 'images'), exist_ok=True)
  
    # cover the range of random samples
    # write to lists
    for i in range(samples):
      d, v, f = self.__digits(
        path = path
      )
      
      # add data to list
      text = np.append(text, v)
      filename = np.append(filename, f)

    # convert to dataframe and save in image directory
    df = pd.DataFrame(
      np.column_stack(
        [filename, text]), 
        columns=['file_name', 'text']
      )
    
    df.to_csv(
      os.path.join(path, 'labels.csv'),
      index = False
    )
