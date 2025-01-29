import os, random
import cv2
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.core.transforms_interface import ImageOnlyTransform
from weahtr.utils import *

# define binarization to be included in the training transform
class Binarize(ImageOnlyTransform):
   def __init__(self, window_size, C, p = 0.5):
      super().__init__(p = 1)
      self.window_size = window_size
      self.C = C
      self.p = p
   def apply(self, img, **params):
      if np.random.uniform(0, 1) > self.p:
       img_new = binarize(
         img,
         window_size = self.window_size,
         C = self.C
        )
       img_new = cv2.cvtColor(img_new, cv2.COLOR_GRAY2BGR)
      else:
       img_new = img
      return img_new

# transformation compose including custom
# binarization function - for consistency
train_transform = A.Compose([
  A.OneOf([
    A.CLAHE(
        clip_limit=(1, 4),
        tile_grid_size=(8, 8),
        p=1),
    Binarize(window_size = 91, C = 6, p = 0.5),
    A.OneOf([
      A.RandomRain(
        brightness_coefficient=1.0,
        drop_length=2,
        drop_width=2,
        drop_color = (0, 0, 0),
        blur_value=1,
        rain_type = 'drizzle',
        p=0.05),
      A.RandomShadow(p=0.5),
      A.PixelDropout(p=0.5),
    ], p=1),
    A.OneOf([
      A.PixelDropout(dropout_prob=0.5,drop_value=255,p=1),
      A.RandomRain(
        brightness_coefficient=1.0,
        drop_length=2,
        drop_width=2,
        drop_color = (255, 255, 255),
        blur_value=1,
        rain_type = 'default',
        p=0.5),
    ], p=1),
  ], p=1),
  A.OneOf([
    A.Affine(
      translate_percent = 0.025,
      rotate = [-5, 5],
      border_mode=cv2.BORDER_CONSTANT,
      fill = 255,
      p = 0.5
    ),
    A.Affine(
      shear=random.randint(-5, 5),
      border_mode = cv2.BORDER_CONSTANT,
      fill = 255,
      p = 0.5
    )
  ], p=1),
  A.Blur(
    blur_limit = 5,
    p=0.25
  ),
])

#--- transforms for the number generator

transform_number = A.Compose([
  # always resize to a fixed 40px
  # first
  A.Resize(
    height=65,
    width=65,
    interpolation=1,
    p=1.0
  ),
  A.Rotate(
    limit=(-4, 4),
    interpolation=1,
    border_mode=3,
    p=0.5,
  ),
  # 20% tolerance on scaling
  # to introduce size variability
  A.RandomScale(
    scale_limit=0.2, 
    interpolation=1,
    p=0.8,
  ),
  A.GaussianBlur(
    blur_limit=(5, 13),
    p=0.5
  )
])

transform_sign = A.Compose([
  # always resize to a fixed 40px
  # first
  A.Resize(
    height=30,
    width=30,
    interpolation=1,
    p=1.0
  ),
  A.Rotate(
    limit=(-4, 4),
    interpolation=1,
    border_mode=3,
    p=0.5,
  ),
  # 20% tolerance on scaling
  # to introduce size variability
  A.RandomScale(
    scale_limit=0.2, 
    interpolation=1,
    p=0.8,
  ),
  A.GaussianBlur(
    blur_limit=(1, 11),
    p=0.5
  )
])

# random subsample of a background
# grid, including mild rotation
transform_grid = A.Compose([
  A.Rotate(
    limit=(-10, 10),
    interpolation=1,
    border_mode=3,
    p=0.2,
  ),
  A.RandomCrop(
        height=200,
        width=200,
        p=1.0
   )
])

transform_image = A.Compose([
  A.GaussNoise(
    std_range=[0.2, 0.44],
    mean_range = [0,0],
    p = 0.5
  ),
  A.RandomBrightnessContrast(
    brightness_limit = [-0.5, 0.5],
    contrast_limit = [-0.5, 0.5],
    p=0.5
  )
])
