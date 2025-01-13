import os, random
import cv2
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.core.transforms_interface import ImageOnlyTransform
from weahtr.utils import *

# define binarization to be included in the transform
class Binarize(ImageOnlyTransform):
   def __init__(self, window_size, C, p = 0.5):
      super().__init__(always_apply = True, p=1)
      self.window_size = window_size
      self.C = C
      self.p = p
   def apply(self, img, **params):
      if np.random.uniform(0, 1) > self.p:
       img_new = binarize(img, window_size = self.window_size, C = self.C)
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
        rain_type = None,
        p=0.5),
    ], p=1),
  ], p=1),
  A.OneOf([
    A.ShiftScaleRotate(
      shift_limit=0.1,
      scale_limit=0.2,
      rotate_limit=5,
      border_mode=cv2.BORDER_CONSTANT,
      value=(255,255,255),
      p=0.5),
    A.Affine(
      shear=random.randint(-5, 5),
      mode=cv2.BORDER_CONSTANT,
      cval=(255,255,255),
      p=0.5)
  ], p=1),
  A.Blur(blur_limit=5,p=0.25),
])
