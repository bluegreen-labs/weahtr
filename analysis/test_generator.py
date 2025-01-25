#!/usr/bin/env python

# import libraries
import os
from weahtr.generator import *

# docker root path
base = "/data/"

# initiate the setup
# listing the UNIPEN path
# generating numbers up to 3 digits long
# with a decimal separator and random sign
g = generator(
  data_path = os.path.join(base, "data/"),
  background = os.path.join(base, "data/grid_background.jpg"),
  values = 3,
  decimal = True,
  sign = True
  )

# generate 10 random samples and put output
# in the designated path
g.generate(
  samples = 10,
  path = os.path.join(base, "output")
)
