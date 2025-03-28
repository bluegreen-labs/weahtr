# weaHTR workflow

A workflow for handwritten text recognition (HTR) of weather records. The
presented workflow provides a set of end to end instructions to follow in order
to automatically transcribe your weather records using machine learning.

> [!CAUTION]
> This python package is a consolidation of previously dispersed proof-of-concept
> script components written across years. Although consolidated, gaps in 
> documentation and usability considerations still exist. For example, paths 
> are not always validated and arguments might not have defaults. Read the below
> documentation carefully before proceeding. The software is provided AS IS and 
> no responsibility is taken in the event of data loss.

## Introduction

Vast amounts of historical weather records remain archived and unexplored.
Despite advances in the recognition of (handwritten) text in general the case of
tabulated weather data remains challenging. The data is exact, provides limited
contextual clues and errors propagate due to easy misinterpretation of table
layouts due to messy formats, deterioration of paper and other factors. Retaining
context is therefore key to quality assurance of the data retained within these
weather records.

### Template matching

Although machine learning (ML) layout detection presents itself as feasible part
of a transcription workflow a template matching approach is given priority.
Layout matching with a good accuracy (90%) on vast numbers of records could still
corrupt large amounts of data which would be need to be filtered (detected)
post-hoc, due to its unsupervised nature.

The template matching methods requires more up-front work, but flags poor 
template matches earlier in the processing chain making it easier to detect
errors and screen for quality. Faulty layout detection can be partial, where only
a part of the table is found, making absolute classifications of faulty or
correct data hard.

Furthermore, the law of large numbers makes small error rates affect a
significant amount of data. Partial matches and shifts in both columns or rows of
a table also propagate throughout the chronological order of the remaining data.
Unlike text data, where contextual clues can allow for meaningful re-orderning
of sentences the nature of numeric data is such that the encoding of the position
within a table must be absolute. Therefore, a preference is given to a
semi-supervised template matching and table detection workflow.

### Transcription models

Transcription models can be flexibly deployed, where the default model is trained
upon thousands of handwritten table entries from the COBECORE project where either
the Transformer based [TrOCR model](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/TrOCR/Fine_tune_TrOCR_on_IAM_Handwriting_Database_using_native_PyTorch.ipynb) or
[Tesseract](https://github.com/tesseract-ocr/tesseract) is used.

## Installation & Use

Clone the package to your local directory.

```bash
git clone https://github.com/bluegreen-labs/weahtr.git
```

### Setup compute environments

The Dockerfile included provides a (GPU) torch setup. You can build this docker
image using the below command. This will download the NVIDIA CUDA drivers for GPU
support, the tidyverse, rstudio IDE and quarto publishing environment. Note that
this setup will require some time to build given the the large downloads
involved. Once build locally no further downloads will be required.

```bash
docker build -f Dockerfile -t weahtr .
```

The default install above provides support for [PyLaia](https://github.com/jpuigcerver/PyLaia)
and [Tesseract](https://tesseract-ocr.github.io/). If you want support for the
[Kraken](https://kraken.re/main/index.html) environment use the following code:

```bash
docker build -f Dockerfile_kraken -t weahtr .
```

> [!NOTE]
> Both the PyLaia and Kraken environments support various open source models
> You can list all available Kraken models by using the command line:
> ```bash
> kraken list
> ```
> PyLaia models can be found on [HugginFace](https://huggingface.co/Teklia).

Make sure to have interfacing libraries running, when relying on different 
docker base images.

```bash
apt-get install nvidia-container-runtime
```

To spin up a GPU docker image and drop into the command prompt use in the 
project directory:

```bash
docker run -it --rm --gpus all -v $(pwd):/data --tmpfs /tmp_images weahtr bash
```

> [!NOTE]
> The command spins up a ramdisk `/tmp_images` for use with `Pylaia`. This is
> is advised as the pylaia routine would generate large amounts of files to
> be written to disk. This genenerates wear on disks and limits I/O. You can
> alter these settings (the img_dir parameter) in the config file, but it
> is not advised.

For independent installs using conda

```bash
conda env create -f environment.yml
```

> [!NOTE]
> Repeatedly building a docker image can result in a large cache being created
> easily 10x the data of the image (which in itself is multiple GB in size).
>
> If you find you are running out of storage space out of the blue, check the
> docker build cache, and prune it.
> 
> ```bash
> docker buildx prune -f
> ```

### Loading the package locally

For now, no online `pip` based install is supported. You can install the package 
by working in editor mode (if under development), or through a linked `pip`
install. This works both in a Docker or in normal install, given that requirement
are met. Editing mode also allows you to dynamically change the code. Note that
the package must be available on the mounted Docker volume.

For editor mode use:
```bash
pip install -e /path/to/pkg
```

For a static local install use:
```bash
pip install weahtr --no-index --find-links /path/to/pkg
```

> [!NOTE]
> When using a Docker image you will have to install the package in every new
> session. A final Docker image with the library installed will be available when
> the package matures.

## Workflow

The workflow consists of six steps:

1. Sort your data into particular tabular formats, these should be unique.
2. Create empty, reference, template images

You can create black and white version using the algorithm included
in the library using the code below.

```python
# import libraries
import glob
from weahtr.utils import *
import cv2

# returns an image cropping any dark borders
# if they exist
image = cv2.imread("your_empty_table.png")
image, _, _ = replace_matte(image)

# binarize (black/white) the data
image = binarize(image)
cv2.imwrite("template.png", image)
```

Further cleanup to create and empty template, using the eraser tool, of values 
in individual cells if no empty table is available can be done in the image
manipulation tool [GIMP](https://www.gimp.org/))

3. [Mark rows and columns on the template images in GIMP](https://github.com/bluegreen-labs/weahtr_guides)
4. [Edit the configuration YAML file](https://github.com/bluegreen-labs/weahtr/blob/main/demo/format_1/format_1.yml)
5. Template match the data to find the tables (three methods are provided)

```python
# import libraries
import glob
from weahtr import *
from weahtr.utils import *

# list files to process
images =  glob.glob("./demo/format_1/images/*.jpg", recursive=True)

# initiate the setup
t = template.template(
  images = images,
  template = "./demo/format_1.jpg",
  config = "./demo/format_1/format_1.yml"
  )

# match all templates, write homography data to file
t.match(method = "features")
```

The data is processed and stored in an output directory keeping track of each
image (table). Preview options are provided to generate an image to screen the
template matching before proceeding to the final transcription step (6). 

All operations are **non-destructive** on the original data, and relatively
little additional image files are created. For example, any image registration is
stored a [homography]() file (a translation matrix), which is a small text file.
This homography file is used, when available (i.e. pre-calculated), to find the
table with data values.

6. Run the OCR/HTR transcription model on the matched data

```python
# using the tesseract setup
t.process(
  model = "tesseract"
)

# custom functions can be passed
# to enhance text at the end
# of the processing chain
# the function must take a single
# image as input and return
# an RGB (3 band) image
labels = t.process(
  model = "tesseract",
  f = remove_lines # included function to blend column and row lines
)
```

The ability to store the homography file allows you to pre-calculate all table
template matching / image registration up front, so this step does not need to 
be repeated when running the table through different OCR/HTR transcription
models. This should allow you to iterate quickly over various ML models once the
image registration is completed and saved to file. Finally, after every run
it is advised to screen the logs and visual output for quality control

## Training models

```python
# import libraries
import glob, os
from weahtr import *
from weahtr.utils import *
import pandas as pd

# file with two columns:
# 'file_name' and 'text'
# the file name only lists the
# image file name (not the absolute long form path)
df = pd.read_csv("labels.csv")

# setup model training
m = weahtr.model(
  model = "trocr", # model to use
  config = "config.yml", # config file
  labels = df, # data frame with the labels
  images = "/image/path" # path with the images listed in the data frame
)

# initiate a training run
m.train()
```

### Number generator

In order to increase representation of handwritten text recognition one can use
synthetic data as compiled from handwritten text databases. The package includes
a functions `generate()` in the generator class which generates random numbers
and their matching table of labels in a CSV file. The generator uses MNIST and
[UNIPEN](https://github.com/sueiras/handwritting_characters_database) data 
formatted by Sueiras et al. 2016 to compile these random images. 
The UNIPEN data only include decimals (.,) and signs
(-) to be used as additional modifiers, where MNIST data is dynamically loaded
from the Torchvision library to include all handwritten MNIST numbers.

> [!WARNING]
> Paths are not validated for now. Make sure paths exist.

```python
# import libraries
from weahtr.generator import generator

# initiate the setup
# listing the UNIPEN path
# generating numbers up to 3 digits long
# with a decimal separator and random sign
g = generator(
  data_path = "/data/data/", # When using the Docker image
  background = "background_image.png",
  values = 3,
  decimal = True,
  sign = True
  )

# generate 10 random samples and put output
# in the designated path (your home directory)
g.generate(
  samples = 10,
  path = "~"
)
```

## Table layout mapping methods (details)

The package includes three methods to align the source image with
a known template or guides, there is a `table`, `fft` and the `features`
method. 

### Table method

The `table` recognition method is based on polygon dilation and erosion, to
calculate the four corners of the largest table in an image. This assumes
that the tables are well defined (bounded by a well defined grid). In addition,
one needs a JSON guides file covering the full extent of the table. This
therefore also requires you to set the rows to skip in the processing of the
table content, should you use the template for the further division of the
table in rows an columns. The latter, is not required as there is an automatic
row and column detection routine which can be used.

Generally, this method is fast but can be unreliable on very noisy data. It
also requires the tables to be relatively well constrained. For example, the
routine **will fail or return faulty registrations on tables with only row dividers**.

### Fast Fourier Transform (FFT) method

The `fft` method uses translation and rotation invariant mapping within the 
frequency domain to align two images, the source image and an empty template
of a table you want to reference.

This method is relatively slow, but rather robust to noisy input. In short,
reference templates which are approximately the same as the source (content)
you want to reference will work to some extent.

In this case a guides file with the location of cells in the table is only
required for the transcription processing of the cell content, not the 
table matching.

### Feature matching method

The `features` method uses the correspondence between key image features (i.e
recognizable patterns in the image) to align both images correctly. This method
is computationally expensive but if it works well also the most accurate way
of referencing data.

As with the `fft` method, a guides file with the location of cells in the table
is only required for the transcription processing of the cell content, not the 
table matching.

## Demo folder

The demo folder containins sub-folders with various [COBECORE data](https://cobecore.org/) to
test the toolbox on. Every sub-folder describes a particular table format from
the larger COBECORE dataset. In addition, for testing and scalability purposes
I've included three test images of the [ReData Zooniverse](https://www.zooniverse.org/projects/meteonetwork/redata) project which aims
at recovering Italian climate data.

### Sub-folder content

Every sub-folder contains a pre-configured workflow, which includes:

- the {format}_workflow.py python script calling the library
- the {format}.yml configuration file for a particular experiment
- the {format}.jpg empty template file for `fft` and `features` based template matching
- the {format}.json file containing the location of rows and columns 
  - annotated on the {format}.jpg template image using [the GIMP guides plugin](https://github.com/bluegreen-labs/weahtr_guides)

The python script is configured to be run in the included Docker which
can be compiled using the included Docker file. By default only slices are
returned, for running the model set `slices=False` and select a transcription
model in the configuration file. Note that `CUDA` acceleration is set by default
and might not be available on all systems. Adjust the setting accordingly to
`default` or `cpu`.

The included model (i.e. tesseract based) might not work for you particular 
data. I refer to [HTR/OCR workshop notes](https://bluegreen-labs.github.io/text_recognition_and_analysis/) on how and if to retrain models.
The library includes support for training your own TrOCR model, once more I
refer to the workshop notes. However, trained output using the library can
be called by referencing the final TrOCR model output in the `YML` config file.

## References

J. Sueiras, et al.: "Using Synthetic Character Database for Training Deep Learning
Models Applied to Offline Handwritten Character Recognition", 
Proc. Intl. Conf.Intelligent Systems Design and Applications (ISDA), Springer, 2016.
