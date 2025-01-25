# weaHTR workflow

A workflow for handwritten text recognition (HTR) of weather records. The
presented workflow provides a set of end to end instructions to follow in order
to automatically transcribe your weather records using machine learning.

> [!CAUTION]
> This python package is a consolidation of previously dispersed proof-of-concept
> script components written across years.
> Although consolidated, gaps in documentation and usability considerations still
> exist. For example, paths are not always validated and arguments might not have
> defaults.

## Introduction

Vast amounts of historical weather records remain archived and unexplored.
Despite advances in the recognition of (handwritten) text in general the case of
tabulated weather data remains challenging. The data is exact, provides limited
contextual clues and errors propagate due to easy misinterpretation of table
layouts due to messy formats, deterioration of paper and other factors. Retaining
context is therefore key to quality assurance of the data retained within these
weather records.

### Template matching

Although layout detection presents itself as feasible part of this workflow a
template matching approach is given priority. Layout matching with a good
accuracy (90%) on vast numbers of records could still corrupt large amounts of
data which would be need to be filtered (detected) post-hoc, due to its
unsupervised nature.

The template matching methods requires more up-front work, but flags poor template matches earlier in the processing chain making it easier to detect
errors and screen for quality. Faulty layout detection can be partial, where only
a part of the table is found, making absolute classifications of faulty or
correct data hard.

Furthermore, the law of large numbers makes small error rates affect a
significant amount of data. Partial matches and shifts in both columns or rows of
a table also propagate throughout the chronological order of the remaining data.

### Transcription models

Transcription models can be flexibly deployed, where the default model is trained upon thousands of handwritten table entries from the COBECORE project where either the Transformer based [TrOCR model](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/TrOCR/Fine_tune_TrOCR_on_IAM_Handwriting_Database_using_native_PyTorch.ipynb) or [Tesseract](https://github.com/tesseract-ocr/tesseract) is used.

## Installation & Use

Clone the package to your local directory.

```bash
git clone https://github.com/bluegreen-labs/weaHTR_workflow.git
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

Make sure to have interfacing libraries running, when relying on different 
docker base images.

```bash
apt-get install nvidia-container-runtime
```

To spin up a GPU docker image and drop into the command prompt use in the 
project directory:

```bash
docker run -it --rm --gpus all -v $(pwd):/data weahtr bash
```

For independent installs using conda

```bash
conda env create -f environment.yml
```

### Loading the package locally

For now no `pip` based install is supported. You can install the package 
by working in editor mode (if under development), or through a linked `pip`
install. This works both in a Docker or in normal install, given that requirement
are met. Editting mode also allows you to dynamically change the code.

For editor mode use:
```bash
pip install -e /path/to/pkg
```

For a static local install use:
```bash
pip install mypackage --no-index --find-links /path/to/pkg
```

> [!NOTE]
> When using a Docker image you will have to install the package in every new
> session. A final Docker image with the library installed will be available when
> the package matures.

## Workflow

The workflow consists of six steps:

1. Sort your data into particular tabular formats, these should be unique.
2. [Create empty, reference, template images]()
3. [Mark rows and columns on the template images]()
4. [Edit the configuration YAML file]()
5. Template match the data to find the tables (three methods are provided)

```python
# import libraries
import glob
from weahtr.template import *

# list files to process
images =  glob.glob("./demo_input/format_1/images/*.jpg", recursive=True)

# initiate the setup
t = template.template(
  images = images,
  template = "./demo_input/format_1.jpg",
  config = "./demo_input/format_1/format_1.yml"
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
t.label(method = "tesseract")
```

The ability to store the homography files allows you to pre-calculate all table
template matching / image registration up front, so this step does not need to 
be repeated when running the table through different OCR/HTR transcription
models. This should allow you to iterate quickly over various ML models once the
image registration is completed and saved to file. Finally, after every run
it is adviced to screen the logs and visual output for quality control

## Training models

```python
# import libraries
import glob, os
from weahtr import *
import pandas as pd

# file with two columns:
# 'file_name' and 'text'
# the file name only lists the
# image file name (not the absolute long form path)
df = pd.read_csv("../data/character_training_data/labels.csv")

# setup model training
m = weahtr.model(
  model = "trocr", # model to use
  config = "./demo_input/format_1/format_1.yml", # config file
  labels = df, # data frame with the labels
  images = "image/path" # path with the images listed in the data frame
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
  data_path = "input/data",
  background = "background_image.png",
  values = 3,
  decimal = True,
  sign = True
  )

# generate 10 random samples and put output
# in the designated path
g.generate(
  samples = 10,
  path = "~/output/path/"
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

As with the `fft` method, a guides file with the location of cells in the table is only
required for the transcription processing of the cell content, not the 
table matching.

## References

J. Sueiras, et al.: "Using Synthetic Character Database for Training Deep Learning
Models Applied to Offline Handwritten Character Recognition", Proc. Intl. Conf.Intelligent
Systems Design and Applications (ISDA), Springer, 2016.
