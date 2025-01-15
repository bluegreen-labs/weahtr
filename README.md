# weaHTR workflow

A workflow for handwritten text recognition (HTR) of weather records. The presented workflow provides a set of end to end instructions to follow in order to automatically transcribe your weather records using machine learning.

> [!CAUTION]
> This python package is a consolidation of previously dispersed script components written across years.
> Although consolidated, gaps in documentation and usability considerations exist. For example, paths
> are note always validated and arguments might not have defaults.

## Introduction

Vast amounts of historical weather records remain archived and unexplored. Despite advances in the recognition of (handwritten) text in general the case of tabulated weather data remains challenging. The data is exact, provides limited contextual clues and errors propagate due to easy misinterpretation of table layouts due to messy formats, deterioration of paper and other factors. Retaining context is therefore key to quality assurance of the data retained within these weather records.

### Template matching

Although layout detection presents itself as feasible part of this workflow a template matching approach is given priority. Layout matching with a good accuracy (90%) on vast numbers of records could still corrupt large amounts of data which would be need to be filtered (detected) post-hoc, due to its unsupervised nature.

The template matching methods requires more up-front work, but flags poor template matches earlier in the processing chain making it easier to detect errors and screen for quality. Faulty layout detection can be partial, where only a part of the table is found, making absolute classifications of faulty or correct data hard. 

Furthermore, the law of large numbers makes small error rates affect a significant amount of data. Partial matches and shifts in both columns or rows of a table also propagate throughout the chronological order of the remaining data.

### Transcription models

Transcription models can be flexibly deployed, where the default model is trained upon thousands of handwritten table entries from the COBECORE project where [easyOCR](https://pypi.org/project/easyocr/) is used for conventional type-written text.

## Installation & Use

Clone the package to your local directory.

```bash
git clone https://github.com/bluegreen-labs/weaHTR_workflow.git
```

### Setup compute environments

The Dockerfile included provides a (GPU) torch setup. You can build this docker image using the below command. This will download the NVIDIA CUDA drivers for GPU support, the tidyverse, rstudio IDE and quarto publishing environment. Note that this setup will require some time to build given the the large downloads involved. Once build locally no further downloads will be required.

```bash
docker build -f Dockerfile -t weahtr .
```

Make sure to have interfacing libraries running.

To spin up a GPU docker image use in the project directory:

```bash
docker run -it --rm --gpus all -v $(pwd):/data weahtr bash
```

For independent installs using conda

```bash
conda env create -f environment.yml
```

### Loading the package locally

For now no `pip` based install is supported you can install the package by working
in editor mode (if under development), or through a linked `pip` install.

For editor mode use:

```bash
pip install -e /path/to/pkg
```

For a static local install use:
```bash
pip install mypackage --no-index --find-links /path/to/pkg
```

> When using a Docker image you will have to install the package in every new session.

## Workflow

The workflow consists of four steps:

1. Sort your data into particular tabular formats, these should be unique.
2. [Create empty, reference, template images]()
3. [Mark rows and columns on the template images]()
4. Edit the configuration YAML file
5. Template match the data

```python
# import libraries
import glob
import weahtr

# list files to process
images =  glob.glob("/data/demo_input/format_1/*.jpg", recursive=True)

# initiate the setup
t = template.template(
  images = images,
  template = "/data/demo_input/format_1.jpg",
  config = "/data/demo_input/format_config.yml"
  )

# match all templates, write homography data to file
t.match(method = "features")
```
Details will be stored in the homography directory of the output folder

6. Run the recognition model on the matched data

```python
# using the tesseract setup
t.label(method = "tesseract")
```
7. Screen the logs and visual output for [quality control]()

## Number generator

In order to increase representation of handwritten text recognition one can use
synthetic data as compiled from handwritten text databases. The package includes
a functions `generate()` in the generator class which generates random numbers
and their matching table of labels in a CSV file. The generator uses MNIST and
[UNIPEN](https://github.com/sueiras/handwritting_characters_database) data 
formatted by Sueiras et al. 2016 to compile these random images. 
The UNIPEN data only include decimals (.,) and signs
(-) to be used as additional modifiers, where MNIST data is dynamically loaded
from the torchvision library to include all handwritten MNIST numbers.

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
g.generate(samples = 10, path = "~/output/path/")
```

## References

J. Sueiras, et al.: "Using Synthetic Character Database for Training Deep Learning
Models Applied to Offline Handwritten Character Recognition", Proc. Intl. Conf.Intelligent
Systems Design and Applications (ISDA), Springer, 2016.
