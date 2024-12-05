# weaHTR workflow

A workflow for handwritten text recognition (HTR) of weather records. The presented workflow provides a set of end to end instructions to follow in order to automatically transcribe your weather records using machine learning.

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

The dockerfile included provides a (GPU) torch setup. You can build this docker image using the below command. This will download the NVIDIA CUDA drivers for GPU support, the tidyverse, rstudio IDE and quarto publishing environment. Note that this setup will require some time to build given the the large downloads involved. Once build locally no further downloads will be required.

```
docker build -f Dockerfile -t weahtr .
```

To spin up a GPU docker image use in the project directory:

```
docker run -it --gpus all -v $(pwd):/workspace weahtr bash
```

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
images =  glob.glob("/workspace/demo_input/format_1/*.jpg", recursive=True)

# initiate the setup
t = template.template(
  images = images,
  template = "/docker_data_dir/demo_input/format_1.jpg",
  config = "/docker_data_dir/demo_input/format_config.yml"
  )

# match all templates, write homography datat to file
t.match(method = "features")
```
Details will be stored in the homography directory of the output folder

6. Run the recognition model on the matched data

```python
# using the tesseract setup
t.label(method = "tesseract")
```
7. Screen the logs and visual output for [quality control]()
