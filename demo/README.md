## Demo folder

This is the demo folder containing sub-folders with various [COBECORE data](https://cobecore.org/) to
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

