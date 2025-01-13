# For NVIDIA acceleration make sure to
# enable the NVIDIA container toolkit
# ubuntu/jammy is the default image,
# nvidia/cuda is the old nvidia image
# pytorch the newer pytorch image which
# might conflict with a tensorflow Install
# if acceleration is desired

#FROM ubuntu/jammy
#FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04
#FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel

# copy package content
COPY environment.yml .

# Install base utilities
RUN apt-get update
RUN apt-get install -y build-essential wget software-properties-common

# install libraries
RUN apt-get install -y libgl1 libavcodec-dev libavformat-dev libswscale-dev \
 libgstreamer-plugins-base1.0-dev libgstreamer1.0-dev \
 libgtk2.0-dev libgtk-3-dev libpng-dev libjpeg-dev \
 libopenexr-dev libtiff-dev libwebp-dev

# install tessaract from developer PPA
# with both legacy v3 and v4 support
RUN add-apt-repository ppa:alex-p/tesseract-ocr-devel
RUN apt-get update
RUN apt-get install -y tesseract-ocr

# install miniconda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
RUN /bin/bash ~/miniconda.sh -b -p /opt/conda

# recreate and activate the environment
# suppress TF log level output
RUN /opt/conda/bin/conda env create -f environment.yml
RUN echo "source activate weahtr" > ~/.bashrc
ENV PATH $CONDA_DIR/bin:$PATH

# Set the working directory on start
# assumes that people follow the directions!
WORKDIR /data
