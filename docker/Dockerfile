# Use CUDA 11.5 and Ubuntu 20.04 docker image as base image
FROM nvidia/cuda:11.5.0-devel-ubuntu20.04

# Install required Ubuntu packages
RUN apt-get clean && apt-get update -y -qq
RUN apt-get install -y wget git build-essential

# Install conda
ENV PATH="/root/anaconda3/bin:${PATH}"
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh &&\
    chmod +x Miniconda3-latest-Linux-x86_64.sh &&\
    ./Miniconda3-latest-Linux-x86_64.sh -b -p /root/anaconda3

# Create conda environment with dependencies
COPY environment.yml .
RUN conda env create -f environment.yml
SHELL ["conda", "run", "-n", "mvgpubtree", "/bin/bash", "-c"]

# Clone MVGpuBTree code from github
RUN git clone https://github.com/owensgroup/MVGpuBTree.git

# Activate conda
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "mvgpubtree"]

# Building the image:
# docker build -t mvgpubtree .
# or ...
# docker build --no-cache -t mvgpubtree .

# export the image
# docker save mvgpubtree | gzip > mvgpubtree_pact22.tar.gz

# import the image
# docker load -i mvgpubtree_pact22.tar.gz

# running the image
# docker run -it --name trees --gpus device 0 mvgpubtree /bin/bash
# or ...
# docker run -it --name trees --gpus all mvgpubtree /bin/bash

# Follow the reproduce.md script to build, benchmark, generate figures and tables

# copy results back to host
# docker cp trees:MVGpuBTree/plots/figs .





