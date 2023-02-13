# Use CUDA 11.5 and Ubuntu 20.04 docker image as base image
FROM nvidia/cuda:11.5.0-devel-ubuntu20.04

# Install required Ubuntu packages
RUN apt-get clean && apt-get update -y -qq
RUN apt-get install -y wget git build-essential
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y graphviz python-is-python3

# Install cmake
ARG CMAKE_VERSION=3.25.1
RUN wget https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-x86_64.sh &&\
    chmod +x cmake-${CMAKE_VERSION}-linux-x86_64.sh &&\
    mkdir /opt/cmake && \
    ./cmake-${CMAKE_VERSION}-linux-x86_64.sh  --skip-license --prefix=/opt/cmake  &&\
    ln -s /opt/cmake/bin/* /usr/local/bin/

# Install conda
ENV PATH="/root/anaconda3/bin:${PATH}"
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh &&\
    chmod +x Miniconda3-latest-Linux-x86_64.sh &&\
    ./Miniconda3-latest-Linux-x86_64.sh -b -p /root/anaconda3

# Create conda environment with dependencies
COPY docker/environment.yml .
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
# docker run -it --name trees --gpus device=0 mvgpubtree /bin/bash
# or ...
# docker run -it --name trees --gpus all mvgpubtree /bin/bash

# Follow the reproduce.md script to build, benchmark, generate figures and tables

# copy results back to host
# docker cp trees:MVGpuBTree/plots/figs .





