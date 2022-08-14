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

# WORKDIR  MVGpuBTree
# # Running CMake
# RUN mkdir build && cd build &&\
#     cmake ..

# # Running make
# RUN cd build && make -j

# Activate conda
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "mvgpubtree"]

# Run the unittests
# docker build -t mvgpubtree .
# or ...
# docker build --no-cache -t mvgpubtree .


# docker run -it  --gpus 4 mvgpubtree /bin/bash
# RUN cd MVGpuBTree &&\
#     ./build/bin/unittest_btree &&\
#     ./build/bin/unittest_versioning
# CMD conda init bash
# CMD conda activate mvgpubtree

# Run the reproduce script
# RUN source reproduce.sh

# Plotting
# CMD cd plots

# CMD source artifact.sh

