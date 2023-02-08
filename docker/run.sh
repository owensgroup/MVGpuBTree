#!/bin/bash
git_directoy=$(pwd -P)

docker run -it --rm --name trees --gpus all -v $git_directoy:$git_directoy -w $git_directoy cuda11.8 /bin/bash