#!/bin/bash
git_directoy=$(pwd -P)
image_name="mvgpubtree"
docker run -it --rm --name trees --gpus all -v $git_directoy:$git_directoy -w $git_directoy $image_name /bin/bash