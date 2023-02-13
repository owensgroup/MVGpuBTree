#!/bin/bash
image_name="mvgpubtree"
docker build -t $image_name -f docker/Dockerfile .