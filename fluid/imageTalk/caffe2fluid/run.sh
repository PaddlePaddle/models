#!/bin/bash

proto_file="./model_caffe/VGG_ILSVRC_16_layers.prototxt"
caffemodel_file="./model_caffe/VGG_ILSVRC_16_layers.caffemodel"
weight_file="vgg16.npy"
net_file="tmp_net.py"
python convert.py \
        $proto_file \
        --caffemodel $caffemodel_file \
        --data-output-path $weight_file\
        --code-output-path $net_file

python vgg16_net.py vgg16.npy ./fluid

