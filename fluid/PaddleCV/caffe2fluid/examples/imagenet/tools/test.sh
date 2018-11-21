#!/bin/bash

#
#script to test all models
#

models="alexnet vgg16 googlenet resnet152 resnet101 resnet50"
for i in $models;do
    echo "begin to process $i"
    bash ./tools/diff.sh $i 2>&1
    echo "finished to process $i with ret[$?]"
done
