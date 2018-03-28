#!/bin/bash

#function:
#   a tool used to:
#       1, convert a caffe model
#       2, do inference using this model
#
#usage:
#   bash run.sh resnet50 ./models.caffe/resnet50 ./models/resnet50
#

#set -x
if [[ $# -lt 3 ]];then
    echo "usage:"
    echo "  bash $0 [model_name] [cf_model_path] [pd_model_path] [only_convert]"
    echo "  eg: bash $0 resnet50 ./models.caffe/resnet50 ./models/resnet50"
    exit 1
else
    model_name=$1
    cf_model_path=$2
    pd_model_path=$3
    only_convert=$4
fi

proto_file=$cf_model_path/${model_name}.prototxt
caffemodel_file=$cf_model_path/${model_name}.caffemodel
weight_file=$pd_model_path/${model_name}.npy
net_file=$pd_model_path/${model_name}.py

if [[ ! -e $proto_file ]];then
    echo "not found prototxt[$proto_file]"
    exit 1
fi

if [[ ! -e $caffemodel_file ]];then
    echo "not found caffemodel[$caffemodel_file]"
    exit 1
fi

if [[ ! -e $pd_model_path ]];then
    mkdir $pd_model_path
fi

PYTHON=`which cfpython`
if [[ -z $PYTHON ]];then
    PYTHON=`which python`
fi
$PYTHON ../../convert.py \
        $proto_file \
        --caffemodel $caffemodel_file \
        --data-output-path $weight_file\
        --code-output-path $net_file

ret=$?
if [[ $ret -ne 0 ]];then
    echo "failed to convert caffe model[$cf_model_path]"
    exit $ret
else
    echo "succeed to convert caffe model[$cf_model_path] to fluid model[$pd_model_path]"
fi

if [[ -z $only_convert ]];then
    PYTHON=`which pdpython`
    if [[ -z $PYTHON ]];then
        PYTHON=`which python`
    fi
    imgfile="data/65.jpeg"
    net_name=`grep "name" $proto_file | head -n1 | perl -ne 'if(/\"([^\"]+)\"/){ print $1."\n";}'`
    $PYTHON ./infer.py $net_file $weight_file $imgfile $net_name
    ret=$?
fi
exit $ret
