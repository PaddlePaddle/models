#!/bin/bash

#function:
#   a tool used to compare all layers' results
#

if [[ $# -ne 1 ]];then
    echo "usage:"
    echo "  bash $0 [model_name]"
    echo "  eg:bash $0 alexnet"
    exit 1
fi

model_name=$1
prototxt="models.caffe/$model_name/${model_name}.prototxt"
layers=$(cat $prototxt | perl -ne 'if(/^\s+name\s*:\s*\"([^\"]+)/){print $1."\n";}')

for i in $layers;do
    cf_npy="results/${model_name}.caffe/${i}.npy"
    pd_npy="results/${model_name}.paddle/${i}.npy"

    if [[ ! -e $cf_npy ]];then
        echo "caffe's result not exist[$cf_npy]"
        continue
    fi

    if [[ ! -e $pd_npy ]];then
        echo "paddle's result not exist[$pd_npy]"
        continue
    fi

    python compare.py $cf_npy $pd_npy no_exception
    if [[ $? -eq 0 ]];then
        echo "succeed to compare layer[$i]"
    else
        echo "failed to compare layer[$i]"
    fi

done
