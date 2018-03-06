#!/bin/bash

#function:
#   convert a caffe model
#   eg:
#       bash ./convert.sh ./model.caffe/lenet.prototxt ./model.caffe/lenet.caffemodel lenet.py lenet.npy

if [[ $# -ne 4 ]];then
    echo "usage:"
    echo "  bash $0 [PROTOTXT] [CAFFEMODEL] [PY_NAME] [WEIGHT_NAME]"
    echo "  eg: bash $0 lenet.prototxt lenet.caffemodel lenet.py lenet.npy"
    exit 1
fi

WORK_ROOT=$(dirname `readlink -f ${BASH_SOURCE[0]}`)
if [[ -z $PYTHON ]];then
    PYTHON=`which python`
fi

PROTOTXT=$1
CAFFEMODEL=$2
PY_NAME=$3
WEIGHT_NAME=$4
CONVERTER_PY="$WORK_ROOT/../../convert.py"

$PYTHON $CONVERTER_PY $PROTOTXT --caffemodel $CAFFEMODEL --code-output-path=$PY_NAME --data-output-path=$WEIGHT_NAME
ret=$?
if [[ $ret -eq 0 ]];then
    echo "succeed to convert caffe model[$CAFFEMODEL, $PROTOTXT] to paddle model[$PY_NAME, $WEIGHT_NAME]"
else
    echo "failed to convert caffe model[$CAFFEMODEL, $PROTOTXT]"
fi
exit $ret
