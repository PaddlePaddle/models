#!/bin/bash

#function:
#   script used to generate caffepb.py from caffe.proto using protoc
#

PROTOC=`which protoc`
if [[ -z $PROTOC ]];then
    echo "not found protoc, you should first install it following this[https://github.com/google/protobuf/releases]"
    exit 1
fi

WORK_ROOT=$(dirname `readlink -f "$BASH_SOURCE[0]"`)
PY_NAME="$WORK_ROOT/caffepb.py"
$PROTOC --proto_path=$WORK_ROOT --python_out=$WORK_ROOT $WORK_ROOT/caffe.proto
ret=$?

if [ $ret -eq 0 ];then
    mv $WORK_ROOT/caffe_pb2.py $PY_NAME
fi

if [ -e "$PY_NAME" ];then
    echo "succeed to generate [$PY_NAME]"
    exit 0
else
    echo "failed to generate [$PY_NAME]"
fi
exit $ret
