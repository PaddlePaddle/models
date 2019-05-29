#!/bin/bash

#function:
#   prepare testing data for ppdetection
#

root=$(dirname `readlink -f ${BASH_SOURCE}[0]`)
cwd=`pwd`

if [[ $cwd != $root ]];then
    pushd $root 2>&1 1>/dev/null
fi

test_coco_data_url="http://filecenter.matrix.baidu.com/api/v1/file/wanglong03/coco.test.zip/20190529130653/download"
test_coco_data="coco.test.zip"
if [ ! -e 'coco.test' ];then
    wget $test_coco_data_url
    if [ -e $test_coco_data ];then
        echo "succeed to download ${test_coco_data}, so unzip it"
        unzip ${test_coco_data}
    else
        echo "failed to download ${test_coco_data}"
    fi
else
    echo "coco.test directory already exist!"
fi
