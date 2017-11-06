#!/bin/bash
if [[ -d data ]] && [[ -d embedding ]] && [[ -d evaluation ]]; then
  echo "data exist"
  exit 0
else
  wget -c http://paddlepaddle.bj.bcebos.com/dataset/webqa/WebQA.v1.0.zip
fi

if [[ `md5sum -c md5sum.txt` =~ 'OK' ]] ; then
    unzip WebQA.v1.0.zip
    mv WebQA.v1.0/* .
    rmdir WebQA.v1.0
    rm WebQA.v1.0.zip
else
  echo "download data error!" >> /dev/stderr
  exit 1
fi

