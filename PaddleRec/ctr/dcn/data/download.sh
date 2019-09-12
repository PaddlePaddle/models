#!/bin/bash

workdir=$(cd $(dirname $0); pwd)

cd $workdir

trainfile='train.txt'

echo "data dir:" ${workdir}

cd $workdir

echo "download data starting..."
wget --no-check-certificate -c https://s3-eu-west-1.amazonaws.com/kaggle-display-advertising-challenge-dataset/dac.tar.gz
echo "download finished"

echo "extracting ..."
tar xzvf dac.tar.gz
wc -l $trainfile | awk '{print $1}' > line_nums.log

echo "extract finished"
echo "total records: "`cat line_nums.log`
echo "done"
