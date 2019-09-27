#!/bin/bash

wget --no-check-certificate https://s3-eu-west-1.amazonaws.com/kaggle-display-advertising-challenge-dataset/dac.tar.gz
tar zxf dac.tar.gz >/dev/null 2>&1
rm -f dac.tar.gz

mkdir raw
mv ./*.txt raw/
