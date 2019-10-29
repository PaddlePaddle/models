#!/bin/bash

wget --no-check-certificate https://s3-eu-west-1.amazonaws.com/kaggle-display-advertising-challenge-dataset/dac.tar.gz
wget --no-check-certificate https://paddlerec.bj.bcebos.com/deepfm%2Ffeat_dict_10.pkl2 -O ./aid_data/feat_dict_10.pkl2 || rm -f ./aid_data/feat_dict_10.pkl2
tar zxf dac.tar.gz >/dev/null 2>&1
rm -f dac.tar.gz

python preprocess.py
rm *.txt
rm -r raw_data
