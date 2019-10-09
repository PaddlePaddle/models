#!/bin/bash

# download small demo dataset
wget --no-check-certificate https://paddlerec.bj.bcebos.com/deepfm%2Fdist_data_demo.tar.gz -O dist_data_demo.tar.gz
tar xzvf dist_data_demo.tar.gz
# preprocess demo dataset
python dist_preprocess.py
