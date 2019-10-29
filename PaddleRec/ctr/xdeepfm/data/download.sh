#!/bin/bash

if [ ! -d "train_data" ]; then
    mkdir train_data
fi

if [ ! -d "test_data" ]; then
    mkdir test_data
fi

wget --no-check-certificate https://paddlerec.bj.bcebos.com/xdeepfm%2Fev -O ./test_data/ev
wget --no-check-certificate https://paddlerec.bj.bcebos.com/xdeepfm%2Ftr -O ./train_data/tr
