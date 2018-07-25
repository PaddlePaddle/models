#!/bin/bash

rm -rf *_factor.txt
model_file='model.py'
python $model_file --batch_size 256 --pass_num 2 --device CPU
