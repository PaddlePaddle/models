#!/bin/bash

# This file is only used for continuous evaluation.

rm -rf *_factor.txt
model_file='model.py'
python $model_file --batch_size 128 --pass_num 5 --device CPU | python _ce.py
