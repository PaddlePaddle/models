###!/bin/bash
####This file is only used for continuous evaluation.

model_file='train.py'
python $model_file --pass_num 1 --learning_rate 0.001 --save_interval 10 --enable_ce | python _ce.py
