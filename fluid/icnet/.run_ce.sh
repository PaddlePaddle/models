#!/bin/bash

# This file is only used for continuous evaluation.
rm -rf ./ck
mkdir ck
python train.py --use_gpu=True --checkpoint_path="./ck"; python eval.py  --model_path="./ck/100" | python _ce.py
