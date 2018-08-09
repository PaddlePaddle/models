###!/bin/bash
####This file is only used for continuous evaluation.

export CE_MODE_X=1
python train.py cnn  | python _ce.py
