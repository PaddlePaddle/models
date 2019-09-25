#!/bin/bash

python train.py --dict_path data/test_build_dict --train_data_dir data/convert_text8/ --with_speed --num_passes 10 --use_iterable_py_reader $@
