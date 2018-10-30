#!/bin/bash

# This file is only used for continuous evaluation.

export CUDA_VISIBLE_DEVICES=0,1,2,3

python train.py --pretrained_model=checkpoint/resnet50/115/ --lr=0.001 --train_batch_size=40 --test_batch_size=10 --loss_name=quadrupletloss --model_save_dir="output_quadruplet" --model=ResNet50 --num_epochs=120 --enable_ce=True | python _ce.py
