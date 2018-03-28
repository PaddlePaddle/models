#!/bin/bash

wget --no-check-certificate https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json -O train.json
wget --no-check-certificate https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json -O dev.json

wget http://nlp.stanford.edu/data/glove.840B.300d.zip
unzip glove.840B.300d.zip
