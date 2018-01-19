#!/usr/bin/env bash

CUR_PATH=`pwd`
git clone https://github.com/moses-smt/mosesdecoder.git
git clone https://github.com/rizar/actor-critic-public

export MOSES=`pwd`/mosesdecoder
export LVSR=`pwd`/actor-critic-public

cd actor-critic-public/exp/ted
sh create_dataset.sh

cd $CUR_PATH
mkdir data
cp actor-critic-public/exp/ted/prep/*-* data/
cp actor-critic-public/exp/ted/vocab.* data/

cd data
python ../preprocess.py

cd ..
rm -rf actor-critic-public mosesdecoder
