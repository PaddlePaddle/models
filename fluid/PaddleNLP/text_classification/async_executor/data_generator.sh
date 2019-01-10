#!/bin/bash

pushd .
cd ./data_generator

# wget "http://ai.stanford.edu/%7Eamaas/data/sentiment/aclImdb_v1.tar.gz"
if [ ! -f aclImdb_v1.tar.gz ]; then
    wget "http://10.64.74.104:8080/paddle/dataset/imdb/aclImdb_v1.tar.gz"
fi
tar zxvf aclImdb_v1.tar.gz

mkdir train_data
python build_raw_data.py train | python splitfile.py 12 train_data

mkdir test_data
python build_raw_data.py test | python splitfile.py 12 test_data

/opt/python27/bin/python IMDB.py train_data
/opt/python27/bin/python IMDB.py test_data

mv ./output_dataset/train_data ../
mv ./output_dataset/test_data ../
cp aclImdb/imdb.vocab ../

rm -rf ./output_dataset
rm -rf train_data
rm -rf test_data
rm -rf aclImdb
popd
