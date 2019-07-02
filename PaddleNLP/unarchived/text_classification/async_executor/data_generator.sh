#!/usr/bin/env bash

# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

pushd .
cd ./data_generator

# wget "http://ai.stanford.edu/%7Eamaas/data/sentiment/aclImdb_v1.tar.gz"
if [ ! -f aclImdb_v1.tar.gz ]; then
    wget "http://ai.stanford.edu/%7Eamaas/data/sentiment/aclImdb_v1.tar.gz"
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
