# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved
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


# Please download the Quora dataset firstly from https://drive.google.com/file/d/0B0PlTAo--BnaQWlsZl9FZ3l1c28/view?usp=sharing
# to the ROOT_DIR: $HOME/.cache/paddle/dataset

DATA_DIR=$HOME/.cache/paddle/dataset
wget --directory-prefix=$DATA_DIR http://nlp.stanford.edu/data/glove.840B.300d.zip

unzip $DATA_DIR/glove.840B.300d.zip

# The finally dataset dir should be like

# $HOME/.cache/paddle/dataset
# |- Quora_question_pair_partition
#     |- train.tsv
#     |- test.tsv
#     |- dev.tsv
#     |- readme.txt
#     |- wordvec.txt
# |- glove.840B.300d.txt
