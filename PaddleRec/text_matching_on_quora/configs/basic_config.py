#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
from __future__ import print_function


class config(object):
    def __init__(self):
        self.batch_size = 128
        self.epoch_num = 50

        self.optimizer_type = 'adam'  # sgd, adagrad

        # pretrained word embedding 
        self.use_pretrained_word_embedding = True
        # when employing pretrained word embedding,  
        # out of vocabulary words' embedding is initialized with uniform or normal numbers
        self.OOV_fill = 'uniform'
        self.embedding_norm = False

        # or else, use padding and masks for sequence data
        self.use_lod_tensor = True

        # lr = lr * lr_decay after each epoch
        self.lr_decay = 1
        self.learning_rate = 0.001

        self.save_dirname = 'model_dir'

        self.train_samples_num = 384348
        self.duplicate_data = False

        self.metric_type = ['accuracy']

    def list_config(self):
        print("config", self.__dict__)

    def has_member(self, var_name):
        return var_name in self.__dict__


if __name__ == "__main__":
    basic = config()
    basic.list_config()
    basic.ahh = 2
    basic.list_config()
