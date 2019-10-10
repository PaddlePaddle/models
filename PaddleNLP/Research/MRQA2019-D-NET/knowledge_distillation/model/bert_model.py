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

import os
import sys
import argparse
import collections
import numpy as np
import multiprocessing
from copy import deepcopy as copy

import paddle
import paddle.fluid as fluid

from model.bert import BertModel
from utils.configure import JsonConfig


class ModelBERT(object):

    def __init__(
        self, 
        conf, 
        name = "", 
        is_training = False,
        base_model = None):

        # the name of this task
        # name is used for identifying parameters
        self.name = name

        # deep copy the configure of model
        self.conf = copy(conf)

        self.is_training = is_training

        ## the overall loss of this task
        self.loss = None

        ## outputs may be useful for the other models
        self.outputs = {}

        ## the prediction of this task
        self.predict = []

    def create_model(self, 
                      args,
                      reader_input,
                      base_model = None):
        """
            given the base model, reader_input
            return the create fn for create this model
        """

        def _create_model():
        
            src_ids, pos_ids, sent_ids, input_mask = reader_input

            bert_conf = JsonConfig(self.conf["bert_conf_file"])
            self.bert = BertModel(
                src_ids = src_ids,
                position_ids = pos_ids,
                sentence_ids = sent_ids,
                input_mask = input_mask,
                config = bert_conf,
                use_fp16 = args.use_fp16,
                model_name = self.name)
                    
            self.loss = None
            self.outputs = {
                "sequence_output":self.bert.get_sequence_output(),
            }

        return _create_model


    def get_output(self, name):
        return self.outputs[name]


    def get_outputs(self):
        return self.outputs

    def get_predict(self):
        return self.predict


if __name__ == "__main__":
    
    bert_model = ModelBERT(conf = {"json_conf_path" : "./data/pretrained_models/squad2_model/bert_config.json"})

    
    


