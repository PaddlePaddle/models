#encoding=utf8

import os
import sys
import argparse
from copy import deepcopy as copy
import numpy as np
import paddle
import paddle.fluid as fluid
import collections
import multiprocessing

from pdnlp.nets.bert import BertModel
from pdnlp.toolkit.configure import JsonConfig

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
                "sequence_output": self.bert.get_sequence_output(),
                # "pooled_output": self.bert.get_pooled_output()
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

    
    


