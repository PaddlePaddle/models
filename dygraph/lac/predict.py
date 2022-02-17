# -*- coding: UTF-8 -*-
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

import argparse
import os
import time
import sys

import paddle.fluid as fluid
import paddle
import utils
import reader
import math
from sequence_labeling import lex_net, Chunk_eval
parser = argparse.ArgumentParser(__doc__)
# 1. model parameters
utils.load_yaml(parser, 'conf/args.yaml')
args = parser.parse_args()

def do_infer(args):
    dataset = reader.Dataset(args)

    if args.use_cuda:
        place = fluid.CUDAPlace(fluid.dygraph.parallel.Env().dev_id) \
        if args.use_data_parallel else fluid.CUDAPlace(0)
    else:
        place = fluid.CPUPlace()

    with fluid.dygraph.guard(place):
        infer_loader = reader.create_dataloader(
            args,
            file_name=args.infer_data,
            place=place,
            model='lac',
            reader=dataset,
            mode='infer')
        model = lex_net(args, dataset.vocab_size, dataset.num_labels)
        load_path = args.init_checkpoint
        state_dict, _ = fluid.dygraph.load_dygraph(load_path)
        #import ipdb; ipdb.set_trace()
        state_dict["linear_chain_crf.weight"]=state_dict["crf_decoding.weight"]
        model.set_dict(state_dict)
        model.eval()
        chunk_eval = Chunk_eval(int(math.ceil((dataset.num_labels - 1) / 2.0)), "IOB")
        chunk_evaluator = fluid.metrics.ChunkEvaluator()
        chunk_evaluator.reset()

        def input_check(data):
       	    if data.lod()[0][-1] == 0:
                return data[0]['words']
            return None
            
        def infer_process(reader):
            results = []
           
            for batch in reader():
                # import ipdb; ipdb.set_trace()
                words, length = batch
                #crf_decode = input_check(words)
                #if crf_decode:
                #    results += utils.parse_result(crf_decode, crf_decode, dataset)
                #    continue
			              
                crf_decode = model(words, length=length)
                results += utils.parse_padding_result(words.numpy(), crf_decode.numpy(), length.numpy(), dataset)          
            return results
            
        result = infer_process(infer_loader)
        for sent, tags in result:
            result_list = ['(%s, %s)' % (ch, tag) for ch, tag in zip(sent, tags)]
            print(''.join(result_list).encode('utf-8'))

if __name__ == '__main__':
    args = parser.parse_args()
    do_infer(args)
