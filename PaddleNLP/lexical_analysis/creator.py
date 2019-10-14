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
# -*- coding: UTF-8 -*-
"""
The function lex_net(args) define the lexical analysis network structure
"""
import sys
import os
import math

import paddle
import paddle.fluid as fluid
from paddle.fluid.initializer import NormalInitializer

from reader import Dataset
sys.path.append("..")
from models.sequence_labeling import nets
from models.representation.ernie import ernie_encoder
from preprocess.ernie import task_reader

def create_model(args,  vocab_size, num_labels, mode = 'train'):
    """create lac model"""

    # model's input data
    words = fluid.layers.data(name='words', shape=[-1, 1], dtype='int64',lod_level=1)
    targets = fluid.layers.data(name='targets', shape=[-1, 1], dtype='int64', lod_level= 1)

    # for inference process
    if mode=='infer':
        crf_decode = nets.lex_net(words, args, vocab_size, num_labels, for_infer=True, target=None)
        return { "feed_list":[words],"words":words, "crf_decode":crf_decode,}

    # for test or train process
    avg_cost, crf_decode = nets.lex_net(words, args, vocab_size, num_labels, for_infer=False, target=targets)

    (precision, recall, f1_score, num_infer_chunks, num_label_chunks,
     num_correct_chunks) = fluid.layers.chunk_eval(
        input=crf_decode,
        label=targets,
        chunk_scheme="IOB",
        num_chunk_types=int(math.ceil((num_labels - 1) / 2.0)))
    chunk_evaluator = fluid.metrics.ChunkEvaluator()
    chunk_evaluator.reset()

    ret = {
        "feed_list":[words, targets],
        "words": words,
        "targets": targets,
        "avg_cost":avg_cost,
        "crf_decode": crf_decode,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "chunk_evaluator": chunk_evaluator,
        "num_infer_chunks": num_infer_chunks,
        "num_label_chunks": num_label_chunks,
        "num_correct_chunks": num_correct_chunks
    }
    return  ret



def create_pyreader(args, file_name, feed_list, place, model='lac', reader=None, return_reader=False, mode='train'):
    # init reader
    pyreader = fluid.io.PyReader(
        feed_list=feed_list,
        capacity=300,
        use_double_buffer=True,
        iterable=True
    )
    if model == 'lac':
        if reader==None:
            reader = Dataset(args)
        # create lac pyreader
        if mode == 'train':
            pyreader.decorate_sample_list_generator(
                paddle.batch(
                    paddle.reader.shuffle(
                        reader.file_reader(file_name),
                        buf_size=args.traindata_shuffle_buffer
                    ),
                    batch_size=args.batch_size
                ),
                places=place
            )
        else:
            pyreader.decorate_sample_list_generator(
                paddle.batch(
                    reader.file_reader(file_name, mode=mode),
                    batch_size=args.batch_size
                ),
                places=place
            )

    elif model == 'ernie':
        # create ernie pyreader
        if reader==None:
            reader = task_reader.SequenceLabelReader(
                vocab_path=args.vocab_path,
                label_map_config=args.label_map_config,
                max_seq_len=args.max_seq_len,
                do_lower_case=args.do_lower_case,
                in_tokens=False,
                random_seed=args.random_seed)

        if mode == 'train':
            pyreader.decorate_batch_generator(
                reader.data_generator(
                    file_name, args.batch_size, args.epoch, shuffle=True, phase="train"
                ),
                places=place
            )
        else:
            pyreader.decorate_batch_generator(
                reader.data_generator(
                    file_name, args.batch_size, epoch=1, shuffle=False, phase=mode
                ),
                places=place
            )

    if return_reader:
        return pyreader, reader
    else:
        return pyreader

def create_ernie_model(args, ernie_config):

    """
    Create Model for LAC based on ERNIE encoder
    """
    # ERNIE's input data
    src_ids = fluid.layers.data(name='src_ids', shape=[args.max_seq_len, 1], dtype='int64',lod_level=0)
    sent_ids = fluid.layers.data(name='sent_ids', shape=[args.max_seq_len, 1], dtype='int64',lod_level=0)
    pos_ids = fluid.layers.data(name='pos_ids', shape=[args.max_seq_len, 1], dtype='int64',lod_level=0)
    input_mask = fluid.layers.data(name='input_mask', shape=[args.max_seq_len, 1], dtype='int64',lod_level=0)
    padded_labels =fluid.layers.data(name='padded_labels', shape=[args.max_seq_len, 1], dtype='int64',lod_level=0)
    seq_lens = fluid.layers.data(name='seq_lens', shape=[-1], dtype='int64',lod_level=0)
    squeeze_labels = fluid.layers.squeeze(padded_labels, axes=[-1])

    ernie_inputs = {
        "src_ids": src_ids,
        "sent_ids": sent_ids,
        "pos_ids": pos_ids,
        "input_mask": input_mask,
        "seq_lens": seq_lens
    }
    embeddings = ernie_encoder(ernie_inputs, ernie_config=ernie_config)

    token_embeddings = embeddings["token_embeddings"]

    emission = fluid.layers.fc(
        size=args.num_labels,
        input=token_embeddings,
        param_attr=fluid.ParamAttr(
            initializer=fluid.initializer.Uniform(
                low=-args.init_bound, high=args.init_bound),
            regularizer=fluid.regularizer.L2DecayRegularizer(
                regularization_coeff=1e-4)),
        num_flatten_dims=2)

    crf_cost = fluid.layers.linear_chain_crf(
        input=emission,
        label=padded_labels,
        param_attr=fluid.ParamAttr(
            name='crfw',
            learning_rate=args.crf_learning_rate),
        length=seq_lens)
    avg_cost = fluid.layers.mean(x=crf_cost)
    crf_decode = fluid.layers.crf_decoding(
            input=emission, param_attr=fluid.ParamAttr(name='crfw'),length=seq_lens)

    (precision, recall, f1_score, num_infer_chunks, num_label_chunks,
     num_correct_chunks) = fluid.layers.chunk_eval(
        input=crf_decode,
        label=squeeze_labels,
        chunk_scheme="IOB",
        num_chunk_types=int(math.ceil((args.num_labels - 1) / 2.0)),
        seq_length=seq_lens)
    chunk_evaluator = fluid.metrics.ChunkEvaluator()
    chunk_evaluator.reset()

    ret = {
        "feed_list": [src_ids, sent_ids, pos_ids, input_mask, padded_labels, seq_lens],
        "words":src_ids,
        "labels":padded_labels,
        "avg_cost":avg_cost,
        "crf_decode":crf_decode,
        "precision" : precision,
        "recall": recall,
        "f1_score": f1_score,
        "chunk_evaluator":chunk_evaluator,
        "num_infer_chunks":num_infer_chunks,
        "num_label_chunks":num_label_chunks,
        "num_correct_chunks":num_correct_chunks
    }

    return ret
