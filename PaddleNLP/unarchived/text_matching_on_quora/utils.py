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
"""
This module provides utilities for data generator and optimizer definition 
"""

import sys
import time
import numpy as np

import paddle.fluid as fluid
import paddle
import quora_question_pairs


def to_lodtensor(data, place):
    """
    convert to LODtensor
    """
    seq_lens = [len(seq) for seq in data]
    cur_len = 0
    lod = [cur_len]
    for l in seq_lens:
        cur_len += l
        lod.append(cur_len)
    flattened_data = np.concatenate(data, axis=0).astype("int64")
    flattened_data = flattened_data.reshape([len(flattened_data), 1])
    res = fluid.LoDTensor()
    res.set(flattened_data, place)
    res.set_lod([lod])
    return res


def getOptimizer(global_config):
    """
    get Optimizer by config
    """
    if global_config.optimizer_type == "adam":
        optimizer = fluid.optimizer.Adam(
            learning_rate=fluid.layers.exponential_decay(
                learning_rate=global_config.learning_rate,
                decay_steps=global_config.train_samples_num //
                global_config.batch_size,
                decay_rate=global_config.lr_decay))
    elif global_config.optimizer_type == "sgd":
        optimizer = fluid.optimizer.SGD(
            learning_rate=fluid.layers.exponential_decay(
                learning_rate=global_config.learning_rate,
                decay_steps=global_config.train_samples_num //
                global_config.batch_size,
                decay_rate=global_config.lr_decay))

    elif global_config.optimizer_type == "adagrad":
        optimizer = fluid.optimizer.Adagrad(
            learning_rate=fluid.layers.exponential_decay(
                learning_rate=global_config.learning_rate,
                decay_steps=global_config.train_samples_num //
                global_config.batch_size,
                decay_rate=global_config.lr_decay))

    return optimizer


def get_pretrained_word_embedding(word2vec, word2id, config):
    """get pretrained embedding in shape [config.dict_dim, config.emb_dim]"""
    print("preparing pretrained word embedding ...")
    assert (config.dict_dim >= len(word2id))
    word2id = sorted(word2id.items(), key=lambda x: x[1])
    words = [x[0] for x in word2id]
    words = words + ['<not-a-real-words>'] * (config.dict_dim - len(words))
    pretrained_emb = []
    for _, word in enumerate(words):
        if word in word2vec:
            assert (len(word2vec[word] == config.emb_dim))
            if config.embedding_norm:
                pretrained_emb.append(word2vec[word] /
                                      np.linalg.norm(word2vec[word]))
            else:
                pretrained_emb.append(word2vec[word])
        elif config.OOV_fill == 'uniform':
            pretrained_emb.append(
                np.random.uniform(
                    -0.05, 0.05, size=[config.emb_dim]).astype(np.float32))
        elif config.OOV_fill == 'normal':
            pretrained_emb.append(
                np.random.normal(
                    loc=0.0, scale=0.1, size=[config.emb_dim]).astype(
                        np.float32))
        else:
            print("Unkown OOV fill method: ", OOV_fill)
            exit()
    word_embedding = np.stack(pretrained_emb)
    return word_embedding


def getDict(data_type="quora_question_pairs"):
    """
    get word2id dict from quora dataset
    """
    print("Generating word dict...")
    if data_type == "quora_question_pairs":
        word_dict = quora_question_pairs.word_dict()
    else:
        raise RuntimeError("No such dataset")
    print("Vocab size: ", len(word_dict))
    return word_dict


def duplicate(reader):
    """
    duplicate the quora qestion pairs since there are 2 questions in a sample
    Input: reader, which yield (question1, question2, label)
    Output: reader, which yield (question1, question2, label) and yield (question2, question1, label)
    """

    def duplicated_reader():
        for data in reader():
            (q1, q2, label) = data
            yield (q1, q2, label)
            yield (q2, q1, label)

    return duplicated_reader


def pad(reader, PAD_ID):
    """
    Input: reader, yield batches of [(question1, question2, label), ... ]
    Output: padded_reader, yield batches of [(padded_question1, padded_question2, mask1, mask2, label), ... ]
    """

    assert (isinstance(PAD_ID, int))

    def padded_reader():
        for batch in reader():
            max_len1 = max([len(data[0]) for data in batch])
            max_len2 = max([len(data[1]) for data in batch])

            padded_batch = []
            for data in batch:
                question1, question2, label = data
                seq_len1 = len(question1)
                seq_len2 = len(question2)
                mask1 = [1] * seq_len1 + [0] * (max_len1 - seq_len1)
                mask2 = [1] * seq_len2 + [0] * (max_len2 - seq_len2)
                padded_question1 = question1 + [PAD_ID] * (max_len1 - seq_len1)
                padded_question2 = question2 + [PAD_ID] * (max_len2 - seq_len2)
                padded_question1 = [
                    [x] for x in padded_question1
                ]  # last dim of questions must be 1, according to fluid's request
                padded_question2 = [[x] for x in padded_question2]
                assert (len(mask1) == max_len1)
                assert (len(mask2) == max_len2)
                assert (len(padded_question1) == max_len1)
                assert (len(padded_question2) == max_len2)
                padded_batch.append(
                    (padded_question1, padded_question2, mask1, mask2, label))
            yield padded_batch

    return padded_reader


def prepare_data(data_type,
                 word_dict,
                 batch_size,
                 buf_size=50000,
                 duplicate_data=False,
                 use_pad=False):
    """
    prepare data
    """

    PAD_ID = word_dict['<pad>']

    if data_type == "quora_question_pairs":
        # train/dev/test reader are batched iters which yield a batch of (question1, question2, label) each time
        # qestion1 and question2 are lists of word ID
        # label is 0 or 1
        # for example: ([1, 3, 2], [7, 5, 4, 99], 1)

        def prepare_reader(reader):
            if duplicate_data:
                reader = duplicate(reader)
            reader = paddle.batch(
                paddle.reader.shuffle(
                    reader, buf_size=buf_size),
                batch_size=batch_size,
                drop_last=False)
            if use_pad:
                reader = pad(reader, PAD_ID=PAD_ID)
            return reader

        train_reader = prepare_reader(quora_question_pairs.train(word_dict))
        dev_reader = prepare_reader(quora_question_pairs.dev(word_dict))
        test_reader = prepare_reader(quora_question_pairs.test(word_dict))

    else:
        raise RuntimeError("no such dataset")

    return train_reader, dev_reader, test_reader
