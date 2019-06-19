#!/usr/bin/env python
# -*- coding: UTF-8 -*-
################################################################################
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
File: source/utils/utils.py
"""
from __future__ import print_function

import argparse
import numpy as np
import paddle.fluid as fluid
import paddle.fluid.layers as layers


def str2bool(v):
    """ str2bool """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def load_id2str_dict(vocab_file):
    """ load id2str dict """
    id_dict_array = []
    with open(vocab_file, 'r') as fr:
        for line in fr:
            line = line.strip()
            id_dict_array.append(line)

    return id_dict_array


def load_str2id_dict(vocab_file):
    """ load str2id dict """
    words_dict = {}
    with open(vocab_file, 'r') as fr:
        for line in fr:
            word = line.strip()
            words_dict[word] = len(words_dict)

    return words_dict


def log_softmax(x):
    """ log softmax """
    t1 = layers.exp(x)
    t1 = layers.reduce_sum(t1, dim=-1)
    t1 = layers.log(t1)
    return layers.elementwise_sub(x, t1, axis=0)


def id_to_text(ids, id_dict_array):
    """ convert id seq to str seq """
    res = []
    for i in ids:
        res.append(id_dict_array[i])

    return ' '.join(res)


def pad_to_bath_size(src_ids, src_len, trg_ids, trg_len, kn_ids, kn_len, batch_size):
    """ pad to bath size for knowledge corpus"""
    real_len = src_ids.shape[0]

    def pad(old):
        """ pad """
        old_shape = list(old.shape)
        old_shape[0] = batch_size
        new_val = np.zeros(old_shape, dtype=old.dtype)
        new_val[:real_len] = old
        for i in range(real_len, batch_size):
            new_val[i] = old[-1]
        return new_val

    new_src_ids = pad(src_ids)
    new_src_len = pad(src_len)
    new_trg_ids = pad(trg_ids)
    new_trg_len = pad(trg_len)
    new_kn_ids = pad(kn_ids)
    new_kn_len = pad(kn_len)

    return [new_src_ids, new_src_len, new_trg_ids, new_trg_len, new_kn_ids, new_kn_len]


def to_lodtensor(data, seq_lens, place):
    """ convert to LoDTensor """
    cur_len = 0
    lod = [cur_len]

    data_array = []
    for idx, seq in enumerate(seq_lens):
        if seq > 0:
            data_array.append(data[idx, :seq])

            cur_len += seq
            lod.append(cur_len)
        else:
            data_array.append(np.zeros([1, 1], dtype='int64'))
            cur_len += 1
            lod.append(cur_len)
    flattened_data = np.concatenate(data_array, axis=0).astype("int64")
    flattened_data = flattened_data.reshape([len(flattened_data), 1])
    res = fluid.LoDTensor()
    res.set(flattened_data, place)

    res.set_lod([lod])
    return res


def len_to_mask(len_seq, max_len=None):
    """ len to mask """
    if max_len is None:
        max_len = np.max(len_seq)

    mask = np.zeros((len_seq.shape[0], max_len), dtype='float32')

    for i, l in enumerate(len_seq):
        mask[i, :l] = 1.0

    return mask


def build_data_feed(data, place,
                    batch_size=128,
                    is_training=False,
                    bow_max_len=30,
                    pretrain_epoch=False):
    """ build data feed """
    src_ids, src_len, trg_ids, trg_len, kn_ids, kn_len = data

    real_size = src_ids.shape[0]
    if src_ids.shape[0] < batch_size:
        if not is_training:
            src_ids, src_len, trg_ids, trg_len, kn_ids, kn_len = \
                pad_to_bath_size(src_ids, src_len, trg_ids, trg_len, kn_ids, kn_len, batch_size)
        else:
            return None

    enc_input = np.expand_dims(src_ids[:, 1: -1], axis=2)
    enc_mask = len_to_mask(src_len - 2)

    tar_input = np.expand_dims(trg_ids[:, 1: -1], axis=2)
    tar_mask = len_to_mask(trg_len - 2)
    cue_input = np.expand_dims(kn_ids.reshape((-1, kn_ids.shape[-1]))[:, 1:-1], axis=2)
    cue_mask = len_to_mask(kn_len.reshape(-1) - 2)
    memory_mask = np.equal(kn_len, 0).astype('float32')

    enc_memory_mask = 1.0 - enc_mask

    if not is_training:
        return {'enc_input': to_lodtensor(enc_input, src_len - 2, place),
                'enc_mask': enc_mask,
                'cue_input': to_lodtensor(cue_input, kn_len.reshape(-1) - 2, place),
                'cue_last_mask': np.not_equal(kn_len.reshape(-1), 0).astype('float32'),
                'memory_mask': memory_mask,
                'enc_memory_mask': enc_memory_mask,
                }, real_size

    dec_input = np.expand_dims(trg_ids[:, :-1], axis=2)
    dec_mask = len_to_mask(trg_len - 1)

    target_label = trg_ids[:, 1:]
    target_mask = len_to_mask(trg_len - 1)

    bow_label = target_label[:, :-1]
    bow_label = np.pad(bow_label, ((0, 0), (0, bow_max_len - bow_label.shape[1])), 'constant', constant_values=(0))
    bow_mask = np.pad(np.not_equal(bow_label, 0).astype('float32'), ((0, 0), (0, bow_max_len - bow_label.shape[1])),
                      'constant', constant_values=(0.0))

    if not pretrain_epoch:
        kl_and_nll_factor = np.ones([1], dtype='float32')
    else:
        kl_and_nll_factor = np.zeros([1], dtype='float32')

    return {'enc_input': to_lodtensor(enc_input, src_len - 2, place),
            'enc_mask': enc_mask,
            'cue_input': to_lodtensor(cue_input, kn_len.reshape(-1) - 2, place),
            'cue_last_mask': np.not_equal(kn_len.reshape(-1), 0).astype('float32'),
            'memory_mask': memory_mask,
            'enc_memory_mask': enc_memory_mask,
            'tar_input': to_lodtensor(tar_input, trg_len - 2, place),
            'bow_label': bow_label,
            'bow_mask': bow_mask,
            'target_label': target_label,
            'target_mask': target_mask,
            'dec_input': dec_input,
            'dec_mask': dec_mask,
            'kl_and_nll_factor': kl_and_nll_factor}


def load_embedding(embedding_file, vocab_file):
    """ load pretrain embedding from file """
    words_dict = load_str2id_dict(vocab_file)
    coverage = 0
    print("Building word embeddings from '{}' ...".format(embedding_file))
    with open(embedding_file, "r") as f:
        num, dim = map(int, f.readline().strip().split())
        embeds = [[0] * dim] * len(words_dict)
        for line in f:
            w, vs = line.rstrip().split(" ", 1)
            if w in words_dict:
                try:
                    vs = [float(x) for x in vs.split(" ")]
                except Exception:
                    vs = []
                if len(vs) == dim:
                    embeds[words_dict[w]] = vs
                    coverage += 1
    rate = coverage * 1.0 / len(embeds)
    print("{} words have pretrained {}-D word embeddings (coverage: {:.3f})".format( \
        coverage, dim, rate))

    return np.array(embeds).astype('float32')


def init_embedding(embedding_file, vocab_file, init_scale, shape):
    """ init embedding by pretrain file or random """
    if embedding_file != "":
        try:
            emb_np = load_embedding(embedding_file, vocab_file)
        except:
            print(("load init emb file failed", embedding_file))
            raise Exception("load embedding file failed")

        if emb_np.shape != shape:
            print(("shape not match", emb_np.shape, shape))
            raise Exception("shape not match")

        zero_count = 0
        for i in range(emb_np.shape[0]):
            if np.sum(emb_np[i]) == 0:
                zero_count += 1
                emb_np[i] = np.random.uniform(-init_scale, init_scale, emb_np.shape[1:]).astype('float32')
    else:
        print("random init embeding")
        emb_np = np.random.uniform(-init_scale, init_scale, shape).astype('float32')

    return emb_np
