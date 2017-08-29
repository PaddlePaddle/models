#!/usr/bin/env python
#coding=utf-8
import os
import sys
import gzip
import logging
import numpy as np

import paddle.v2 as paddle
from paddle.v2.layer import parse_network
import reader

from model import GNR
from train import choose_samples
from config import ModelConfig, TrainerConfig
from beam_decoding import BeamDecoding

logger = logging.getLogger("paddle")
logger.setLevel(logging.INFO)


def load_reverse_dict(dict_file):
    word_dict = {}
    with open(dict_file, "r") as fin:
        for idx, line in enumerate(fin):
            word_dict[idx] = line.strip()
    return word_dict


def print_result(test_batch, predicted_ans, ids_2_word, print_top_k=1):
    for i, sample in enumerate(test_batch):
        query_words = [ids_2_word[ids] for ids in sample[0]]
        print("query:\t%s" % (" ".join(query_words)))

        print("documents:")
        for j, sen in enumerate(sample[1]):
            sen_words = [ids_2_word[ids] for ids in sen]
            start = sample[4]
            end = sample[4] + sample[5] + 1
            print("%d\t%s" % (j, " ".join(sen_words)))
        print("gold:\t[%d %d %d] %s" % (
            sample[3], sample[4], sample[5], " ".join(
                [ids_2_word[ids] for ids in sample[1][sample[3]][start:end]])))

        print("predicted:")
        for k in range(print_top_k):
            label = predicted_ans[i][k]["label"]
            start = label[1]
            end = label[1] + label[2] + 1
            ans_words = [
                ids_2_word[ids] for ids in sample[1][label[0]][start:end]
            ]
            print("%.4f\t[%d %d %d] %s" %
                  (predicted_ans[i][k]["score"], label[0], label[1], label[2],
                   " ".join(ans_words)))
        print("\n")


def infer_a_batch(inferer, test_batch, ids_2_word, out_layer_count):
    outs = inferer.infer(input=test_batch, flatten_result=False, field="value")
    decoder = BeamDecoding([sample[1] for sample in test_batch], *outs)
    print_result(test_batch, decoder.decoding(), ids_2_word, print_top_k=10)


def infer(model_path, data_dir, test_batch_size, config):
    assert os.path.exists(model_path), "The model does not exist."
    paddle.init(use_gpu=True, trainer_count=1)

    ids_2_word = load_reverse_dict(config.dict_path)

    outputs = GNR(config, is_infer=True)

    # load the trained models
    parameters = paddle.parameters.Parameters.from_tar(
        gzip.open(model_path, "r"))
    logger.info("loading parameter is done.")

    inferer = paddle.inference.Inference(
        output_layer=outputs, parameters=parameters)

    _, valid_samples = choose_samples(data_dir)
    test_reader = reader.data_reader(valid_samples, is_train=False)

    test_batch = []
    for i, item in enumerate(test_reader()):
        test_batch.append(item)
        if len(test_batch) == test_batch_size:
            infer_a_batch(inferer, test_batch, ids_2_word, len(outputs))
            test_batch = []

    if len(test_batch):
        infer_a_batch(inferer, test_batch, ids_2_word, len(outputs))
        test_batch = []


if __name__ == "__main__":
    # infer("models/round1/pass_00000.tar.gz", TrainerConfig.data_dir,
    infer("models/round2_on_cpu/pass_00000.tar.gz", TrainerConfig.data_dir,
          TrainerConfig.test_batch_size, ModelConfig)
