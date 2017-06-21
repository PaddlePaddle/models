#!/usr/bin/env python
import os
from network_conf import *


def infer_a_batch(inferer, test_batch, beam_size, src_dict, trg_dict):
    beam_result = inferer.infer(input=test_batch, field=["prob", "id"])

    # the delimited element of generated sequences is -1,
    # the first element of each generated sequence is the sequence length
    seq_list, seq = [], []
    for w in beam_result[1]:
        if w != -1:
            seq.append(w)
        else:
            seq_list.append(" ".join([trg_dict.get(w) for w in seq[1:]]))
            seq = []

    prob = beam_result[0]
    for i, sample in enumerate(test_batch):
        print("src:", " ".join([src_dict.get(w) for w in sample[0]]), "\n")
        for j in xrange(beam_size):
            print("prob = %f:" % (prob[i][j]), seq_list[i * beam_size + j])
        print("\n")


def generate(source_dict_dim, target_dict_dim, model_path, batch_size):
    """
    Generating function for NMT

    :param source_dict_dim: size of source dictionary
    :type source_dict_dim: int
    :param target_dict_dim: size of target dictionary
    :type target_dict_dim: int
    :param model_path: path for inital model
    :type model_path: string
    """

    assert os.path.exists(model_path), "trained model does not exist."

    # step 1: prepare dictionary
    src_dict, trg_dict = paddle.dataset.wmt14.get_dict(source_dict_dim)
    beam_size = 5

    # step 2: load the trained model
    paddle.init(use_gpu=True, trainer_count=1)
    with gzip.open(model_path) as f:
        parameters = paddle.parameters.Parameters.from_tar(f)
    beam_gen = seq2seq_net(
        source_dict_dim,
        target_dict_dim,
        beam_size=beam_size,
        max_length=100,
        is_generating=True)
    inferer = paddle.inference.Inference(
        output_layer=beam_gen, parameters=parameters)

    # step 3: iterating over the testing dataset
    test_batch = []
    for idx, item in enumerate(paddle.dataset.wmt14.gen(source_dict_dim)()):
        test_batch.append([item[0]])
        if len(test_batch) == batch_size:
            infer_a_batch(inferer, test_batch, beam_size, src_dict, trg_dict)
            test_batch = []

    if len(test_batch):
        infer_a_batch(inferer, test_batch, beam_size, src_dict, trg_dict)
        test_batch = []


if __name__ == "__main__":
    generate(
        source_dict_dim=3000,
        target_dict_dim=3000,
        batch_size=5,
        model_path="models/nmt_without_att_params_batch_00001.tar.gz")
