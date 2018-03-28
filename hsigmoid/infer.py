import os
import logging
import gzip
import click

import paddle.v2 as paddle
from network_conf import ngram_lm

logger = logging.getLogger("paddle")
logger.setLevel(logging.WARNING)


def decode_result(infer_res, dict_size):
    """
    Inferring probabilities are orginized as a complete binary tree.
    The actual labels are leaves (indices are counted from class number).
    This function travels paths decoded from inferring results.
    If the probability >0.5 then go to right child, otherwise go to left child.

    param infer_res: inferring result
    param dict_size: class number
    return predict_lbls: actual class
    """
    predict_lbls = []
    infer_res = infer_res > 0.5
    for i, probs in enumerate(infer_res):
        idx = 0
        result = 1
        while idx < len(probs):
            result <<= 1
            if probs[idx]:
                result |= 1
            if probs[idx]:
                idx = idx * 2 + 2  # right child
            else:
                idx = idx * 2 + 1  # left child

        predict_lbl = result - dict_size
        predict_lbls.append(predict_lbl)
    return predict_lbls


def infer_a_batch(batch_ins, idx_word_dict, dict_size, inferer):
    infer_res = inferer.infer(input=batch_ins)

    predict_lbls = decode_result(infer_res, dict_size)
    predict_words = [idx_word_dict[lbl] for lbl in predict_lbls]  # map to word

    # Ouput format: word1 word2 word3 word4 -> predict label
    for i, ins in enumerate(batch_ins):
        print(" ".join([idx_word_dict[w]
                        for w in ins]) + " -> " + predict_words[i])


@click.command("infer")
@click.option(
    "--model_path",
    default="",
    help="The path of the trained model for generation.")
@click.option(
    "--batch_size",
    default=1,
    help="The number of testing examples in one forward batch in inferring.")
@click.option(
    "--use_gpu", default=False, help="Whether to use GPU in inference or not.")
@click.option(
    "--trainer_count",
    default=1,
    help="Whether to use GPU in inference or not.")
def infer(model_path, batch_size, use_gpu, trainer_count):
    assert os.path.exists(model_path), "The trained model does not exist."
    assert (batch_size and trainer_count and batch_size >= trainer_count), (
        "batch_size and trainer_count must both be greater than 0. "
        "And batch_size must be equal to "
        "or greater than trainer_count.")

    paddle.init(use_gpu=use_gpu, trainer_count=trainer_count)
    word_dict = paddle.dataset.imikolov.build_dict(min_word_freq=2)
    dict_size = len(word_dict)
    prediction_layer = ngram_lm(
        is_train=False, hidden_size=256, embed_size=32, dict_size=dict_size)

    with gzip.open(model_path, "r") as f:
        parameters = paddle.parameters.Parameters.from_tar(f)

    inferer = paddle.inference.Inference(
        output_layer=prediction_layer, parameters=parameters)
    idx_word_dict = dict((v, k) for k, v in word_dict.items())

    batch_ins = []
    for ins in paddle.dataset.imikolov.test(word_dict, 5)():
        batch_ins.append(ins[:-1])
        if len(batch_ins) == batch_size:
            infer_a_batch(batch_ins, idx_word_dict, dict_size, inferer)
            batch_ins = []

    if len(batch_ins) > 0:
        infer_a_batch(batch_ins, idx_word_dict, dict_size, inferer)


if __name__ == "__main__":
    infer()
