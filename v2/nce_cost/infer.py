import os
import gzip
import click
import numpy as np

import paddle.v2 as paddle
from network_conf import ngram_lm


def infer_a_batch(inferer, test_batch, id_to_word):
    probs = inferer.infer(input=test_batch)
    for i, res in enumerate(zip(test_batch, probs)):
        maxid = res[1].argsort()[-1]
        print("%.4f\t%s\t%s" % (res[1][maxid], id_to_word[maxid],
                                " ".join([id_to_word[w] for w in res[0]])))


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

    word_to_id = paddle.dataset.imikolov.build_dict()
    id_to_word = dict((v, k) for k, v in word_to_id.items())
    dict_size = len(word_to_id)

    paddle.init(use_gpu=use_gpu, trainer_count=trainer_count)

    # load the trained model.
    with gzip.open(model_path) as f:
        parameters = paddle.parameters.Parameters.from_tar(f)
    prediction_layer = ngram_lm(
        is_train=False, hidden_size=128, emb_size=512, dict_size=dict_size)
    inferer = paddle.inference.Inference(
        output_layer=prediction_layer, parameters=parameters)

    test_batch = []
    for idx, item in enumerate(paddle.dataset.imikolov.test(word_to_id, 5)()):
        test_batch.append((item[:4]))
        if len(test_batch) == batch_size:
            infer_a_batch(inferer, test_batch, id_to_word)
            infer_data = []

    if len(test_batch):
        infer_a_batch(inferer, test_batch, id_to_word)
        infer_data = []
        infer_data_label = []


if __name__ == "__main__":
    infer()
