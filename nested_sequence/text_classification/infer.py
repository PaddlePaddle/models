import sys
import os
import gzip
import click

import paddle.v2 as paddle

import reader
from network_conf import nest_net
from utils import logger, load_dict


@click.command('infer')
@click.option(
    "--data_path",
    default=None,
    help=("path of data for inference (default: None). "
          "if this parameter is not set, "
          "imdb test dataset will be used."))
@click.option(
    "--model_path",
    type=str,
    default='models/params_pass_00000.tar.gz',
    help=("path of saved model. "
          "(default: 'models/params_pass_00000.tar.gz')"))
@click.option(
    "--word_dict_path",
    type=str,
    default=None,
    help=("path of word dictionary (default: None)."
          "if this parameter is not set, imdb dataset will be used."))
@click.option(
    "--class_num", type=int, default=2, help="class number (default: 2).")
@click.option(
    "--batch_size",
    type=int,
    default=32,
    help="the number of examples in one batch (default: 32).")
def infer(data_path, model_path, word_dict_path, batch_size, class_num):
    def _infer_a_batch(inferer, test_batch, ids_2_word):
        probs = inferer.infer(input=test_batch, field=["value"])
        assert len(probs) == len(test_batch)
        for word_ids, prob in zip(test_batch, probs):
            sent_ids = []
            for sent in word_ids[0]:
                sent_ids.extend(sent)
            word_text = " ".join([ids_2_word[id] for id in sent_ids])
            print("%s\t%s\t%s" % (prob.argmax(),
                                  " ".join(["{:0.4f}".format(p)
                                            for p in prob]), word_text))

    assert os.path.exists(model_path), "the trained model does not exist."
    logger.info("begin to predict...")
    use_default_data = (data_path is None)

    if use_default_data:
        word_dict = reader.imdb_word_dict()
        word_reverse_dict = dict((value, key)
                                 for key, value in word_dict.iteritems())
        test_reader = reader.imdb_test(word_dict)
        class_num = 2
    else:
        assert os.path.exists(
            word_dict_path), "the word dictionary file does not exist"

        word_dict = load_dict(word_dict_path)
        word_reverse_dict = dict((value, key)
                                 for key, value in word_dict.iteritems())

        test_reader = reader.infer_reader(data_path, word_dict)()

    dict_dim = len(word_dict)
    prob_layer = nest_net(dict_dim, class_num=class_num, is_infer=True)

    # initialize PaddlePaddle
    paddle.init(use_gpu=True, trainer_count=4)

    # load the trained models
    parameters = paddle.parameters.Parameters.from_tar(
        gzip.open(model_path, "r"))
    inferer = paddle.inference.Inference(
        output_layer=prob_layer, parameters=parameters)

    test_batch = []
    for idx, item in enumerate(test_reader):
        test_batch.append([item[0]])
        if len(test_batch) == batch_size:
            _infer_a_batch(inferer, test_batch, word_reverse_dict)
            test_batch = []

    if len(test_batch):
        _infer_a_batch(inferer, test_batch, word_reverse_dict)
        test_batch = []


if __name__ == "__main__":
    infer()
