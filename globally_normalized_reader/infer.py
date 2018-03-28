#coding=utf-8

import os
import sys
import argparse
import gzip
import logging
import numpy as np

import paddle.v2 as paddle
from paddle.v2.layer import parse_network
import reader

from model import GNR
from train import choose_samples
from config import ModelConfig
from beam_decoding import BeamDecoding

logger = logging.getLogger("paddle")
logger.setLevel(logging.INFO)


def parse_cmd():
    """
    Build the command line arguments parser for inferring task.
    """
    parser = argparse.ArgumentParser(
        description="Globally Normalized Reader in PaddlePaddle.")
    parser.add_argument(
        "--model_path",
        required=True,
        type=str,
        help="Path of the trained model to evaluate.",
        default="")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path of the training and testing data.",
        default="")
    parser.add_argument(
        "--batch_size",
        type=int,
        required=False,
        help="The batch size for inferring.",
        default=1)
    parser.add_argument(
        "--use_gpu",
        type=int,
        required=False,
        help="Whether to run the inferring on GPU.",
        default=0)
    parser.add_argument(
        "--trainer_count",
        type=int,
        required=False,
        help=("The thread number used in inferring. When set "
              "use_gpu=True, the trainer_count cannot excess "
              "the gpu device number in your computer."),
        default=1)
    return parser.parse_args()


def load_reverse_dict(dict_file):
    """ Build the dict which is used to map the word index to word string.

    The keys are word index and the values are word strings.

    Arguments:
        - dict_file:    The path of a word dictionary.
    """
    word_dict = {}
    with open(dict_file, "r") as fin:
        for idx, line in enumerate(fin):
            word_dict[idx] = line.strip()
    return word_dict


def print_result(test_batch, predicted_ans, ids_2_word, print_top_k=1):
    """ Print the readable predicted answers.

    Format of the output:
        query:\tthe input query.
        documents:\n
        0\tthe first sentence in the document.
        1\tthe second sentence in the document.
        ...
        gold:\t[i j k] the answer words.
            (i: the sentence index;
             j: the start span index;
             k: the end span index)
        top answers:
        score0\t[i j k] the answer with the highest score.
        score1\t[i j k] the answer with the second highest score.
            (i, j, k has a same meaning as in gold.)
        ...

        By default, top 10 answers will be printed.

    Arguments:
        - test_batch:     A test batch returned by reader.
        - predicted_ans:  The beam decoding results.
        - ids_2_word:     The dict whose key is word index and the values are
                          word strings.
        - print_top_k:    Indicating how many answers will be printed.
    """

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

        print("top answers:")
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
    """ Call the PaddlePaddle's infer interface to infer by batch.

    Arguments:
        - inferer:         The PaddlePaddle Inference object.
        - test_batch:      A test batch returned by reader.
        - ids_2_word:      The dict whose key is word index and the values are
                           word strings.
        - out_layer_count: The number of output layers in the inferring process.
    """

    outs = inferer.infer(input=test_batch, flatten_result=False, field="value")
    decoder = BeamDecoding([sample[1] for sample in test_batch], *outs)
    print_result(test_batch, decoder.decoding(), ids_2_word, print_top_k=10)


def infer(model_path,
          data_dir,
          batch_size,
          config,
          use_gpu=False,
          trainer_count=1):
    """ The inferring process.

    Arguments:
        - model_path:      The path of trained model.
        - data_dir:        The directory path of test data.
        - batch_size:      The batch_size.
        - config:          The model configuration.
        - use_gpu:         Whether to run the inferring on GPU.
        - trainer_count:   The thread number used in inferring. When set
                           use_gpu=True, the trainer_count cannot excess
                           the gpu device number in your computer.
    """

    assert os.path.exists(model_path), "The model does not exist."
    paddle.init(use_gpu=use_gpu, trainer_count=trainer_count)

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
        if len(test_batch) == batch_size:
            infer_a_batch(inferer, test_batch, ids_2_word, len(outputs))
            test_batch = []

    if len(test_batch):
        infer_a_batch(inferer, test_batch, ids_2_word, len(outputs))
        test_batch = []


def main(args):
    infer(
        model_path=args.model_path,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        config=ModelConfig,
        use_gpu=args.use_gpu,
        trainer_count=args.trainer_count)


if __name__ == "__main__":
    args = parse_cmd()
    main(args)
