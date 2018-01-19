"""
    Contains infering script for machine translation with external memory.
"""
import distutils.util
import argparse
import gzip
import random

import paddle.v2 as paddle
from external_memory import ExternalMemory
from model import memory_enhanced_seq2seq
from data_utils import reader_append_wrapper

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--dict_size",
    default=30000,
    type=int,
    help="Vocabulary size. (default: %(default)s)")
parser.add_argument(
    "--word_vec_dim",
    default=512,
    type=int,
    help="Word embedding size. (default: %(default)s)")
parser.add_argument(
    "--hidden_size",
    default=1024,
    type=int,
    help="Hidden cell number in RNN. (default: %(default)s)")
parser.add_argument(
    "--memory_slot_num",
    default=8,
    type=int,
    help="External memory slot number. (default: %(default)s)")
parser.add_argument(
    "--beam_size",
    default=3,
    type=int,
    help="Beam search width. (default: %(default)s)")
parser.add_argument(
    "--use_gpu",
    default=False,
    type=distutils.util.strtobool,
    help="Use gpu or not. (default: %(default)s)")
parser.add_argument(
    "--trainer_count",
    default=1,
    type=int,
    help="Trainer number. (default: %(default)s)")
parser.add_argument(
    "--batch_size",
    default=5,
    type=int,
    help="Batch size. (default: %(default)s)")
parser.add_argument(
    "--infer_data_num",
    default=3,
    type=int,
    help="Instance num to infer. (default: %(default)s)")
parser.add_argument(
    "--model_filepath",
    default="checkpoints/params.latest.tar.gz",
    type=str,
    help="Model filepath. (default: %(default)s)")
parser.add_argument(
    "--memory_perturb_stddev",
    default=0.1,
    type=float,
    help="Memory perturb stddev for memory initialization."
    "(default: %(default)s)")
args = parser.parse_args()


def parse_beam_search_result(beam_result, dictionary):
    """
    Beam search result parser.
    """
    sentence_list = []
    sentence = []
    for word in beam_result[1]:
        if word != -1:
            sentence.append(word)
        else:
            sentence_list.append(' '.join(
                [dictionary.get(word) for word in sentence[1:]]))
            sentence = []
    beam_probs = beam_result[0]
    beam_size = len(beam_probs[0])
    beam_sentences = [
        sentence_list[i:i + beam_size]
        for i in range(0, len(sentence_list), beam_size)
    ]
    return beam_probs, beam_sentences


def infer():
    """
    For inferencing.
    """
    # create network config
    source_words = paddle.layer.data(
        name="source_words",
        type=paddle.data_type.integer_value_sequence(args.dict_size))
    beam_gen = memory_enhanced_seq2seq(
        encoder_input=source_words,
        decoder_input=None,
        decoder_target=None,
        hidden_size=args.hidden_size,
        word_vec_dim=args.word_vec_dim,
        dict_size=args.dict_size,
        is_generating=True,
        beam_size=args.beam_size)

    # load parameters
    parameters = paddle.parameters.Parameters.from_tar(
        gzip.open(args.model_filepath))

    # prepare infer data
    infer_data = []
    random.seed(0)  # for keeping consitancy for multiple runs
    bounded_memory_perturbation = [[
        random.gauss(0, args.memory_perturb_stddev)
        for i in xrange(args.hidden_size)
    ] for j in xrange(args.memory_slot_num)]
    test_append_reader = reader_append_wrapper(
        reader=paddle.dataset.wmt14.test(args.dict_size),
        append_tuple=(bounded_memory_perturbation, ))
    for i, item in enumerate(test_append_reader()):
        if i < args.infer_data_num:
            infer_data.append((
                item[0],
                item[3], ))

    # run inference
    beam_result = paddle.infer(
        output_layer=beam_gen,
        parameters=parameters,
        input=infer_data,
        field=['prob', 'id'])

    # parse beam result and print
    source_dict, target_dict = paddle.dataset.wmt14.get_dict(args.dict_size)
    beam_probs, beam_sentences = parse_beam_search_result(beam_result,
                                                          target_dict)
    for i in xrange(args.infer_data_num):
        print "\n***************************************************\n"
        print "src:", ' '.join(
            [source_dict.get(word) for word in infer_data[i][0]]), "\n"
        for j in xrange(args.beam_size):
            print "prob = %f : %s" % (beam_probs[i][j], beam_sentences[i][j])


def main():
    paddle.init(use_gpu=args.use_gpu, trainer_count=args.trainer_count)
    infer()


if __name__ == '__main__':
    main()
