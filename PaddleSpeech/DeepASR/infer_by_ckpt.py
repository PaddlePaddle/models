from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import numpy as np
import argparse
import time

import paddle.fluid as fluid
import data_utils.augmentor.trans_mean_variance_norm as trans_mean_variance_norm
import data_utils.augmentor.trans_add_delta as trans_add_delta
import data_utils.augmentor.trans_splice as trans_splice
import data_utils.augmentor.trans_delay as trans_delay
import data_utils.async_data_reader as reader
from data_utils.util import lodtensor_to_ndarray, split_infer_result
from model_utils.model import stacked_lstmp_model
from post_latgen_faster_mapped import Decoder
from tools.error_rate import char_errors


def parse_args():
    parser = argparse.ArgumentParser("Run inference by using checkpoint.")
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='The sequence number of a batch data. (default: %(default)d)')
    parser.add_argument(
        '--beam_size',
        type=int,
        default=11,
        help='The beam size for decoding. (default: %(default)d)')
    parser.add_argument(
        '--minimum_batch_size',
        type=int,
        default=1,
        help='The minimum sequence number of a batch data. '
        '(default: %(default)d)')
    parser.add_argument(
        '--frame_dim',
        type=int,
        default=80,
        help='Frame dimension of feature data. (default: %(default)d)')
    parser.add_argument(
        '--stacked_num',
        type=int,
        default=5,
        help='Number of lstmp layers to stack. (default: %(default)d)')
    parser.add_argument(
        '--proj_dim',
        type=int,
        default=512,
        help='Project size of lstmp unit. (default: %(default)d)')
    parser.add_argument(
        '--hidden_dim',
        type=int,
        default=1024,
        help='Hidden size of lstmp unit. (default: %(default)d)')
    parser.add_argument(
        '--class_num',
        type=int,
        default=1749,
        help='Number of classes in label. (default: %(default)d)')
    parser.add_argument(
        '--num_threads',
        type=int,
        default=10,
        help='The number of threads for decoding. (default: %(default)d)')
    parser.add_argument(
        '--device',
        type=str,
        default='GPU',
        choices=['CPU', 'GPU'],
        help='The device type. (default: %(default)s)')
    parser.add_argument(
        '--parallel', action='store_true', help='If set, run in parallel.')
    parser.add_argument(
        '--mean_var',
        type=str,
        default='data/global_mean_var',
        help="The path for feature's global mean and variance. "
        "(default: %(default)s)")
    parser.add_argument(
        '--infer_feature_lst',
        type=str,
        default='data/infer_feature.lst',
        help='The feature list path for inference. (default: %(default)s)')
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='./checkpoint',
        help="The checkpoint path to init model. (default: %(default)s)")
    parser.add_argument(
        '--trans_model',
        type=str,
        default='./graph/trans_model',
        help="The path to vocabulary. (default: %(default)s)")
    parser.add_argument(
        '--vocabulary',
        type=str,
        default='./graph/words.txt',
        help="The path to vocabulary. (default: %(default)s)")
    parser.add_argument(
        '--graphs',
        type=str,
        default='./graph/TLG.fst',
        help="The path to TLG graphs for decoding. (default: %(default)s)")
    parser.add_argument(
        '--log_prior',
        type=str,
        default="./logprior",
        help="The log prior probs for training data. (default: %(default)s)")
    parser.add_argument(
        '--acoustic_scale',
        type=float,
        default=0.2,
        help="Scaling factor for acoustic likelihoods. (default: %(default)f)")
    parser.add_argument(
        '--post_matrix_path',
        type=str,
        default=None,
        help="The path to output post prob matrix. (default: %(default)s)")
    parser.add_argument(
        '--decode_to_path',
        type=str,
        default='./decoding_result.txt',
        required=True,
        help="The path to output the decoding result. (default: %(default)s)")
    args = parser.parse_args()
    return args


def print_arguments(args):
    print('-----------  Configuration Arguments -----------')
    for arg, value in sorted(vars(args).iteritems()):
        print('%s: %s' % (arg, value))
    print('------------------------------------------------')


class PostMatrixWriter:
    """ The writer for outputing the post probability matrix
    """

    def __init__(self, to_path):
        self._to_path = to_path
        with open(self._to_path, "w") as post_matrix:
            post_matrix.seek(0)
            post_matrix.truncate()

    def write(self, keys, probs):
        with open(self._to_path, "a") as post_matrix:
            if isinstance(keys, str):
                keys, probs = [keys], [probs]

            for key, prob in zip(keys, probs):
                post_matrix.write(key + " [\n")
                for i in range(prob.shape[0]):
                    for j in range(prob.shape[1]):
                        post_matrix.write(str(prob[i][j]) + " ")
                    post_matrix.write("\n")
                post_matrix.write("]\n")


class DecodingResultWriter:
    """ The writer for writing out decoding results
    """

    def __init__(self, to_path):
        self._to_path = to_path
        with open(self._to_path, "w") as decoding_result:
            decoding_result.seek(0)
            decoding_result.truncate()

    def write(self, results):
        with open(self._to_path, "a") as decoding_result:
            if isinstance(results, str):
                decoding_result.write(results.encode("utf8") + "\n")
            else:
                for result in results:
                    decoding_result.write(result.encode("utf8") + "\n")


def infer_from_ckpt(args):
    """Inference by using checkpoint."""

    if not os.path.exists(args.checkpoint):
        raise IOError("Invalid checkpoint!")

    feature = fluid.data(
        name='feature',
        shape=[None, 3, 11, args.frame_dim],
        dtype='float32',
        lod_level=1)
    label = fluid.data(
        name='label', shape=[None, 1], dtype='int64', lod_level=1)

    prediction, avg_cost, accuracy = stacked_lstmp_model(
        feature=feature,
        label=label,
        hidden_dim=args.hidden_dim,
        proj_dim=args.proj_dim,
        stacked_num=args.stacked_num,
        class_num=args.class_num,
        parallel=args.parallel)

    infer_program = fluid.default_main_program().clone()

    # optimizer, placeholder
    optimizer = fluid.optimizer.Adam(
        learning_rate=fluid.layers.exponential_decay(
            learning_rate=0.0001,
            decay_steps=1879,
            decay_rate=1 / 1.2,
            staircase=True))
    optimizer.minimize(avg_cost)

    place = fluid.CPUPlace() if args.device == 'CPU' else fluid.CUDAPlace(0)
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    # load checkpoint.
    fluid.io.load_persistables(exe, args.checkpoint)

    # init decoder
    decoder = Decoder(args.trans_model, args.vocabulary, args.graphs,
                      args.log_prior, args.beam_size, args.acoustic_scale)

    ltrans = [
        trans_add_delta.TransAddDelta(2, 2),
        trans_mean_variance_norm.TransMeanVarianceNorm(args.mean_var),
        trans_splice.TransSplice(5, 5), trans_delay.TransDelay(5)
    ]

    feature_t = fluid.LoDTensor()
    label_t = fluid.LoDTensor()

    # infer data reader
    infer_data_reader = reader.AsyncDataReader(
        args.infer_feature_lst, drop_frame_len=-1, split_sentence_threshold=-1)
    infer_data_reader.set_transformers(ltrans)

    decoding_result_writer = DecodingResultWriter(args.decode_to_path)
    post_matrix_writer = None if args.post_matrix_path is None \
                         else PostMatrixWriter(args.post_matrix_path)

    for batch_id, batch_data in enumerate(
            infer_data_reader.batch_iterator(args.batch_size,
                                             args.minimum_batch_size)):
        # load_data
        (features, labels, lod, name_lst) = batch_data
        features = np.reshape(features, (-1, 11, 3, args.frame_dim))
        features = np.transpose(features, (0, 2, 1, 3))
        feature_t.set(features, place)
        feature_t.set_lod([lod])
        label_t.set(labels, place)
        label_t.set_lod([lod])

        results = exe.run(infer_program,
                          feed={"feature": feature_t,
                                "label": label_t},
                          fetch_list=[prediction, avg_cost, accuracy],
                          return_numpy=False)

        probs, lod = lodtensor_to_ndarray(results[0])
        infer_batch = split_infer_result(probs, lod)

        print("Decoding batch %d ..." % batch_id)
        decoded = decoder.decode_batch(name_lst, infer_batch, args.num_threads)

        decoding_result_writer.write(decoded)

        if args.post_matrix_path is not None:
            post_matrix_writer.write(name_lst, infer_batch)


if __name__ == '__main__':
    import paddle
    paddle.enable_static()
    args = parse_args()
    print_arguments(args)

    infer_from_ckpt(args)
