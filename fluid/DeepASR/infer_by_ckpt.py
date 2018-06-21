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
from decoder.post_latgen_faster_mapped import Decoder
from data_utils.util import lodtensor_to_ndarray
from model_utils.model import stacked_lstmp_model
from data_utils.util import split_infer_result
from tools.error_rate import char_errors


def parse_args():
    parser = argparse.ArgumentParser("Run inference by using checkpoint.")
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='The sequence number of a batch data. (default: %(default)d)')
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
        '--learning_rate',
        type=float,
        default=0.00016,
        help='Learning rate used to train. (default: %(default)f)')
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
        default='data/global_mean_var_search26kHr',
        help="The path for feature's global mean and variance. "
        "(default: %(default)s)")
    parser.add_argument(
        '--infer_feature_lst',
        type=str,
        default='data/infer_feature.lst',
        help='The feature list path for inference. (default: %(default)s)')
    parser.add_argument(
        '--infer_label_lst',
        type=str,
        default='data/infer_label.lst',
        help='The label list path for inference. (default: %(default)s)')
    parser.add_argument(
        '--ref_txt',
        type=str,
        default='data/text.test',
        help='The reference text for decoding. (default: %(default)s)')
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
        '--target_trans',
        type=str,
        default="./decoder/target_trans.txt",
        help="The path to target transcription. (default: %(default)s)")
    parser.add_argument(
        '--post_matrix_path',
        type=str,
        default=None,
        help="The path to output post prob matrix. (default: %(default)s)")
    args = parser.parse_args()
    return args


def print_arguments(args):
    print('-----------  Configuration Arguments -----------')
    for arg, value in sorted(vars(args).iteritems()):
        print('%s: %s' % (arg, value))
    print('------------------------------------------------')


def get_trg_trans(args):
    trans_dict = {}
    with open(args.target_trans) as trg_trans:
        line = trg_trans.readline()
        while line:
            items = line.strip().split()
            key = items[0]
            trans_dict[key] = ''.join(items[1:])
            line = trg_trans.readline()
    return trans_dict


def out_post_matrix(key, prob):
    with open(args.post_matrix_path, "a") as post_matrix:
        post_matrix.write(key + " [\n")
        for i in range(prob.shape[0]):
            for j in range(prob.shape[1]):
                post_matrix.write(str(prob[i][j]) + " ")
            post_matrix.write("\n")
        post_matrix.write("]\n")


def infer_from_ckpt(args):
    """Inference by using checkpoint."""

    if not os.path.exists(args.checkpoint):
        raise IOError("Invalid checkpoint!")

    prediction, avg_cost, accuracy = stacked_lstmp_model(
        frame_dim=args.frame_dim,
        hidden_dim=args.hidden_dim,
        proj_dim=args.proj_dim,
        stacked_num=args.stacked_num,
        class_num=args.class_num,
        parallel=args.parallel)

    infer_program = fluid.default_main_program().clone()

    optimizer = fluid.optimizer.Adam(learning_rate=args.learning_rate)
    optimizer.minimize(avg_cost)

    place = fluid.CPUPlace() if args.device == 'CPU' else fluid.CUDAPlace(0)
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    #trg_trans = get_trg_trans(args)
    # load checkpoint.
    fluid.io.load_persistables(exe, args.checkpoint)

    # init decoder
    decoder = Decoder(args.trans_model, args.vocabulary, args.graphs,
                      args.log_prior, args.acoustic_scale)

    ltrans = [
        trans_add_delta.TransAddDelta(2, 2),
        trans_mean_variance_norm.TransMeanVarianceNorm(args.mean_var),
        trans_splice.TransSplice(5, 5), trans_delay.TransDelay(5)
    ]

    feature_t = fluid.LoDTensor()
    label_t = fluid.LoDTensor()

    # infer data reader
    infer_data_reader = reader.AsyncDataReader(
        args.infer_feature_lst,
        args.infer_label_lst,
        drop_frame_len=-1,
        split_sentence_threshold=-1)
    infer_data_reader.set_transformers(ltrans)
    infer_costs, infer_accs = [], []
    total_edit_dist, total_ref_len = 0.0, 0
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
        infer_costs.append(lodtensor_to_ndarray(results[1])[0])
        infer_accs.append(lodtensor_to_ndarray(results[2])[0])

        probs, lod = lodtensor_to_ndarray(results[0])
        infer_batch = split_infer_result(probs, lod)

        decoded = decoder.decode_batch(name_lst, infer_batch, args.num_threads)
        for res in decoded:
            print(res.encode("utf8"))
        if args.post_matrix_path is not None:
            for index, sample in enumerate(infer_batch):
                key = name_lst[index]
                out_post_matrix(key, sample)
            '''
            hyp = decoder.decode(key, sample)
            edit_dist, ref_len = char_errors(ref.decode("utf8"), hyp)
            total_edit_dist += edit_dist
            total_ref_len += ref_len
            print(key + "|Ref:", ref)
            print(key + "|Hyp:", hyp.encode("utf8"))
            print("Instance CER: ", edit_dist / ref_len)
            '''
        #print("batch: ", batch_id)

        #print("Total CER = %f" % (total_edit_dist / total_ref_len))


if __name__ == '__main__':
    args = parse_args()
    print_arguments(args)

    infer_from_ckpt(args)
