from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import paddle.fluid as fluid
import data_utils.augmentor.trans_mean_variance_norm as trans_mean_variance_norm
import data_utils.augmentor.trans_add_delta as trans_add_delta
import data_utils.augmentor.trans_splice as trans_splice
import data_utils.async_data_reader as reader
from data_utils.util import lodtensor_to_ndarray
from data_utils.util import split_infer_result


def parse_args():
    parser = argparse.ArgumentParser("Inference for stacked LSTMP model.")
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='The sequence number of a batch data. (default: %(default)d)')
    parser.add_argument(
        '--device',
        type=str,
        default='GPU',
        choices=['CPU', 'GPU'],
        help='The device type. (default: %(default)s)')
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
        '--infer_model_path',
        type=str,
        default='./infer_models/deep_asr.pass_0.infer.model/',
        help='The directory for loading inference model. '
        '(default: %(default)s)')
    args = parser.parse_args()
    return args


def print_arguments(args):
    print('-----------  Configuration Arguments -----------')
    for arg, value in sorted(vars(args).iteritems()):
        print('%s: %s' % (arg, value))
    print('------------------------------------------------')


def infer(args):
    """ Gets one batch of feature data and predicts labels for each sample.
    """

    if not os.path.exists(args.infer_model_path):
        raise IOError("Invalid inference model path!")

    place = fluid.CUDAPlace(0) if args.device == 'GPU' else fluid.CPUPlace()
    exe = fluid.Executor(place)

    # load model
    [infer_program, feed_dict,
     fetch_targets] = fluid.io.load_inference_model(args.infer_model_path, exe)

    ltrans = [
        trans_add_delta.TransAddDelta(2, 2),
        trans_mean_variance_norm.TransMeanVarianceNorm(args.mean_var),
        trans_splice.TransSplice()
    ]

    infer_data_reader = reader.AsyncDataReader(args.infer_feature_lst,
                                               args.infer_label_lst)
    infer_data_reader.set_transformers(ltrans)

    feature_t = fluid.LoDTensor()
    one_batch = infer_data_reader.batch_iterator(args.batch_size, 1).next()

    (features, labels, lod) = one_batch
    feature_t.set(features, place)
    feature_t.set_lod([lod])

    results = exe.run(infer_program,
                      feed={feed_dict[0]: feature_t},
                      fetch_list=fetch_targets,
                      return_numpy=False)

    probs, lod = lodtensor_to_ndarray(results[0])
    preds = probs.argmax(axis=1)
    infer_batch = split_infer_result(preds, lod)
    for index, sample in enumerate(infer_batch):
        print("result %d: " % index, sample, '\n')


if __name__ == '__main__':
    import paddle
    paddle.enable_static()
    args = parse_args()
    print_arguments(args)
    infer(args)
