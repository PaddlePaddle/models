"""Trainer for DeepSpeech2 model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import distutils.util
import multiprocessing
import paddle.v2 as paddle
from model import DeepSpeech2Model
from data_utils.data import DataGenerator
import utils


def print_arguments(args):
    print("-----  Configuration Arguments -----")
    for arg, value in vars(args).iteritems():
        print("%s: %s" % (arg, value))
    print("------------------------------------")


def add_arg(argname, type, default, help, **kwargs):
    type = distutils.util.strtobool if type == bool else type
    parser.add_argument(
        "--" + argname,
        default=default,
        type=type,
        help=help + ' Default: %(default)s.',
        **kwargs)


NUM_CPU = multiprocessing.cpu_count() // 2

parser = argparse.ArgumentParser(description=__doc__)
# yapf: disable
add_arg('batch_size',       int,    256,    "Minibatch size.")
add_arg('num_passes',       int,    200,    "# of training epochs.")
add_arg('num_iter_print',   int,    100,    "Every # iterations for printing "
                                            "train cost.")
add_arg('batch_size',       int,    256,    "Minibatch size.")
add_arg('num_conv_layers',  int,    2,      "# of convolution layers.")
add_arg('num_rnn_layers',   int,    3,      "# of recurrent layers.")
add_arg('rnn_layer_size',   int,    2048,   "# of recurrent cells per layer.")
# yapf: disable
add_arg('share_weights',    bool,   True,   "Share input-hidden weights across "
                                            "bi-directional RNNs. Not for GRU.")
add_arg('use_gru',          bool,   False,  "Use GRUs instead of Simple RNNs.")
add_arg('learning_rate',    float,  5e-4,   "Learning rate.")
add_arg('use_gpu',          bool,   True,   "Use GPU or not.")
add_arg('use_sortagrad',    bool,   True,   "Use SortaGrad or not.")
add_arg('specgram_type',    str,    'linear',
    "Audio Feature type.",
    choices=['linear', 'mfcc'])
add_arg(
    'shuffle_method',
    str,
    'batch_shuffle_clipped',
    "Shuffle method.",
    choices=['instance_shuffle', 'batch_shuffle', 'batch_shuffle_clipped'])
add_arg('max_duration', float, 27.0, "Longest audio duration allowed.")
add_arg('min_duration', float, 0.0, "Shortest audio duration allowed.")
add_arg('trainer_count', int, 8, "# of Trainers (CPUs or GPUs).")
add_arg('parallels_data', int, NUM_CPU, "# of CPUs for data preprocessing.")
add_arg('mean_std_fiepath', str, 'mean_std.npz',
        "Manifest filepath for normalizer's stats.")
parser.add_argument(
    "--train_manifest_path",
    default='datasets/manifest.train',
    type=str,
    help="Manifest path for training. (default: %(default)s)")
parser.add_argument(
    "--dev_manifest_path",
    default='datasets/manifest.dev',
    type=str,
    help="Manifest path for validation. (default: %(default)s)")
parser.add_argument(
    "--vocab_filepath",
    default='datasets/vocab/eng_vocab.txt',
    type=str,
    help="Vocabulary filepath. (default: %(default)s)")
parser.add_argument(
    "--init_model_path",
    default=None,
    type=str,
    help="If set None, the training will start from scratch. "
    "Otherwise, the training will resume from "
    "the existing model of this path. (default: %(default)s)")
parser.add_argument(
    "--output_model_dir",
    default="./checkpoints",
    type=str,
    help="Directory for saving models. (default: %(default)s)")
parser.add_argument(
    "--augmentation_config",
    default=open('conf/augmentation.config', 'r').read(),
    type=str,
    help="Augmentation configuration in json-format. "
    "(default: %(default)s)")
parser.add_argument(
    "--is_local",
    default=True,
    type=distutils.util.strtobool,
    help="Set to false if running with pserver in paddlecloud. "
    "(default: %(default)s)")
args = parser.parse_args()


def train():
    """DeepSpeech2 training."""
    train_generator = DataGenerator(
        vocab_filepath=args.vocab_filepath,
        mean_std_filepath=args.mean_std_filepath,
        augmentation_config=args.augmentation_config,
        max_duration=args.max_duration,
        min_duration=args.min_duration,
        specgram_type=args.specgram_type,
        num_threads=args.parallels_data)
    dev_generator = DataGenerator(
        vocab_filepath=args.vocab_filepath,
        mean_std_filepath=args.mean_std_filepath,
        augmentation_config="{}",
        specgram_type=args.specgram_type,
        num_threads=args.parallels_data)
    train_batch_reader = train_generator.batch_reader_creator(
        manifest_path=args.train_manifest_path,
        batch_size=args.batch_size,
        min_batch_size=args.trainer_count,
        sortagrad=args.use_sortagrad if args.init_model_path is None else False,
        shuffle_method=args.shuffle_method)
    dev_batch_reader = dev_generator.batch_reader_creator(
        manifest_path=args.dev_manifest_path,
        batch_size=args.batch_size,
        min_batch_size=1,  # must be 1, but will have errors.
        sortagrad=False,
        shuffle_method=None)

    ds2_model = DeepSpeech2Model(
        vocab_size=train_generator.vocab_size,
        num_conv_layers=args.num_conv_layers,
        num_rnn_layers=args.num_rnn_layers,
        rnn_layer_size=args.rnn_layer_size,
        use_gru=args.use_gru,
        pretrained_model_path=args.init_model_path,
        share_rnn_weights=args.share_weights)
    ds2_model.train(
        train_batch_reader=train_batch_reader,
        dev_batch_reader=dev_batch_reader,
        feeding_dict=train_generator.feeding,
        learning_rate=args.learning_rate,
        gradient_clipping=400,
        num_passes=args.num_passes,
        num_iterations_print=args.num_iter_print,
        output_model_dir=args.output_model_dir,
        is_local=args.is_local)


def main():
    utils.print_arguments(args)
    paddle.init(use_gpu=args.use_gpu, trainer_count=args.trainer_count)
    train()


if __name__ == '__main__':
    main()
