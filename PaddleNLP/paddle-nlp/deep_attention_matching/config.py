"""
Deep Attention Matching Network
"""

import argparse
import six

def parse_args():
    """
    Deep Attention Matching Network Config
    """
    parser = argparse.ArgumentParser("DAM Config")

    parser.add_argument(
        '--do_train', 
        type=bool, 
        default=False, 
        help='Whether to perform training.')
    parser.add_argument(
        '--do_test', 
        type=bool, 
        default=False, 
        help='Whether to perform training.')

    parser.add_argument(
        '--batch_size',
        type=int,
        default=256,
        help='Batch size for training. (default: %(default)d)')
    parser.add_argument(
        '--num_scan_data',
        type=int,
        default=2,
        help='Number of pass for training. (default: %(default)d)')
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=1e-3,
        help='Learning rate used to train. (default: %(default)f)')
    parser.add_argument(
        '--data_path',
        type=str,
        default="data/data_small.pkl",
        help='Path to training data. (default: %(default)s)')
    parser.add_argument(
        '--save_path',
        type=str,
        default="saved_models",
        help='Path to save trained models. (default: %(default)s)')
    parser.add_argument(
        '--model_path',
        type=str,
        default=None,
        help='Path to load well-trained models. (default: %(default)s)')
    parser.add_argument(
        '--use_cuda',
        action='store_true',
        help='If set, use cuda for training.')
    parser.add_argument(
        '--use_pyreader',
        action='store_true',
        help='If set, use pyreader for reading data.')
    parser.add_argument(
        '--ext_eval',
        action='store_true',
        help='If set, use MAP, MRR ect for evaluation.')
    parser.add_argument(
        '--max_turn_num',
        type=int,
        default=9,
        help='Maximum number of utterances in context.')
    parser.add_argument(
        '--max_turn_len',
        type=int,
        default=50,
        help='Maximum length of setences in turns.')
    parser.add_argument(
        '--word_emb_init',
        type=str,
        default=None,
        help='Path to the initial word embedding.')
    parser.add_argument(
        '--vocab_size',
        type=int,
        default=434512,
        help='The size of vocabulary.')
    parser.add_argument(
        '--emb_size',
        type=int,
        default=200,
        help='The dimension of word embedding.')
    parser.add_argument(
        '--_EOS_',
        type=int,
        default=28270,
        help='The id for the end of sentence in vocabulary.')
    parser.add_argument(
        '--stack_num',
        type=int,
        default=5,
        help='The number of stacked attentive modules in network.')
    parser.add_argument(
        '--channel1_num',
        type=int,
        default=32,
        help="The channels' number of the 1st conv3d layer's output.")
    parser.add_argument(
        '--channel2_num',
        type=int,
        default=16,
        help="The channels' number of the 2nd conv3d layer's output.")
    args = parser.parse_args()
    return args


def print_arguments(args):
    """
    Print Config
    """
    print('-----------  Configuration Arguments -----------')
    for arg, value in sorted(six.iteritems(vars(args))):
        print('%s: %s' % (arg, value))
    print('------------------------------------------------')
