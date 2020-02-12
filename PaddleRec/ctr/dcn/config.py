import argparse
"""
global params
"""


def boolean_string(s):
    if s.lower() not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s.lower() == 'true'


def parse_args():
    parser = argparse.ArgumentParser(description="PaddleFluid DCN demo")
    parser.add_argument(
        '--train_data_dir',
        type=str,
        default='data/train',
        help='The path of train data')
    parser.add_argument(
        '--test_valid_data_dir',
        type=str,
        default='data/test_valid',
        help='The path of test and valid data')
    parser.add_argument(
        '--vocab_dir',
        type=str,
        default='data/vocab',
        help='The path of generated vocabs')
    parser.add_argument(
        '--cat_feat_num',
        type=str,
        default='data/cat_feature_num.txt',
        help='The path of generated cat_feature_num.txt')
    parser.add_argument(
        '--batch_size', type=int, default=512, help="Batch size")
    parser.add_argument(
        '--steps',
        type=int,
        default=150000,
        help="Early stop steps in training. If set, num_epoch will not work")
    parser.add_argument('--num_epoch', type=int, default=2, help="train epoch")
    parser.add_argument(
        '--model_output_dir',
        type=str,
        default='models',
        help='The path for model to store')
    parser.add_argument(
        '--num_thread', type=int, default=20, help='The number of threads')
    parser.add_argument('--test_epoch', type=str, default='1')
    parser.add_argument(
        '--dnn_hidden_units',
        nargs='+',
        type=int,
        default=[1024, 1024],
        help='DNN layers and hidden units')
    parser.add_argument(
        '--cross_num',
        type=int,
        default=6,
        help='The number of Cross network layers')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument(
        '--l2_reg_cross',
        type=float,
        default=1e-5,
        help='Cross net l2 regularizer coefficient')
    parser.add_argument(
        '--use_bn',
        type=boolean_string,
        default=True,
        help='Whether use batch norm in dnn part')
    parser.add_argument(
        '--is_sparse',
        action='store_true',
        required=False,
        default=False,
        help='embedding will use sparse or not, (default: False)')
    parser.add_argument(
        '--clip_by_norm', type=float, default=100.0, help="gradient clip norm")
    parser.add_argument('--print_steps', type=int, default=100)
    parser.add_argument(
        '--enable_ce', action='store_true', help='If set, run the task with continuous evaluation logs.')

    return parser.parse_args()
