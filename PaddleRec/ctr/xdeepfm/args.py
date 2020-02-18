import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="PaddlePaddle CTR example")
    parser.add_argument(
        '--train_data_dir',
        type=str,
        default='data/train_data',
        help='The path of train data (default: data/train_data)')
    parser.add_argument(
        '--test_data_dir',
        type=str,
        default='data/test_data',
        help='The path of test data (default: models)')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=100,
        help="The size of mini-batch (default:100)")
    parser.add_argument(
        '--embedding_size',
        type=int,
        default=10,
        help="The size for embedding layer (default:10)")
    parser.add_argument(
        '--num_epoch',
        type=int,
        default=10,
        help="The number of epochs to train (default: 10)")
    parser.add_argument(
        '--model_output_dir',
        type=str,
        required=True,
        help='The path for model to store (default: models)')
    parser.add_argument(
        '--num_thread',
        type=int,
        default=1,
        help='The number of threads (default: 10)')
    parser.add_argument('--test_epoch', type=str, default='1')
    parser.add_argument(
        '--layer_sizes_dnn',
        nargs='+',
        type=int,
        default=[10, 10, 10],
        help='The size of each layers')
    parser.add_argument(
        '--layer_sizes_cin',
        nargs='+',
        type=int,
        default=[10, 10],
        help='The size of each layers')
    parser.add_argument(
        '--act',
        type=str,
        default='relu',
        help='The activation of each layers (default: relu)')
    parser.add_argument(
        '--lr', type=float, default=1e-1, help='Learning rate (default: 1e-4)')
    parser.add_argument(
        '--reg', type=float, default=1e-4, help=' (default: 1e-4)')
    parser.add_argument('--num_field', type=int, default=39)
    parser.add_argument('--num_feat', type=int, default=28651)
    parser.add_argument(
        '--model_name',
        type=str,
        default='ctr_xdeepfm_model',
        help='The name of model (default: ctr_xdeepfm_model)')
    parser.add_argument('--use_gpu', type=int, default=1)
    parser.add_argument('--print_steps', type=int, default=50)
    parser.add_argument(
        '--is_sparse',
        action='store_true',
        required=False,
        default=False,
        help='embedding will use sparse or not, (default: False)')
    parser.add_argument(
        '--enable_ce', action='store_true', help='If set, run the task with continuous evaluation logs.')

    return parser.parse_args()
