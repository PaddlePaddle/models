import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="PaddlePaddle CTR example")
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
        default=30,
        help="The number of epochs to train (default: 10)")
    parser.add_argument(
        '--model_output_dir',
        type=str,
        required=True,
        help='The path for model to store (default: models)')
    parser.add_argument(
        '--num_thread',
        type=int,
        default=10,
        help='The number of threads (default: 10)')
    parser.add_argument('--test_epoch', type=str, default='1')
    parser.add_argument(
        '--layer_sizes',
        nargs='+',
        type=int,
        default=[400, 400, 400],
        help='The size of each layers (default: [400, 400, 400])')
    parser.add_argument(
        '--act',
        type=str,
        default='relu',
        help='The activation of each layers (default: relu)')
    parser.add_argument(
        '--lr', type=float, default=1e-4, help='Learning rate (default: 1e-4)')
    parser.add_argument(
        '--reg', type=float, default=1e-4, help=' (default: 1e-4)')
    parser.add_argument('--num_field', type=int, default=39)
    parser.add_argument('--num_feat', type=int, default=1086460)  # 2090493

    return parser.parse_args()
