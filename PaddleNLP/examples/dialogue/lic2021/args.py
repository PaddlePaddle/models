import argparse


def parse_args():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "--save_dir",
        default="./checkpoints",
        type=str,
        help="The directory where the checkpoints will be saved.")
    parser.add_argument(
        "--output_dir",
        default="./output",
        type=str,
        help="The directory where the infer result will be saved.")
    parser.add_argument(
        "--train_data_path",
        type=str,
        default="./datasets/train.shuffle.txt",
        help="Specify the path to load train data.")
    parser.add_argument(
        "--valid_data_path",
        type=str,
        default="./datasets/valid.txt",
        help="Specify the path to load valid data.")
    parser.add_argument(
        "--test_data_path",
        type=str,
        default="./datasets/test.txt",
        help="Specify the path to load test data.")
    parser.add_argument(
        "--vocab_file", default=None, type=str, help="The vocabulary filepath.")
    parser.add_argument(
        "--init_from_ckpt",
        default=None,
        type=str,
        help="The path of checkpoint to be loaded.")
    parser.add_argument(
        "--logging_steps",
        default=500,
        type=int,
        help="Log every X updates steps.")
    parser.add_argument(
        "--save_steps",
        default=8000,
        type=int,
        help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--seed", default=11, type=int, help="Random seed for initialization.")
    parser.add_argument(
        "--n_gpus",
        default=1,
        type=int,
        help="The number of gpus to use, 0 for cpu.")
    parser.add_argument(
        "--batch_size",
        default=8192,
        type=int,
        help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--infer_batch_size",
        default=4,
        type=int,
        help="Batch size per GPU/CPU for infer.")
    parser.add_argument(
        "--lr", default=1e-5, type=float, help="The initial learning rate.")
    parser.add_argument(
        "--weight_decay",
        default=0.01,
        type=float,
        help="The weight decay for optimizer.")
    parser.add_argument(
        "--epochs",
        default=10,
        type=int,
        help="Total number of training epochs to perform.")
    parser.add_argument(
        "--warmup_steps",
        default=4000,
        type=float,
        help="The number of warmup steps.")
    parser.add_argument(
        '--max_grad_norm',
        default=0.1,
        type=float,
        help='The max value of grad norm.')
    parser.add_argument(
        '--num_layers',
        default=12,
        type=int,
        help='The number of layers in Transformer encoder.')
    parser.add_argument(
        '--d_model',
        default=768,
        type=int,
        help='The expected feature size in the Transformer input and output.')
    parser.add_argument(
        '--nhead',
        default=12,
        type=int,
        help='The number of heads in multi-head attention(MHA).')
    parser.add_argument(
        '--dropout',
        default=0.1,
        type=float,
        help='The dropout probability used in network.')
    parser.add_argument(
        '--activation',
        default='gelu',
        type=str,
        help='The activation function in the feedforward network.')
    parser.add_argument(
        '--normalize_before',
        default=True,
        type=eval,
        help='whether to put layer normalization into preprocessing of MHA and FFN sub-layers.'
    )
    parser.add_argument(
        '--type_size', default=2, type=int, help='The number of input type.')
    parser.add_argument(
        '--max_seq_len',
        default=512,
        type=int,
        help='The max length of input sequence.')
    parser.add_argument(
        '--sort_pool_size',
        default=65536,
        type=int,
        help='The pool size for sort in build batch data.')
    parser.add_argument(
        '--topk',
        default=5,
        type=int,
        help='The number of highest probability vocabulary tokens to keep for top-k sampling.'
    )
    parser.add_argument(
        '--min_dec_len',
        default=1,
        type=int,
        help='The minimum sequence length of generation.')
    parser.add_argument(
        '--max_dec_len',
        default=64,
        type=int,
        help='The maximum sequence length of generation.')
    parser.add_argument(
        '--num_samples',
        default=20,
        type=int,
        help='The decode numbers in generation.')

    args = parser.parse_args()
    return args


def print_args(args):
    print('-----------  Configuration Arguments -----------')
    for arg, value in sorted(vars(args).items()):
        print('%s: %s' % (arg, value))
    print('------------------------------------------------')
