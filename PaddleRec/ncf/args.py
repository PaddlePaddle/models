import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run GMF.")
    parser.add_argument('--path', nargs='?', default='Data/',  help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='ml-1m', help='Choose a dataset.')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size.')
    parser.add_argument('--test_batch_size', type=int, default=100, help='Batch size.')
    parser.add_argument('--num_factors', type=int, default=8, help='Embedding size.')
    parser.add_argument('--num_users', type=int, default=6040, help='num_users')
    parser.add_argument('--num_items', type=int, default=3706, help='num_users')
    parser.add_argument('--num_neg', type=int, default=4, help='Number of negative instances to pair with a positive instance.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--train_data_path', type=str, default="Data/train_data.csv", help='train_data_path')
    parser.add_argument('--test_data_path', type=str, default="Data/test.txt", help='train_data_path')
    parser.add_argument('--model_dir', type=str, default="model_dir", help='model_dir.')
    parser.add_argument('--use_gpu', type=int, default=0, help='use_gpu')
    parser.add_argument('--GMF', type=int, default=0, help='GMF')
    parser.add_argument('--MLP', type=int, default=0, help='MLP')
    parser.add_argument('--NeuMF', type=int, default=0, help='NeuMF')
    parser.add_argument('--layers', nargs='?', default=[64,32,16,8],
                        help="MLP layers. Note that the first layer is the concatenation of user and item embeddings. So layers[0]/2 is the embedding size.")
    return parser.parse_args()