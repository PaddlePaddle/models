import argparse

def parse_args():
        parser = argparse.ArgumentParser(description="PaddlePaddle CTR example")
        parser.add_argument(
            '--train_data_path',
            type=str,
            default='./data/raw/train.txt',
            help="The path of training dataset")
        parser.add_argument(
            '--test_data_path',
            type=str,
            default='./data/raw/valid.txt',
            help="The path of testing dataset")
        parser.add_argument(
            '--batch_size',
            type=int,
            default=1000,
            help="The size of mini-batch (default:1000)")
        parser.add_argument(
            '--embedding_size',
            type=int,
            default=16,
            help="The size for embedding layer (default:10)")
        parser.add_argument(
            '--num_passes',
            type=int,
            default=10,
            help="The number of passes to train (default: 10)")
        parser.add_argument(
            '--model_output_dir',
            type=str,
            default='models',
            help='The path for model to store (default: models)')
        parser.add_argument(
            '--sparse_feature_dim',
            type=int,
            default=1000001,
            help='sparse feature hashing space for index processing')
        parser.add_argument(
            '--is_local',
            type=int,
            default=1,
            help='Local train or distributed train (default: 1)')
        parser.add_argument(
            '--cloud_train',
            type=int,
            default=0,
            help='Local train or distributed train on paddlecloud (default: 0)')
        parser.add_argument(
            '--async_mode',
            action='store_true',
            default=False,
            help='Whether start pserver in async mode to support ASGD')
        parser.add_argument(
            '--no_split_var',
            action='store_true',
            default=False,
            help='Whether split variables into blocks when update_method is pserver')
        parser.add_argument(
            '--role',
            type=str,
            default='pserver', # trainer or pserver
            help='The path for model to store (default: models)')
        parser.add_argument(
            '--endpoints',
            type=str,
            default='127.0.0.1:6000',
            help='The pserver endpoints, like: 127.0.0.1:6000,127.0.0.1:6001')
        parser.add_argument(
            '--current_endpoint',
            type=str,
            default='127.0.0.1:6000',
            help='The path for model to store (default: 127.0.0.1:6000)')
        parser.add_argument(
            '--trainer_id',
            type=int,
            default=0,
            help='The path for model to store (default: models)')
        parser.add_argument(
            '--trainers',
            type=int,
            default=1,
            help='The num of trianers, (default: 1)')
        return parser.parse_args()
