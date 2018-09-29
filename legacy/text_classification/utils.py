import logging
import os
import argparse
from collections import defaultdict

logger = logging.getLogger("paddle")
logger.setLevel(logging.INFO)


def parse_train_cmd():
    parser = argparse.ArgumentParser(
        description="PaddlePaddle text classification example.")
    parser.add_argument(
        "--nn_type",
        type=str,
        help=("A flag that defines which type of network to use, "
              "available: [dnn, cnn]."),
        default="dnn")
    parser.add_argument(
        "--train_data_dir",
        type=str,
        required=False,
        help=("The path of training dataset (default: None). If this parameter "
              "is not set, paddle.dataset.imdb will be used."),
        default=None)
    parser.add_argument(
        "--test_data_dir",
        type=str,
        required=False,
        help=("The path of testing dataset (default: None). If this parameter "
              "is not set, paddle.dataset.imdb will be used."),
        default=None)
    parser.add_argument(
        "--word_dict",
        type=str,
        required=False,
        help=("The path of word dictionary (default: None). If this parameter "
              "is not set, paddle.dataset.imdb will be used. If this parameter "
              "is set, but the file does not exist, word dictionay "
              "will be built from the training data automatically."),
        default=None)
    parser.add_argument(
        "--label_dict",
        type=str,
        required=False,
        help=("The path of label dictionay (default: None).If this parameter "
              "is not set, paddle.dataset.imdb will be used. If this parameter "
              "is set, but the file does not exist, word dictionay "
              "will be built from the training data automatically."),
        default=None)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="The number of training examples in one forward/backward pass.")
    parser.add_argument(
        "--num_passes",
        type=int,
        default=10,
        help="The number of passes to train the model.")
    parser.add_argument(
        "--model_save_dir",
        type=str,
        required=False,
        help=("The path to save the trained models."),
        default="models")

    return parser.parse_args()


def build_dict(data_dir,
               save_path,
               use_col=0,
               cutoff_fre=0,
               insert_extra_words=[]):
    values = defaultdict(int)

    for file_name in os.listdir(data_dir):
        file_path = os.path.join(data_dir, file_name)
        if not os.path.isfile(file_path):
            continue
        with open(file_path, "r") as fdata:
            for line in fdata:
                line_splits = line.strip().split("\t")
                if len(line_splits) < use_col: continue
                for w in line_splits[use_col].split():
                    values[w] += 1

    with open(save_path, "w") as f:
        for w in insert_extra_words:
            f.write("%s\t-1\n" % (w))

        for v, count in sorted(
                values.iteritems(), key=lambda x: x[1], reverse=True):
            if count < cutoff_fre:
                break
            f.write("%s\t%d\n" % (v, count))


def load_dict(dict_path):
    return dict((line.strip().split("\t")[0], idx)
                for idx, line in enumerate(open(dict_path, "r").readlines()))


def load_reverse_dict(dict_path):
    return dict((idx, line.strip().split("\t")[0])
                for idx, line in enumerate(open(dict_path, "r").readlines()))
