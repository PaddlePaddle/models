# -*- coding: utf-8 -*

import re
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Paddle Fluid word2 vector preprocess")
    parser.add_argument(
        '--data_path',
        type=str,
        required=True,
        help="The path of training dataset")
    parser.add_argument(
        '--dict_path',
        type=str,
        default='./dict',
        help="The path of generated dict")
    parser.add_argument(
        '--freq',
        type=int,
        default=5,
        help="If the word count is less then freq, it will be removed from dict")

    return parser.parse_args()


def preprocess(data_path, dict_path, freq):
    """
    proprocess the data, generate dictionary and save into dict_path.
    :param data_path: the input data path.
    :param dict_path: the generated dict path. the data in dict is "word count"
    :param freq:
    :return:
    """
    # word to count
    word_count = dict()

    with open(data_path) as f:
        for line in f:
            line = line.lower()
            line = re.sub("[^a-z ]", "", line)
            words = line.split()
            for item in words:
                if item in word_count:
                    word_count[item] = word_count[item] + 1
                else:
                    word_count[item] = 1
    item_to_remove = []
    for item in word_count:
        if word_count[item] <= freq:
            item_to_remove.append(item)
    for item in item_to_remove:
        del word_count[item]

    with open(dict_path, 'w+') as f:
        for k, v in word_count.items():
            f.write(str(k) + " " + str(v) + '\n')


if __name__ == "__main__":
    args = parse_args()
    preprocess(args.data_path, args.dict_path, args.freq)
