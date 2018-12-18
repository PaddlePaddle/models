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
    parser.add_argument(
        '--is_local',
        action='store_true',
        required=False,
        default=False,
        help='Local train or not, (default: False)')

    return parser.parse_args()

pattern = re.compile("[^a-z] ")
def text_strip(text):
    return pattern.sub("", text)


def build_Huffman(word_count, max_code_length):

    MAX_CODE_LENGTH = max_code_length
    sorted_by_freq = sorted(word_count.items(), key=lambda x: x[1])
    count = list()
    vocab_size = len(word_count)
    parent = [-1] * 2 * vocab_size
    code = [-1] * MAX_CODE_LENGTH
    point = [-1] * MAX_CODE_LENGTH
    binary = [-1] * 2 * vocab_size
    word_code_len = dict()
    word_code = dict()
    word_point = dict()
    i = 0
    for a in range(vocab_size):
        count.append(word_count[sorted_by_freq[a][0]])

    for a in range(vocab_size):
        word_point[sorted_by_freq[a][0]] = [-1] * MAX_CODE_LENGTH
        word_code[sorted_by_freq[a][0]] = [-1] * MAX_CODE_LENGTH

    for k in range(vocab_size):
        count.append(1e15)

    pos1 = vocab_size - 1
    pos2 = vocab_size
    min1i = 0
    min2i = 0
    b = 0

    for r in range(vocab_size):
        if pos1 >= 0:
            if count[pos1] < count[pos2]:
                min1i = pos1
                pos1 = pos1 - 1
            else:
                min1i = pos2
                pos2 = pos2 + 1
        else:
            min1i = pos2
            pos2 = pos2 + 1
        if pos1 >= 0:
            if count[pos1] < count[pos2]:
                min2i = pos1
                pos1 = pos1 - 1
            else:
                min2i = pos2
                pos2 = pos2 + 1
        else:
            min2i = pos2
            pos2 = pos2 + 1

        count[vocab_size + r] = count[min1i] + count[min2i]

        #record the parent of left and right child
        parent[min1i] = vocab_size + r
        parent[min2i] = vocab_size + r
        binary[min1i] = 0  #left branch has code 0
        binary[min2i] = 1  #right branch has code 1

    for a in range(vocab_size):
        b = a
        i = 0
        while True:
            code[i] = binary[b]
            point[i] = b
            i = i + 1
            b = parent[b]
            if b == vocab_size * 2 - 2:
                break

        word_code_len[sorted_by_freq[a][0]] = i
        word_point[sorted_by_freq[a][0]][0] = vocab_size - 2

        for k in range(i):
            word_code[sorted_by_freq[a][0]][i - k - 1] = code[k]

            # only non-leaf nodes will be count in
            if point[k] - vocab_size >= 0:
                word_point[sorted_by_freq[a][0]][i - k] = point[k] - vocab_size

    return word_point, word_code, word_code_len


def preprocess(data_path, dict_path, freq, is_local):
    """
    proprocess the data, generate dictionary and save into dict_path.
    :param data_path: the input data path.
    :param dict_path: the generated dict path. the data in dict is "word count"
    :param freq:
    :return:
    """
    # word to count
    word_count = dict()

    if is_local:
        for i in range(1, 100):
            with open(data_path + "/news.en-000{:0>2d}-of-00100".format(
                    i)) as f:
                for line in f:
                    line = line.lower()
                    line = text_strip(line)
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

    path_table, path_code, word_code_len = build_Huffman(word_count, 40)

    with open(dict_path, 'w+') as f:
        for k, v in word_count.items():
            f.write(str(k) + " " + str(v) + '\n')

    with open(dict_path + "_ptable", 'w+') as f2:
        for pk, pv in path_table.items():
            f2.write(str(pk) + ":" + ' '.join((str(x) for x in pv)) + '\n')

    with open(dict_path + "_pcode", 'w+') as f3:
        for pck, pcv in path_table.items():
            f3.write(str(pck) + ":" + ' '.join((str(x) for x in pcv)) + '\n')


if __name__ == "__main__":
    args = parse_args()
    preprocess(args.data_path, args.dict_path, args.freq, args.is_local)
