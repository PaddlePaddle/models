# -*- coding: utf-8 -*
import os
import random
import re
import six
import argparse
import io
import math
prog = re.compile("[^a-z ]", flags=0)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Paddle Fluid word2 vector preprocess")
    parser.add_argument(
        '--build_dict_corpus_dir', type=str, help="The dir of corpus")
    parser.add_argument(
        '--input_corpus_dir', type=str, help="The dir of input corpus")
    parser.add_argument(
        '--output_corpus_dir', type=str, help="The dir of output corpus")
    parser.add_argument(
        '--dict_path',
        type=str,
        default='./dict',
        help="The path of dictionary ")
    parser.add_argument(
        '--min_count',
        type=int,
        default=5,
        help="If the word count is less then min_count, it will be removed from dict"
    )
    parser.add_argument(
        '--downsample',
        type=float,
        default=0.001,
        help="filter word by downsample")
    parser.add_argument(
        '--filter_corpus',
        action='store_true',
        default=False,
        help='Filter corpus')
    parser.add_argument(
        '--build_dict',
        action='store_true',
        default=False,
        help='Build dict from corpus')
    return parser.parse_args()


def text_strip(text):
    #English Preprocess Rule
    return prog.sub("", text.lower())


# Shameless copy from Tensorflow https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/data_generators/text_encoder.py
# Unicode utility functions that work with Python 2 and 3
def native_to_unicode(s):
    if _is_unicode(s):
        return s
    try:
        return _to_unicode(s)
    except UnicodeDecodeError:
        res = _to_unicode(s, ignore_errors=True)
        return res


def _is_unicode(s):
    if six.PY2:
        if isinstance(s, unicode):
            return True
    else:
        if isinstance(s, str):
            return True
    return False


def _to_unicode(s, ignore_errors=False):
    if _is_unicode(s):
        return s
    error_mode = "ignore" if ignore_errors else "strict"
    return s.decode("utf-8", errors=error_mode)


def filter_corpus(args):
    """
    filter corpus and convert id.
    """
    word_count = dict()
    word_to_id_ = dict()
    word_all_count = 0
    id_counts = []
    word_id = 0
    #read dict
    with io.open(args.dict_path, 'r', encoding='utf-8') as f:
        for line in f:
            word, count = line.split()[0], int(line.split()[1])
            word_count[word] = count
            word_to_id_[word] = word_id
            word_id += 1
            id_counts.append(count)
            word_all_count += count

    #write word2id file
    print("write word2id file to : " + args.dict_path + "_word_to_id_")
    with io.open(
            args.dict_path + "_word_to_id_", 'w+', encoding='utf-8') as fid:
        for k, v in word_to_id_.items():
            fid.write(k + " " + str(v) + '\n')
    #filter corpus and convert id
    if not os.path.exists(args.output_corpus_dir):
        os.makedirs(args.output_corpus_dir)
    for file in os.listdir(args.input_corpus_dir):
        with io.open(args.output_corpus_dir + '/convert_' + file, "w") as wf:
            with io.open(
                    args.input_corpus_dir + '/' + file, encoding='utf-8') as rf:
                print(args.input_corpus_dir + '/' + file)
                for line in rf:
                    signal = False
                    line = text_strip(line)
                    words = line.split()
                    for item in words:
                        if item in word_count:
                            idx = word_to_id_[item]
                        else:
                            idx = word_to_id_[native_to_unicode('<UNK>')]
                        count_w = id_counts[idx]
                        corpus_size = word_all_count
                        keep_prob = (
                            math.sqrt(count_w /
                                      (args.downsample * corpus_size)) + 1
                        ) * (args.downsample * corpus_size) / count_w
                        r_value = random.random()
                        if r_value > keep_prob:
                            continue
                        wf.write(_to_unicode(str(idx) + " "))
                        signal = True
                    if signal:
                        wf.write(_to_unicode("\n"))


def build_dict(args):
    """
    proprocess the data, generate dictionary and save into dict_path.
    :param corpus_dir: the input data dir.
    :param dict_path: the generated dict path. the data in dict is "word count"
    :param min_count:
    :return:
    """
    # word to count

    word_count = dict()

    for file in os.listdir(args.build_dict_corpus_dir):
        with io.open(
                args.build_dict_corpus_dir + "/" + file, encoding='utf-8') as f:
            print("build dict : ", args.build_dict_corpus_dir + "/" + file)
            for line in f:
                line = text_strip(line)
                words = line.split()
                for item in words:
                    if item in word_count:
                        word_count[item] = word_count[item] + 1
                    else:
                        word_count[item] = 1

    item_to_remove = []
    for item in word_count:
        if word_count[item] <= args.min_count:
            item_to_remove.append(item)

    unk_sum = 0
    for item in item_to_remove:
        unk_sum += word_count[item]
        del word_count[item]
    #sort by count
    word_count[native_to_unicode('<UNK>')] = unk_sum
    word_count = sorted(
        word_count.items(), key=lambda word_count: -word_count[1])

    with io.open(args.dict_path, 'w+', encoding='utf-8') as f:
        for k, v in word_count:
            f.write(k + " " + str(v) + '\n')


if __name__ == "__main__":
    args = parse_args()
    if args.build_dict:
        build_dict(args)
    elif args.filter_corpus:
        filter_corpus(args)
    else:
        print(
            "error command line, please choose --build_dict or --filter_corpus")
