import os
import logging
from collections import defaultdict

logger = logging.getLogger("paddle")
logger.setLevel(logging.INFO)


def build_word_dict(data_dir, save_path, use_col=1, cutoff_fre=1):
    """
    Build word dictionary from training data.
    :param data_dir: The directory of training dataset.
    :type data_dir: str
    :params save_path: The path where the word dictionary will be saved.
    :type save_path: str
    :params use_col: The index of text juring line split.
    :type use_col: int
    :params cutoff_fre: The word will not be added to dictionary if it's
                    frequency is less than cutoff_fre.
    :type cutoff_fre: int
    """
    values = defaultdict(int)

    for file_name in os.listdir(data_dir):
        file_path = os.path.join(data_dir, file_name)
        if not os.path.isfile(file_path):
            continue
        with open(file_path, "r") as fdata:
            for line in fdata:
                line_splits = line.strip().split("\t")
                if len(line_splits) < use_col:
                    continue
                doc = line_splits[use_col]
                for sent in doc.strip().split("."):
                    for w in sent.split():
                        values[w] += 1

    values['<unk>'] = cutoff_fre
    with open(save_path, "w") as f:
        for v, count in sorted(
                values.iteritems(), key=lambda x: x[1], reverse=True):
            if count < cutoff_fre:
                break
            f.write("%s\t%d\n" % (v, count))


def build_label_dict(data_dir, save_path, use_col=0):
    """
    Build label dictionary from training data.
    :param data_dir: The directory of training dataset.
    :type data_dir: str
    :params save_path: The path where the label dictionary will be saved.
    :type save_path: str
    :params use_col: The index of label juring line split.
    :type use_col: int
    """
    values = defaultdict(int)

    for file_name in os.listdir(data_dir):
        file_path = os.path.join(data_dir, file_name)
        if not os.path.isfile(file_path):
            continue
        with open(file_path, "r") as fdata:
            for line in fdata:
                line_splits = line.strip().split("\t")
                if len(line_splits) < use_col:
                    continue
                values[line_splits[use_col]] += 1

    with open(save_path, "w") as f:
        for v, count in sorted(
                values.iteritems(), key=lambda x: x[1], reverse=True):
            f.write("%s\t%d\n" % (v, count))


def load_dict(dict_path):
    """
    Load word dictionary from dictionary path.
    :param dict_path: The path of word dictionary.
    :type data_dir: str
    """
    return dict((line.strip().split("\t")[0], idx)
                for idx, line in enumerate(open(dict_path, "r").readlines()))


def load_reverse_dict(dict_path):
    """
    Load the reversed word dictionary from dictionary path.
    Index of each word is saved in key of the dictionary and the
    corresponding word saved in value of the dictionary.
    :param dict_path: The path of word dictionary.
    :type data_dir: str
    """
    return dict((idx, line.strip().split("\t")[0])
                for idx, line in enumerate(open(dict_path, "r").readlines()))
