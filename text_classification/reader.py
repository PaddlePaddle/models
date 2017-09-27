import os


def train_reader(data_dir, word_dict, label_dict):
    """
    Reader interface for training data

    :param data_dir: data directory
    :type data_dir: str
    :param word_dict: path of word dictionary,
        the dictionary must has a "UNK" in it.
    :type word_dict: Python dict
    :param label_dict: path of label dictionary
    :type label_dict: Python dict
    """

    def reader():
        UNK_ID = word_dict["<UNK>"]
        word_col = 1
        lbl_col = 0

        for file_name in os.listdir(data_dir):
            with open(os.path.join(data_dir, file_name), "r") as f:
                for line in f:
                    line_split = line.strip().split("\t")
                    word_ids = [
                        word_dict.get(w, UNK_ID)
                        for w in line_split[word_col].split()
                    ]
                    yield word_ids, label_dict[line_split[lbl_col]]

    return reader


def test_reader(data_dir, word_dict):
    """
    Reader interface for testing data

    :param data_dir: data directory.
    :type data_dir: str
    :param word_dict: path of word dictionary,
        the dictionary must has a "UNK" in it.
    :type word_dict: Python dict
    """

    def reader():
        UNK_ID = word_dict["<UNK>"]
        word_col = 1

        for file_name in os.listdir(data_dir):
            with open(os.path.join(data_dir, file_name), "r") as f:
                for line in f:
                    line_split = line.strip().split("\t")
                    if len(line_split) < word_col: continue
                    word_ids = [
                        word_dict.get(w, UNK_ID)
                        for w in line_split[word_col].split()
                    ]
                    yield word_ids, line_split[word_col]

    return reader
