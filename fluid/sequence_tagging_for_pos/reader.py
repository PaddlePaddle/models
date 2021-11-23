# Copyright(c) 2018 PaddlePaddle.  All rights reserved.
# Created on 2018
#
# Author:Lin_Bo
# Version 1.0
# filename: reader.py
import os


def train_reader(data_dir, word_dict, label_dict, window_size=5):
    """
    Reader interface for training data

    :param data_dir: data directory
    :type data_dir: str
    :param word_dict: path of word dictionary,
        the dictionary must has a "UNK" in it.
    :type word_dict: Python dict
    :param label_dict: path of label dictionary
    :type label_dict: Python dict
    :param window_size: the sequence window size
    :type window_size: int
    """

    def reader():
        UNK_WID = word_dict["<UNK>"]
        UNK_LID = label_dict["<UNK>"]
        word_col, lbl_col = 0, 1
        interest_word_window = int(window_size / 2)

        for file_name in os.listdir(data_dir):
            with open(os.path.join(data_dir, file_name), "r") as f:
                for line in f:
                    line_split = line.encode("utf-8").strip().split()

                    ##sentence with a special "PADDING" word
                    #      replicated window_size/2 times at the begining and end
                    # "PADDING" at the begining
                    word_ids, label_ids = [UNK_WID] * interest_word_window, [
                        UNK_LID
                    ] * interest_word_window

                    for item in line_split:
                        try:
                            items = item.split("/")
                            w = word_dict.get(items[word_col].strip(), UNK_WID)
                            l = label_dict.get(items[lbl_col].strip(), UNK_LID)
                            word_ids.append(w)
                            label_ids.append(l)
                        except:
                            continue

                    #"PADDING" at the end
                    word_ids += [UNK_WID] * interest_word_window
                    label_ids += [UNK_LID] * interest_word_window

                    if len(word_ids) < interest_word_window:
                        continue

                    if len(word_ids) < window_size:
                        yield word_ids + [UNK_WID] * (
                            window_size - len(word_ids)
                        ), label_ids[interest_word_window]

                    for i in range(len(word_ids) - window_size):
                        yield word_ids[i:i + window_size], label_ids[
                            i + interest_word_window]

    return reader
