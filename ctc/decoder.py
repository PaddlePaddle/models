"""Contains various CTC decoders."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from itertools import groupby
import numpy as np


def ctc_greedy_decoder(probs_seq, vocabulary):
    """CTC greedy (best path) decoder.
    Path consisting of the most probable tokens are further post-processed to
    remove consecutive repetitions and all blanks.
    :param probs_seq: 2-D list of probabilities over the vocabulary for each
                      character. Each element is a list of float probabilities
                      for one character.
    :type probs_seq: list
    :param vocabulary: Vocabulary list.
    :type vocabulary: list
    :return: Decoding result string.
    :rtype: baseline
    """
    # dimension verification
    for probs in probs_seq:
        if not len(probs) == len(vocabulary) + 1:
            raise ValueError("probs_seq dimension mismatchedd with vocabulary")
    # argmax to get the best index for each time step
    max_index_list = list(np.array(probs_seq).argmax(axis=1))
    # remove consecutive duplicate indexes
    index_list = [index_group[0] for index_group in groupby(max_index_list)]
    # remove blank indexes
    blank_index = len(vocabulary)
    index_list = [index for index in index_list if index != blank_index]
    # convert index list to string
    return ''.join([vocabulary[index] for index in index_list])
