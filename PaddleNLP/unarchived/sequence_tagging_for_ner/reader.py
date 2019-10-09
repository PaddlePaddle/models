#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Conll03 dataset.
"""

from utils import *

__all__ = ["data_reader"]


def canonicalize_digits(word):
    if any([c.isalpha() for c in word]): return word
    word = re.sub("\d", "DG", word)
    if word.startswith("DG"):
        word = word.replace(",", "")  # remove thousands separator
    return word


def canonicalize_word(word, wordset=None, digits=True):
    word = word.lower()
    if digits:
        if (wordset != None) and (word in wordset): return word
        word = canonicalize_digits(word)  # try to canonicalize numbers
    if (wordset == None) or (word in wordset): return word
    else: return "UUUNKKK"  # unknown token


def data_reader(data_file, word_dict, label_dict):
    """
    The dataset can be obtained according to http://www.clips.uantwerpen.be/conll2003/ner/.
    It returns a reader creator, each sample in the reader includes:
    word id sequence, label id sequence and raw sentence.

    :return: reader creator
    :rtype: callable
    """

    def reader():
        UNK_IDX = word_dict["UUUNKKK"]

        sentence = []
        labels = []
        with open(data_file, "r") as f:
            for line in f:
                if len(line.strip()) == 0:
                    if len(sentence) > 0:
                        word_idx = [
                            word_dict.get(
                                canonicalize_word(w, word_dict), UNK_IDX)
                            for w in sentence
                        ]
                        mark = [1 if w[0].isupper() else 0 for w in sentence]
                        label_idx = [label_dict[l] for l in labels]
                        yield word_idx, mark, label_idx
                    sentence = []
                    labels = []
                else:
                    segs = line.strip().split()
                    sentence.append(segs[0])
                    # transform I-TYPE to BIO schema
                    if segs[-1] != "O" and (len(labels) == 0 or
                                            labels[-1][1:] != segs[-1][1:]):
                        labels.append("B" + segs[-1][1:])
                    else:
                        labels.append(segs[-1])

    return reader
