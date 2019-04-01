# -*- coding: utf-8 -*

import numpy as np
import preprocess
import logging
import math
import random
import io

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("fluid")
logger.setLevel(logging.INFO)


class NumpyRandomInt(object):
    def __init__(self, a, b, buf_size=1000):
        self.idx = 0
        self.buffer = np.random.random_integers(a, b, buf_size)
        self.a = a
        self.b = b

    def __call__(self):
        if self.idx == len(self.buffer):
            self.buffer = np.random.random_integers(self.a, self.b,
                                                    len(self.buffer))
            self.idx = 0

        result = self.buffer[self.idx]
        self.idx += 1
        return result


class Word2VecReader(object):
    def __init__(self,
                 dict_path,
                 data_path,
                 filelist,
                 trainer_id,
                 trainer_num,
                 window_size=5):
        self.window_size_ = window_size
        self.data_path_ = data_path
        self.filelist = filelist
        self.word_to_id_ = dict()
        self.id_to_word = dict()
        self.word_count = dict()
        self.trainer_id = trainer_id
        self.trainer_num = trainer_num

        word_all_count = 0
        id_counts = []
        word_id = 0

        with io.open(dict_path, 'r', encoding='utf-8') as f:
            for line in f:
                word, count = line.split()[0], int(line.split()[1])
                self.word_count[word] = count
                self.word_to_id_[word] = word_id
                self.id_to_word[word_id] = word  #build id to word dict
                word_id += 1
                id_counts.append(count)
                word_all_count += count

        self.word_all_count = word_all_count
        self.corpus_size_ = word_all_count
        self.dict_size = len(self.word_to_id_)
        self.id_counts_ = id_counts
        #write word2id file
        print("write word2id file to : " + dict_path + "_word_to_id_")
        with io.open(dict_path + "_word_to_id_", 'w+', encoding='utf-8') as f6:
            for k, v in self.word_to_id_.items():
                f6.write(k + " " + str(v) + '\n')

        print("corpus_size:", self.corpus_size_)
        self.id_frequencys = [
            float(count) / word_all_count for count in self.id_counts_
        ]
        print("dict_size = " + str(self.dict_size) + " word_all_count = " + str(
            word_all_count))

        self.random_generator = NumpyRandomInt(1, self.window_size_ + 1)

    def get_context_words(self, words, idx):
        """
        Get the context word list of target word.
        words: the words of the current line
        idx: input word index
        window_size: window size
        """
        target_window = self.random_generator()
        start_point = idx - target_window  # if (idx - target_window) > 0 else 0
        if start_point < 0:
            start_point = 0
        end_point = idx + target_window
        targets = words[start_point:idx] + words[idx + 1:end_point + 1]
        return targets

    def train(self):
        def nce_reader():
            for file in self.filelist:
                with io.open(
                        self.data_path_ + "/" + file, 'r',
                        encoding='utf-8') as f:
                    logger.info("running data in {}".format(self.data_path_ +
                                                            "/" + file))
                    count = 1
                    for line in f:
                        if self.trainer_id == count % self.trainer_num:
                            word_ids = [int(w) for w in line.split()]
                            for idx, target_id in enumerate(word_ids):
                                context_word_ids = self.get_context_words(
                                    word_ids, idx)
                                for context_id in context_word_ids:
                                    yield [target_id], [context_id]
                        count += 1

        return nce_reader
