# -*- coding: utf-8 -*

import numpy as np
import preprocess


class Word2VecReader(object):
    def __init__(self, dict_path, data_path, filelist, window_size=5):
        self.window_size_ = window_size
        self.data_path_ = data_path
        self.filelist = filelist
        self.num_non_leaf = 0
        self.word_to_id_ = dict()
        self.id_to_word = dict()
        self.word_to_path = dict()
        self.word_to_code = dict()

        word_all_count = 0
        word_counts = []
        word_id = 0

        with open(dict_path, 'r') as f:
            for line in f:
                word, count = line.split()[0], int(line.split()[1])
                self.word_to_id_[word] = word_id
                self.id_to_word[word_id] = word  #build id to word dict
                word_id += 1
                word_counts.append(count)
                word_all_count += count

        with open(dict_path + "_word_to_id_", 'w+') as f6:
            for k, v in self.word_to_id_.items():
                f6.write(str(k) + " " + str(v) + '\n')

        self.dict_size = len(self.word_to_id_)
        self.word_frequencys = [
            float(count) / word_all_count for count in word_counts
        ]
        print("dict_size = " + str(
            self.dict_size)) + " word_all_count = " + str(word_all_count)

        with open(dict_path + "_ptable", 'r') as f2:
            for line in f2:
                self.word_to_path[line.split(":")[0]] = np.fromstring(
                    line.split(':')[1], dtype=int, sep=' ')
                self.num_non_leaf = np.fromstring(
                    line.split(':')[1], dtype=int, sep=' ')[0]
        print("word_ptable dict_size = " + str(len(self.word_to_path)))

        with open(dict_path + "_pcode", 'r') as f3:
            for line in f3:
                self.word_to_code[line.split(":")[0]] = np.fromstring(
                    line.split(':')[1], dtype=int, sep=' ')
        print("word_pcode dict_size = " + str(len(self.word_to_code)))

    def get_context_words(self, words, idx, window_size):
        """
        Get the context word list of target word.

        words: the words of the current line
        idx: input word index
        window_size: window size
        """
        target_window = np.random.randint(1, window_size + 1)
        # need to keep in mind that maybe there are no enough words before the target word.
        start_point = idx - target_window if (idx - target_window) > 0 else 0
        end_point = idx + target_window
        # context words of the target word
        targets = set(words[start_point:idx] + words[idx + 1:end_point + 1])
        return list(targets)

    def train(self, with_hs):
        def _reader():
            for file in self.filelist:
                with open(self.data_path_ + "/" + file, 'r') as f:
                    for line in f:
                        line = preprocess.text_strip(line)
                        word_ids = [
                            self.word_to_id_[word] for word in line.split()
                            if word in self.word_to_id_
                        ]
                        for idx, target_id in enumerate(word_ids):
                            context_word_ids = self.get_context_words(
                                word_ids, idx, self.window_size_)
                            for context_id in context_word_ids:
                                yield [target_id], [context_id]

        def _reader_hs():
            for file in self.filelist:
                with open(self.data_path_ + "/" + file, 'r') as f:
                    for line in f:
                        line = preprocess.text_strip(line)
                        word_ids = [
                            self.word_to_id_[word] for word in line.split()
                            if word in self.word_to_id_
                        ]
                        for idx, target_id in enumerate(word_ids):
                            context_word_ids = self.get_context_words(
                                word_ids, idx, self.window_size_)
                            for context_id in context_word_ids:
                                yield [target_id], [context_id], [
                                    self.word_to_code[self.id_to_word[
                                        context_id]]
                                ], [
                                    self.word_to_path[self.id_to_word[
                                        context_id]]
                                ]

        if not with_hs:
            return _reader
        else:
            return _reader_hs


if __name__ == "__main__":
    window_size = 10

    reader = Word2VecReader("data/enwik9_dict", "data/enwik9", window_size)
    i = 0
    for x, y in reader.train()():
        print("x: " + str(x))
        print("y: " + str(y))
        print("\n")
        if i == 10:
            exit(0)
        i += 1
