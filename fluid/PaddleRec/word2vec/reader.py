# -*- coding: utf-8 -*

import numpy as np
"""
enwik9 dataset

http://mattmahoney.net/dc/enwik9.zip
"""


class Word2VecReader(object):
    def __init__(self, dict_path, data_path, window_size=5):
        self.window_size_ = window_size
        self.data_path_ = data_path
        self.word_to_id_ = dict()

        word_id = 0
        with open(dict_path, 'r') as f:
            for line in f:
                self.word_to_id_[line.split()[0]] = word_id
                word_id += 1
        self.dict_size = len(self.word_to_id_)
        print("dict_size = " + str(self.dict_size))

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

    def train(self):
        def _reader():
            with open(self.data_path_, 'r') as f:
                for line in f:
                    word_ids = [
                        self.word_to_id_[word] for word in line.split()
                        if word in self.word_to_id_
                    ]
                    for idx, target_id in enumerate(word_ids):
                        context_word_ids = self.get_context_words(
                            word_ids, idx, self.window_size_)
                        for context_id in context_word_ids:
                            yield [target_id], [context_id]

        return _reader


if __name__ == "__main__":
    epochs = 10
    batch_size = 1000
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
