# -*- coding: utf-8 -*

import numpy as np
import preprocess
import logging
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
        self.num_non_leaf = 0
        self.word_to_id_ = dict()
        self.id_to_word = dict()
        self.word_count = dict()
        self.word_to_path = dict()
        self.word_to_code = dict()
        self.trainer_id = trainer_id
        self.trainer_num = trainer_num

        word_all_count = 0
        word_counts = []
        word_id = 0

        with io.open(dict_path, 'r', encoding='utf-8') as f:
            for line in f:
                word, count = line.split()[0], int(line.split()[1])
                self.word_count[word] = count
                self.word_to_id_[word] = word_id
                self.id_to_word[word_id] = word  #build id to word dict
                word_id += 1
                word_counts.append(count)
                word_all_count += count

        with io.open(dict_path + "_word_to_id_", 'w+', encoding='utf-8') as f6:
            for k, v in self.word_to_id_.items():
                f6.write(k + " " + str(v) + '\n')

        self.dict_size = len(self.word_to_id_)
        self.word_frequencys = [
            float(count) / word_all_count for count in word_counts
        ]
        print("dict_size = " + str(
            self.dict_size)) + " word_all_count = " + str(word_all_count)

        with io.open(dict_path + "_ptable", 'r', encoding='utf-8') as f2:
            for line in f2:
                self.word_to_path[line.split('\t')[0]] = np.fromstring(
                    line.split('\t')[1], dtype=int, sep=' ')
                self.num_non_leaf = np.fromstring(
                    line.split('\t')[1], dtype=int, sep=' ')[0]
        print("word_ptable dict_size = " + str(len(self.word_to_path)))

        with io.open(dict_path + "_pcode", 'r', encoding='utf-8') as f3:
            for line in f3:
                self.word_to_code[line.split('\t')[0]] = np.fromstring(
                    line.split('\t')[1], dtype=int, sep=' ')
        print("word_pcode dict_size = " + str(len(self.word_to_code)))
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

        return set(targets)

    def train(self, with_hs):
        def _reader():
            for file in self.filelist:
                with io.open(
                        self.data_path_ + "/" + file, 'r',
                        encoding='utf-8') as f:
                    logger.info("running data in {}".format(self.data_path_ +
                                                            "/" + file))
                    count = 1
                    for line in f:
                        if self.trainer_id == count % self.trainer_num:
                            line = preprocess.strip_lines(line, self.word_count)
                            word_ids = [
                                self.word_to_id_[word] for word in line.split()
                                if word in self.word_to_id_
                            ]
                            for idx, target_id in enumerate(word_ids):
                                context_word_ids = self.get_context_words(
                                    word_ids, idx)
                                for context_id in context_word_ids:
                                    yield [target_id], [context_id]
                        else:
                            pass
                        count += 1

        def _reader_hs():
            for file in self.filelist:
                with io.open(
                        self.data_path_ + "/" + file, 'r',
                        encoding='utf-8') as f:
                    logger.info("running data in {}".format(self.data_path_ +
                                                            "/" + file))
                    count = 1
                    for line in f:
                        if self.trainer_id == count % self.trainer_num:
                            line = preprocess.strip_lines(line, self.word_count)
                            word_ids = [
                                self.word_to_id_[word] for word in line.split()
                                if word in self.word_to_id_
                            ]
                            for idx, target_id in enumerate(word_ids):
                                context_word_ids = self.get_context_words(
                                    word_ids, idx)
                                for context_id in context_word_ids:
                                    yield [target_id], [context_id], [
                                        self.word_to_path[self.id_to_word[
                                            target_id]]
                                    ], [
                                        self.word_to_code[self.id_to_word[
                                            target_id]]
                                    ]
                        else:
                            pass
                        count += 1

        if not with_hs:
            return _reader
        else:
            return _reader_hs

    def async_train(self, with_hs):
        def _reader():
            write_f = list()
            for i in range(20):
                write_f.append(
                    io.open(
                        "./async_data/async_" + str(i), 'w+', encoding='utf-8'))
            for file in self.filelist:
                with io.open(
                        self.data_path_ + "/" + file, 'r',
                        encoding='utf-8') as f:
                    logger.info("running data in {}".format(self.data_path_ +
                                                            "/" + file))
                    count = 1
                    file_spilt_count = 0
                    for line in f:
                        if self.trainer_id == count % self.trainer_num:
                            line = preprocess.strip_lines(line, self.word_count)
                            word_ids = [
                                self.word_to_id_[word] for word in line.split()
                                if word in self.word_to_id_
                            ]
                            for idx, target_id in enumerate(word_ids):
                                context_word_ids = self.get_context_words(
                                    word_ids, idx)
                                for context_id in context_word_ids:
                                    content = "1" + " " + str(
                                        target_id) + " " + "1" + " " + str(
                                            context_id) + '\n'
                                    write_f[file_spilt_count %
                                            20].write(content.decode('utf-8'))
                                    file_spilt_count += 1
                        else:
                            pass
                        count += 1
            for i in range(20):
                write_f[i].close()

        def _reader_hs():
            write_f = list()
            for i in range(20):
                write_f.append(
                    io.open(
                        "./async_data/async_" + str(i), 'w+', encoding='utf-8'))

            for file in self.filelist:
                with io.open(
                        self.data_path_ + "/" + file, 'r',
                        encoding='utf-8') as f:
                    logger.info("running data in {}".format(self.data_path_ +
                                                            "/" + file))
                    count = 1
                    file_spilt_count = 0
                    for line in f:
                        if self.trainer_id == count % self.trainer_num:
                            line = preprocess.strip_lines(line, self.word_count)
                            word_ids = [
                                self.word_to_id_[word] for word in line.split()
                                if word in self.word_to_id_
                            ]
                            for idx, target_id in enumerate(word_ids):
                                context_word_ids = self.get_context_words(
                                    word_ids, idx)
                                for context_id in context_word_ids:
                                    path = [
                                        str(i)
                                        for i in self.word_to_path[
                                            self.id_to_word[target_id]]
                                    ]
                                    code = [
                                        str(j)
                                        for j in self.word_to_code[
                                            self.id_to_word[target_id]]
                                    ]
                                    content = str(1) + " " + str(
                                        target_id
                                    ) + " " + str(1) + " " + str(
                                        context_id
                                    ) + " " + str(len(path)) + " " + ' '.join(
                                        path) + " " + str(len(
                                            code)) + " " + ' '.join(code) + '\n'
                                    write_f[file_spilt_count %
                                            20].write(content.decode('utf-8'))
                                    file_spilt_count += 1
                        else:
                            pass
                        count += 1
            for i in range(20):
                write_f[i].close()

        if not with_hs:
            _reader()
        else:
            _reader_hs()


if __name__ == "__main__":
    window_size = 5

    reader = Word2VecReader(
        "./data/1-billion_dict",
        "./data/1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled/",
        ["news.en-00001-of-00100"], 0, 1)

    i = 0
    # print(reader.train(True))
    for x, y, z, f in reader.train(True)():
        print("x: " + str(x))
        print("y: " + str(y))
        print("path: " + str(z))
        print("code: " + str(f))
        print("\n")
        if i == 10:
            exit(0)
        i += 1
