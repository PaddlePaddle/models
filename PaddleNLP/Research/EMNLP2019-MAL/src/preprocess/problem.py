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

from gen_utils import get_or_generate_vocab
from gen_utils import txt_line_iterator
import os, sys
from gen_utils import txt2txt_encoder
from gen_utils import txt2txt_generator
from text_encoder import TokenTextEncoder
import logging

LOG_FORMAT = "[%(asctime)s %(levelname)s %(filename)s:%(lineno)d] %(message)s"
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=LOG_FORMAT)

class GenSubword(object):
    """
    gen subword
    """

    def __init__(self,
                 vocab_size=8000,
                 training_dataset_filenames="train.txt"):
        """

        :param vocab_size:
        :param vocab_name:
        :param training_dataset_filenames: list
        """
        self.vocab_size = vocab_size
        self.vocab_name = "vocab.%s" % self.vocab_size
        if not isinstance(training_dataset_filenames, list):
            training_dataset_filenames = [training_dataset_filenames]
        self.training_dataset_filenames = training_dataset_filenames

    def generate_data(self, data_dir, tmp_dir):
        """

        :param data_dir: target dir(includes vocab file)
        :param tmp_dir: original dir(includes training dataset filenames)
        :return:
        """
        data_set = [["", self.training_dataset_filenames]]
        source_vocab = get_or_generate_vocab(
            data_dir,
            tmp_dir,
            self.vocab_name,
            self.vocab_size,
            data_set,
            file_byte_budget=1e8)
        source_vocab.store_to_file(os.path.join(data_dir, self.vocab_name))


class SubwordVocabProblem(object):
    """subword input"""

    def __init__(self,
                 source_vocab_size=8000,
                 target_vocab_size=8000,
                 source_train_filenames="train.src",
                 target_train_filenames="train.tgt",
                 source_dev_filenames="dev.src",
                 target_dev_filenames="dev.tgt",
                 one_vocab=False):
        """

        :param source_vocab_size:
        :param target_vocab_size:
        :param source_train_filenames:
        :param target_train_filenames:
        :param source_dev_filenames:
        :param target_dev_filenames:
        """
        self.source_vocab_size = source_vocab_size
        self.target_vocab_size = target_vocab_size
        self.source_vocab_name = "vocab.source.%s" % self.source_vocab_size
        self.target_vocab_name = "vocab.target.%s" % self.target_vocab_size
        if not isinstance(source_train_filenames, list):
            source_train_filenames = [source_train_filenames]
        if not isinstance(target_train_filenames, list):
            target_train_filenames = [target_train_filenames]
        if not isinstance(source_dev_filenames, list):
            source_dev_filenames = [source_dev_filenames]
        if not isinstance(target_dev_filenames, list):
            target_dev_filenames = [target_dev_filenames]
        self.source_train_filenames = source_train_filenames
        self.target_train_filenames = target_train_filenames
        self.source_dev_filenames = source_dev_filenames
        self.target_dev_filenames = target_dev_filenames
        self.one_vocab = one_vocab

    def generate_data(self, data_dir, tmp_dir, is_train=True):
        """

        :param data_dir:
        :param tmp_dir:
        :return:
        """
        self.source_train_ds = [["", self.source_train_filenames]]
        self.target_train_ds = [["", self.target_train_filenames]]
        logging.info("building source vocab ...")
        logging.info(self.one_vocab)
        if not self.one_vocab:
            source_vocab = get_or_generate_vocab(data_dir, tmp_dir,
                                                 self.source_vocab_name,
                                                 self.source_vocab_size,
                                                 self.source_train_ds,
                                                 file_byte_budget=1e8)
            logging.info("building target vocab ...")
            target_vocab = get_or_generate_vocab(data_dir, tmp_dir,
                                                 self.target_vocab_name,
                                                 self.target_vocab_size,
                                                 self.target_train_ds,
                                                 file_byte_budget=1e8)
        else:
            train_ds = [["", self.source_train_filenames + self.target_train_filenames]]
            source_vocab = get_or_generate_vocab(data_dir, tmp_dir,
                                                 self.source_vocab_name,
                                                 self.source_vocab_size,
                                                 train_ds,
                                                 file_byte_budget=1e8)
            target_vocab = source_vocab
            target_vocab.store_to_file(os.path.join(data_dir, self.target_vocab_name))
        pair_filenames = [self.source_train_filenames, self.target_train_filenames]
        if not is_train:
            pair_filenames = [self.source_dev_filenames, self.target_dev_filenames]
        self.compile_data(tmp_dir, pair_filenames, is_train)
        source_fname = "train.lang1" if is_train else "dev.lang1"
        target_fname = "train.lang2" if is_train else "dev.lang2"
        source_fname = os.path.join(tmp_dir, source_fname)
        target_fname = os.path.join(tmp_dir, target_fname)
        return txt2txt_encoder(txt2txt_generator(source_fname, target_fname),
                               source_vocab,
                               target_vocab)

    def compile_data(self, tmp_dir, pair_filenames, is_train=True):
        """
        combine the input files
        :param tmp_dir:
        :param pair_filenames:
        :param is_train:
        :return:
        """
        filename = "train.lang1" if is_train else "dev.lang1"
        out_file_1 = open(os.path.join(tmp_dir, filename), "w")
        filename = "train.lang2" if is_train else "dev.lang2"
        out_file_2 = open(os.path.join(tmp_dir, filename), "w")
        for file1, file2 in zip(pair_filenames[0], pair_filenames[1]):
            for line in txt_line_iterator(os.path.join(tmp_dir, file1)):
                out_file_1.write(line + "\n")
            for line in txt_line_iterator(os.path.join(tmp_dir, file2)):
                out_file_2.write(line + "\n")
        out_file_2.close()
        out_file_1.close()


class TokenVocabProblem(object):
    """token input"""

    def __init__(self,
                 source_vocab_size=8000,
                 target_vocab_size=8000,
                 source_train_filenames="train.src",
                 target_train_filenames="train.tgt",
                 source_dev_filenames="dev.src",
                 target_dev_filenames="dev.tgt",
                 one_vocab=False):
        """

        :param source_vocab_size:
        :param target_vocab_size:
        :param source_train_filenames:
        :param target_train_filenames:
        :param source_dev_filenames:
        :param target_dev_filenames:
        """
        self.source_vocab_size = source_vocab_size
        self.target_vocab_size = target_vocab_size
        self.source_vocab_name = "vocab.source.%s" % self.source_vocab_size
        self.target_vocab_name = "vocab.target.%s" % self.target_vocab_size
        if not isinstance(source_train_filenames, list):
            source_train_filenames = [source_train_filenames]
        if not isinstance(target_train_filenames, list):
            target_train_filenames = [target_train_filenames]
        if not isinstance(source_dev_filenames, list):
            source_dev_filenames = [source_dev_filenames]
        if not isinstance(target_dev_filenames, list):
            target_dev_filenames = [target_dev_filenames]
        self.source_train_filenames = source_train_filenames
        self.target_train_filenames = target_train_filenames
        self.source_dev_filenames = source_dev_filenames
        self.target_dev_filenames = target_dev_filenames
        self.one_vocab = one_vocab


    def add_exsits_vocab(self, filename):
        """
        :param filename
        """
        token_list = []
        with open(filename) as f:
            for line in f:
                line = line.strip()
                token_list.append(line)
        token_list.append("UNK")
        return token_list


    def generate_data(self, data_dir, tmp_dir, is_train=True):
        """

        :param data_dir:
        :param tmp_dir:
        :return:
        """
        self.source_train_ds = [["", self.source_train_filenames]]
        self.target_train_ds = [["", self.target_train_filenames]]

        pair_filenames = [self.source_train_filenames, self.target_train_filenames]
        if not is_train:
            pair_filenames = [self.source_dev_filenames, self.target_dev_filenames]
        self.compile_data(tmp_dir, pair_filenames, is_train)
        source_fname = "train.lang1" if is_train else "dev.lang1"
        target_fname = "train.lang2" if is_train else "dev.lang2"
        source_fname = os.path.join(tmp_dir, source_fname)
        target_fname = os.path.join(tmp_dir, target_fname)
        if is_train:
            source_vocab_path = os.path.join(data_dir, self.source_vocab_name)
            target_vocab_path = os.path.join(data_dir, self.target_vocab_name)
            if not self.one_vocab:
                if os.path.exists(source_vocab_path) and os.path.exists(target_vocab_path):
                    logging.info("found source vocab ...")
                    source_vocab = TokenTextEncoder(None, vocab_list=self.add_exsits_vocab(source_vocab_path))

                    logging.info("found target vocab ...")
                    target_vocab = TokenTextEncoder(None, vocab_list=self.add_exsits_vocab(target_vocab_path))
                else:
                    logging.info("building source vocab ...")
                    source_vocab = TokenTextEncoder.build_from_corpus(source_fname,
                                                                      self.source_vocab_size)
                    os.makedirs(data_dir)
                    logging.info("building target vocab ...")
                    target_vocab = TokenTextEncoder.build_from_corpus(target_fname,
                                                                      self.target_vocab_size)
            else:
                if os.path.exists(source_vocab_path):
                    logging.info("found source vocab ...")
                    source_vocab = TokenTextEncoder(None, vocab_list=self.add_exsits_vocab(source_vocab_path))
                else:
                    source_vocab = TokenTextEncoder.build_from_corpus([source_fname, target_fname],
                                                                      self.source_vocab_size)
                    logging.info("building target vocab ...")
                target_vocab = source_vocab

            source_vocab.store_to_file(source_vocab_path)
            target_vocab.store_to_file(target_vocab_path)
        else:
            source_vocab = TokenTextEncoder(os.path.join(data_dir, self.source_vocab_name))
            target_vocab = TokenTextEncoder(os.path.join(data_dir, self.target_vocab_name))

        return txt2txt_encoder(txt2txt_generator(source_fname, target_fname),
                               source_vocab,
                               target_vocab)

    def compile_data(self, tmp_dir, pair_filenames, is_train=True):
        """
        combine the input files
        :param tmp_dir:
        :param pair_filenames:
        :param is_train:
        :return:
        """
        filename = "train.lang1" if is_train else "dev.lang1"
        out_file_1 = open(os.path.join(tmp_dir, filename), "w")
        filename = "train.lang2" if is_train else "dev.lang2"
        out_file_2 = open(os.path.join(tmp_dir, filename), "w")
        for file1, file2 in zip(pair_filenames[0], pair_filenames[1]):
            for line in txt_line_iterator(os.path.join(tmp_dir, file1)):
                out_file_1.write(line + "\n")
            for line in txt_line_iterator(os.path.join(tmp_dir, file2)):
                out_file_2.write(line + "\n")
        out_file_2.close()
        out_file_1.close()


if __name__ == "__main__":
    gen_sub = GenSubword().generate_data("train_data", "../asr/")
