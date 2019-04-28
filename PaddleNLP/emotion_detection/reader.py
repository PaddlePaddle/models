"""
EmoTect Reader, data converters for classification data.
"""
import os

import paddle
import paddle.fluid as fluid
import numpy as np

from utils import load_vocab
from utils import data_reader

class EmoTectProcessor(object):
    """
    Processor class for data convertors for EmoTect.
    """
    def __init__(self,
                 data_dir,
                 vocab_path,
                 random_seed=None):
        self.data_dir = data_dir
        self.vocab = load_vocab(vocab_path)
        self.num_examples = {"train": -1, "dev": -1, "test": -1, "infer": -1}
        np.random.seed(random_seed)

    def get_train_examples(self, data_dir, epoch=1):
        """
        Load training examples
        """
        return data_reader(os.path.join(self.data_dir, "train.tsv"),
                    self.vocab, self.num_examples, "train", epoch)

    def get_dev_examples(self, data_dir):
        """
        Load dev examples
        """
        return data_reader(os.path.join(self.data_dir, "dev.tsv"),
                    self.vocab, self.num_examples, "dev")

    def get_test_examples(self, data_dir):
        """
        Load test examples
        """
        return data_reader(os.path.join(self.data_dir, "test.tsv"),
                    self.vocab, self.num_examples, "test")

    def get_infer_examples(self, data_dir):
        """
        Load infer querys
        """
        return data_reader(os.path.join(self.data_dir, "infer.tsv"),
                    self.vocab, self.num_examples, "infer")

    def get_labels(self):
        """
        Return Labels
        """
        return ["0", "1", "2"]

    def get_num_examples(self, phase):
        """
        Return num of examples in train, dev, test set
        """
        if phase not in ['train', 'dev', 'test', 'infer']:
            raise ValueError(
                "Unknown phase, which should be in ['train', 'dev', 'test', 'infer'].")
        return self.num_examples[phase]

    def get_train_progress(self):
        """
        Get train progress
        """
        return self.current_train_example, self.current_train_epoch

    def data_generator(self, batch_size, phase='train', epoch=1):
        """
        Generate data for train, dev or test
        """
        if phase == "train":
            return paddle.batch(self.get_train_examples(self.data_dir, epoch), batch_size)
        elif phase == "dev":
            return paddle.batch(self.get_dev_examples(self.data_dir), batch_size)
        elif phase == "test":
            return paddle.batch(self.get_test_examples(self.data_dir), batch_size)
        elif phase == "infer":
            return paddle.batch(self.get_infer_examples(self.data_dir), batch_size)
        else:
            raise ValueError(
                "Unknown phase, which should be in ['train', 'dev', 'test', 'infer'].")

