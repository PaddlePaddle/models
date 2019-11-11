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
""" data reader for CoKE
"""

from __future__ import print_function
from __future__ import division

import numpy as np
import six
import collections
import logging

from reader.batching import prepare_batch_data

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.info(logger.getEffectiveLevel())

RawExample = collections.namedtuple("RawExample", ["token_ids", "mask_type"])


def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text.decode("utf-8", "ignore")
        elif isinstance(text, unicode):
            return text
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")


#def printable_text(text):
#    """Returns text encoded in a way suitable for print or `tf.logging`."""
#
#    # These functions want `str` for both Python2 and Python3, but in one case
#    # it's a Unicode string and in the other it's a byte string.
#    if six.PY3:
#        if isinstance(text, str):
#            return text
#        elif isinstance(text, bytes):
#            return text.decode("utf-8", "ignore")
#        else:
#            raise ValueError("Unsupported string type: %s" % (type(text)))
#    elif six.PY2:
#        if isinstance(text, str):
#            return text
#        elif isinstance(text, unicode):
#            return text.encode("utf-8")
#        else:
#            raise ValueError("Unsupported string type: %s" % (type(text)))
#    else:
#        raise ValueError("Not running on Python2 or Python 3?")


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    fin = open(vocab_file)
    for num, line in enumerate(fin):
        items = line.strip().split("\t")
        if len(items) > 2:
            break
        token = items[0]
        index = items[1] if len(items) == 2 else num
        token = token.strip()
        vocab[token] = int(index)
    return vocab


#def convert_by_vocab(vocab, items):
#    """Converts a sequence of [tokens|ids] using the vocab."""
#    output = []
#    for item in items:
#        output.append(vocab[item])
#    return output


def convert_tokens_to_ids(vocab, tokens):
    """Converts a sequence of tokens into ids using the vocab."""
    output = []
    for item in tokens:
        output.append(vocab[item])
    return output


class KBCDataReader(object):
    """ DataReader
    """

    def __init__(self,
                 vocab_path,
                 data_path,
                 max_seq_len=3,
                 batch_size=4096,
                 is_training=True,
                 shuffle=True,
                 dev_count=1,
                 epoch=10,
                 vocab_size=-1):
        self.vocab = load_vocab(vocab_path)
        if vocab_size > 0:
            assert len(self.vocab) == vocab_size, \
                "Assert Error! Input vocab_size(%d) is not consistant with voab_file(%d)" % \
                (vocab_size, len(self.vocab))
        self.pad_id = self.vocab["[PAD]"]
        self.mask_id = self.vocab["[MASK]"]

        self.max_seq_len = max_seq_len
        self.batch_size = batch_size

        self.is_training = is_training
        self.shuffle = shuffle
        self.dev_count = dev_count
        self.epoch = epoch
        if not is_training:
            self.shuffle = False
            self.dev_count = 1
            self.epoch = 1

        self.examples = self.read_example(data_path)
        self.total_instance = len(self.examples)

        self.current_epoch = -1
        self.current_instance_index = -1

    def get_progress(self):
        """return current progress of traning data
        """
        return self.current_instance_index, self.current_epoch

    def line2tokens(self, line):
        tokens = line.split("\t")
        return tokens

    def read_example(self, input_file):
        """Reads the input file into a list of examples."""
        examples = []
        with open(input_file, "r") as f:
            for line in f.readlines():
                line = convert_to_unicode(line.strip())
                tokens = self.line2tokens(line)
                assert len(tokens) <= (self.max_seq_len + 1), \
                    "Expecting at most [max_seq_len + 1]=%d tokens each line, current tokens %d" \
                    % (self.max_seq_len + 1, len(tokens))
                token_ids = convert_tokens_to_ids(self.vocab, tokens[:-1])
                if len(token_ids) <= 0:
                    continue
                examples.append(
                    RawExample(
                        token_ids=token_ids, mask_type=tokens[-1]))
                # if len(examples) <= 10:
                #     logger.info("*** Example ***")
                #     logger.info("tokens: %s" % " ".join([printable_text(x) for x in tokens]))
                #     logger.info("token_ids: %s" % " ".join([str(x) for x in token_ids]))
        return examples

    def data_generator(self):
        """ wrap the batch data generator
        """
        range_list = [i for i in range(self.total_instance)]

        def wrapper():
            """ wrapper batch data
            """

            def reader():
                for epoch_index in range(self.epoch):
                    self.current_epoch = epoch_index
                    if self.shuffle is True:
                        np.random.shuffle(range_list)
                    for idx, sample in enumerate(range_list):
                        self.current_instance_index = idx
                        yield self.examples[sample]

            def batch_reader(reader, batch_size):
                """reader generator for batches of examples
                :param reader: reader generator for one example
                :param batch_size: int batch size
                :return: a list of examples for batch data
                """
                batch = []
                for example in reader():
                    token_ids = example.token_ids
                    mask_type = example.mask_type
                    example_out = [token_ids] + [mask_type]
                    to_append = len(batch) < batch_size
                    if to_append is False:
                        yield batch
                        batch = [example_out]
                    else:
                        batch.append(example_out)
                if len(batch) > 0:
                    yield batch

            all_device_batches = []
            for batch_data in batch_reader(reader, self.batch_size):
                batch_data = prepare_batch_data(
                    batch_data,
                    max_len=self.max_seq_len,
                    pad_id=self.pad_id,
                    mask_id=self.mask_id)
                if len(all_device_batches) < self.dev_count:
                    all_device_batches.append(batch_data)

                if len(all_device_batches) == self.dev_count:
                    for batch in all_device_batches:
                        yield batch
                    all_device_batches = []

        return wrapper


class PathqueryDataReader(KBCDataReader):
    def __init__(self,
                 vocab_path,
                 data_path,
                 max_seq_len=3,
                 batch_size=4096,
                 is_training=True,
                 shuffle=True,
                 dev_count=1,
                 epoch=10,
                 vocab_size=-1):

        KBCDataReader.__init__(self, vocab_path, data_path, max_seq_len,
                               batch_size, is_training, shuffle, dev_count,
                               epoch, vocab_size)

    def line2tokens(self, line):
        tokens = []
        s, path, o, mask_type = line.split("\t")
        path_tokens = path.split(",")
        tokens.append(s)
        tokens.extend(path_tokens)
        tokens.append(o)
        tokens.append(mask_type)
        return tokens
