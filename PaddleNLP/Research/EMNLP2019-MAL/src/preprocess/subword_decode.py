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

import os
import sys
import logging
import argparse
from text_encoder import SubwordTextEncoder
from text_encoder import EOS_ID


def get_or_generate_vocab(data_dir, tmp_dir, vocab_filename, vocab_size,
                          sources, file_byte_budget=1e6):
    """Generate a vocabulary from the datasets in sources."""

    def generate():
        """Generate lines for vocabulary generation."""
        logging.info("Generating vocab from: %s", str(sources))
        for source in sources:
            for lang_file in source[1]:
                logging.info("Reading file: %s" % lang_file)

                filepath = os.path.join(tmp_dir, lang_file)
                with open(filepath, mode="r") as source_file:
                    file_byte_budget_ = file_byte_budget
                    counter = 0
                    countermax = int(os.path.getsize(filepath) / file_byte_budget_ / 2)
                    logging.info("countermax: %d" % countermax)
                    for line in source_file:
                        if counter < countermax:
                            counter += 1
                        else:
                            if file_byte_budget_ <= 0:
                                break
                            line = line.strip()
                            file_byte_budget_ -= len(line)
                            counter = 0
                            yield line

    return get_or_generate_vocab_inner(data_dir, vocab_filename, vocab_size,
                                       generate())


def get_or_generate_vocab_inner(data_dir, vocab_filename, vocab_size,
                                generator, max_subtoken_length=None,
                                reserved_tokens=None):
    """Inner implementation for vocab generators.

    Args:
      data_dir: The base directory where data and vocab files are stored. If None,
        then do not save the vocab even if it doesn't exist.
      vocab_filename: relative filename where vocab file is stored
      vocab_size: target size of the vocabulary constructed by SubwordTextEncoder
      generator: a generator that produces tokens from the vocabulary
      max_subtoken_length: an optional integer.  Set this to a finite value to
        avoid quadratic costs during vocab building.
      reserved_tokens: List of reserved tokens. `text_encoder.RESERVED_TOKENS`
        should be a prefix of `reserved_tokens`. If `None`, defaults to
        `RESERVED_TOKENS`.

    Returns:
      A SubwordTextEncoder vocabulary object.
    """
    if data_dir and vocab_filename:
        vocab_filepath = os.path.join(data_dir, vocab_filename)
        if os.path.exists(vocab_filepath):
            logging.info("Found vocab file: %s", vocab_filepath)
            return SubwordTextEncoder(vocab_filepath)
    else:
        vocab_filepath = None

    logging.info("Generating vocab file: %s", vocab_filepath)
    vocab = SubwordTextEncoder.build_from_generator(
        generator, vocab_size, max_subtoken_length=max_subtoken_length,
        reserved_tokens=reserved_tokens)

    if vocab_filepath:
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        vocab.store_to_file(vocab_filepath)

    return vocab


def txt_line_iterator(fname):
    """
    generator for line
    :param fname:
    :return:
    """
    with open(fname, 'r') as f:
        for line in f:
            yield line.strip()


def txt2txt_generator(source_fname, target_fname):
    """

    :param source_fname:
    :param target_fname:
    :return:
    """
    for source, target in zip(
            txt_line_iterator(source_fname),
            txt_line_iterator(target_fname)
    ):
        yield {"inputs": source, "targets": target}


def txt2txt_encoder(sample_generator, vocab, target_vocab=None):
    """

    :param sample_generator:
    :param vocab:
    :param target_vocab:
    :return:
    """
    target_vocab = target_vocab or vocab
    for sample in sample_generator:
        sample["inputs"] = vocab.encode(sample["inputs"])
        sample["inputs"].append(EOS_ID)
        sample["targets"] = target_vocab.encode(sample["targets"])
        sample["targets"].append(EOS_ID)
        yield sample


def txt_encoder(filename, batch_size=1, vocab=None):
    """

    :param sample_generator:
    :param vocab:
    :return:
    """
    def pad_mini_batch(batch):
        """

        :param batch:
        :return:
        """
        lens = map(lambda x: len(x), batch)
        max_len = max(lens)
        for i in range(len(batch)):
            batch[i] = batch[i] + [0] * (max_len - lens[i])
        return batch

    fp = open(filename, 'r')
    samples = []
    batches = []
    ct = 0
    for sample in fp:
        sample = sample.strip()

        if vocab:
            sample = vocab.encode(sample)
        else:
            sample = [int(s) for s in sample]
        #sample.append(EOS_ID)
        batches.append(sample)
        ct += 1
        if ct % batch_size == 0:
            batches = pad_mini_batch(batches)
            samples.extend(batches)
            batches = []
    if ct % batch_size != 0:
        batches += [batches[-1]] * (batch_size - ct % batch_size)
        batches = pad_mini_batch(batches)
        samples.extend(batches)
    return samples

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Tips for generating testset")
    parser.add_argument(
        "--vocab",
        type=str,
        required=True,
        help="The path of source vocab.")
    
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="The path of testset.")

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="The path of result.")

    args = parser.parse_args()
    subword = SubwordTextEncoder(args.vocab)

    samples = []
    with open(args.input, 'r') as f:
        for line in f:
            line = line.strip()
            ids_list = [int(num) for num in line.split(" ")]
            samples.append(ids_list)

    with open(args.output, 'w') as f:
        for sample in samples:
            ret = subword.decode(sample)
            f.write("%s\n" % ret)
