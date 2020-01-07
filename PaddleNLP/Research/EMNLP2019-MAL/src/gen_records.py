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

import logging
import os, sys
import random
import six
import ast

class TRDataGen(object):
    """record data generator
    """

    def __init__(self, num_shards, data_dir):
        self.num_shards = num_shards
        self.data_dir = data_dir

    def gen_data_fnames(self, is_train=True):
        """generate filenames for train and valid
        return:
            train_filenames, valid_filenames
        """
        if not os.path.isdir(self.data_dir):
            try:
                os.mkdir(self.data_dir)
            except Exception as e:
                raise ValueError("%s is exists as one file", self.data_dir)
        if is_train:
            train_prefix = os.path.join(self.data_dir, "translate-train-%05d-of_unshuffle")
            return [train_prefix % i for i in xrange(self.num_shards)]
        return [os.path.join(self.data_dir, "translate-dev-00000-of_unshuffle")]

    def generate(self, data_list, is_train=True, is_shuffle=True):
        """generating record file
        :param data_list:
        :param is_train:
        :return:
        """
        output_filename = self.gen_data_fnames(is_train)
        #writers = [tf.python_io.TFRecordWriter(fname) for fname in output_filename]
        writers = [open(fname, 'w') for fname in output_filename]
        ct = 0
        shard = 0
        for case in data_list:
            ct += 1
            if ct % 10000 == 0:
                logging.info("Generating case %s ." % ct)

            example = self.to_example(case)
            writers[shard].write(example.strip() + "\n")
            if is_train:
                shard = (shard + 1) % self.num_shards
        logging.info("Generating case %s ." % ct)
        for writer in writers:
            writer.close()
        if is_shuffle:
            self.shuffle_dataset(output_filename)

    def to_example(self, dictionary):
        """
        :param source:
        :param target:
        :return:
        """

        if "inputs" not in dictionary or "targets" not in dictionary:
            raise ValueError("Empty generated field: inputs or target")
        
        inputs = " ".join(str(x) for x in dictionary["inputs"])
        targets = " ".join(str(x) for x in dictionary["targets"])
        return inputs + "\t" + targets

    def shuffle_dataset(self, filenames):
        """
        :return:
        """
        logging.info("Shuffling data...")
        for fname in filenames:
            records = self.read_records(fname)
            random.shuffle(records)
            out_fname = fname.replace("_unshuffle", "-shuffle")
            self.write_records(records, out_fname)
            os.remove(fname)

    def read_records(self, filename):
        """
        :param filename:
        :return:
        """
        records = []
        with open(filename, 'r') as reader:
            for record in reader:
                records.append(record)
                if len(records) % 100000 == 0:
                    logging.info("read: %d", len(records))
        return records

    def write_records(self, records, out_filename):
        """
        :param records:
        :param out_filename:
        :return:
        """
        with open(out_filename, 'w') as f:
            for count, record in enumerate(records):
                f.write(record)
                if count > 0 and count % 100000 == 0:
                    logging.info("write: %d", count)


if __name__ == "__main__":
    from preprocess.problem import SubwordVocabProblem
    from preprocess.problem import TokenVocabProblem
    import argparse
    
    parser = argparse.ArgumentParser("Tips for generating subword.")
    parser.add_argument(
        "--tmp_dir",
        type=str,
        required=True,
        help="dir that includes original corpus.")

    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="dir that generates training files")

    parser.add_argument(
        "--source_train_files",
        type=str,
        required=True,
        help="train file for source")

    parser.add_argument(
        "--target_train_files",
        type=str,
        required=True,
        help="train file for target")

    parser.add_argument(
        "--source_vocab_size",
        type=int,
        required=True,
        help="source_vocab_size")

    parser.add_argument(
        "--target_vocab_size",
        type=int,
        required=True,
        help="target_vocab_size")

    parser.add_argument(
        "--num_shards",
        type=int,
        default=100,
        help="number of shards")
    
    parser.add_argument(
        "--subword",
        type=ast.literal_eval,
        default=False,
        help="subword")

    parser.add_argument(
        "--token",
        type=ast.literal_eval,
        default=False,
        help="token")
    
    parser.add_argument(
        "--onevocab",
        type=ast.literal_eval,
        default=False,
        help="share vocab")

    args = parser.parse_args()
    print args

    gen = TRDataGen(args.num_shards, args.data_dir)
    source_train_files = args.source_train_files.split(",")
    target_train_files = args.target_train_files.split(",")
    if args.token == args.subword:
        print "one of subword or token is True"
        import sys

        sys.exit(1)
    
    LOG_FORMAT = "[%(asctime)s %(levelname)s %(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=LOG_FORMAT)
    
    if args.subword:
        problem = SubwordVocabProblem(args.source_vocab_size,
                                      args.target_vocab_size,
                                      source_train_files,
                                      target_train_files,
                                      None,
                                      None,
                                      args.onevocab)
    else:
        problem = TokenVocabProblem(args.source_vocab_size, 
                                    args.target_vocab_size, 
                                    source_train_files, 
                                    target_train_files, 
                                    None,
                                    None,
                                    args.onevocab)

    gen.generate(problem.generate_data(args.data_dir, args.tmp_dir, True), True, True)
