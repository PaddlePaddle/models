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
Preprocess script.
"""

import os
import argparse

from plato.args import str2bool
from plato.args import parse_args
from plato.data.dataset import Dataset
from plato.data.field import BPETextField


def main():
    parser = argparse.ArgumentParser()
    
    BPETextField.add_cmdline_argument(parser)
    Dataset.add_cmdline_argument(parser)
    
    args = parse_args(parser)
    
    raw_train_file = os.path.join(args.data_dir, "dial.train")
    raw_valid_file = os.path.join(args.data_dir, "dial.valid")
    raw_test_file = os.path.join(args.data_dir, "dial.test")
    train_file = raw_train_file + f".{args.tokenizer_type}.jsonl"
    valid_file = raw_valid_file + f".{args.tokenizer_type}.jsonl"
    test_file = raw_test_file + f".{args.tokenizer_type}.jsonl"
    
    bpe = BPETextField(args.BPETextField)
    
    BUILD_EXAMPLES_FN = {
        "multi": bpe.build_examples_multi_turn,
        "multi_knowledge": bpe.build_examples_multi_turn_with_knowledge
    }
    build_examples_fn = BUILD_EXAMPLES_FN[args.data_type]
    
    if os.path.exists(raw_valid_file) and not os.path.exists(valid_file):
        valid_examples = build_examples_fn(raw_valid_file, data_type="valid")
        bpe.save_examples(valid_examples, valid_file)
    
    if os.path.exists(raw_test_file) and not os.path.exists(test_file):
        test_examples = build_examples_fn(raw_test_file, data_type="test")
        bpe.save_examples(test_examples, test_file)
    
    if os.path.exists(raw_train_file) and not os.path.exists(train_file):
        train_examples = build_examples_fn(raw_train_file, data_type="train")
        bpe.save_examples(train_examples, train_file)

    return


if __name__ == "__main__":
    main()
