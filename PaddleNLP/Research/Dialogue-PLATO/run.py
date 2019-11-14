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
Running scripts.
"""

import argparse
import json
import os

import numpy as np
import paddle.fluid as fluid

from plato.args import parse_args
from plato.args import str2bool
from plato.data.data_loader import DataLoader
from plato.data.dataset import Dataset
from plato.data.dataset import LazyDataset
from plato.data.field import BPETextField
from plato.trainer import Trainer
from plato.models.model_base import ModelBase
from plato.models.generator import Generator
import plato.modules.parallel as parallel


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--do_train", type=str2bool, default=False,
                        help="Whether to run trainning.")
    parser.add_argument("--do_test", type=str2bool, default=False,
                        help="Whether to run evaluation on the test dataset.")
    parser.add_argument("--do_infer", type=str2bool, default=False,
                        help="Whether to run inference on the test dataset.")
    parser.add_argument("--num_infer_batches", type=int, default=None,
                        help="The number of batches need to infer.\n"
                        "Stay 'None': infer on entrie test dataset.")
    parser.add_argument("--hparams_file", type=str, default=None,
                        help="Loading hparams setting from file(.json format).")
    BPETextField.add_cmdline_argument(parser)
    Dataset.add_cmdline_argument(parser)
    Trainer.add_cmdline_argument(parser)
    ModelBase.add_cmdline_argument(parser)
    Generator.add_cmdline_argument(parser)

    hparams = parse_args(parser)

    if hparams.hparams_file and os.path.exists(hparams.hparams_file):
        print(f"Loading hparams from {hparams.hparams_file} ...")
        hparams.load(hparams.hparams_file)
        print(f"Loaded hparams from {hparams.hparams_file}")

    print(json.dumps(hparams, indent=2))

    if not os.path.exists(hparams.save_dir):
        os.makedirs(hparams.save_dir)
    hparams.save(os.path.join(hparams.save_dir, "hparams.json"))
    
    bpe = BPETextField(hparams.BPETextField)
    hparams.Model.num_token_embeddings = bpe.vocab_size

    generator = Generator.create(hparams.Generator, bpe=bpe)

    COLLATE_FN = {
        "multi": bpe.collate_fn_multi_turn,
        "multi_knowledge": bpe.collate_fn_multi_turn_with_knowledge
    }
    collate_fn = COLLATE_FN[hparams.data_type]

    # Loading datasets
    if hparams.do_train:
        raw_train_file = os.path.join(hparams.data_dir, "dial.train")
        train_file = raw_train_file + f".{hparams.tokenizer_type}.jsonl"
        assert os.path.exists(train_file), f"{train_file} isn't exist"
        train_dataset = LazyDataset(train_file)
        train_loader = DataLoader(train_dataset, hparams.Trainer, collate_fn=collate_fn, is_train=True)
        raw_valid_file = os.path.join(hparams.data_dir, "dial.valid")
        valid_file = raw_valid_file + f".{hparams.tokenizer_type}.jsonl"
        assert os.path.exists(valid_file), f"{valid_file} isn't exist"
        valid_dataset = LazyDataset(valid_file)
        valid_loader = DataLoader(valid_dataset, hparams.Trainer, collate_fn=collate_fn)

    if hparams.do_infer or hparams.do_test:
        raw_test_file = os.path.join(hparams.data_dir, "dial.test")
        test_file = raw_test_file + f".{hparams.tokenizer_type}.jsonl"
        assert os.path.exists(test_file), f"{test_file} isn't exist"
        test_dataset = LazyDataset(test_file)
        test_loader = DataLoader(test_dataset, hparams.Trainer, collate_fn=collate_fn, is_test=hparams.do_infer)

    def to_tensor(array):
        array = np.expand_dims(array, -1)
        return fluid.dygraph.to_variable(array)

    if hparams.use_data_distributed:
        place = fluid.CUDAPlace(parallel.Env().dev_id)
    else:
        place = fluid.CUDAPlace(0)

    with fluid.dygraph.guard(place):
        # Construct Model
        model = ModelBase.create("Model", hparams, generator=generator)

        # Construct Trainer
        trainer = Trainer(model, to_tensor, hparams.Trainer)

        if hparams.do_train:
            # Training process
            for epoch in range(hparams.num_epochs):
                trainer.train_epoch(train_loader, valid_loader)

        if hparams.do_test:
            # Validation process
            trainer.evaluate(test_loader, need_save=False)

        if hparams.do_infer:
            # Inference process
            def split(xs, sep, pad):
                """ Split id list by separator. """
                out, o = [], []
                for x in xs:
                    if x == pad:
                        continue
                    if x != sep:
                        o.append(x)
                    else:
                        if len(o) > 0:
                            out.append(list(o))
                            o = []
                if len(o) > 0:
                    out.append(list(o))
                assert(all(len(o) > 0 for o in out))
                return out

            def parse_context(batch):
                """ Parse context. """
                return bpe.denumericalize([split(xs, bpe.eos_id, bpe.pad_id)
                                           for xs in batch.tolist()])

            def parse_text(batch):
                """ Parse text. """
                return bpe.denumericalize(batch.tolist())

            infer_parse_dict = {
                "src": parse_context,
                "tgt": parse_text,
                "preds": parse_text
            }
            trainer.infer(test_loader, infer_parse_dict, num_batches=hparams.num_infer_batches)


if __name__ == "__main__":
    main()
