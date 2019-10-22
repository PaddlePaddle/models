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
Field class
"""

from itertools import chain
import json
import numpy as np
import pickle
import time
from tqdm import tqdm

from tokenizer import Tokenizer
from args import str2bool


def max_lens(X):
    lens = [len(X)]
    while isinstance(X[0], list):
        lens.append(max(map(len, X)))
        X = [x for xs in X for x in xs]
    return lens


def list2np(X, padding=0, dtype="int64"):
    shape = max_lens(X)
    ret = np.full(shape, padding, dtype=np.int32)

    if len(shape) == 1:
        ret = np.array(X)
    elif len(shape) == 2:
        for i, x in enumerate(X):
            ret[i, :len(x)] = np.array(x)
    elif len(shape) == 3:
        for i, xs in enumerate(X):
            for j, x in enumerate(xs):
                ret[i, j, :len(x)] = np.array(x)
    return ret.astype(dtype)

class BPETextField(object):

    pad_token = "[PAD]"
    bos_token = "[BOS]"
    eos_token = "[EOS]"
    unk_token = "[UNK]"

    @classmethod
    def add_cmdline_argument(cls, parser):
        group = parser.add_argument_group("BPETextField")
        group.add_argument("--vocab_path", type=str, required=True,
                           help="The vocabulary file path.")
        group.add_argument("--filtered", type=str2bool, default=False,
                           help="Whether to filter the data with too long utterance/context. "
                           "If the data is unfiltered, it will be truncated.")
        group.add_argument("--max_len", type=int, default=256,
                           help="The maximum length of context or knowledges.")
        group.add_argument("--min_utt_len", type=int, default=1,
                           help="The minimum length of utterance.")
        group.add_argument("--max_utt_len", type=int, default=50,
                           help="The maximum length of utterance.")
        group.add_argument("--min_ctx_turn", type=int, default=1,
                           help="The minimum turn of context.")
        group.add_argument("--max_ctx_turn", type=int, default=16,
                           help="The maximum turn of context.")
        group.add_argument("--max_knowledge_num", type=int, default=16,
                           help="The maximum number of knowledges.")
        group.add_argument("--max_knowledge_len", type=int, default=16,
                           help="The maximum length of each knowledges")
        return group

    def __init__(self, hparam):
        special_tokens = [self.pad_token, self.bos_token, self.eos_token, self.unk_token]
        self.tokenizer = Tokenizer(vocab_path=hparam.vocab_path, special_tokens=special_tokens)

        self.filtered = hparam.filtered
        self.max_len = hparam.max_len
        self.min_utt_len = hparam.min_utt_len
        self.max_utt_len = hparam.max_utt_len
        self.min_ctx_turn = hparam.min_ctx_turn
        self.max_ctx_turn = hparam.max_ctx_turn - 1 # subtract reply turn
        self.max_knowledge_num = hparam.max_knowledge_num
        self.max_knowledge_len = hparam.max_knowledge_len
        return

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size

    @property
    def num_specials(self):
        return len(self.special_tokens)

    @property
    def pad_id(self):
        return self.tokenizer.convert_tokens_to_ids([self.pad_token])[0]

    @property
    def bos_id(self):
        return self.tokenizer.convert_tokens_to_ids([self.bos_token])[0]

    @property
    def eos_id(self):
        return self.tokenizer.convert_tokens_to_ids([self.eos_token])[0]

    @property
    def unk_id(self):
        return self.tokenizer.convert_tokens_to_ids([self.unk_token])[0]

    @property
    def bot_id(self):
        return 0

    @property
    def user_id(self):
        return 1

    @property
    def knowledge_id(self):
        return 2

    def numericalize(self, tokens):
        assert isinstance(tokens, list)
        if len(tokens) == 0:
            return []
        element = tokens[0]
        if isinstance(element, list):
            return [self.numericalize(s) for s in tokens]
        else:
            return self.tokenizer.convert_tokens_to_ids(tokens)

    def denumericalize(self, numbers):
        assert isinstance(numbers, list)
        if len(numbers) == 0:
            return []
        element = numbers[0]
        if isinstance(element, list):
            return [self.denumericalize(x) for x in numbers]
        else:
            return self.tokenizer.decode(
                numbers, ignore_tokens=[self.bos_token, self.eos_token, self.pad_token])

    def save_examples(self, examples, filename):
        print(f"Saving examples to '{filename}' ...")
        start = time.time()
        if filename.endswith("pkl"):
            with open(filename, "wb") as fp:
                pickle.dump(examples, fp)
        elif filename.endswith("jsonl"):
            with open(filename, "w", encoding="utf-8") as fp:
                for ex in examples:
                    fp.write(json.dumps(ex) + "\n")
        else:
            raise ValueError(f"Unsport file format: {filename}")
        elapsed = time.time() - start
        print(f"Saved {len(examples)} examples (elapsed {elapsed:.2f}s)")

    def load_examples(self, filename):
        print(f"Loading examples from '{filename}' ...")
        start = time.time()
        if filename.endswith("pkl"):
            with open(filename, "rb") as fp:
                examples = pickle.load(fp)
        else:
            with open(filename, "r", encoding="utf-8") as fp:
                examples = list(map(lambda s: json.loads(s.strip()), fp))
        elapsed = time.time() - start
        print(f"Loaded {len(examples)} examples (elapsed {elapsed:.2f}s)")
        return examples

    def utt_filter_pred(self, utt):
        return self.min_utt_len <= len(utt) \
            and (not self.filtered or len(utt) <= self.max_utt_len)

    def utts_filter_pred(self, utts):
        return self.min_ctx_turn <= len(utts) \
            and (not self.filtered or len(utts) <= self.max_ctx_turn)

    def build_examples_multi_turn(self, data_file, data_type="train"):
        print(f"Reading examples from '{data_file}' ...")
        examples = []
        ignored = 0

        with open(data_file, "r", encoding="utf-8") as f:
            for line in tqdm(f, total=None):
                src, tgt = line.strip("\n").split("\t")
                tgt = self.tokenizer.tokenize(tgt)
                src = [self.tokenizer.tokenize(s) for s in src.split(" __eou__ ")]

                if (self.utts_filter_pred(src) and all(map(self.utt_filter_pred, src))
                        and self.utt_filter_pred(tgt)) or data_type == "test":
                    src = [s[-self.max_utt_len:] for s in src[-self.max_ctx_turn:]]
                    src = [self.numericalize(s) + [self.eos_id] for s in src]
                    tgt = [self.bos_id] + self.numericalize(tgt) + [self.eos_id]
                    if data_type != "test":
                        tgt = tgt[:self.max_utt_len + 2]
                    ex = {"src": src, "tgt": tgt}
                    examples.append(ex)
                else:
                    ignored += 1
        print(f"Built {len(examples)} {data_type.upper()} examples ({ignored} filtered)")
        return examples

    def build_examples_multi_turn_with_knoledge(self, data_file, data_type="train"):
        print(f"Reading examples from '{data_file}' ...")
        examples = []
        ignored = 0

        with open(data_file, "r", encoding="utf-8") as f:
            for line in tqdm(f, total=None):
                knowledge, src, tgt = line.strip("\n").split("\t")
                tgt = self.tokenizer.tokenize(tgt)
                knowledge = [self.tokenizer.tokenize(k) for k in knowledge.split(" __eou__ ")]
                knowledge = [k[:self.max_knowledge_len]
                             for k in knowledge[-self.max_knowledge_num:]]
                src = [self.tokenizer.tokenize(s) for s in src.split(" __eou__ ")]

                if (self.utts_filter_pred(src) and all(map(self.utt_filter_pred, src)) 
                        and self.utt_filter_pred(tgt)) or data_type == "test":
                    src = [s[-self.max_utt_len:] for s in src[-self.max_ctx_turn:]]
                    src = [self.numericalize(s) + [self.eos_id] for s in src]
                    knowledge = [self.numericalize(k) + [self.eos_id] for k in knowledge]
                    tgt = [self.bos_id] + self.numericalize(tgt) + [self.eos_id]
                    if data_type != "test":
                        tgt = tgt[:self.max_utt_len + 2]
                    ex = {"src": src, "knowledge": knowledge, "tgt": tgt}
                    examples.append(ex)
                else:
                    ignored += 1
        print(f"Built {len(examples)} {data_type.upper()} examples ({ignored} filtered)")
        return examples

    def collate_fn_multi_turn(self, samples):
        batch_size = len(samples)

        src = [sp["src"] for sp in samples]

        src_token, src_pos, src_turn, src_role = [], [], [], []
        for utts in src:
            utt_lens = [len(utt) for utt in utts]

            # Token ids
            src_token.append(list(chain(*utts))[-self.max_len:])

            # Position ids
            pos = [list(range(l)) for l in utt_lens]
            src_pos.append(list(chain(*pos))[-self.max_len:])

            # Turn ids
            turn = [[len(utts) - i] * l for i, l in enumerate(utt_lens)]
            src_turn.append(list(chain(*turn))[-self.max_len:])

            # Role ids
            role = [[self.bot_id if (len(utts) - i) % 2 == 0 else self.user_id] * l
                    for i, l in enumerate(utt_lens)]
            src_role.append(list(chain(*role))[-self.max_len:])

        src_token = list2np(src_token, padding=self.pad_id)
        src_pos = list2np(src_pos, padding=self.pad_id)
        src_turn = list2np(src_turn, padding=self.pad_id)
        src_role = list2np(src_role, padding=self.pad_id)

        batch = {}
        batch["src_token"] = src_token
        batch["src_mask"] = (src_token != self.pad_id).astype("int64")
        batch["src_pos"] = src_pos
        batch["src_type"] = src_role
        batch["src_turn"] = src_turn

        if "tgt" in samples[0]:
            tgt = [sp["tgt"] for sp in samples]

            # Token ids & Label ids
            tgt_token = list2np(tgt, padding=self.pad_id)

            # Position ids
            tgt_pos = np.zeros_like(tgt_token)
            tgt_pos[:] = np.arange(tgt_token.shape[1], dtype=tgt_token.dtype)

            # Turn ids
            tgt_turn = np.zeros_like(tgt_token)

            # Role ids
            tgt_role = np.full_like(tgt_token, self.bot_id)

            batch["tgt_token"] = tgt_token
            batch["tgt_mask"] = (tgt_token != self.pad_id).astype("int64")
            batch["tgt_pos"] = tgt_pos
            batch["tgt_type"] = tgt_role
            batch["tgt_turn"] = tgt_turn

        return batch, batch_size

    def collate_fn_multi_turn_with_knowledge(self, samples):
        batch_size = len(samples)

        src = [sp["src"] for sp in samples]
        knowledge = [sp["knowledge"] for sp in samples]

        src_token, src_pos, src_turn, src_role = [], [], [], []
        for utts, ks in zip(src, knowledge):
            utt_lens = [len(utt) for utt in utts]
            k_lens = [len(k) for k in ks]

            # Token ids
            token = list(chain(*utts))[-self.max_len:]
            token.extend(list(chain(*ks))[-self.max_len:])
            src_token.append(token)

            # Position ids
            pos = list(chain(*[list(range(l)) for l in utt_lens]))[-self.max_len:]
            pos.extend(list(chain(*[list(range(l)) for l in k_lens]))[-self.max_len:])
            src_pos.append(pos)

            # Turn ids
            turn = list(chain(*[[len(utts) - i] * l for i, l in enumerate(utt_lens)]))[-self.max_len:]
            turn.extend(list(chain(*[[i] * l for i, l in enumerate(k_lens)]))[-self.max_len:])
            src_turn.append(turn)

            # Role ids
            role = list(chain(*[[self.bot_id if (len(utts)-i) % 2 == 0 else self.user_id] * l
                                for i, l in enumerate(utt_lens)]))[-self.max_len:]
            role.extend(list(chain(*[[self.knowledge_id] * l for l in k_lens]))[-self.max_len:])
            src_role.append(role)
        
        src_token = list2np(src_token, padding=self.pad_id)
        src_pos = list2np(src_pos, padding=self.pad_id)
        src_turn = list2np(src_turn, padding=self.pad_id)
        src_role = list2np(src_role, padding=self.pad_id)

        batch = {}
        batch["src_token"] = src_token
        batch["src_mask"] = (src_token != self.pad_id).astype("int64")
        batch["src_pos"] = src_pos
        batch["src_type"] = src_role
        batch["src_turn"] = src_turn

        if "tgt" in samples[0]:
            tgt = [sp["tgt"] for sp in samples]

            # Token ids & Label ids
            tgt_token = list2np(tgt, padding=self.pad_id)

            # Position ids
            tgt_pos = np.zeros_like(tgt_token)
            tgt_pos[:] = np.arange(tgt_token.shape[1], dtype=tgt_token.dtype)

            # Turn ids
            tgt_turn = np.zeros_like(tgt_token)

            # Role ids
            tgt_role = np.full_like(tgt_token, self.bot_id)

            batch["tgt_token"] = tgt_token
            batch["tgt_mask"] = (tgt_token != self.pad_id).astype("int64")
            batch["tgt_pos"] = tgt_pos
            batch["tgt_type"] = tgt_role
            batch["tgt_turn"] = tgt_turn

        return batch, batch_size
