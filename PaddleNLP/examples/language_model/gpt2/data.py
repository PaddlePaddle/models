# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import json
import nltk
import numpy as np
import pandas as pd
import paddle


class GPT2Dataset(paddle.io.Dataset):
    def __init__(self,
                 file_path,
                 tokenizer,
                 max_seq_len=1024,
                 weighted=False,
                 sample_across_doc=True,
                 random_across_doc_sampling=True,
                 reset_attenion_mask=False,
                 reset_position_id=False,
                 mode="train"):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.reset_attenion_mask = reset_attenion_mask
        self.reset_position_id = reset_position_id
        self.example_texts = []
        self._read_json()
        self.eos_id = tokenizer.get_command("eos").Id
        print("the eos is:{}".format(self.eos_id))

    def _read_json(self):
        nltk.download("punkt")
        with open(self.file_path, "r") as input_file:
            for line in input_file.readlines():
                # if "</doc>" in line:
                #     continue
                # if "<doc" in line:
                #     continue
                # if len(line) < 50:
                #     continue
                json_data = json.loads(line)
                sent_list = []
                for line in json_data['text'].split('\n'):
                    if line != '\n':
                        if len(line) < 10:
                            continue
                    sent_list.extend(nltk.tokenize.sent_tokenize(line))
                if len(sent_list) < 1:
                    continue
                self.example_texts.append("\n".join(sent_list))
                #self.example_texts.append(line.strip())

    def _pad_seq(self, seq):
        total_tokens = self.max_seq_len + 1
        num_pad_tokens = max(0, total_tokens - len(seq))
        seq += [self.tokenizer.get_command('pad').Id] * (num_pad_tokens)
        return seq

    def _construct_sample(self, tokens):
        labels = tokens[1:]
        tokens = tokens[:-1]
        seq_length = len(tokens)
        # attention mask for the attention calulate
        attention_mask = np.tri(seq_length, seq_length).reshape(
            (1, seq_length, seq_length))

        # the pad and eod tokens do not contribute the loss
        loss_mask = np.ones(seq_length, dtype="float32")
        loss_mask[np.where(np.array(tokens) == self.eos_id)] = 0.0
        position_ids = np.arange(0, seq_length, dtype="int64")

        if self.reset_attenion_mask or self.reset_position_id:
            eos_indices = position_ids[np.weher(tokens == eod_token)]
            prev_index = 0
            for i in range(eos_indices.size()[0]):
                pos_id = eos_indices[i]
                if self.reset_attention_mask:
                    attention_mask[0, (pos_id + 1):, :(pos_id + 1)] = 0
                if self.reset_position_ids:
                    position_ids[(pos_id + 1):] -= (pos_id + 1 - prev_index)
                    prev_index = i + 1
        attention_mask = (attention_mask - 1.0) * 10000.0
        attention_mask = attention_mask.astype("float32")
        return [tokens, loss_mask, attention_mask, position_ids, labels]

    def __getitem__(self, index):
        raw_text = self.example_texts[index]
        tokenization = self.tokenizer.encode(raw_text)
        tokenization.append(self.tokenizer.get_command('eos'))
        tokens = tokenization.tokenization
        num_tokens = len(tokens)
        # truncate the tokens
        tokens_to_remove = num_tokens - self.max_seq_len - 1
        if tokens_to_remove > 0:
            remove_left_tokens = 0  #rng.randint(tokens_to_remove + 1)
            tokens = tokens[remove_left_tokens:]
            remove_right_rokens = len(tokens) - self.max_seq_len - 1
            if remove_right_rokens > 0:
                tokens = tokens[:-remove_right_rokens]
        tokens = self._pad_seq(tokens)
        return self._construct_sample(tokens)

    def __len__(self):
        return len(self.example_texts)
