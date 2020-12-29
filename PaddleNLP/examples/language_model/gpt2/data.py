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
                 max_seq_len=1024,
                 weighted=False,
                 sample_across_doc=True,
                 random_across_doc_sampling=True):
        self.file_path = file_path
        self.num_samples = num_samples
        if num_samples is None:
            self.num_samples = 1000 * self.ds_len
        self.max_seq_len = max_seq_len
        self.example_texts = []
        self._read_json()

    def _read_json(self):
        nltk.download("punkt")
        with open(self.file_path, "r") as input_file:
            for line in input_file.readlines():
                json_data = json.loads(line)
                sent_list = []
                for line in json_data['text'].split('\n'):
                    if line != '\n':
                        if len(line) < 10:
                            continue
                    sent_list.extend(nltk.tokenize.sent_tokenize(line))
                if len(sent_list) < 1:
                    continue
                self.example_texts.append("".join(sent_list))

    def __getitem__(self, index):
        return self.example_texts[index]

    def __len__(self):
        return len(self.example_texts)
