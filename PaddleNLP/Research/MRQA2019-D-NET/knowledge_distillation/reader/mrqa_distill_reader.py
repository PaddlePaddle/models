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
"""Run MRQA"""

import six
import math
import json
import random
import collections
import numpy as np

from utils import tokenization
from utils.batching import prepare_batch_data


class DataProcessorDistill(object): 
    def __init__(self): 
        self.num_examples = -1
        self.current_train_example = -1
        self.current_train_epoch = -1

    def get_features(self, data_path): 
        with open(data_path, 'r') as fr: 
            for line in fr: 
                yield line.strip()

    def data_generator(self, 
                       data_file, 
                       batch_size, 
                       max_len, 
                       in_tokens, 
                       dev_count,
                       epochs,
                       shuffle): 
        self.num_examples = len([ "" for line in open(data_file,"r")])
        def batch_reader(data_file, in_tokens, batch_size): 
            batch = []
            index = 0
            for feature in self.get_features(data_file): 
                to_append = len(batch) < batch_size
                if to_append: 
                    batch.append(feature)
                else: 
                    yield batch
                    batch = []
            if len(batch) > 0: 
                yield batch

        def wrapper(): 
            for epoch in range(epochs): 
                all_batches = []
                for batch_data in batch_reader(data_file, in_tokens, batch_size): 
                    batch_data_segment = []
                    for feature in batch_data: 
                        data = json.loads(feature.strip())
                        example_index = data['example_index']
                        unique_id = data['unique_id']
                        input_ids = data['input_ids']
                        position_ids = data['position_ids']
                        input_mask = data['input_mask']
                        segment_ids = data['segment_ids']
                        start_position = data['start_position']
                        end_position = data['end_position']
                        start_logits = data['start_logits']
                        end_logits = data['end_logits']
                        instance = [input_ids, position_ids, segment_ids, input_mask, start_logits, end_logits, start_position, end_position]
                        batch_data_segment.append(instance)
                    batch_data = batch_data_segment
                    src_ids = [inst[0] for inst in batch_data]
                    pos_ids = [inst[1] for inst in batch_data]
                    sent_ids = [inst[2] for inst in batch_data]
                    input_mask = [inst[3] for inst in batch_data]
                    start_logits = [inst[4] for inst in batch_data]
                    end_logits = [inst[5] for inst in batch_data]
                    src_ids = np.array(src_ids).astype("int64").reshape([-1, max_len, 1])
                    pos_ids = np.array(pos_ids).astype("int64").reshape([-1, max_len, 1])
                    sent_ids = np.array(sent_ids).astype("int64").reshape([-1, max_len, 1])
                    input_mask = np.array(input_mask).astype("float32").reshape([-1, max_len, 1])
                    start_logits = np.array(start_logits).astype("float32").reshape([-1, max_len])
                    end_logits = np.array(end_logits).astype("float32").reshape([-1, max_len])
                    start_positions = [inst[6] for inst in batch_data]
                    end_positions = [inst[7] for inst in batch_data]
                    start_positions = np.array(start_positions).astype("int64").reshape([-1, 1])
                    end_positions = np.array(end_positions).astype("int64").reshape([-1, 1])
                    batch_data = [src_ids, pos_ids, sent_ids, input_mask, start_logits, end_logits, start_positions, end_positions]

                    if len(all_batches) < dev_count:
                        all_batches.append(batch_data)

                    if len(all_batches) == dev_count: 
                        for batch in all_batches: 
                            yield batch
                        all_batches = []
        return wrapper
