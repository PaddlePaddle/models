#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
"""Create model for dialogue task."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle.fluid as fluid

from bert import BertModel


def create_model(args,
                 pyreader_name,
                 bert_config,
                 num_labels,
                 paradigm_inst,
                 is_prediction=False): 
    """create dialogue task model"""
    if args.task_name == 'atis_slot': 
        label_dim = [-1, args.max_seq_len]
        lod_level = 1
    elif args.task_name in ['dstc2', 'dstc2_asr', 'multi-woz']: 
        label_dim = [-1, num_labels]
        lod_level = 0
    else: 
        label_dim = [-1, 1]
        lod_level = 0
    pyreader = fluid.layers.py_reader(
        capacity=50,
        shapes=[[-1, args.max_seq_len, 1], [-1, args.max_seq_len, 1],
                [-1, args.max_seq_len, 1], [-1, args.max_seq_len, 1], label_dim],
        dtypes=['int64', 'int64', 'int64', 'float32', 'int64'],
        lod_levels=[0, 0, 0, 0, lod_level],
        name=pyreader_name,
        use_double_buffer=True)

    (src_ids, pos_ids, sent_ids, input_mask, 
     labels) = fluid.layers.read_file(pyreader)

    bert = BertModel(
        src_ids=src_ids,
        position_ids=pos_ids,
        sentence_ids=sent_ids,
        input_mask=input_mask,
        config=bert_config,
        use_fp16=args.use_fp16)

    params = {'num_labels': num_labels,
              'src_ids': src_ids,
              'pos_ids': pos_ids,
              'sent_ids': sent_ids,
              'input_mask': input_mask,
              'labels': labels,
              'is_prediction': is_prediction}

    if is_prediction: 
        results = paradigm_inst.paradigm(bert, params)
        results['pyreader'] = pyreader
        return results

    results = paradigm_inst.paradigm(bert, params)
    results['pyreader'] = pyreader
    return results


