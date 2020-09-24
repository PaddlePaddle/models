# -*- coding=utf8 -*-
"""
#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import json
import pickle
import time
import random
import os
import numpy as np
import sys
import paddle.fluid.incubate.data_generator as dg


class TDMDataset(dg.MultiSlotStringDataGenerator):
    """
    DacDataset: inheritance MultiSlotDataGeneratior, Implement data reading
    Help document: http://wiki.baidu.com/pages/viewpage.action?pageId=728820675
    """

    def infer_reader(self, infer_file_list, batch):
        """
            Read test_data line by line & yield batch
            """

        def local_iter():
            """Read file line by line"""
            for fname in infer_file_list:
                with open(fname, "r") as fin:
                    for line in fin:
                        one_data = (line.strip('\n')).split('\t')
                        input_emb = one_data[0].split(' ')

                        yield [input_emb]

        import paddle
        batch_iter = fluid.io.batch(local_iter, batch)
        return batch_iter

    def generate_sample(self, line):
        """
        Read the data line by line and process it as a dictionary
        """

        def iterator():
            """
            This function needs to be implemented by the user, based on data format
            """
            features = (line.strip('\n')).split('\t')
            input_emb = features[0].split(' ')
            item_label = [features[1]]

            feature_name = ["input_emb", "item_label"]
            yield zip(feature_name, [input_emb] + [item_label])

        return iterator


if __name__ == "__main__":
    import paddle
    paddle.enable_static()
    d = TDMDataset()
    d.run_from_stdin()
