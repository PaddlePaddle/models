# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

import logging
import sys

import paddle.fluid.incubate.data_generator as data_generator

logging.basicConfig()
logger = logging.getLogger("paddle")
logger.setLevel(logging.INFO)


class CriteoDataset(data_generator.MultiSlotDataGenerator):
    def __init__(self, sparse_feature_dim, trainer_id, is_train, trainer_num):
        super(CriteoDataset, self).__init__()
        self.cont_min_ = [0, -3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.cont_max_ = [20, 600, 100, 50, 64000, 500, 100, 50, 500, 10, 10, 10, 50]
        self.cont_diff_ = [20, 603, 100, 50, 64000, 500, 100, 50, 500, 10, 10, 10, 50]
        self.hash_dim_ = sparse_feature_dim
        # here, training data are lines with line_index < train_idx_
        self.train_idx_ = 41256555
        self.continuous_range_ = range(1, 14)
        self.categorical_range_ = range(14, 40)
        self.trainer_id_ = trainer_id
        self.line_idx_ = 0
        self.is_train_ = is_train
        self.trainer_num_ = trainer_num

    def generate_sample(self, line):
        def iter():
            self.line_idx_ += 1
            if self.is_train_ and self.line_idx_ > self.train_idx_:
                return
            elif not is_train and self.line_idx_ <= self.train_idx_:
                return
            if self.line_idx_ % self.trainer_num_ != self.trainer_id_:
                return
            features = line.rstrip('\n').split('\t')
            ret_result = []

            dense_feature = []
            for idx in self.continuous_range_:
                if features[idx] == '':
                    dense_feature.append(0.0)
                else:
                    dense_feature.append((float(features[idx]) - self.cont_min_[idx - 1]) / self.cont_diff_[idx - 1])
            ret_result.append(("dense_feature", dense_feature))
            for idx in self.categorical_range_:
                ret_result.append((str(idx - 13), [hash(str(idx) + features[idx]) % self.hash_dim_]))
            ret_result.append(("label", [int(features[0])]))

            yield tuple(ret_result)

        return iter


if __name__ == "__main__":
    sparse_feature_dim = int(sys.argv[1])
    trainer_id = int(sys.argv[2])
    is_train = bool(sys.argv[3])
    trainer_num = int(sys.argv[4])

    pairwise_reader = CriteoDataset(sparse_feature_dim, trainer_id, is_train, trainer_num)
    pairwise_reader.run_from_stdin()
