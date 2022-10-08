# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import os.path
import sys

import numpy as np

from .utils import init_logger, check_print_diff
from .compare import compute_diff, check_data


class ReprodDiffHelper:
    def load_info(self, path: str):
        """
        加载字典文件
        :param path:
        :return:
        """
        assert os.path.exists(path)
        data = np.load(path, allow_pickle=True).tolist()
        return data

    def compare_info(self, info1: dict, info2: dict):
        """
        对比diff
        :param info1:
        :param info2:
        :return:
        """
        assert isinstance(info1, dict) and isinstance(info2, dict)
        check_data(info1, info2)
        self.diff_dict = compute_diff(info1, info2)

    def report(self,
               diff_method='mean',
               diff_threshold: float=1e-6,
               path: str="./diff.txt"):
        """
        可视化diff，保存到文件或者到屏幕
        :param diff_threshold:
        :param path:
        :return:
        """

        logger = init_logger(path)

        passed = check_print_diff(
            self.diff_dict,
            diff_method=diff_method,
            diff_threshold=diff_threshold,
            print_func=logger.info)
        if passed:
            logger.info('diff check passed')
        else:
            logger.info('diff check failed')

        return
