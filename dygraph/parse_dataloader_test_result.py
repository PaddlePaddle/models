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

from __future__ import print_function

import os
import sys
import logging
import paddle.compat as cpt

TEST_LOG_DIR = './dataloader_test_log'

def parse_test_result(dirname):
    if not os.path.exists(dirname):
        logging.warning("log dir is not exist.")
        return

    for parent, _, filenames in os.walk(dirname):
        for filename in filenames:
          filepath = os.path.join(parent, filename)
          if not os.path.isfile(filepath):
              continue
          with open(filepath, 'rb') as f:
              for line in f.readlines():
                  if line.startswith(b'total train time:'):
                      print("%s - %s" % (filepath, cpt.to_text(line)[:-1]))

if __name__ == "__main__":
    parse_test_result(TEST_LOG_DIR)