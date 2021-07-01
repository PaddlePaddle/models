# Copyright (c) 2021  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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
import logging
import os
import sys
from typing import Optional


def get_logger(name: Optional[str] = None,
               use_error_log: bool = False,
               log_dir: Optional[os.PathLike] = None,
               log_file_name: Optional[str] = None):

    if name is None:
        name = __file__

    logger = logging.getLogger(name)
    logging_level = getattr(logging, 'INFO')
    logger.setLevel(logging_level)

    formatter = logging.Formatter(
        fmt='%(asctime)s %(filename)s: %(levelname)s: %(message)s')
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging_level)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)

    if log_dir:  #logging to file
        if log_file_name is None:
            log_file_name = 'log'
        log_file = os.path.join(log_dir, log_file_name + '.txt')
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging_level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    if use_error_log:
        stderr_handler = logging.StreamHandler(sys.stderr)
        stderr_handler.setLevel(logging.WARNING)
        stderr_handler.setFormatter(formatter)
        logger.addHandler(stderr_handler)

    logger.propagate = False
    return logger
