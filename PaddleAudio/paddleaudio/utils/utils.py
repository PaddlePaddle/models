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

__all__ = [
    'ParameterError',
    'get_logger',
    'Timer',
    'seconds_to_hms',
    'download_and_decompress',
    'load_state_dict_from_url',
    'default_logger',
]

import logging
import math
import os
import sys
import time
from typing import Dict, List, Optional

from paddle.framework import load as load_state_dict
from paddle.utils import download


class ParameterError(Exception):
    """Exception class for Parameter checking"""
    pass


def get_logger(name: Optional[str] = None,
               use_error_log: bool = False,
               log_dir: Optional[os.PathLike] = None,
               log_file_name: Optional[str] = None):

    if name is None:
        name = __file__

    def list_handlers(logger):
        return {str(h) for h in logger.handlers}

    logger = logging.getLogger(name)
    logging_level = getattr(logging, 'INFO')
    logger.setLevel(logging_level)

    formatter = logging.Formatter(
        fmt='%(asctime)s %(filename)s: %(levelname)s: %(message)s')
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging_level)
    stdout_handler.setFormatter(formatter)
    if str(stdout_handler) not in list_handlers(logger):
        logger.addHandler(stdout_handler)
    if log_dir:  #logging to file
        if log_file_name is None:
            log_file_name = 'log'
        log_file = os.path.join(log_dir, log_file_name + '.txt')
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging_level)
        fh.setFormatter(formatter)
        if str(fh) not in list_handlers(logger):
            logger.addHandler(fh)

    if use_error_log:
        stderr_handler = logging.StreamHandler(sys.stderr)
        stderr_handler.setLevel(logging.WARNING)
        stderr_handler.setFormatter(formatter)
        if str(stderr_handler) not in list_handlers(logger):
            logger.addHandler(stderr_handler)

    logger.propagate = 0
    return logger


class Timer(object):
    '''Calculate runing speed and estimated time of arrival(ETA)'''
    def __init__(self, total_step: int):
        self.total_step = total_step
        self.last_start_step = 0
        self.current_step = 0
        self._is_running = True

    def start(self):
        self.last_time = time.time()
        self.start_time = time.time()

    def stop(self):
        self._is_running = False
        self.end_time = time.time()

    def count(self) -> int:
        if not self.current_step >= self.total_step:
            self.current_step += 1
        return self.current_step

    @property
    def timing(self) -> float:
        run_steps = self.current_step - self.last_start_step
        self.last_start_step = self.current_step
        time_used = time.time() - self.last_time
        self.last_time = time.time()
        return run_steps / time_used

    @property
    def is_running(self) -> bool:
        return self._is_running

    @property
    def eta(self) -> str:
        if not self.is_running:
            return '00:00:00'
        scale = self.total_step / self.current_step
        remaining_time = (time.time() - self.start_time) * scale
        return seconds_to_hms(remaining_time)


def seconds_to_hms(seconds: int) -> str:
    '''Convert the number of seconds to hh:mm:ss'''
    h = math.floor(seconds / 3600)
    m = math.floor((seconds - h * 3600) / 60)
    s = int(seconds - h * 3600 - m * 60)
    hms_str = '{:0>2}:{:0>2}:{:0>2}'.format(h, m, s)
    return hms_str


def download_and_decompress(archives: List[Dict[str, str]], path: str):
    """
    Download archieves and decompress to specific path.
    """
    if not os.path.isdir(path):
        os.makedirs(path)

    for archive in archives:
        assert 'url' in archive and 'md5' in archive, \
            'Dictionary keys of "url" and "md5" are required in the archive, but got: {list(archieve.keys())}'

        download.get_path_from_url(archive['url'], path, archive['md5'])


def load_state_dict_from_url(url: str, path: str, md5: str = None):
    """
    Download and load a state dict from url
    """
    if not os.path.isdir(path):
        os.makedirs(path)

    download.get_path_from_url(url, path, md5)
    return load_state_dict(os.path.join(path, os.path.basename(url)))


default_logger = get_logger(__file__)
download.logger = default_logger
