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
    'decompress',
    'download_and_decompress',
    'load_state_dict_from_url',
    'default_logger',
    'USER_HOME',
    'PPAUDIO_HOME',
    'MODEL_HOME',
    'DATA_HOME',
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


def decompress(file: str, path: str = os.PathLike):
    """
    Extracts all files to specific path from a compressed file.
    """
    assert os.path.isfile(file), "File: {} not exists.".format(file)

    if path is None:
        download._decompress(file)
    else:
        if not os.path.isdir(path):
            os.makedirs(path)

        tmp_file = os.path.join(path, os.path.basename(file))
        os.rename(file, tmp_file)
        download._decompress(tmp_file)
        os.rename(tmp_file, file)


def download_and_decompress(archives: List[Dict[str, str]],
                            path: os.PathLike,
                            decompress: bool = True):
    """
    Download archieves and decompress to specific path.
    """
    if not os.path.isdir(path):
        os.makedirs(path)

    for archive in archives:
        assert 'url' in archive and 'md5' in archive, \
            'Dictionary keys of "url" and "md5" are required in the archive, but got: {list(archieve.keys())}'

        download.get_path_from_url(
            archive['url'], path, archive['md5'], decompress=decompress)


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


def _get_user_home():
    return os.path.expanduser('~')


def _get_ppaudio_home():
    if 'PPAUDIO_HOME' in os.environ:
        home_path = os.environ['PPAUDIO_HOME']
        if os.path.exists(home_path):
            if os.path.isdir(home_path):
                return home_path
            else:
                raise RuntimeError(
                    'The environment variable PPAUDIO_HOME {} is not a directory.'
                    .format(home_path))
        else:
            return home_path
    return os.path.join(_get_user_home(), '.paddleaudio')


def _get_sub_home(directory):
    home = os.path.join(_get_ppaudio_home(), directory)
    if not os.path.exists(home):
        os.makedirs(home)
    return home


'''
PPAUDIO_HOME     -->  the root directory for storing PaddleAudio related data. Default to ~/.paddleaudio. Users can change the
├                            default value through the PPAUDIO_HOME environment variable.
├─ MODEL_HOME    -->  Store model files.
└─ DATA_HOME     -->  Store automatically downloaded datasets.
'''
USER_HOME = _get_user_home()
PPAUDIO_HOME = _get_ppaudio_home()
MODEL_HOME = _get_sub_home('models')
DATA_HOME = _get_sub_home('datasets')
