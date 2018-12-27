# coding:utf8
# ==============================================================================
# Copyright 2017 Baidu.com, Inc. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
This package implements some utility functions shared by PaddlePaddle
and Tensorflow model implementations.

Authors: liuyuan(liuyuan04@baidu.com)
Date:    2017/10/06 18:23:06
"""

from .dureader_eval import compute_bleu_rouge
from .dureader_eval import normalize
from .preprocess import find_fake_answer
from .preprocess import find_best_question_match

__all__ = [
    'compute_bleu_rouge',
    'normalize',
    'find_fake_answer',
    'find_best_question_match',
]
