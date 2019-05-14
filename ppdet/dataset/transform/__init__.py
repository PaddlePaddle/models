# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

# function:
#   transform a dataset to another one

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from . import transformer
from . import operator

def transform(source, ops_conf, worker_args=None):
    """ transform data in 'source' using a mapper defined by 'ops_conf'

    Args:
        @source (instance of Dataset): input data sample
        @ops_conf (list of op configs): used to build a mapper which accept a sample and return a transformed sample

    Returns:
        instance of 'Dataset'
    """
    mapper = operator.build(ops_conf)
    if worker_args is None:
        return transformer.Transformer(
            source, mapper)
    else:
        return transformer.FastTransformer(
            source, mapper, worker_args)


__all__ = ['transformer']
