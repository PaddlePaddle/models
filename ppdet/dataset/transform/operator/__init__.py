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

import copy
from . import base
from . import arrange_sample
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

__all__ = ['build']


def build(ops, context=None):
    """ Build a mapper for operators in 'ops'

    Args:
        ops (list of base.BaseOperator or list of op dict): 
            configs for oprators, eg:
            [{'name': 'DecodeImage', 'params': {'to_rgb': True}}, {xxx}]
        context (dict): a context object for mapper

    Returns:
        a mapper function which accept one argument 'sample' and
        return the processed result
    """
    new_ops = []
    for _dict in ops:
        new_dict = {}
        for i, j in _dict.items():
            new_dict[i.lower()] = j
        new_ops.append(new_dict)
    ops = new_ops
    op_funcs = []
    op_repr = []
    for op in ops:
        if type(op) is dict and 'op' in op:
            op_func = getattr(base.BaseOperator, op['op'])
            params = copy.deepcopy(op)
            del params['op']
            o = op_func(**params)
        elif not isinstance(op, base.BaseOperator):
            op_func = getattr(base.BaseOperator, op['name'])
            params = {} if 'params' not in op else op['params']
            o = op_func(**params)
        else:
            assert isinstance(
                op, base.BaseOperator), 'invalid operator when build ops'
            o = op
        op_funcs.append(o)
        op_repr.append('{%s}' % str(o))
    op_repr = '[%s]' % ','.join(op_repr)

    def _mapper(sample):
        ctx = {} if context is None else copy.deepcopy(context)
        for f in op_funcs:
            try:
                out = f(sample, ctx)
                sample = out
            except Exception as e:
                logger.warn('failed to map operator[%s] with exception[%s]' \
                    % (f, str(e)))
        return out

    _mapper.ops = op_repr
    return _mapper


for nm in base.registered_ops:
    op = getattr(base.BaseOperator, nm)
    locals()[nm] = op

__all__ += base.registered_ops
