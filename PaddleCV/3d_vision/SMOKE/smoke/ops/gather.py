# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

"""
The same function as torch.gather.
Note that: In PaddlePaddle2.0, paddle.gather is different with torch.gather
"""

import paddle

def gather_op(x, dim, index):

    dtype_mapping = {"VarType.INT32": "int32", "VarType.INT64": "int64", "paddle.int32": "int32", "paddle.int64": "int64"}
    if dim < 0:
        dim += len(x.shape)

    x_range = list(range(len(x.shape)))
    x_range[0] = dim
    x_range[dim] = 0
    x_swaped = paddle.transpose(x, perm=x_range)

    index_range = list(range(len(index.shape)))
    index_range[0] = dim
    index_range[dim] = 0
    index_swaped = paddle.transpose(index, perm=index_range)

    dtype = dtype_mapping[str(index.dtype)]
    x_shape = paddle.shape(x_swaped)
    index_shape = paddle.shape(index_swaped)
    prod = paddle.prod(x_shape, dtype=dtype) / x_shape[0]

    x_swaped_flattend = paddle.flatten(x_swaped)
    index_swaped_flattend = paddle.flatten(index_swaped)
    index_swaped_flattend *= prod

    bias = paddle.arange(start=0, end=prod, dtype=dtype)
    bias = paddle.reshape(bias, x_shape[1:])
    bias = paddle.crop(bias, index_shape[1:])
    bias = paddle.flatten(bias)
    bias = paddle.tile(bias, [index_shape[0]])

    index_swaped_flattend += bias

    gathered = paddle.index_select(x_swaped_flattend, index_swaped_flattend)
    gathered = paddle.reshape(gathered, index_swaped.shape)

    out = paddle.transpose(gathered, perm=x_range)

    return out