#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from paddle import ParamAttr
from paddle.nn import Layer
import paddle


def _convert_param_attr_to_list(param_attr, n):
    if isinstance(param_attr, (list, tuple)):
        assert len(param_attr) == n, (
            "length of param_attr should be %d when it is a list/tuple" % n)
        param_attrs = []
        for attr in param_attr:
            if isinstance(attr, bool):
                if attr:
                    param_attrs.append(ParamAttr._to_attr(None))
                else:
                    param_attrs.append(False)
            else:
                param_attrs.append(ParamAttr._to_attr(attr))
    elif isinstance(param_attr, bool):
        param_attrs = []
        if param_attr:
            param_attrs = [ParamAttr._to_attr(None) for i in range(n)]
        else:
            param_attrs = [False] * n
    else:
        param_attrs = []
        attr = ParamAttr._to_attr(param_attr)
        for i in range(n):
            attr_i = copy.deepcopy(attr)
            if attr.name:
                attr_i.name = attr_i.name + "_" + str(i)
            param_attrs.append(attr_i)
    return param_attrs


class Linear3D(Layer):
    def __init__(self,
                 hidden_size,
                 num_attention_heads,
                 size_per_head,
                 weight_attr=None,
                 bias_attr=None):
        super(Linear3D, self).__init__()
        self._dtype = self._helper.get_default_dtype()
        self._weight_attr = weight_attr
        self._bias_attr = bias_attr
        self.weight = self.create_parameter(
            shape=[hidden_size, hidden_size],
            attr=self._weight_attr,
            dtype=self._dtype,
            is_bias=False)
        self.bias = self.create_parameter(
            shape=[hidden_size],
            attr=self._bias_attr,
            dtype=self._dtype,
            is_bias=True)
        self.size_per_head = size_per_head
        self.num_attention_heads = num_attention_heads
        self.hidden_size = hidden_size

    def forward(self, input):
        # abc,cde->adbe
        reshape_input = paddle.unsqueeze(input, 1)
        reshape_w = paddle.reshape(
            self.weight,
            [self.hidden_size, self.num_attention_heads, self.size_per_head])
        reshape_w = paddle.transpose(reshape_w, [1, 0, 2])
        reshape_w = paddle.unsqueeze(reshape_w, 0)
        result = paddle.matmul(reshape_input, reshape_w)
        reshape_b = paddle.reshape(
            self.bias, [1, self.num_attention_heads, 1, self.size_per_head])
        result += reshape_b
        return result
