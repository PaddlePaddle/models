#!/usr/bin/env python
# -*- coding: utf-8 -*-
################################################################################
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
################################################################################
"""
Wrappers for fluid.layers. It helps to easily share parameters between layers.
"""

import operator
from collections import OrderedDict

import paddle.fluid.layers as layers
import paddle.fluid.unique_name as unique_name
from paddle.fluid.param_attr import ParamAttr


def update_attr(attr, name, prefix=None, suffix="W"):
    if attr == False:
        return False

    if prefix:
        name = prefix + "." + name
    new_name = unique_name.generate(name + "." + suffix)
    if attr is None:
        attr = ParamAttr(name=new_name)
    elif attr.name is None:
        attr.name = new_name
    return attr


class BaseLayer(object):
    def __init__(self):
        self._parameters = OrderedDict()
        self._layers = OrderedDict()

    def __getattr__(self, name):
        if '_parameters' in self.__dict__:
            _parameters = self.__dict__['_parameters']
            if name in _parameters:
                return _parameters[name]
        if '_layers' in self.__dict__:
            _layers = self.__dict__['_layers']
            if name in _layers:
                return _layers[name]
        if name in self.__dict__:
            return self.__dict__[name]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, name))

    def __setattr__(self, name, value):
        def remove_from(*dicts):
            for d in dicts:
                if name in d:
                    del d[name]

        if isinstance(value, ParamAttr):
            self._parameters[name] = value
            remove_from(self.__dict__, self._layers)
        elif isinstance(value, BaseLayer):
            self._layers[name] = value
            remove_from(self.__dict__, self._parameters)
        else:
            object.__setattr__(self, name, value)

    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class LayerList(BaseLayer):
    def __init__(self, layers):
        super(LayerList, self).__init__()
        self += layers

    def _get_abs_string_index(self, idx):
        """Get the absolute index for the list of layers"""
        idx = operator.index(idx)
        if not (-len(self) <= idx < len(self)):
            raise IndexError('index {} is out of range'.format(idx))
        if idx < 0:
            idx += len(self)
        return str(idx)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return self.__class__(list(self._layers.values())[idx])
        else:
            return self._layers[self._get_abs_string_index(idx)]

    def __setitem__(self, idx, layer):
        idx = self._get_abs_string_index(idx)
        return setattr(self, str(idx), layer)

    def __delitem__(self, idx):
        if isinstance(idx, slice):
            for k in range(len(self._layers))[idx]:
                delattr(self, str(k))
        else:
            delattr(self, self._get_abs_string_index(idx))
        # To preserve numbering, self._layers is being reconstructed with layers after deletion
        str_indices = [str(i) for i in range(len(self._layers))]
        self._layers = OrderedDict(
            list(zip(str_indices, self._layers.values())))

    def __len__(self):
        return len(self._layers)

    def __iter__(self):
        return iter(self._layers.values())

    def __iadd__(self, layers):
        return self.extend(layers)

    def extend(self, layers):
        if not isinstance(layers, (list, tuple)):
            raise TypeError("LayerList.extend should be called with a "
                            "list or tuple, but got " + type(layers).__name__)
        offset = len(self)
        for i, layer in enumerate(layers):
            self._layers[str(offset + i)] = layer
        return self


class Sequential(BaseLayer):
    def __init__(self, *layers):
        super(Sequential, self).__init__()
        for idx, layer in enumerate(layers):
            if not isinstance(layer, BaseLayer):
                raise TypeError("{} is not a BaseLayer subclass".format(
                    type(layer)))
            self._layers[str(idx)] = layer

    def __call__(self, input):
        for layer in self._layers.values():
            input = layer(input)
        return input


class Embedding(BaseLayer):
    def __init__(self,
                 size,
                 is_sparse=False,
                 is_distributed=False,
                 padding_idx=None,
                 param_attr=None,
                 dtype='float32',
                 name=None):
        super(Embedding, self).__init__()
        self.name = name or "Embedding"
        self.size = size
        self.is_sparse = is_sparse
        self.is_distributed = False
        self.padding_idx = padding_idx
        self.param_attr = update_attr(param_attr, self.name, suffix="W")
        self.dtype = dtype

    def __call__(self, input):
        return layers.embedding(
            input=input,
            size=self.size,
            is_sparse=self.is_sparse,
            is_distributed=self.is_distributed,
            padding_idx=self.padding_idx,
            param_attr=self.param_attr,
            dtype=self.dtype)


class FC(BaseLayer):
    def __init__(self,
                 size,
                 num_flatten_dims=1,
                 param_attr=None,
                 bias_attr=None,
                 act=None,
                 is_test=False,
                 name=None):
        super(FC, self).__init__()
        self.name = name or "FC"
        self.size = size
        self.num_flatten_dims = num_flatten_dims
        self.param_attr = update_attr(param_attr, self.name, suffix="W")
        self.bias_attr = update_attr(bias_attr, self.name, suffix="b")
        self.act = act
        self.is_test = False

    def __call__(self, input, name=None):
        assert not isinstance(input, (list, tuple))
        return layers.fc(input=input,
                         size=self.size,
                         num_flatten_dims=self.num_flatten_dims,
                         param_attr=self.param_attr,
                         bias_attr=self.bias_attr,
                         act=self.act,
                         is_test=self.is_test,
                         name=name)


class DynamicGRU(BaseLayer):
    def __init__(self,
                 hidden_dim,
                 param_attr=None,
                 bias_attr=None,
                 input_param_attr=None,
                 input_bias_attr=None,
                 is_reverse=False,
                 gate_activation='sigmoid',
                 candidate_activation='tanh',
                 origin_mode=False,
                 name=None):
        super(DynamicGRU, self).__init__()
        self.name = name or "DynamicGRU"
        self.hidden_dim = hidden_dim
        self.param_attr = update_attr(param_attr, self.name, suffix="hidden.W")
        self.bias_attr = update_attr(bias_attr, self.name, suffix="hidden.b")
        self.input_param_attr = update_attr(
            input_param_attr, self.name, suffix="input.W")
        self.input_bias_attr = update_attr(
            input_bias_attr, self.name, suffix="input.b")
        self.is_reverse = is_reverse
        self.gate_activation = gate_activation
        self.candidate_activation = candidate_activation
        self.origin_mode = origin_mode

    def __call__(self, input, state=None):
        gru_input = layers.fc(input=input,
                              size=self.hidden_dim * 3,
                              param_attr=self.input_param_attr,
                              bias_attr=self.input_bias_attr)
        return layers.dynamic_gru(
            input=gru_input,
            size=self.hidden_dim,
            param_attr=self.param_attr,
            bias_attr=self.bias_attr,
            is_reverse=self.is_reverse,
            gate_activation=self.gate_activation,
            candidate_activation=self.candidate_activation,
            h_0=state,
            origin_mode=self.origin_mode)


class GRU(BaseLayer):
    def __init__(self,
                 hidden_dim,
                 num_layers=1,
                 bidirectional=False,
                 dropout=0.0,
                 name=None):
        super(GRU, self).__init__()
        if dropout > 0 and num_layers == 1:
            raise ValueError(
                "Non-zero dropout expects num_layers greater than 1")
        self.name = name or "GRU"
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.dropout = dropout
        rnns = []
        for l in range(num_layers):
            inners = []
            inners.append(
                DynamicGRU(
                    hidden_dim=hidden_dim, name="{}_l{}".format(self.name, l)))
            if bidirectional:
                inners.append(
                    DynamicGRU(
                        hidden_dim=hidden_dim,
                        name="{}_l{}_reverse".format(self.name, l),
                        is_reverse=True))
            rnns.append(LayerList(inners))
        self.rnns = LayerList(rnns)

    def __call__(self, input, hidden=None):
        if hidden is not None:
            assert len(hidden) == self.num_layers
            assert len(hidden[0]) == self.num_directions
        else:
            hidden = [[None] * self.num_directions] * self.num_layers
        new_hidden = []
        for l in range(self.num_layers):
            layer_output = []
            layer_hidden = []
            for i, inner in enumerate(self.rnns[l]):
                output = inner(input, hidden[l][i])
                layer_output.append(output)
                if inner.is_reverse:
                    layer_hidden.append(layers.sequence_first_step(output))
                else:
                    layer_hidden.append(layers.sequence_last_step(output))
            input = layers.concat(layer_output, axis=1)
            if self.dropout > 0 and l + 1 < self.num_layers:
                input = layers.dropout(
                    input,
                    dropout_prob=self.dropout,
                    dropout_implementation='upscale_in_train')
            new_hidden.append(layers.concat(layer_hidden, axis=1))
        return input, new_hidden


class GRUCell(BaseLayer):
    def __init__(self,
                 hidden_dim,
                 param_attr=None,
                 bias_attr=None,
                 input_param_attr=None,
                 input_bias_attr=None,
                 activation='tanh',
                 gate_activation='sigmoid',
                 origin_mode=False,
                 name=None):
        super(GRUCell, self).__init__()
        self.name = name or "GRUCell"
        self.hidden_dim = hidden_dim
        self.param_attr = update_attr(param_attr, self.name, suffix="hidden.W")
        self.bias_attr = update_attr(bias_attr, self.name, suffix="hidden.b")
        self.input_param_attr = update_attr(
            input_param_attr, self.name, suffix="input.W")
        self.input_bias_attr = update_attr(
            input_bias_attr, self.name, suffix="input.b")
        self.activation = activation
        self.gate_activation = gate_activation
        self.origin_mode = origin_mode

    def __call__(self, input, hidden):
        gru_input = layers.fc(input=input,
                              size=self.hidden_dim * 3,
                              param_attr=self.input_param_attr,
                              bias_attr=self.input_bias_attr)
        new_hidden, _, _ = layers.gru_unit(
            input=gru_input,
            hidden=hidden,
            size=self.hidden_dim * 3,
            param_attr=self.param_attr,
            bias_attr=self.bias_attr,
            activation=self.activation,
            gate_activation=self.gate_activation,
            origin_mode=self.origin_mode)
        return new_hidden, new_hidden


class StackedGRUCell(BaseLayer):
    def __init__(self, hidden_dim, num_layers=1, dropout=0.0, name=None):
        super(StackedGRUCell, self).__init__()
        if dropout > 0 and num_layers == 1:
            raise ValueError(
                "Non-zero dropout expects num_layers greater than 1")

        self.name = name or "StackedGRUCell"
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        cells = [
            GRUCell(
                hidden_dim=hidden_dim, name="{}_l{}".format(self.name, l))
            for l in range(self.num_layers)
        ]
        self.cells = LayerList(cells)

    def __call__(self, input, hidden):
        assert len(hidden) == self.num_layers
        new_hidden = []
        for cell, hid in zip(self.cells, hidden):
            input, new_hid = cell(input, hid)
            new_hidden += [new_hid]
            if self.dropout > 0:
                input = layers.dropout(
                    input,
                    dropout_prob=self.dropout,
                    dropout_implementation='upscale_in_train')
        output = new_hidden[-1]
        return output, new_hidden


class Dropout(BaseLayer):
    def __init__(self, dropout_prob, is_test=False, seed=None, name=None):
        super(Dropout, self).__init__()
        self.dropout_prob = dropout_prob
        self.is_test = is_test
        self.seed = seed
        self.name = name

    def __call__(self, input):
        if self.dropout_prob > 0.0:
            return layers.dropout(
                input,
                dropout_prob=self.dropout_prob,
                is_test=self.is_test,
                seed=self.seed,
                name=self.name,
                dropout_implementation='upscale_in_train')
        else:
            return input
