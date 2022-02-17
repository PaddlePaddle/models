#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
"""
network layers
"""
import collections
import contextlib
import inspect
import six
import sys
from functools import partial
from functools import reduce
import numpy as np
import paddle
import paddle.fluid as fluid
from paddle.fluid import layers
import paddle.fluid.param_attr as attr
import paddle.fluid.layers.utils as utils
from paddle.fluid.dygraph import Embedding, Conv2D, GRUUnit, Layer, to_variable
from paddle.fluid.layers.utils import map_structure, flatten, pack_sequence_as


class EmbeddingLayer(object):
    """
    Embedding Layer class
    """

    def __init__(self, dict_size, emb_dim, name="emb", padding_idx=None):
        """
        initialize
        """
        self.dict_size = dict_size
        self.emb_dim = emb_dim
        self.name = name
        self.padding_idx = padding_idx

    def ops(self):
        """
        operation
        """
        emb = Embedding(
            size=[self.dict_size, self.emb_dim],
            is_sparse=True,
            padding_idx=self.padding_idx,
            param_attr=attr.ParamAttr(
                name=self.name, initializer=fluid.initializer.Xavier()))

        return emb


class FCLayer(object):
    """
    Fully Connect Layer class
    """

    def __init__(self, fc_dim, act, name="fc"):
        """
        initialize
        """
        self.fc_dim = fc_dim
        self.act = act
        self.name = name

    def ops(self):
        """
        operation
        """
        fc = FC(size=self.fc_dim,
                param_attr=attr.ParamAttr(name="%s.w" % self.name),
                bias_attr=attr.ParamAttr(name="%s.b" % self.name),
                act=self.act)
        return fc


class DynamicGRULayer(object):
    """
    Dynamic GRU Layer class
    """

    def __init__(self, gru_dim, name="dyn_gru"):
        """
        initialize
        """
        self.gru_dim = gru_dim
        self.name = name

    def ops(self):
        """
        operation
        """
        gru = DynamicGRU(
            size=self.gru_dim,
            param_attr=attr.ParamAttr(name="%s.w" % self.name),
            bias_attr=attr.ParamAttr(name="%s.b" % self.name))
        return gru


class DynamicLSTMLayer(object):
    """
    Dynamic LSTM Layer class
    """

    def __init__(self, lstm_dim, name="dyn_lstm", is_reverse=False):
        """
        initialize
        """
        self.lstm_dim = lstm_dim
        self.name = name
        self.is_reverse = is_reverse

    def ops(self):
        """
        operation
        """
        lstm_cell = BasicLSTMUnit(
            hidden_size=self.lstm_dim, input_size=self.lstm_dim * 4)
        lstm = RNN(cell=lstm_cell, time_major=True, is_reverse=self.is_reverse)
        return lstm


class DataLayer(object):
    """
    Data Layer class
    """

    def __init__(self):
        """
        initialize
        """
        pass

    def ops(self, name, shape, dtype, lod_level=0):
        """
        operation
        """
        data = fluid.data(
            name=name, shape=shape, dtype=dtype, lod_level=lod_level)
        return data


class ConcatLayer(object):
    """
    Connection Layer class
    """

    def __init__(self, axis):
        """
        initialize
        """
        self.axis = axis

    def ops(self, inputs):
        """
        operation
        """
        concat = fluid.layers.concat(inputs, axis=self.axis)
        return concat


class ReduceMeanLayer(object):
    """
    Reduce Mean Layer class
    """

    def __init__(self):
        """
        initialize
        """
        pass

    def ops(self, input):
        """
        operation
        """
        mean = fluid.layers.reduce_mean(input)
        return mean


class CrossEntropyLayer(object):
    """
    Cross Entropy Calculate Layer
    """

    def __init__(self, name="cross_entropy"):
        """
        initialize
        """
        pass

    def ops(self, input, label):
        """
        operation
        """
        loss = fluid.layers.cross_entropy(input=input, label=label)
        return loss


class SoftmaxWithCrossEntropyLayer(object):
    """
    Softmax with Cross Entropy Calculate Layer
    """

    def __init__(self, name="softmax_with_cross_entropy"):
        """
        initialize
        """
        pass

    def ops(self, input, label):
        """
        operation
        """
        loss = fluid.layers.softmax_with_cross_entropy(
            logits=input, label=label)
        return loss


class CosSimLayer(object):
    """
    Cos Similarly Calculate Layer
    """

    def __init__(self):
        """
        initialize
        """
        pass

    def ops(self, x, y):
        """
        operation
        """
        sim = fluid.layers.cos_sim(x, y)
        return sim


class ElementwiseMaxLayer(object):
    """
    Elementwise Max Layer class
    """

    def __init__(self):
        """
        initialize
        """
        pass

    def ops(self, x, y):
        """
        operation
        """
        max = fluid.layers.elementwise_max(x, y)
        return max


class ElementwiseAddLayer(object):
    """
    Elementwise Add Layer class
    """

    def __init__(self):
        """
        initialize
        """
        pass

    def ops(self, x, y):
        """
        operation
        """
        add = fluid.layers.elementwise_add(x, y)
        return add


class ElementwiseSubLayer(object):
    """
    Elementwise Add Layer class
    """

    def __init__(self):
        """
        initialize
        """
        pass

    def ops(self, x, y):
        """
        operation
        """
        sub = fluid.layers.elementwise_sub(x, y)
        return sub


class ConstantLayer(object):
    """
    Generate A Constant Layer class
    """

    def __init__(self):
        """
        initialize
        """
        pass

    def ops(self, input, shape, dtype, value):
        """
        operation
        """
        shape = list(shape)
        input_shape = fluid.layers.shape(input)
        shape[0] = input_shape[0]
        constant = fluid.layers.fill_constant(shape, dtype, value)
        return constant


class SigmoidLayer(object):
    """
    Sigmoid Layer class
    """

    def __init__(self):
        """
        initialize
        """
        pass

    def ops(self, input):
        """
        operation
        """
        sigmoid = fluid.layers.sigmoid(input)
        return sigmoid


class SoftsignLayer(object):
    """
    Softsign Layer class
    """

    def __init__(self):
        """
        initialize
        """
        pass

    def ops(self, input):
        """
        operation
        """
        softsign = fluid.layers.softsign(input)
        return softsign


class SimpleConvPool(Layer):
    def __init__(self, num_channels, num_filters, filter_size, use_cudnn=False):
        super(SimpleConvPool, self).__init__()
        self._conv2d = Conv2D(
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=filter_size,
            padding=[1, 1],
            use_cudnn=use_cudnn,
            act='relu')

    def forward(self, inputs):
        x = self._conv2d(inputs)
        x = fluid.layers.reduce_max(x, dim=-1)
        x = fluid.layers.reshape(x, shape=[x.shape[0], -1])
        return x


class FC(Layer):
    """
    This interface is used to construct a callable object of the ``FC`` class.
    For more details, refer to code examples.
    It creates a fully connected layer in the network. It can take
    one or multiple ``Tensor`` as its inputs. It creates a Variable called weights for each input tensor,
    which represents a fully connected weight matrix from each input unit to
    each output unit. The fully connected layer multiplies each input tensor
    with its corresponding weight to produce an output Tensor with shape [N, `size`],
    where N is batch size. If multiple input tensors are given, the results of
    multiple output tensors with shape [N, `size`] will be summed up. If ``bias_attr``
    is not None, a bias variable will be created and added to the output.
    Finally, if ``act`` is not None, it will be applied to the output as well.
    When the input is single ``Tensor`` :
    .. math::
        Out = Act({XW + b})
    When the input are multiple ``Tensor`` :
    .. math::
        Out = Act({\sum_{i=0}^{N-1}X_iW_i + b})
    In the above equation:
    * :math:`N`: Number of the input. N equals to len(input) if input is list of ``Tensor`` .
    * :math:`X_i`: The i-th input ``Tensor`` .
    * :math:`W_i`: The i-th weights matrix corresponding i-th input tensor.
    * :math:`b`: The bias parameter created by this layer (if needed).
    * :math:`Act`: The activation function.
    * :math:`Out`: The output ``Tensor`` .
    See below for an example.
    .. code-block:: text
        Given:
            data_1.data = [[[0.1, 0.2]]]
            data_1.shape = (1, 1, 2) # 1 is batch_size
            data_2.data = [[[0.1, 0.2, 0.3]]]
            data_2.shape = (1, 1, 3) # 1 is batch_size
            fc = FC("fc", 2, num_flatten_dims=2)
            out = fc(input=[data_1, data_2])
        Then:
            out.data = [[[0.182996 -0.474117]]]
            out.shape = (1, 1, 2)
    Parameters:
        
        size(int): The number of output units in this layer.
        num_flatten_dims (int, optional): The fc layer can accept an input tensor with more than
            two dimensions. If this happens, the multi-dimension tensor will first be flattened
            into a 2-dimensional matrix. The parameter `num_flatten_dims` determines how the input
            tensor is flattened: the first `num_flatten_dims` (inclusive, index starts from 1)
            dimensions will be flatten to form the first dimension of the final matrix (height of
            the matrix), and the rest `rank(X) - num_flatten_dims` dimensions are flattened to
            form the second dimension of the final matrix (width of the matrix). For example, suppose
            `X` is a 5-dimensional tensor with a shape [2, 3, 4, 5, 6], and `num_flatten_dims` = 3.
            Then, the flattened matrix will have a shape [2 x 3 x 4, 5 x 6] = [24, 30]. Default: 1
        param_attr (ParamAttr or list of ParamAttr, optional): The parameter attribute for learnable
            weights(Parameter) of this layer. Default: None.
        bias_attr (ParamAttr or list of ParamAttr, optional): The attribute for the bias
            of this layer. If it is set to False, no bias will be added to the output units.
            If it is set to None, the bias is initialized zero. Default: None.
        act (str, optional): Activation to be applied to the output of this layer. Default: None.
        is_test(bool, optional): A flag indicating whether execution is in test phase. Default: False.
        dtype(str, optional): Dtype used for weight, it can be "float32" or "float64". Default: "float32".
    Attribute:
        **weight** (list of Parameter): the learnable weights of this layer.
        **bias** (Parameter or None): the learnable bias of this layer.
    Returns:
        None
    
    Examples:
        .. code-block:: python
          from paddle.fluid.dygraph.base import to_variable
          import paddle.fluid as fluid
          from paddle.fluid.dygraph import FC
          import numpy as np
          data = np.random.uniform(-1, 1, [30, 10, 32]).astype('float32')
          with fluid.dygraph.guard():
              fc = FC("fc", 64, num_flatten_dims=2)
              data = to_variable(data)
              conv = fc(data)
    """

    def __init__(self,
                 size,
                 num_flatten_dims=1,
                 param_attr=None,
                 bias_attr=None,
                 act=None,
                 is_test=False,
                 dtype="float32"):
        super(FC, self).__init__(dtype)

        self._size = size
        self._num_flatten_dims = num_flatten_dims
        self._dtype = dtype
        self._param_attr = param_attr
        self._bias_attr = bias_attr
        self._act = act
        self.__w = list()

    def _build_once(self, input):
        i = 0
        for inp, param in self._helper.iter_inputs_and_params(input,
                                                              self._param_attr):
            input_shape = inp.shape

            param_shape = [
                reduce(lambda a, b: a * b, input_shape[self._num_flatten_dims:],
                       1)
            ] + [self._size]
            self.__w.append(
                self.add_parameter(
                    '_w%d' % i,
                    self.create_parameter(
                        attr=param,
                        shape=param_shape,
                        dtype=self._dtype,
                        is_bias=False)))
            i += 1

        size = list([self._size])
        self._b = self.create_parameter(
            attr=self._bias_attr, shape=size, dtype=self._dtype, is_bias=True)

    # TODO(songyouwei): We should remove _w property
    @property
    def _w(self, i=0):
        return self.__w[i]

    @_w.setter
    def _w(self, value, i=0):
        assert isinstance(self.__w[i], Variable)
        self.__w[i].set_value(value)

    @property
    def weight(self):
        if len(self.__w) > 1:
            return self.__w
        else:
            return self.__w[0]

    @weight.setter
    def weight(self, value):
        if len(self.__w) == 1:
            self.__w[0] = value

    @property
    def bias(self):
        return self._b

    @bias.setter
    def bias(self, value):
        self._b = value

    def forward(self, input):
        mul_results = list()
        i = 0
        for inp, param in self._helper.iter_inputs_and_params(input,
                                                              self._param_attr):
            tmp = self._helper.create_variable_for_type_inference(self._dtype)
            self._helper.append_op(
                type="mul",
                inputs={"X": inp,
                        "Y": self.__w[i]},
                outputs={"Out": tmp},
                attrs={
                    "x_num_col_dims": self._num_flatten_dims,
                    "y_num_col_dims": 1
                })
            i += 1
            mul_results.append(tmp)

        if len(mul_results) == 1:
            pre_bias = mul_results[0]
        else:
            pre_bias = self._helper.create_variable_for_type_inference(
                self._dtype)
            self._helper.append_op(
                type="sum",
                inputs={"X": mul_results},
                outputs={"Out": pre_bias},
                attrs={"use_mkldnn": False})

        if self._b is not None:
            pre_activation = self._helper.create_variable_for_type_inference(
                dtype=self._dtype)
            self._helper.append_op(
                type='elementwise_add',
                inputs={'X': [pre_bias],
                        'Y': [self._b]},
                outputs={'Out': [pre_activation]},
                attrs={'axis': self._num_flatten_dims})
        else:
            pre_activation = pre_bias
        # Currently, we don't support inplace in dygraph mode
        return self._helper.append_activation(pre_activation, act=self._act)


class DynamicGRU(Layer):
    def __init__(self,
                 size,
                 param_attr=None,
                 bias_attr=None,
                 is_reverse=False,
                 gate_activation='sigmoid',
                 candidate_activation='tanh',
                 origin_mode=False,
                 init_size=None):
        super(DynamicGRU, self).__init__()
        self.gru_unit = GRUUnit(
            size * 3,
            param_attr=param_attr,
            bias_attr=bias_attr,
            activation=candidate_activation,
            gate_activation=gate_activation,
            origin_mode=origin_mode)
        self.size = size
        self.is_reverse = is_reverse

    def forward(self, inputs, h_0):
        hidden = h_0
        res = []
        for i in range(inputs.shape[1]):
            if self.is_reverse:
                i = inputs.shape[1] - 1 - i
            input_ = inputs[:, i:i + 1, :]
            input_ = fluid.layers.reshape(
                input_, [-1, input_.shape[2]], inplace=False)
            hidden, reset, gate = self.gru_unit(input_, hidden)
            hidden_ = fluid.layers.reshape(
                hidden, [-1, 1, hidden.shape[1]], inplace=False)
            res.append(hidden_)
        if self.is_reverse:
            res = res[::-1]
        res = fluid.layers.concat(res, axis=1)
        return res


class RNNUnit(Layer):
    def get_initial_states(self,
                           batch_ref,
                           shape=None,
                           dtype=None,
                           init_value=0,
                           batch_dim_idx=0):
        """
        Generate initialized states according to provided shape, data type and
        value.

        Parameters:
            batch_ref: A (possibly nested structure of) tensor variable[s].
                The first dimension of the tensor will be used as batch size to
                initialize states.
            shape: A (possiblely nested structure of) shape[s], where a shape is
                represented as a list/tuple of integer). -1(for batch size) will
                beautomatically inserted if shape is not started with it. If None,
                property `state_shape` will be used. The default value is None.
            dtype: A (possiblely nested structure of) data type[s]. The structure
                must be same as that of `shape`, except when all tensors' in states
                has the same data type, a single data type can be used. If None and
                property `cell.state_shape` is not available, float32 will be used
                as the data type. The default value is None.
            init_value: A float value used to initialize states.
        
        Returns:
            Variable: tensor variable[s] packed in the same structure provided \
                by shape, representing the initialized states.
        """
        # TODO: use inputs and batch_size
        batch_ref = flatten(batch_ref)[0]

        def _is_shape_sequence(seq):
            if sys.version_info < (3, ):
                integer_types = (
                    int,
                    long, )
            else:
                integer_types = (int, )
            """For shape, list/tuple of integer is the finest-grained objection"""
            if (isinstance(seq, list) or isinstance(seq, tuple)):
                if reduce(lambda flag, x: isinstance(x, integer_types) and flag,
                          seq, True):
                    return False
            # TODO: Add check for the illegal
            if isinstance(seq, dict):
                return True
            return (isinstance(seq, collections.Sequence) and
                    not isinstance(seq, six.string_types))

        class Shape(object):
            def __init__(self, shape):
                self.shape = shape if shape[0] == -1 else ([-1] + list(shape))

        # nested structure of shapes
        states_shapes = self.state_shape if shape is None else shape
        is_sequence_ori = utils.is_sequence
        utils.is_sequence = _is_shape_sequence
        states_shapes = map_structure(lambda shape: Shape(shape), states_shapes)
        utils.is_sequence = is_sequence_ori

        # nested structure of dtypes
        try:
            states_dtypes = self.state_dtype if dtype is None else dtype
        except NotImplementedError:  # use fp32 as default
            states_dtypes = "float32"
        if len(flatten(states_dtypes)) == 1:
            dtype = flatten(states_dtypes)[0]
            states_dtypes = map_structure(lambda shape: dtype, states_shapes)

        init_states = map_structure(
            lambda shape, dtype: fluid.layers.fill_constant_batch_size_like(
                input=batch_ref,
                shape=shape.shape,
                dtype=dtype,
                value=init_value,
                input_dim_idx=batch_dim_idx), states_shapes, states_dtypes)
        return init_states

    @property
    def state_shape(self):
        """
        Abstract method (property).
        Used to initialize states.
        A (possiblely nested structure of) shape[s], where a shape is represented
        as a list/tuple of integers (-1 for batch size would be automatically
        inserted into a shape if shape is not started with it). 
        Not necessary to be implemented if states are not initialized by
        `get_initial_states` or the `shape` argument is provided when using
        `get_initial_states`.
        """
        raise NotImplementedError(
            "Please add implementaion for `state_shape` in the used cell.")

    @property
    def state_dtype(self):
        """
        Abstract method (property).
        Used to initialize states.
        A (possiblely nested structure of) data types[s]. The structure must be
        same as that of `shape`, except when all tensors' in states has the same
        data type, a signle data type can be used.
        Not necessary to be implemented if states are not initialized
        by `get_initial_states` or the `dtype` argument is provided when using
        `get_initial_states`.
        """
        raise NotImplementedError(
            "Please add implementaion for `state_dtype` in the used cell.")


class BasicLSTMUnit(RNNUnit):
    """
    ****
    BasicLSTMUnit class, Using basic operator to build LSTM
    The algorithm can be described as the code below.
        .. math::
           i_t &= \sigma(W_{ix}x_{t} + W_{ih}h_{t-1} + b_i)
           f_t &= \sigma(W_{fx}x_{t} + W_{fh}h_{t-1} + b_f + forget_bias )
           o_t &= \sigma(W_{ox}x_{t} + W_{oh}h_{t-1} + b_o)
           \\tilde{c_t} &= tanh(W_{cx}x_t + W_{ch}h_{t-1} + b_c)
           c_t &= f_t \odot c_{t-1} + i_t \odot \\tilde{c_t}
           h_t &= o_t \odot tanh(c_t)
        - $W$ terms denote weight matrices (e.g. $W_{ix}$ is the matrix
          of weights from the input gate to the input)
        - The b terms denote bias vectors ($bx_i$ and $bh_i$ are the input gate bias vector).
        - sigmoid is the logistic sigmoid function.
        - $i, f, o$ and $c$ are the input gate, forget gate, output gate,
          and cell activation vectors, respectively, all of which have the same size as
          the cell output activation vector $h$.
        - The :math:`\odot` is the element-wise product of the vectors.
        - :math:`tanh` is the activation functions.
        - :math:`\\tilde{c_t}` is also called candidate hidden state,
          which is computed based on the current input and the previous hidden state.
    Args:
        hidden_size (integer): The hidden size used in the Unit.
        param_attr(ParamAttr|None): The parameter attribute for the learnable
            weight matrix. Note:
            If it is set to None or one attribute of ParamAttr, lstm_unit will
            create ParamAttr as param_attr. If the Initializer of the param_attr
            is not set, the parameter is initialized with Xavier. Default: None.
        bias_attr (ParamAttr|None): The parameter attribute for the bias
            of LSTM unit.
            If it is set to None or one attribute of ParamAttr, lstm_unit will 
            create ParamAttr as bias_attr. If the Initializer of the bias_attr
            is not set, the bias is initialized as zero. Default: None.
        gate_activation (function|None): The activation function for gates (actGate).
                                  Default: 'fluid.layers.sigmoid'
        activation (function|None): The activation function for cells (actNode).
                             Default: 'fluid.layers.tanh'
        forget_bias(float|1.0): forget bias used when computing forget gate
        dtype(string): data type used in this unit
    """

    def __init__(self,
                 hidden_size,
                 input_size,
                 param_attr=None,
                 bias_attr=None,
                 gate_activation=None,
                 activation=None,
                 forget_bias=1.0,
                 dtype='float32'):
        super(BasicLSTMUnit, self).__init__(dtype)

        self._hidden_size = hidden_size
        self._param_attr = param_attr
        self._bias_attr = bias_attr
        self._gate_activation = gate_activation or layers.sigmoid
        self._activation = activation or layers.tanh
        self._forget_bias = layers.fill_constant(
            [1], dtype=dtype, value=forget_bias)
        self._forget_bias.stop_gradient = False
        self._dtype = dtype
        self._input_size = input_size

        self._weight = self.create_parameter(
            attr=self._param_attr,
            shape=[
                self._input_size + self._hidden_size, 4 * self._hidden_size
            ],
            dtype=self._dtype)

        self._bias = self.create_parameter(
            attr=self._bias_attr,
            shape=[4 * self._hidden_size],
            dtype=self._dtype,
            is_bias=True)

    def forward(self, input, state):
        pre_hidden, pre_cell = state
        concat_input_hidden = layers.concat([input, pre_hidden], axis=1)

        gate_input = layers.matmul(x=concat_input_hidden, y=self._weight)

        gate_input = layers.elementwise_add(gate_input, self._bias)
        i, j, f, o = layers.split(gate_input, num_or_sections=4, dim=-1)
        new_cell = layers.elementwise_add(
            layers.elementwise_mul(
                pre_cell,
                layers.sigmoid(layers.elementwise_add(f, self._forget_bias))),
            layers.elementwise_mul(layers.sigmoid(i), layers.tanh(j)))
        new_hidden = layers.tanh(new_cell) * layers.sigmoid(o)

        return new_hidden, [new_hidden, new_cell]

    @property
    def state_shape(self):
        return [[self._hidden_size], [self._hidden_size]]


class RNN(Layer):
    def __init__(self, cell, is_reverse=False, time_major=False, **kwargs):
        super(RNN, self).__init__()
        self.cell = cell
        if not hasattr(self.cell, "call"):
            self.cell.call = self.cell.forward
        self.is_reverse = is_reverse
        self.time_major = time_major
        self.batch_index, self.time_step_index = (1, 0) if time_major else (0,
                                                                            1)

    def forward(self,
                inputs,
                initial_states=None,
                sequence_length=None,
                **kwargs):
        if fluid.in_dygraph_mode():

            class OutputArray(object):
                def __init__(self, x):
                    self.array = [x]

                def append(self, x):
                    self.array.append(x)

            def _maybe_copy(state, new_state, step_mask):
                # TODO: use where_op
                new_state = fluid.layers.elementwise_mul(
                    new_state, step_mask,
                    axis=0) - fluid.layers.elementwise_mul(
                        state, (step_mask - 1), axis=0)
                return new_state

            flat_inputs = flatten(inputs)
            batch_size, time_steps = (
                flat_inputs[0].shape[self.batch_index],
                flat_inputs[0].shape[self.time_step_index])

            if initial_states is None:
                initial_states = self.cell.get_initial_states(
                    batch_ref=inputs, batch_dim_idx=self.batch_index)

            if not self.time_major:
                inputs = map_structure(
                    lambda x: fluid.layers.transpose(x, [1, 0] + list(
                        range(2, len(x.shape)))), inputs)

            if sequence_length is not None:
                mask = fluid.layers.sequence_mask(
                    sequence_length,
                    maxlen=time_steps,
                    dtype=flatten(initial_states)[0].dtype)
                mask = fluid.layers.transpose(mask, [1, 0])

            if self.is_reverse:
                inputs = map_structure(lambda x: fluid.layers.reverse(x, axis=[0]), inputs)
                mask = fluid.layers.reverse(
                    mask, axis=[0]) if sequence_length is not None else None

            states = initial_states
            outputs = []
            for i in range(time_steps):
                step_inputs = map_structure(lambda x: x[i], inputs)
                step_outputs, new_states = self.cell(step_inputs, states,
                                                     **kwargs)
                if sequence_length is not None:
                    new_states = map_structure(
                        partial(
                            _maybe_copy, step_mask=mask[i]),
                        states,
                        new_states)
                states = new_states
                if i == 0:
                    outputs = map_structure(lambda x: OutputArray(x),
                                            step_outputs)
                else:
                    map_structure(lambda x, x_array: x_array.append(x),
                                  step_outputs, outputs)

            final_outputs = map_structure(
                lambda x: fluid.layers.stack(x.array, axis=self.time_step_index
                                             ), outputs)

            if self.is_reverse:
                final_outputs = map_structure(
                    lambda x: fluid.layers.reverse(x, axis=self.time_step_index
                                                   ), final_outputs)

            final_states = new_states
        else:
            final_outputs, final_states = fluid.layers.rnn(
                self.cell,
                inputs,
                initial_states=initial_states,
                sequence_length=sequence_length,
                time_major=self.time_major,
                is_reverse=self.is_reverse,
                **kwargs)
        return final_outputs, final_states


class EncoderCell(RNNUnit):
    def __init__(self, num_layers, input_size, hidden_size, dropout_prob=0.):
        super(EncoderCell, self).__init__()
        self.num_layers = num_layers
        self.dropout_prob = dropout_prob

        self.lstm_cells = list()
        for i in range(self.num_layers):
            self.lstm_cells.append(
                self.add_sublayer("layer_%d" % i,
                                  BasicLSTMUnit(input_size if i == 0 else
                                                hidden_size, hidden_size)))

    def forward(self, step_input, states):
        new_states = []
        for i in range(self.num_layers):
            out, new_state = self.lstm_cells[i](step_input, states[i])
            step_input = layers.dropout(
                out, self.dropout_prob) if self.dropout_prob > 0 else out
            new_states.append(new_state)
        return step_input, new_states

    @property
    def state_shape(self):
        return [cell.state_shape for cell in self.lstm_cells]


class BasicGRUUnit(Layer):
    """
    ****
    BasicGRUUnit class, using basic operators to build GRU
    The algorithm can be described as the equations below.
        .. math::
            u_t & = actGate(W_ux xu_{t} + W_uh h_{t-1} + b_u)
            r_t & = actGate(W_rx xr_{t} + W_rh h_{t-1} + b_r)
            m_t & = actNode(W_cx xm_t + W_ch dot(r_t, h_{t-1}) + b_m)
            h_t & = dot(u_t, h_{t-1}) + dot((1-u_t), m_t)
    Args:
        hidden_size (integer): The hidden size used in the Unit.
        param_attr(ParamAttr|None): The parameter attribute for the learnable
            weight matrix. Note:
            If it is set to None or one attribute of ParamAttr, gru_unit will
            create ParamAttr as param_attr. If the Initializer of the param_attr
            is not set, the parameter is initialized with Xavier. Default: None.
        bias_attr (ParamAttr|None): The parameter attribute for the bias
            of GRU unit.
            If it is set to None or one attribute of ParamAttr, gru_unit will 
            create ParamAttr as bias_attr. If the Initializer of the bias_attr
            is not set, the bias is initialized zero. Default: None.
        gate_activation (function|None): The activation function for gates (actGate).
                                  Default: 'fluid.layers.sigmoid'
        activation (function|None): The activation function for cell (actNode).
                             Default: 'fluid.layers.tanh'
        dtype(string): data type used in this unit
    Examples:
        .. code-block:: python
            import paddle.fluid.layers as layers
            from paddle.fluid.contrib.layers import BasicGRUUnit
            input_size = 128
            hidden_size = 256
            input = layers.data( name = "input", shape = [-1, input_size], dtype='float32')
            pre_hidden = layers.data( name = "pre_hidden", shape=[-1, hidden_size], dtype='float32')
            gru_unit = BasicGRUUnit( "gru_unit", hidden_size )
            new_hidden = gru_unit( input, pre_hidden )
    """

    def __init__(self,
                 hidden_size,
                 input_size,
                 param_attr=None,
                 bias_attr=None,
                 gate_activation=None,
                 activation=None,
                 dtype='float32'):
        super(BasicGRUUnit, self).__init__(dtype)

        self._hiden_size = hidden_size
        self._input_size = input_size
        self._param_attr = param_attr
        self._bias_attr = bias_attr
        self._gate_activation = gate_activation or layers.sigmoid
        self._activation = activation or layers.tanh
        self._dtype = dtype

        self._gate_weight = self.create_parameter(
            attr=self._param_attr,
            shape=[self._input_size + self._hiden_size, 2 * self._hiden_size],
            dtype=self._dtype)

        self._candidate_weight = self.create_parameter(
            attr=self._param_attr,
            shape=[self._input_size + self._hiden_size, self._hiden_size],
            dtype=self._dtype)

        self._gate_bias = self.create_parameter(
            attr=self._bias_attr,
            shape=[2 * self._hiden_size],
            dtype=self._dtype,
            is_bias=True)
        self._candidate_bias = self.create_parameter(
            attr=self._bias_attr,
            shape=[self._hiden_size],
            dtype=self._dtype,
            is_bias=True)

    def forward(self, input, state):
        pre_hidden = state
        concat_input_hidden = fluid.layers.concat([input, pre_hidden], axis=1)

        gate_input = layers.matmul(x=concat_input_hidden, y=self._gate_weight)

        gate_input = layers.elementwise_add(gate_input, self._gate_bias)

        gate_input = self._gate_activation(gate_input)
        r, u = layers.split(gate_input, num_or_sections=2, dim=1)

        r_hidden = r * pre_hidden

        candidate = layers.matmul(
            layers.concat([input, r_hidden], 1), self._candidate_weight)
        candidate = layers.elementwise_add(candidate, self._candidate_bias)

        c = self._activation(candidate)
        new_hidden = u * pre_hidden + (1 - u) * c

        return new_hidden
