# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn.utils import weight_norm

__all__ = [
    'BoWEncoder', 'CNNEncoder', 'GRUEncoder', 'LSTMEncoder', 'RNNEncoder',
    'TCNEncoder'
]


class BoWEncoder(nn.Layer):
    """
    A `BoWEncoder` takes as input a sequence of vectors and returns a
    single vector, which simply sums the embeddings of a sequence across the time dimension. 
    The input to this module is of shape `(batch_size, num_tokens, emb_dim)`, 
    and the output is of shape `(batch_size, emb_dim)`.

    Args:
        # TODO: unify the docstring style with PaddlePaddle.
        emb_dim(obj:`int`, required): It is the input dimension to the encoder.
    """

    def __init__(self, emb_dim):
        super().__init__()
        self._emb_dim = emb_dim

    def get_input_dim(self):
        """
        Returns the dimension of the vector input for each element in the sequence input
        to a `BoWEncoder`. This is not the shape of the input tensor, but the
        last element of that shape.
        """
        return self._emb_dim

    def get_output_dim(self):
        """
        Returns the dimension of the final vector output by this `BoWEncoder`.  This is not
        the shape of the returned tensor, but the last element of that shape.
        """
        return self._emb_dim

    def forward(self, inputs, mask=None):
        """
        It simply sums the embeddings of a sequence across the time dimension.

        Args:
            inputs (obj: `paddle.Tensor`): Shape as `(batch_size, num_tokens, emb_dim)`
            mask (obj: `paddle.Tensor`, optional, defaults to `None`): Shape same as `inputs`. Its each elements identify whether is padding token or not. 
                If True, not padding token. If False, padding token.

        Returns:
            summed (obj: `paddle.Tensor`): Shape of `(batch_size, emb_dim)`. The result vector of BagOfEmbedding.

        """
        if mask is not None:
            inputs = inputs * mask

        # Shape: (batch_size, embedding_dim)
        summed = inputs.sum(axis=1)
        return summed


class CNNEncoder(nn.Layer):
    """
    A `CNNEncoder` takes as input a sequence of vectors and returns a
    single vector, a combination of multiple convolution layers and max pooling layers.
    The input to this module is of shape `(batch_size, num_tokens, emb_dim)`, 
    and the output is of shape `(batch_size, ouput_dim)` or `(batch_size, len(ngram_filter_sizes) * num_filter)`.

    The CNN has one convolution layer for each ngram filter size. Each convolution operation gives
    out a vector of size num_filter. The number of times a convolution layer will be used
    is `num_tokens - ngram_size + 1`. The corresponding maxpooling layer aggregates all these
    outputs from the convolution layer and outputs the max.

    This operation is repeated for every ngram size passed, and consequently the dimensionality of
    the output after maxpooling is `len(ngram_filter_sizes) * num_filter`.  This then gets
    (optionally) projected down to a lower dimensional output, specified by `output_dim`.

    We then use a fully connected layer to project in back to the desired output_dim.  For more
    details, refer to "A Sensitivity Analysis of (and Practitioners’ Guide to) Convolutional Neural
    Networks for Sentence Classification", Zhang and Wallace 2016, particularly Figure 1.
    ref: https://arxiv.org/abs/1510.03820

    Args:
        emb_dim(object:`int`, required):
            This is the input dimension to the encoder.
        num_filter(object:`int`, required):
            This is the output dim for each convolutional layer, which is the number of "filters"
            learned by that layer.
        ngram_filter_sizes(object: `Tuple[int]`, optional, default to `(2, 3, 4, 5)`):
            This specifies both the number of convolutional layers we will create and their sizes.  The
            default of `(2, 3, 4, 5)` will have four convolutional layers, corresponding to encoding
            ngrams of size 2 to 5 with some number of filters.
        conv_layer_activation(object: `str`, optional, default to `tanh`):
            Activation to use after the convolution layers.
        output_dim(object: `int`, optional, default to `None`):
            After doing convolutions and pooling, we'll project the collected features into a vector of
            this size.  If this value is `None`, we will just return the result of the max pooling,
            giving an output of shape `len(ngram_filter_sizes) * num_filter`.

    """

    def __init__(self,
                 emb_dim,
                 num_filter,
                 ngram_filter_sizes=(2, 3, 4, 5),
                 conv_layer_activation=nn.Tanh(),
                 output_dim=None,
                 **kwargs):
        super().__init__()
        self._emb_dim = emb_dim
        self._num_filter = num_filter
        self._ngram_filter_sizes = ngram_filter_sizes
        self._activation = conv_layer_activation
        self._output_dim = output_dim

        self.convs = [
            nn.Conv2D(
                in_channels=1,
                out_channels=self._num_filter,
                kernel_size=(i, self._emb_dim),
                **kwargs) for i in self._ngram_filter_sizes
        ]

        maxpool_output_dim = self._num_filter * len(self._ngram_filter_sizes)
        if self._output_dim:
            self.projection_layer = nn.Linear(maxpool_output_dim,
                                              self._output_dim)
        else:
            self.projection_layer = None
            self._output_dim = maxpool_output_dim

    def get_input_dim(self):
        """
        Returns the dimension of the vector input for each element in the sequence input
        to a `CNNEncoder`. This is not the shape of the input tensor, but the
        last element of that shape.
        """
        return self._emb_dim

    def get_output_dim(self):
        """
        Returns the dimension of the final vector output by this `CNNEncoder`.  This is not
        the shape of the returned tensor, but the last element of that shape.
        """
        return self._output_dim

    def forward(self, inputs, mask=None):
        """
        The combination of multiple convolution layers and max pooling layers.

        Args:
            inputs (obj: `paddle.Tensor`, required): Shape as `(batch_size, num_tokens, emb_dim)`
            mask (obj: `paddle.Tensor`, optional, defaults to `None`): Shape same as `inputs`. 
                Its each elements identify whether is padding token or not. 
                If True, not padding token. If False, padding token.

        Returns:
            result (obj: `paddle.Tensor`): If output_dim is None, the result shape 
                is of `(batch_size, output_dim)`; if not, the result shape 
                is of `(batch_size, len(ngram_filter_sizes) * num_filter)`.

        """
        if mask is not None:
            inputs = inputs * mask

        # Shape: (batch_size, 1, num_tokens, emb_dim) = (N, C, H, W)
        inputs = inputs.unsqueeze(1)

        # If output_dim is None, result shape of (batch_size, len(ngram_filter_sizes) * num_filter));
        # else, result shape of (batch_size, output_dim).
        convs_out = [
            self._activation(conv(inputs)).squeeze(3) for conv in self.convs
        ]
        maxpool_out = [
            F.max_pool1d(
                t, kernel_size=t.shape[2]).squeeze(2) for t in convs_out
        ]
        result = paddle.concat(maxpool_out, axis=1)

        if self.projection_layer is not None:
            result = self.projection_layer(result)
        return result


class GRUEncoder(nn.Layer):
    """
    A GRUEncoder takes as input a sequence of vectors and returns a
    single vector, which is a combination of multiple GRU layers.
    The input to this module is of shape `(batch_size, num_tokens, input_size)`, 
    The output is of shape `(batch_size, hidden_size*2)` if GRU is bidirectional;
    If not, output is of shape `(batch_size, hidden_size)`.

    Paddle's GRU have two outputs: the hidden state for every time step at last layer, 
    and the hidden state at the last time step for every layer.
    If `pooling_type` is None, we perform the pooling on the hidden state of every time 
    step at last layer to create a single vector. If not None, we use the hidden state 
    of the last time step at last layer as a single output (shape of `(batch_size, hidden_size)`); 
    And if direction is bidirectional, the we concat the hidden state of the last forward 
    gru and backward gru layer to create a single vector (shape of `(batch_size, hidden_size*2)`).

    Args:
        input_size (obj:`int`, required): The number of expected features in the input (the last dimension).
        hidden_size (obj:`int`, required): The number of features in the hidden state.
        num_layers (obj:`int`, optional, defaults to 1): Number of recurrent layers. 
            E.g., setting num_layers=2 would mean stacking two GRUs together to form a stacked GRU, 
            with the second GRU taking in outputs of the first GRU and computing the final results.
        direction (obj:`str`, optional, defaults to obj:`forwrd`): The direction of the network. 
            It can be "forward" and "bidirectional".
            When "bidirectional", the way to merge outputs of forward and backward is concatenating.
        dropout (obj:`float`, optional, defaults to 0.0): If non-zero, introduces a Dropout layer 
            on the outputs of each GRU layer except the last layer, with dropout probability equal to dropout.
        pooling_type (obj: `str`, optional, defaults to obj:`None`): If `pooling_type` is None, 
            then the GRUEncoder will return the hidden state of the last time step at last layer as a single vector.
            If pooling_type is not None, it must be one of `sum`, `max` and `mean`. Then it will be pooled on 
            the GRU output (the hidden state of every time step at last layer) to create a single vector.

    """

    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers=1,
                 direction="forward",
                 dropout=0.0,
                 pooling_type=None,
                 **kwargs):
        super().__init__()
        self._input_size = input_size
        self._hidden_size = hidden_size
        self._direction = direction
        self._pooling_type = pooling_type

        self.gru_layer = nn.GRU(input_size=input_size,
                                hidden_size=hidden_size,
                                num_layers=num_layers,
                                direction=direction,
                                dropout=dropout,
                                **kwargs)

    def get_input_dim(self):
        """
        Returns the dimension of the vector input for each element in the sequence input
        to a `GRUEncoder`. This is not the shape of the input tensor, but the
        last element of that shape.
        """
        return self._input_size

    def get_output_dim(self):
        """
        Returns the dimension of the final vector output by this `GRUEncoder`.  This is not
        the shape of the returned tensor, but the last element of that shape.
        """
        if self._direction == "bidirectional":
            return self._hidden_size * 2
        else:
            return self._hidden_size

    def forward(self, inputs, sequence_length):
        """
        GRUEncoder takes the a sequence of vectors and and returns a
        single vector, which is a combination of multiple GRU layers.
        The input to this module is of shape `(batch_size, num_tokens, input_size)`, 
        The output is of shape `(batch_size, hidden_size*2)` if GRU is bidirectional;
        If not, output is of shape `(batch_size, hidden_size)`.

        Args:
            inputs (obj:`Paddle.Tensor`, required): Shape as `(batch_size, num_tokens, input_size)`.
            sequence_length (obj:`Paddle.Tensor`, required): Shape as `(batch_size)`.

        Returns:
            last_hidden (obj:`Paddle.Tensor`, required): Shape as `(batch_size, hidden_size)`.
                The hidden state at the last time step for every layer.

        """
        encoded_text, last_hidden = self.gru_layer(
            inputs, sequence_length=sequence_length)
        if not self._pooling_type:
            # We exploit the `last_hidden` (the hidden state at the last time step for every layer)
            # to create a single vector.
            # If gru is not bidirectional, then output is the hidden state of the last time step 
            # at last layer. Output is shape of `(batch_size, hidden_size)`.
            # If gru is bidirectional, then output is concatenation of the forward and backward hidden state 
            # of the last time step at last layer. Output is shape of `(batch_size, hidden_size*2)`.
            if self._direction != 'bidirectional':
                output = last_hidden[-1, :, :]
            else:
                output = paddle.concat(
                    (last_hidden[-2, :, :], last_hidden[-1, :, :]), axis=1)
        else:
            # We exploit the `encoded_text` (the hidden state at the every time step for last layer)
            # to create a single vector. We perform pooling on the encoded text.
            # If gru is not bidirectional, output is shape of `(batch_size, hidden_size)`.
            # If gru is bidirectional, then output is shape of `(batch_size, hidden_size*2)`.
            if self._pooling_type == 'sum':
                output = paddle.sum(encoded_text, axis=1)
            elif self._pooling_type == 'max':
                output = paddle.max(encoded_text, axis=1)
            elif self._pooling_type == 'mean':
                output = paddle.mean(encoded_text, axis=1)
            else:
                raise RuntimeError(
                    "Unexpected pooling type %s ."
                    "Pooling type must be one of sum, max and mean." %
                    self._pooling_type)
        return output


class LSTMEncoder(nn.Layer):
    """
    A LSTMEncoder takes as input a sequence of vectors and returns a
    single vector, which is a combination of multiple LSTM layers.
    The input to this module is of shape `(batch_size, num_tokens, input_size)`, 
    The output is of shape `(batch_size, hidden_size*2)` if LSTM is bidirectional;
    If not, output is of shape `(batch_size, hidden_size)`.

    Paddle's LSTM have two outputs: the hidden state for every time step at last layer, 
    and the hidden state and cell at the last time step for every layer.
    If `pooling_type` is None, we perform the pooling on the hidden state of every time 
    step at last layer to create a single vector. If not None, we use the hidden state 
    of the last time step at last layer as a single output (shape of `(batch_size, hidden_size)`); 
    And if direction is bidirectional, the we concat the hidden state of the last forward 
    lstm and backward lstm layer to create a single vector (shape of `(batch_size, hidden_size*2)`).

    Args:
        input_size (obj:`int`, required): The number of expected features in the input (the last dimension).
        hidden_size (obj:`int`, required): The number of features in the hidden state.
        num_layers (obj:`int`, optional, defaults to 1): Number of recurrent layers. 
            E.g., setting num_layers=2 would mean stacking two LSTMs together to form a stacked LSTM, 
            with the second LSTM taking in outputs of the first LSTM and computing the final results.
        direction (obj:`str`, optional, defaults to obj:`forwrd`): The direction of the network. 
            It can be "forward" and "bidirectional".
            When "bidirectional", the way to merge outputs of forward and backward is concatenating.
        dropout (obj:`float`, optional, defaults to 0.0): If non-zero, introduces a Dropout layer 
            on the outputs of each LSTM layer except the last layer, with dropout probability equal to dropout.
        pooling_type (obj: `str`, optional, defaults to obj:`None`): If `pooling_type` is None, 
            then the LSTMEncoder will return the hidden state of the last time step at last layer as a single vector.
            If pooling_type is not None, it must be one of `sum`, `max` and `mean`. Then it will be pooled on 
            the LSTM output (the hidden state of every time step at last layer) to create a single vector.

    """

    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers=1,
                 direction="forward",
                 dropout=0.0,
                 pooling_type=None,
                 **kwargs):
        super().__init__()
        self._input_size = input_size
        self._hidden_size = hidden_size
        self._direction = direction
        self._pooling_type = pooling_type

        self.lstm_layer = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            direction=direction,
            dropout=dropout,
            **kwargs)

    def get_input_dim(self):
        """
        Returns the dimension of the vector input for each element in the sequence input
        to a `LSTMEncoder`. This is not the shape of the input tensor, but the
        last element of that shape.
        """
        return self._input_size

    def get_output_dim(self):
        """
        Returns the dimension of the final vector output by this `LSTMEncoder`.  This is not
        the shape of the returned tensor, but the last element of that shape.
        """
        if self._direction == "bidirectional":
            return self._hidden_size * 2
        else:
            return self._hidden_size

    def forward(self, inputs, sequence_length):
        """
        LSTMEncoder takes the a sequence of vectors and and returns a
        single vector, which is a combination of multiple LSTM layers.
        The input to this module is of shape `(batch_size, num_tokens, input_size)`, 
        The output is of shape `(batch_size, hidden_size*2)` if LSTM is bidirectional;
        If not, output is of shape `(batch_size, hidden_size)`.

        Args:
            inputs (obj:`Paddle.Tensor`, required): Shape as `(batch_size, num_tokens, input_size)`.
            sequence_length (obj:`Paddle.Tensor`, required): Shape as `(batch_size)`.

        Returns:
            last_hidden (obj:`Paddle.Tensor`, required): Shape as `(batch_size, hidden_size)`.
                The hidden state at the last time step for every layer.

        """
        encoded_text, (last_hidden, last_cell) = self.lstm_layer(
            inputs, sequence_length=sequence_length)
        if not self._pooling_type:
            # We exploit the `last_hidden` (the hidden state at the last time step for every layer)
            # to create a single vector.
            # If lstm is not bidirectional, then output is the hidden state of the last time step 
            # at last layer. Output is shape of `(batch_size, hidden_size)`.
            # If lstm is bidirectional, then output is concatenation of the forward and backward hidden state 
            # of the last time step at last layer. Output is shape of `(batch_size, hidden_size*2)`.
            if self._direction != 'bidirectional':
                output = last_hidden[-1, :, :]
            else:
                output = paddle.concat(
                    (last_hidden[-2, :, :], last_hidden[-1, :, :]), axis=1)
        else:
            # We exploit the `encoded_text` (the hidden state at the every time step for last layer)
            # to create a single vector. We perform pooling on the encoded text.
            # If lstm is not bidirectional, output is shape of `(batch_size, hidden_size)`.
            # If lstm is bidirectional, then output is shape of `(batch_size, hidden_size*2)`.
            if self._pooling_type == 'sum':
                output = paddle.sum(encoded_text, axis=1)
            elif self._pooling_type == 'max':
                output = paddle.max(encoded_text, axis=1)
            elif self._pooling_type == 'mean':
                output = paddle.mean(encoded_text, axis=1)
            else:
                raise RuntimeError(
                    "Unexpected pooling type %s ."
                    "Pooling type must be one of sum, max and mean." %
                    self._pooling_type)
        return output


class RNNEncoder(nn.Layer):
    """
    A RNNEncoder takes as input a sequence of vectors and returns a
    single vector, which is a combination of multiple RNN layers.
    The input to this module is of shape `(batch_size, num_tokens, input_size)`, 
    The output is of shape `(batch_size, hidden_size*2)` if RNN is bidirectional;
    If not, output is of shape `(batch_size, hidden_size)`.

    Paddle's RNN have two outputs: the hidden state for every time step at last layer, 
    and the hidden state at the last time step for every layer.
    If `pooling_type` is None, we perform the pooling on the hidden state of every time 
    step at last layer to create a single vector. If not None, we use the hidden state 
    of the last time step at last layer as a single output (shape of `(batch_size, hidden_size)`); 
    And if direction is bidirectional, the we concat the hidden state of the last forward 
    rnn and backward rnn layer to create a single vector (shape of `(batch_size, hidden_size*2)`).

    Args:
        input_size (obj:`int`, required): The number of expected features in the input (the last dimension).
        hidden_size (obj:`int`, required): The number of features in the hidden state.
        num_layers (obj:`int`, optional, defaults to 1): Number of recurrent layers. 
            E.g., setting num_layers=2 would mean stacking two RNNs together to form a stacked RNN, 
            with the second RNN taking in outputs of the first RNN and computing the final results.
        direction (obj:`str`, optional, defaults to obj:`forwrd`): The direction of the network. 
            It can be "forward" and "bidirectional".
            When "bidirectional", the way to merge outputs of forward and backward is concatenating.
        dropout (obj:`float`, optional, defaults to 0.0): If non-zero, introduces a Dropout layer 
            on the outputs of each RNN layer except the last layer, with dropout probability equal to dropout.
        pooling_type (obj: `str`, optional, defaults to obj:`None`): If `pooling_type` is None, 
            then the RNNEncoder will return the hidden state of the last time step at last layer as a single vector.
            If pooling_type is not None, it must be one of `sum`, `max` and `mean`. Then it will be pooled on 
            the RNN output (the hidden state of every time step at last layer) to create a single vector.

    """

    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers=1,
                 direction="forward",
                 dropout=0.0,
                 pooling_type=None,
                 **kwargs):
        super().__init__()
        self._input_size = input_size
        self._hidden_size = hidden_size
        self._direction = direction
        self._pooling_type = pooling_type

        self.rnn_layer = nn.SimpleRNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            direction=direction,
            dropout=dropout,
            **kwargs)

    def get_input_dim(self):
        """
        Returns the dimension of the vector input for each element in the sequence input
        to a `RNNEncoder`. This is not the shape of the input tensor, but the
        last element of that shape.
        """
        return self._input_size

    def get_output_dim(self):
        """
        Returns the dimension of the final vector output by this `RNNEncoder`.  This is not
        the shape of the returned tensor, but the last element of that shape.
        """
        if self._direction == "bidirectional":
            return self._hidden_size * 2
        else:
            return self._hidden_size

    def forward(self, inputs, sequence_length):
        """
        RNNEncoder takes the a sequence of vectors and and returns a
        single vector, which is a combination of multiple RNN layers.
        The input to this module is of shape `(batch_size, num_tokens, input_size)`, 
        The output is of shape `(batch_size, hidden_size*2)` if RNN is bidirectional;
        If not, output is of shape `(batch_size, hidden_size)`.

        Args:
            inputs (obj:`Paddle.Tensor`, required): Shape as `(batch_size, num_tokens, input_size)`.
            sequence_length (obj:`Paddle.Tensor`, required): Shape as `(batch_size)`.

        Returns:
            last_hidden (obj:`Paddle.Tensor`, required): Shape as `(batch_size, hidden_size)`.
                The hidden state at the last time step for every layer.

        """
        encoded_text, last_hidden = self.rnn_layer(
            inputs, sequence_length=sequence_length)
        if not self._pooling_type:
            # We exploit the `last_hidden` (the hidden state at the last time step for every layer)
            # to create a single vector.
            # If rnn is not bidirectional, then output is the hidden state of the last time step 
            # at last layer. Output is shape of `(batch_size, hidden_size)`.
            # If rnn is bidirectional, then output is concatenation of the forward and backward hidden state 
            # of the last time step at last layer. Output is shape of `(batch_size, hidden_size*2)`.
            if self._direction != 'bidirectional':
                output = last_hidden[-1, :, :]
            else:
                output = paddle.concat(
                    (last_hidden[-2, :, :], last_hidden[-1, :, :]), axis=1)
        else:
            # We exploit the `encoded_text` (the hidden state at the every time step for last layer)
            # to create a single vector. We perform pooling on the encoded text.
            # If rnn is not bidirectional, output is shape of `(batch_size, hidden_size)`.
            # If rnn is bidirectional, then output is shape of `(batch_size, hidden_size*2)`.
            if self._pooling_type == 'sum':
                output = paddle.sum(encoded_text, axis=1)
            elif self._pooling_type == 'max':
                output = paddle.max(encoded_text, axis=1)
            elif self._pooling_type == 'mean':
                output = paddle.mean(encoded_text, axis=1)
            else:
                raise RuntimeError(
                    "Unexpected pooling type %s ."
                    "Pooling type must be one of sum, max and mean." %
                    self._pooling_type)
        return output


class Chomp1d(nn.Layer):
    """
    Remove the elements on the right.

    Args:
        chomp_size ([int]): The number of elements removed.
    """

    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size]


class TemporalBlock(nn.Layer):
    """
    The TCN block, consists of dilated causal conv, relu and residual block. 
    See the Figure 1(b) in https://arxiv.org/pdf/1803.01271.pdf for more details.

    Args:
        n_inputs ([int]): The number of channels in the input tensor.
        n_outputs ([int]): The number of filters.
        kernel_size ([int]): The filter size.
        stride ([int]): The stride size.
        dilation ([int]): The dilation size.
        padding ([int]): The size of zeros to be padded.
        dropout (float, optional): Probability of dropout the units. Defaults to 0.2.
    """

    def __init__(self,
                 n_inputs,
                 n_outputs,
                 kernel_size,
                 stride,
                 dilation,
                 padding,
                 dropout=0.2):

        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(
            nn.Conv1D(
                n_inputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation))
        # Chomp1d is used to make sure the network is causal.
        # We pad by (k-1)*d on the two sides of the input for convolution, 
        # and then use Chomp1d to remove the (k-1)*d output elements on the right.
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(
            nn.Conv1D(
                n_outputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1,
                                 self.dropout1, self.conv2, self.chomp2,
                                 self.relu2, self.dropout2)
        self.downsample = nn.Conv1D(n_inputs, n_outputs,
                                    1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.set_value(
            paddle.tensor.normal(0.0, 0.01, self.conv1.weight.shape))
        self.conv2.weight.set_value(
            paddle.tensor.normal(0.0, 0.01, self.conv2.weight.shape))
        if self.downsample is not None:
            self.downsample.weight.set_value(
                paddle.tensor.normal(0.0, 0.01, self.downsample.weight.shape))

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNEncoder(nn.Layer):
    """
    A `TCNEncoder` takes as input a sequence of vectors and returns a
    single vector, which is the last one time step in the feature map. 
    The input to this module is of shape `(batch_size, num_tokens, input_size)`, 
    and the output is of shape `(batch_size, num_channels[-1])` with a receptive 
    filed:
    
    .. math::
    
        receptive filed = $2 * \sum_{i=0}^{len(num\_channels)-1}2^i(kernel\_size-1)$.
    
    Temporal Convolutional Networks is a simple convolutional architecture. It outperforms canonical recurrent networks
    such as LSTMs in many tasks. See https://arxiv.org/pdf/1803.01271.pdf for more details.

    Args:
        input_size (obj:`int`, required): The number of expected features in the input (the last dimension).
        num_channels (obj:`list` or obj:`tuple`, required): The number of channels in different layer. 
        kernel_size (obj:`int`, optional): The kernel size. Defaults to 2.
        dropout (obj:`float`, optional): The dropout probability. Defaults to 0.2.
    """

    def __init__(self, input_size, num_channels, kernel_size=2, dropout=0.2):
        super(TCNEncoder, self).__init__()
        self._input_size = input_size
        self._output_dim = num_channels[-1]

        layers = nn.LayerList()
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2**i
            in_channels = input_size if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers.append(
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
                    dropout=dropout))

        self.network = nn.Sequential(*layers)

    def get_input_dim(self):
        """
        Returns the dimension of the vector input for each element in the sequence input
        to a `TCNEncoder`. This is not the shape of the input tensor, but the
        last element of that shape.
        """
        return self._input_size

    def get_output_dim(self):
        """
        Returns the dimension of the final vector output by this `TCNEncoder`.  This is not
        the shape of the returned tensor, but the last element of that shape.
        """
        return self._output_dim

    def forward(self, inputs):
        """
        TCNEncoder takes as input a sequence of vectors and returns a
        single vector, which is the last one time step in the feature map. 
        The input to this module is of shape `(batch_size, num_tokens, input_size)`, 
        and the output is of shape `(batch_size, num_channels[-1])` with a receptive 
        filed:
    
        .. math::
        
            receptive filed = $2 * \sum_{i=0}^{len(num\_channels)-1}2^i(kernel\_size-1)$.

        Args:
            inputs (obj:`Paddle.Tensor`, required): The input tensor with shape `[batch_size, num_tokens, input_size]`.

        Returns:
            output (obj:`Paddle.Tensor`): The output tensor with shape `[batch_size, num_channels[-1]]`.
        """
        inputs_t = inputs.transpose([0, 2, 1])
        output = self.network(inputs_t).transpose([2, 0, 1])[-1]
        return output
