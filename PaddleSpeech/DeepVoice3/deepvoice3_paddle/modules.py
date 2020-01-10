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

import paddle
from paddle import fluid
import paddle.fluid.dygraph as dg

import numpy as np

import deepvoice3_paddle.conv as conv
import deepvoice3_paddle.weight_norm as weight_norm


def FC(name_scope,
       in_features,
       size,
       num_flatten_dims=1,
       dropout=0.0,
       epsilon=1e-30,
       act=None,
       is_test=False,
       dtype="float32"):
    """
    A special Linear Layer, when it is used with dropout, the weight is 
    initialized as normal(0, std=np.sqrt((1-dropout) / in_features))
    """

    # stds
    if isinstance(in_features, int):
        in_features = [in_features]
    stds = [np.sqrt((1 - dropout) / in_feature) for in_feature in in_features]
    weight_inits = [
        fluid.initializer.NormalInitializer(scale=std) for std in stds
    ]
    bias_init = fluid.initializer.ConstantInitializer(0.0)

    # param attrs
    weight_attrs = [fluid.ParamAttr(initializer=init) for init in weight_inits]
    bias_attr = fluid.ParamAttr(initializer=bias_init)

    layer = weight_norm.FC(name_scope,
                           size,
                           num_flatten_dims=num_flatten_dims,
                           param_attr=weight_attrs,
                           bias_attr=bias_attr,
                           act=act,
                           dtype=dtype)
    return layer


def Conv1D(name_scope,
           in_channels,
           num_filters,
           filter_size=3,
           dilation=1,
           groups=None,
           causal=False,
           std_mul=1.0,
           dropout=0.0,
           use_cudnn=True,
           act=None,
           dtype="float32"):
    """
    A special Conv1D Layer, when it is used with dropout, the weight is 
    initialized as 
    normal(0, std=np.sqrt(std_mul * (1-dropout) / (filter_size * in_features)))
    """
    # std
    std = np.sqrt((std_mul * (1 - dropout)) / (filter_size * in_channels))
    weight_init = fluid.initializer.NormalInitializer(loc=0.0, scale=std)
    bias_init = fluid.initializer.ConstantInitializer(0.0)

    # param attrs
    weight_attr = fluid.ParamAttr(initializer=weight_init)
    bias_attr = fluid.ParamAttr(initializer=bias_init)

    layer = conv.Conv1D(
        name_scope,
        in_channels,
        num_filters,
        filter_size,
        dilation,
        groups=groups,
        causal=causal,
        param_attr=weight_attr,
        bias_attr=bias_attr,
        use_cudnn=use_cudnn,
        act=act,
        dtype=dtype)
    return layer


def Embedding(name_scope,
              num_embeddings,
              embed_dim,
              is_sparse=False,
              is_distributed=False,
              padding_idx=None,
              std=0.01,
              dtype="float32"):
    # param attrs
    weight_attr = fluid.ParamAttr(initializer=fluid.initializer.Normal(
        scale=std))
    layer = dg.Embedding(
        name_scope, (num_embeddings, embed_dim),
        padding_idx=padding_idx,
        param_attr=weight_attr,
        dtype=dtype)
    return layer


class Conv1DGLU(dg.Layer):
    """
    A Convolution 1D block with GLU activation. It also applys dropout for the 
    input x. It fuses speaker embeddings through a FC activated by softsign. It
    has residual connection from the input x, and scale the output by 
    np.sqrt(0.5).
    """

    def __init__(self,
                 name_scope,
                 n_speakers,
                 speaker_dim,
                 in_channels,
                 num_filters,
                 filter_size,
                 dilation,
                 std_mul=4.0,
                 dropout=0.0,
                 causal=False,
                 residual=True,
                 dtype="float32"):
        super(Conv1DGLU, self).__init__(name_scope, dtype=dtype)

        # conv spec
        self.in_channels = in_channels
        self.n_speakers = n_speakers
        self.speaker_dim = speaker_dim
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.dilation = dilation
        self.causal = causal
        self.residual = residual

        # weight init and dropout
        self.std_mul = std_mul
        self.dropout = dropout

        if residual:
            assert (
                in_channels == num_filters
            ), "this block uses residual connection"\
                "the input_channes should equals num_filters"

        self.conv = Conv1D(
            self.full_name(),
            in_channels,
            2 * num_filters,
            filter_size,
            dilation,
            causal=causal,
            std_mul=std_mul,
            dropout=dropout,
            dtype=dtype)

        if n_speakers > 1:
            assert (speaker_dim is not None
                    ), "speaker embed should not be null in multi-speaker case"
            self.fc = Conv1D(
                self.full_name(),
                speaker_dim,
                num_filters,
                filter_size=1,
                dilation=1,
                causal=False,
                act="softsign",
                dtype=dtype)

    def forward(self, x, speaker_embed_bc1t=None):
        """
        Args:
            x (Variable): Shape(B, C_in, 1, T), the input of Conv1DGLU
                layer, where B means batch_size, C_in means the input channels
                T means input time steps.
            speaker_embed_bct1 (Variable): Shape(B, C_sp, 1, T), expanded
                speaker embed, where C_sp means speaker embedding size. Note
                that when using residual connection, the Conv1DGLU does not
                change the number of channels, so out channels equals input
                channels.

        Returns:
            x (Variable): Shape(B, C_out, 1, T), the output of Conv1DGLU, where
                C_out means the output channels of Conv1DGLU.
        """

        residual = x
        x = fluid.layers.dropout(
            x, self.dropout, dropout_implementation="upscale_in_train")
        x = self.conv(x)

        content, gate = fluid.layers.split(x, num_or_sections=2, dim=1)

        if speaker_embed_bc1t is not None:
            sp = self.fc(speaker_embed_bc1t)
            content = content + sp

        # glu
        x = fluid.layers.elementwise_mul(fluid.layers.sigmoid(gate), content)

        if self.residual:
            x = fluid.layers.scale(x + residual, np.sqrt(0.5))
        return x

    def add_input(self, x, speaker_embed_bc11=None):
        """
        Inputs:
        x: shape(B, num_filters, 1, time_steps)
        speaker_embed_bc11: shape(B, speaker_dim, 1, time_steps)

        Outputs:
        out: shape(B, num_filters, 1, time_steps), where time_steps = 1
        """

        residual = x

        # add step input and produce step output
        x = fluid.layers.dropout(
            x, self.dropout, dropout_implementation="upscale_in_train")
        x = self.conv.add_input(x)

        content, gate = fluid.layers.split(x, num_or_sections=2, dim=1)

        if speaker_embed_bc11 is not None:
            sp = self.fc(speaker_embed_bc11)
            content = content + sp

        x = fluid.layers.elementwise_mul(fluid.layers.sigmoid(gate), content)

        if self.residual:
            x = fluid.layers.scale(x + residual, np.sqrt(0.5))
        return x


def Conv1DTranspose(name_scope,
                    in_channels,
                    num_filters,
                    filter_size,
                    padding=0,
                    stride=1,
                    dilation=1,
                    groups=None,
                    std_mul=1.0,
                    dropout=0.0,
                    use_cudnn=True,
                    act=None,
                    dtype="float32"):
    std = np.sqrt(std_mul * (1 - dropout) / (in_channels * filter_size))
    weight_init = fluid.initializer.NormalInitializer(scale=std)
    weight_attr = fluid.ParamAttr(initializer=weight_init)
    bias_init = fluid.initializer.ConstantInitializer(0.0)
    bias_attr = fluid.ParamAttr(initializer=bias_init)
    layer = conv.Conv1DTranspose(
        name_scope,
        in_channels,
        num_filters,
        filter_size,
        padding=padding,
        stride=stride,
        dilation=dilation,
        groups=groups,
        param_attr=weight_attr,
        bias_attr=bias_attr,
        use_cudnn=use_cudnn,
        act=act,
        dtype=dtype)
    return layer


def compute_position_embedding(rad):
    # rad is a transposed radius, shape(embed_dim, n_vocab)
    embed_dim, n_vocab = rad.shape

    even_dims = dg.to_variable(np.arange(0, embed_dim, 2).astype("int32"))
    odd_dims = dg.to_variable(np.arange(1, embed_dim, 2).astype("int32"))

    even_rads = fluid.layers.gather(rad, even_dims)
    odd_rads = fluid.layers.gather(rad, odd_dims)

    sines = fluid.layers.sin(even_rads)
    cosines = fluid.layers.cos(odd_rads)

    temp = fluid.layers.scatter(rad, even_dims, sines)
    out = fluid.layers.scatter(temp, odd_dims, cosines)
    out = fluid.layers.transpose(out, perm=[1, 0])
    return out


def position_encoding_init(n_position,
                           d_pos_vec,
                           position_rate=1.0,
                           sinusoidal=True):
    """ Init the sinusoid position encoding table """

    # keep idx 0 for padding token position encoding zero vector
    position_enc = np.array([[
        position_rate * pos / np.power(10000, 2 * (i // 2) / d_pos_vec)
        for i in range(d_pos_vec)
    ] if pos != 0 else np.zeros(d_pos_vec) for pos in range(n_position)])

    if sinusoidal:
        position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2])  # dim 2i
        position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2])  # dim 2i+1

    return position_enc


class PositionEmbedding(dg.Layer):
    def __init__(self,
                 name_scope,
                 n_position,
                 d_pos_vec,
                 position_rate=1.0,
                 is_sparse=False,
                 is_distributed=False,
                 param_attr=None,
                 max_norm=None,
                 padding_idx=None,
                 dtype="float32"):
        super(PositionEmbedding, self).__init__(name_scope, dtype=dtype)
        self.embed = dg.Embedding(
            self.full_name(),
            size=(n_position, d_pos_vec),
            is_sparse=is_sparse,
            is_distributed=is_distributed,
            padding_idx=None,
            param_attr=param_attr,
            dtype=dtype)
        self.set_weight(
            position_encoding_init(
                n_position,
                d_pos_vec,
                position_rate=position_rate,
                sinusoidal=False).astype(dtype))

        self._is_sparse = is_sparse
        self._is_distributed = is_distributed
        self._remote_prefetch = self._is_sparse and (not self._is_distributed)
        if self._remote_prefetch:
            assert self._is_sparse is True and self._is_distributed is False

        self._padding_idx = (-1 if padding_idx is None else padding_idx if
                             padding_idx >= 0 else (n_position + padding_idx))
        self._position_rate = position_rate
        self._max_norm = max_norm
        self._dtype = dtype

    def set_weight(self, array):
        assert self.embed.weight.shape == list(
            array.shape), "shape does not match"
        self.embed.weight.set_value(array)

    def forward(self, indices, speaker_position_rate=None):
        """
        Args:
            indices (Variable): Shape (B, T), dtype: int64, position
                indices, where B means the batch size, T means the time steps.
            speaker_position_rate (Variable | float, optional), position
                rate. It can be a float point number or a Variable with 
                shape (1,), then this speaker_position_rate is used for every 
                example. It can also be a Variable with shape (B, 1), which 
                contains a speaker position rate for each speaker.
        Returns:
            out (Variable): Shape(B, C_pos), position embedding, where C_pos 
                means position embedding size.
        """
        rad = fluid.layers.transpose(self.embed.weight, perm=[1, 0])
        batch_size = indices.shape[0]

        if speaker_position_rate is None:
            weight = compute_position_embedding(rad)
            out = self._helper.create_variable_for_type_inference(self._dtype)
            self._helper.append_op(
                type="lookup_table_v2",
                inputs={"Ids": indices,
                        "W": weight},
                outputs={"Out": out},
                attrs={
                    "is_sparse": self._is_sparse,
                    "is_distributed": self._is_distributed,
                    "remote_prefetch": self._remote_prefetch,
                    "padding_idx":
                    self._padding_idx,  # special value for lookup table op
                })
            return out

        elif (np.isscalar(speaker_position_rate) or
              isinstance(speaker_position_rate, fluid.framework.Variable) and
              speaker_position_rate.shape == [1, 1]):
            # # make a weight
            # scale the weight (the operand for sin & cos)
            if np.isscalar(speaker_position_rate):
                scaled_rad = fluid.layers.scale(rad, speaker_position_rate)
            else:
                scaled_rad = fluid.layers.elementwise_mul(
                    rad, speaker_position_rate[0])
            weight = compute_position_embedding(scaled_rad)
            out = self._helper.create_variable_for_type_inference(self._dtype)
            self._helper.append_op(
                type="lookup_table_v2",
                inputs={"Ids": indices,
                        "W": weight},
                outputs={"Out": out},
                attrs={
                    "is_sparse": self._is_sparse,
                    "is_distributed": self._is_distributed,
                    "remote_prefetch": self._remote_prefetch,
                    "padding_idx":
                    self._padding_idx,  # special value for lookup table op
                })
            return out

        elif np.prod(speaker_position_rate.shape) > 1:
            assert speaker_position_rate.shape == [batch_size, 1]
            outputs = []
            for i in range(batch_size):
                rate = speaker_position_rate[i]  # rate has shape [1]
                scaled_rad = fluid.layers.elementwise_mul(rad, rate)
                weight = compute_position_embedding(scaled_rad)
                out = self._helper.create_variable_for_type_inference(
                    self._dtype)
                sequence = indices[i]
                self._helper.append_op(
                    type="lookup_table_v2",
                    inputs={"Ids": sequence,
                            "W": weight},
                    outputs={"Out": out},
                    attrs={
                        "is_sparse": self._is_sparse,
                        "is_distributed": self._is_distributed,
                        "remote_prefetch": self._remote_prefetch,
                        "padding_idx": -1,
                    })
                outputs.append(out)
            out = fluid.layers.stack(outputs)
            return out
        else:
            raise Exception("Then you can just use position rate at init")
