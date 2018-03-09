import math
import numpy as np
import logging
import paddle.v2 as paddle
import paddle.fluid as fluid
import paddle.fluid.layers as pd
import paddle.fluid.nets as nets
import paddle.fluid.framework as framework
import paddle.fluid.debuger as debuger
from paddle.fluid.framework import framework_pb2
from utils import *
'''
Reference the paper: Convolutional Sequence to Sequence Learning



Some difference with the pytorch implementation:

- conv2d padding will change output, need a special strategy
'''


class ConvEncoder:
    def __init__(self,
                 dict_size,
                 embed_dim,
                 max_positions,
                 convolutions,
                 pad_id,
                 pos_pad_id,
                 dropout=0.1):
        self.dropout = dropout
        self.embed_tokens = Embedding(dict_size+1, embed_dim, pad_id)
        self.embed_positions = Embedding(max_positions+1, embed_dim, pos_pad_id)

        in_channels = convolutions[0][0]
        self.fc1 = Linear(in_channels, dropout=dropout)

        self.projections = []
        self.convolutions = []
        for (out_channels, kernel_size) in convolutions:
            pad = (kernel_size - 1) / 2
            # here the in_channels is deduced from the input variable.
            self.projections.append(
                Linear(out_channels) if in_channels != out_channels else None)
            self.convolutions.append(
                Conv1D(
                    in_channels,
                    out_channels * 2,
                    kernel_size,
                    # padding=pad,
                    dropout=dropout))
            in_channels = out_channels

        self.fc2 = Linear(embed_dim)

    def forward(self, src_tokens, src_positions):
        x = self.embed_tokens(src_tokens) + self.embed_positions(src_positions)
        x = Op.dropout(x, self.dropout, is_test=is_test)
        input_embedding = x

        # project to size of convolution
        x = self.fc1(x)

        # # B x T x C -> T x B x C
        # x = Op.transpose(x, 0, 1)

        # temporal convolutions
        for proj, conv in zip(self.projections, self.convolutions):
            residual = x if proj is None else proj(x)
            x = Op.dropout(x, self.dropout)
            x = conv(x)
            x = fluid.nets.glu(x, dim=2)
            x = (x + residual) * math.sqrt(0.5)

        # # T x B x C -> B x T x C
        # x = Op.transpose(x, 1, 2)

        # project back to size of embedding
        x = self.fc2(x)

        y = (x + input_embedding) * math.sqrt(0.5)

        return x, y


class ConvDecoder:
    '''
    decode trainer
    '''

    def __init__(self,
                 dict_size,
                 embed_dim,
                 out_embed_dim,
                 max_positions,
                 convolutions,
                 pad_id,
                 pos_pad_id,
                 attention=True,
                 dropout=0.1,
                 share_embed=False):
        self.dropout = dropout

        in_channels = convolutions[0][0]
        if isinstance(attention, bool):
            attention = [attention] * len(convolutions)

        self.embed_tokens = Embedding(dict_size+1, embed_dim, pad_id)
        self.embed_positions = Embedding(
            max_positions + 1,
            embed_dim,
            pos_pad_id,
        )

        self.fc1 = Linear(in_channels, dropout=dropout)
        self.projections = []
        self.convolutions = []
        self.attention = []
        for i, (out_channels, kernel_size) in enumerate(convolutions):
            pad = kernel_size - 1
            self.projections.append(
                Linear(out_channels) if in_channels != out_channels else None)
            self.convolutions.append(
                Conv1D(
                    in_channels,
                    out_channels * 2,
                    kernel_size,
                    # padding=pad,
                    dropout=dropout))
            self.attention.append(
                AttentionLayer(out_channels, embed_dim)
                if attention[i] else None)
            in_channels = out_channels

        self.fc2 = Linear(out_embed_dim)
        self.fc3 = Linear(dict_size+1, dropout=dropout)

    def forward(self, prev_output_tokens, prev_positions, encoder_out):
        '''
        only works in train mode
        '''
        encoder_a, encoder_b = self._split_encoder_out(encoder_out)

        x = self.embed_tokens(prev_output_tokens)
        x += self.embed_positions(prev_positions)
        x = Op.dropout(x, self.dropout)
        target_embedding = x

        # project to size of convolution
        x = self.fc1(x)

        x = self._transpose_if_training(x)

        # temporal covolutoins
        avg_attn_scores = None
        num_attn_layers = len(self.attention)
        for proj, conv, attention in zip(self.projections, self.convolutions,
                                         self.attention):
            residual = x if proj is None else proj(x)
            x = Op.dropout(x, self.dropout)
            x = conv(x)
            x = fluid.nets.glu(x, dim=2)

            if attention is not None:
                x = self._transpose_if_training(x)

                x, attn_scores = attention(x, target_embedding,
                                           (encoder_a, encoder_b))
                attn_scores = attn_scores / num_attn_layers
                if avg_attn_scores is None:
                    avg_attn_scores = attn_scores
                else:
                    avg_attn_scores += attn_scores

                x = self._transpose_if_training(x)

            x = (x + residual) * math.sqrt(0.5)

        # T x B x C -> B x T x C
        x = self._transpose_if_training(x)

        # project back to size of vocabulary
        x = self.fc2(x)
        x = Op.dropout(x, self.dropout)
        x = self.fc3(x)
        # NOTE this is different from fairseq, the predictions are negative and avg_cost will be nan
        x = Op.softmax(x)
        return x, avg_attn_scores

    def _transpose_if_training(self, x):
        x = Op.transpose(x, [0, 1])
        return x

    def _split_encoder_out(self, encoder_out):
        encoder_a, encoder_b = encoder_out
        encoder_a = Op.transpose(encoder_a, [1, 2])
        result = (encoder_a, encoder_b)
        return result


class AttentionLayer:
    def __init__(self, conv_channels, embed_dim):
        self.in_projection = Linear(embed_dim)
        self.out_projection = Linear(conv_channels)

    def __call__(self, x, target_embedding, encoder_out):
        return self.forward(x, target_embedding, encoder_out)

    def forward(self, x, target_embedding, encoder_out):
        '''
        x is decoder state
        '''
        residual = x
        encoder_a, encoder_b = encoder_out
        # here just a trick
        encoder_a = Op.transpose(encoder_a, [1, 2])
        encoder_b = Op.transpose(encoder_b, [1, 2])

        # di = fc(hi) + gi(decoder embedding)
        x = self.in_projection(x)
        x = (x + target_embedding) * math.sqrt(0.5)
        x = pd.matmul(x, encoder_a, transpose_y=True)

        sz = get_tensor(x).dims
        x = pd.softmax(Op.reshape(x, (sz[0] * sz[1], sz[2])), dim=1)
        x = Op.reshape(x, sz)
        # x = x.view(sz)
        attn_scores = x

        x = pd.matmul(x, encoder_b, transpose_y=True)

        # scale attention output
        s = get_tensor(encoder_b).dims[1]
        x = x * (s * math.sqrt(1. / s))

        x = (self.out_projection(x) + residual) * math.sqrt(0.5)
        return x, attn_scores
