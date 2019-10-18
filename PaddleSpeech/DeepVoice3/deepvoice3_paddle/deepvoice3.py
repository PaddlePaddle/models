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

from itertools import chain
from collections import namedtuple

from paddle import fluid
import paddle.fluid.dygraph as dg

import numpy as np

from deepvoice3_paddle import conv

from deepvoice3_paddle.modules import Embedding, PositionEmbedding
from deepvoice3_paddle.modules import FC, Conv1D, Conv1DGLU, Conv1DTranspose

ConvSpec = namedtuple("ConvSpec", ["out_channels", "filter_size", "dilation"])
WindowRange = namedtuple("WindowRange", ["backward", "ahead"])


def expand_speaker_embed(x, speaker_embed, tdim=-1):
    """
    Expand speaker embeddings for multiple timesteps.
    
    Args:
        x (Variable): A reference Variable used to determine number of timesteps.
        speaker_embed (Variable): Shape(B, C), embeddings of speakers, where
            B means batch_size, C means speaker embedding size.
        tdim (int, optional): The idex of time dimension in x. Defaults to -1,
            which means the last dimension is time dimension.
    
    Returns:
        Variable: Shape(B, C, 1, T), the expanded speaker embeddings, where
            T = x.shape[tdim]. T means number of timesteps.
    
    """

    speaker_embed = fluid.layers.reshape(
        speaker_embed, shape=speaker_embed.shape + [1, 1])
    time_steps = x.shape[tdim]
    speaker_embed_bc1t = fluid.layers.expand(
        speaker_embed, expand_times=[1, 1, 1, time_steps])
    return speaker_embed_bc1t


def gen_mask2(valid_lengths, max_len, dtype="float32"):
    """
    Generate a mask tensor from valid lengths. note that it return a *reverse*
    mask. Indices within valid lengths correspond to 0, and those within
    padding area correspond to 1. 
    
    Assume that valid_lengths = [2,5,7], and max_len = 7, the generated mask is
    [[0, 0, 1, 1, 1, 1, 1],
     [0, 0, 0, 0, 0, 1, 1],
     [0, 0, 0, 0, 0, 0, 0]].

    Args:
        valid_lengths (Variable): Shape(B), dtype: int64. A 1D-Tensor containing
            the valid lengths (timesteps) of each example, where B means
            beatch_size.
        max_len (int): The length (number of timesteps) of the mask.
        dtype (str, optional): A string that specifies the data type of the
            returned mask.

    Returns:
        mask (Variable): A mask computed from valid lengths.
    """
    batch_size = valid_lengths.shape[0]
    mask = fluid.layers.sequence_mask(
        valid_lengths, maxlen=max_len, dtype=dtype)
    mask = 1 - mask
    return mask


def expand_mask(mask, attn):
    """
    Expand a mask for multiple time steps. This function is used
    by the AttentionLayer in the Decoder to expand a mask for every
    timestep in the decoder.

    Args:
        mask (Variable): Shape(B, T_enc), a mask generated with valid
            text lengths, where T_enc means encoder length(time steps).
        attn (Variable): Shape(B, T_dec, T_enc), a Variable stands for 
            the alignment tensor between encoder and decoder, where 
            T_dec means the decoder length(time_steps).

    Returns:
        mask_btc (Variable): shape(B, T_dec, T_enc), the expanded mask.
    """
    decoder_length = attn.shape[1]
    mask = fluid.layers.reshape(mask, [mask.shape[0], 1, mask.shape[1]])
    mask_btc = fluid.layers.expand(mask, expand_times=[1, decoder_length, 1])
    return mask_btc


class Encoder(dg.Layer):
    def __init__(self,
                 name_scope,
                 n_vocab,
                 embed_dim,
                 n_speakers,
                 speaker_dim,
                 padding_idx=None,
                 embedding_weight_std=0.1,
                 convolutions=(ConvSpec(64, 5, 1)) * 7,
                 max_positions=512,
                 dropout=0.1,
                 dtype="float32"):
        super(Encoder, self).__init__(name_scope, dtype=dtype)

        self.dropout = dropout
        self.embedding_weight_std = embedding_weight_std

        self.embed = Embedding(
            self.full_name(),
            n_vocab,
            embed_dim,
            padding_idx=padding_idx,
            std=embedding_weight_std,
            dtype=dtype)

        if n_speakers > 1:
            self.sp_proj1 = Conv1D(
                self.full_name(),
                speaker_dim,
                embed_dim,
                filter_size=1,
                std_mul=1.0,
                dropout=dropout,
                act="softsign",
                dtype=dtype)
            self.sp_proj2 = Conv1D(
                self.full_name(),
                speaker_dim,
                embed_dim,
                filter_size=1,
                std_mul=1.0,
                dropout=dropout,
                act="softsign",
                dtype=dtype)
        self.n_speakers = n_speakers

        self.convolutions = []

        in_channels = embed_dim
        std_mul = 1.0
        for (out_channels, filter_size, dilation) in convolutions:
            # 1 * 1 convolution & relu
            if in_channels != out_channels:
                self.convolutions.append(
                    Conv1D(
                        self.full_name(),
                        in_channels,
                        out_channels,
                        filter_size=1,
                        std_mul=std_mul,
                        act="relu",
                        dtype=dtype))
                in_channels = out_channels
                std_mul = 2.0

            self.convolutions.append(
                Conv1DGLU(
                    self.full_name(),
                    n_speakers,
                    speaker_dim,
                    in_channels,
                    out_channels,
                    filter_size,
                    dilation,
                    std_mul=std_mul,
                    dropout=dropout,
                    causal=False,
                    residual=True,
                    dtype=dtype))
            in_channels = out_channels
            std_mul = 4.0

        self.convolutions.append(
            Conv1D(
                self.full_name(),
                in_channels,
                embed_dim,
                filter_size=1,
                std_mul=std_mul,
                dropout=dropout,
                dtype=dtype))

        for i, layer in enumerate(self.convolutions):
            self.add_sublayer("convolution_{}".format(i), layer)

    def forward(self, x, speaker_embed=None):
        """
        Encode text sequence.
        
        Args:
            x (Variable): Shape(B, T_enc, 1), dtype: int64. Ihe input text
                indices. T_enc means the timesteps of decoder input x.
            speaker_embed (Variable, optional): Shape(Batch_size, speaker_dim),
                dtype: float32. Speaker embeddings. This arg is not None only
                when the model is a multispeaker model.

        Returns:
            keys (Variable), Shape(B, C_emb, 1, T_enc), the encoded
                representation for keys, where C_emb menas the text embedding
                size.
            values (Variable), Shape(B, C_embed, 1, T_enc), the encoded
                representation for values.
        """
        x = self.embed(x)

        x = fluid.layers.dropout(
            x, self.dropout, dropout_implementation="upscale_in_train")
        x = fluid.layers.transpose(
            fluid.layers.reshape(
                x, shape=x.shape + [1]), perm=[0, 2, 3, 1])

        speaker_embed_bc1t = None
        if speaker_embed is not None:
            speaker_embed_bc1t = expand_speaker_embed(x, speaker_embed, tdim=3)

            speaker_embed_bc1t = fluid.layers.dropout(
                speaker_embed_bc1t,
                self.dropout,
                dropout_implementation="upscale_in_train")

            x = x + self.sp_proj1(speaker_embed_bc1t)

        input_embed = x

        for layer in self.convolutions:
            if isinstance(layer, Conv1DGLU):
                x = layer(x, speaker_embed_bc1t)
            else:
                x = layer(x)

        if speaker_embed is not None:
            x = x + self.sp_proj2(speaker_embed_bc1t)

        keys = x
        values = fluid.layers.scale(input_embed + x, scale=np.sqrt(0.5))

        return keys, values

    def freeze_embedding(self):
        """Fix text embedding while training."""
        for param in self.embed.parameters():
            param.trainable = False


class AttentionLayer(dg.Layer):
    def __init__(self,
                 name_scope,
                 conv_channels,
                 embed_dim,
                 dropout=0.0,
                 window_range=WindowRange(-1, 3),
                 key_projection=True,
                 value_projection=True,
                 dtype="float32"):
        super(AttentionLayer, self).__init__(name_scope, dtype=dtype)
        self.query_proj = Conv1D(
            self.full_name(),
            conv_channels,
            embed_dim,
            filter_size=1,
            dtype=dtype)

        if key_projection:
            self.key_proj = Conv1D(
                self.full_name(),
                embed_dim,
                embed_dim,
                filter_size=1,
                dtype=dtype)

        if value_projection:
            self.value_proj = Conv1D(
                self.full_name(),
                embed_dim,
                embed_dim,
                filter_size=1,
                dtype=dtype)

        self.out_proj = Conv1D(
            self.full_name(),
            embed_dim,
            conv_channels,
            filter_size=1,
            dtype=dtype)

        self.key_projection = key_projection
        self.value_projection = value_projection
        self.dropout = dropout
        self.window_range = window_range

    def forward(self, query, encoder_out, mask=None, last_attended=None):
        """
        Compute pooled context representation and alignment scores.
        
        Args:
            query (Variable): shape(B, C_q, 1, T_dec), the query tensor,
                where C_q means the channel of query.
            encoder_out (Tuple(Variable, Variable)): 
                keys (Variable): shape(B, C_emb, 1, T_enc), the key
                    representation from an encoder, where C_emb means
                    text embedding size.
                values (Variable): shape(B, C_emb, 1, T_enc), the value
                    representation from an encoder, where C_emb means
                    text embedding size.
            mask (Variable, optional): Shape(B, T_enc), mask generated with 
                valid text lengths.
            last_attended (int, optional): The position that received most
                attention at last timestep. This is only used at decoding.

        Outpus:
            x (Variable): Shape(B, C_q, 1, T_dec), the context representation
                pooled from attention mechanism.
            attn_scores (Variable): shape(B, T_dec, T_enc), the alignment
                tensor, where T_dec means the number of decoder time steps and 
                T_enc means number the number of decoder time steps.
        """
        keys, values = encoder_out
        residual = query
        if self.value_projection:
            values = self.value_proj(values)

        if self.key_projection:
            keys = self.key_proj(keys)

        x = self.query_proj(query)

        batch_size, conv_channels, _, decoder_length = query.shape
        encoder_length = keys.shape[-1]
        embed_dim = keys.shape[1]

        x = fluid.layers.matmul(
            fluid.layers.reshape(
                x, shape=[batch_size, embed_dim, decoder_length]),
            fluid.layers.reshape(
                keys, shape=[batch_size, embed_dim, encoder_length]),
            transpose_x=True)

        mask_value = -1.0e30
        if mask is not None:
            mask = expand_mask(mask, x)
            neg_inf_mask = fluid.layers.scale(mask, mask_value)
            x = x + neg_inf_mask

        # if last_attended is provided, focus only on a window range around it
        # to enforce monotonic attention.
        if last_attended is not None:
            locality_mask = np.ones(shape=x.shape, dtype=np.float32)
            backward, ahead = self.window_range
            backward = last_attended + backward
            ahead = last_attended + ahead
            if backward < 0:
                backward = 0
            if ahead > x.shape[-1]:
                ahead = x.shape[-1]
            locality_mask[:, :, backward:ahead] = 0.

            locality_mask = dg.to_variable(locality_mask)
            neg_inf_mask = fluid.layers.scale(locality_mask, mask_value)
            x = x + neg_inf_mask

        x = fluid.layers.softmax(x)
        attn_scores = x

        x = fluid.layers.dropout(
            x, self.dropout, dropout_implementation="upscale_in_train")

        x = fluid.layers.matmul(
            fluid.layers.reshape(
                values, shape=[batch_size, embed_dim, encoder_length]),
            x,
            transpose_y=True)

        x = fluid.layers.reshape(x, [batch_size, embed_dim, 1, decoder_length])

        x = fluid.layers.scale(x,
                               encoder_length * np.sqrt(1.0 / encoder_length))

        x = self.out_proj(x)

        x = fluid.layers.scale((x + residual), np.sqrt(0.5))
        return x, attn_scores


class Decoder(dg.Layer):
    def __init__(self,
                 name_scope,
                 n_speakers,
                 speaker_dim,
                 embed_dim,
                 mel_dim=80,
                 r=5,
                 max_positions=512,
                 padding_idx=None,
                 preattention=(ConvSpec(128, 5, 1)) * 4,
                 convolutions=(ConvSpec(128, 5, 1)) * 4,
                 attention=True,
                 dropout=0.1,
                 use_memory_mask=False,
                 force_monotonic_attention=False,
                 query_position_rate=1.0,
                 key_position_rate=1.29,
                 window_range=WindowRange(-1, 3),
                 key_projection=True,
                 value_projection=True,
                 dtype="float32"):
        super(Decoder, self).__init__(name_scope, dtype=dtype)

        self.dropout = dropout
        self.mel_dim = mel_dim
        self.r = r
        self.query_position_rate = query_position_rate
        self.key_position_rate = key_position_rate
        self.window_range = window_range
        self.n_speakers = n_speakers

        conv_channels = convolutions[0].out_channels
        self.embed_query_positions = PositionEmbedding(
            self.full_name(),
            max_positions,
            conv_channels,
            padding_idx=padding_idx,
            dtype=dtype)
        self.embed_keys_positions = PositionEmbedding(
            self.full_name(),
            max_positions,
            embed_dim,
            padding_idx=padding_idx,
            dtype=dtype)

        # Used to compute multiplier for position rate
        if n_speakers > 1:
            self.speaker_proj1 = FC(self.full_name(),
                                    speaker_dim,
                                    1,
                                    act="sigmoid",
                                    dropout=dropout,
                                    dtype=dtype)
            self.speaker_proj2 = FC(self.full_name(),
                                    speaker_dim,
                                    1,
                                    act="sigmoid",
                                    dropout=dropout,
                                    dtype=dtype)

        # prenet
        self.prenet = []
        in_channels = mel_dim * r
        std_mul = 1.0
        for (out_channels, filter_size, dilation) in preattention:
            if in_channels != out_channels:
                # conv1d & relu
                self.prenet.append(
                    Conv1D(
                        self.full_name(),
                        in_channels,
                        out_channels,
                        filter_size=1,
                        std_mul=std_mul,
                        act="relu"))
                in_channels = out_channels
                std_mul = 2.0
            self.prenet.append(
                Conv1DGLU(
                    self.full_name(),
                    n_speakers,
                    speaker_dim,
                    in_channels,
                    out_channels,
                    filter_size,
                    dilation,
                    std_mul=std_mul,
                    dropout=dropout,
                    causal=True,
                    residual=True,
                    dtype=dtype))
            in_channels = out_channels
            std_mul = 4.0
        for i, layer in enumerate(self.prenet):
            self.add_sublayer("prenet_{}".format(i), layer)

        self.use_memory_mask = use_memory_mask
        if isinstance(attention, bool):
            self.attention = [attention] * len(convolutions)
        else:
            self.attention = attention

        if isinstance(force_monotonic_attention, bool):
            self.force_monotonic_attention = [force_monotonic_attention
                                              ] * len(convolutions)
        else:
            self.force_monotonic_attention = force_monotonic_attention

        # causual convolution & attention
        self.conv_attn = []
        for use_attention, (out_channels, filter_size,
                            dilation) in zip(self.attention, convolutions):
            assert (
                in_channels == out_channels
            ), "the stack of convolution & attention does not change channels"
            conv_layer = Conv1DGLU(
                self.full_name(),
                n_speakers,
                speaker_dim,
                in_channels,
                out_channels,
                filter_size,
                dilation,
                std_mul=std_mul,
                dropout=dropout,
                causal=True,
                residual=False,
                dtype=dtype)
            attn_layer = (AttentionLayer(
                self.full_name(),
                out_channels,
                embed_dim,
                dropout=dropout,
                window_range=window_range,
                key_projection=key_projection,
                value_projection=value_projection,
                dtype=dtype) if use_attention else None)
            in_channels = out_channels
            std_mul = 4.0
            self.conv_attn.append((conv_layer, attn_layer))
        for i, (conv_layer, attn_layer) in enumerate(self.conv_attn):
            self.add_sublayer("conv_{}".format(i), conv_layer)
            if attn_layer is not None:
                self.add_sublayer("attn_{}".format(i), attn_layer)

        # 1 * 1 conv to transform channels
        self.last_conv = Conv1D(
            self.full_name(),
            in_channels,
            mel_dim * r,
            filter_size=1,
            std_mul=std_mul,
            dropout=dropout,
            dtype=dtype)

        # mel (before sigmoid) to done hat
        self.fc = Conv1D(
            self.full_name(), mel_dim * r, 1, filter_size=1, dtype=dtype)

        # decoding configs
        self.max_decoder_steps = 200
        self.min_decoder_steps = 10

    def freeze_positional_encoding(self):
        for param in self.embed_query_positions.parameters():
            param.trainable = False
        for param in self.embed_keys_positions.parameters():
            param.trainable = False

    def forward(self,
                encoder_out,
                lengths,
                inputs,
                text_positions,
                frame_positions,
                speaker_embed=None):
        """
        Compute decoder outputs with ground truth mel spectrogram.

        Args:
            encoder_out (Tuple(Variable, Variable)): 
                keys (Variable): shape(B, C_emb, 1, T_enc), the key
                    representation from an encoder, where C_emb means
                    text embedding size.
                values (Variable): shape(B, C_emb, 1, T_enc), the value
                    representation from an encoder, where C_emb means
                    text embedding size.
            lengths (Variable): Shape(batch_size,), dtype: int64, valid lengths
                of text inputs for each example.
            inputs (Variable): Shape(B, C_mel, 1, T_mel), ground truth
                mel-spectrogram, which is used as decoder inputs when training.
            text_positions (Variable): Shape(B, T_enc, 1), dtype: int64.
                Positions indices for text inputs for the encoder, where 
                T_enc means the encoder timesteps.
            frame_positions (Variable): Shape(B, T_dec // r, 1), dtype: 
                int64. Positions indices for each decoder time steps.
            speaker_embed: shape(batch_size, speaker_dim), speaker embedding, 
                only used for multispeaker model.


        Returns:
            outputs (Variable): Shape(B, C_mel * r, 1, T_mel // r). Decoder
                outputs, where C_mel means the channels of mel-spectrogram, r 
                means the outputs per decoder step, T_mel means the length(time
                steps) of mel spectrogram. Note that, when r > 1, the decoder
                outputs r frames of mel spectrogram per step.
            alignments (Variable): Shape(N, B, T_mel // r, T_enc), the alignment
                tensor between the decoder and the encoder, where N means number
                of Attention Layers, T_mel means the length of mel spectrogram,
                r means the outputs per decoder step, T_enc means the encoder
                time steps.
            done (Variable): Shape(B, 1, 1, T_mel // r), probability that the
                outputs should stop.
            decoder_states (Variable): Shape(B, C_dec, 1, T_mel // r), decoder
                hidden states, where C_dec means the channels of decoder states.
        """

        # pack multiple frames if necessary
        B, _, _, T = inputs.shape
        if self.r > 1 and inputs.shape[1] == self.mel_dim:
            if T % self.r != 0:
                inputs = fluid.layers.slice(
                    inputs, axes=[3], starts=[0], ends=[T - T % self.r])
            inputs = fluid.layers.transpose(inputs, [0, 3, 2, 1])
            inputs = fluid.layers.reshape(
                inputs, shape=[B, -1, 1, self.mel_dim * self.r])
            inputs = fluid.layers.transpose(inputs, [0, 3, 2, 1])
        assert inputs.shape[3] == T // self.r

        if speaker_embed is not None:
            speaker_embed_bc1t = expand_speaker_embed(inputs, speaker_embed)
            speaker_embed_bc1t = fluid.layers.dropout(
                speaker_embed_bc1t,
                self.dropout,
                dropout_implementation="upscale_in_train")
        else:
            speaker_embed_bc1t = None

        keys, values = encoder_out

        if self.use_memory_mask and lengths is not None:
            mask = gen_mask2(lengths, keys.shape[-1])
        else:
            mask = None

        if text_positions is not None:
            w = self.key_position_rate
            if self.n_speakers > 1:
                w = w * fluid.layers.reshape(
                    self.speaker_proj1(speaker_embed), [B, -1])
            text_pos_embed = self.embed_keys_positions(text_positions, w)
            text_pos_embed = fluid.layers.transpose(
                fluid.layers.reshape(
                    text_pos_embed, shape=text_pos_embed.shape + [1]),
                perm=[0, 2, 3, 1])
            keys = keys + text_pos_embed

        if frame_positions is not None:
            w = self.query_position_rate
            if self.n_speakers > 1:
                w = w * fluid.layers.reshape(
                    self.speaker_proj2(speaker_embed), [B, -1])
            frame_pos_embed = self.embed_query_positions(frame_positions, w)
            frame_pos_embed = fluid.layers.transpose(
                fluid.layers.reshape(
                    frame_pos_embed, shape=frame_pos_embed.shape + [1]),
                perm=[0, 2, 3, 1])
        else:
            frame_pos_embed = None

        x = inputs
        x = fluid.layers.dropout(
            x, self.dropout, dropout_implementation="upscale_in_train")

        # Prenet
        for layer in self.prenet:
            x = (layer(x, speaker_embed_bc1t)
                 if isinstance(layer, Conv1DGLU) else layer(x))

        # Convolution & Multi-hop Attention
        alignments = []
        for conv, attn in self.conv_attn:
            residual = x
            x = conv(x, speaker_embed_bc1t)
            if attn is not None:
                if frame_pos_embed is not None:
                    x = x + frame_pos_embed
                x, attn_scores = attn(x, (keys, values), mask)
                alignments.append(attn_scores)
            x = fluid.layers.scale(residual + x, scale=np.sqrt(0.5))

        alignments = fluid.layers.stack(alignments)

        decoder_states = x
        x = self.last_conv(x)
        outputs = fluid.layers.sigmoid(x)
        done = fluid.layers.sigmoid(self.fc(x))

        return outputs, alignments, done, decoder_states

    def decode(self,
               encoder_out,
               text_positions,
               speaker_embed=None,
               initial_input=None,
               test_inputs=None):
        """
        Decode without ground truth mel spectrogram.
        
        Args:
            encoder_out (Tuple(Variable, Variable)): 
                keys (Variable): shape(B, C_emb, 1, T_enc), the key
                    representation from an encoder, where C_emb means
                    text embedding size.
                values (Variable): shape(B, C_emb, 1, T_enc), the value
                    representation from an encoder, where C_emb means
                    text embedding size.
            text_positions (Variable): Shape(B, T_enc, 1), dtype: int64.
                Positions indices for text inputs for the encoder, where 
                T_enc means the encoder timesteps.
               
            speaker_embed (Variable): Shape(B, C_sp), where C_sp means 
               speaker embedding size. It is only used for multispeaker model.
            initial_input (Variable, optional): Shape(B, C_mel * r, 1, 1).
               The input for the first time step of the decoder. If r > 0,
               it is a packed r frames of mel spectrograms.
            test_inputs (Variable, optional): Shape(B, C_mel, 1, T_test),
               where T_test means the time steps of test inputs. This is 
               only used for testing this method, the user should just leave
               it None.

        Returns:
            outputs (Variable): Shape(B, C_mel * r, 1, T_mel // r). Decoder
                outputs, where C_mel means the channels of mel-spectrogram, r 
                means the outputs per decoder step, T_mel means the length(time
                steps) of output mel spectrogram. Note that, when r > 1, 
                the decoder outputs r frames of mel spectrogram per step.
            alignments (Variable): Shape(B, T_mel // r, T_enc), the alignment
                tensor between the decoder and the encoder, T_mel means the 
                length of output mel spectrogram, r means the outputs per
                decoder step, T_enc means the encoder time steps.
            done (Variable): Shape(B, 1, 1, T_mel // r), probability that the
                outputs stops.
            decoder_states (Variable): Shape(B, C_dec, 1, T_mel // r), decoder
                hidden states, where C_dec means the channels of decoder states.
        """
        self.start_new_sequence()
        keys, values = encoder_out
        B = keys.shape[0]
        assert B == 1, "now only supports single instance inference"
        mask = None  # no mask because we use single instance decoding

        w = self.key_position_rate
        if speaker_embed is not None:
            if self.n_speakers > 1:
                w = w * fluid.layers.reshape(
                    self.speaker_proj1(speaker_embed), shape=[B, -1])
            speaker_embed_bc11 = fluid.layers.reshape(
                speaker_embed, shape=[B, speaker_embed.shape[1], 1, 1])
        else:
            speaker_embed_bc11 = None

        if text_positions is not None:
            text_pos_embed = self.embed_keys_positions(text_positions, w)
            text_pos_embed = fluid.layers.transpose(
                fluid.layers.reshape(
                    text_pos_embed, shape=text_pos_embed.shape + [1]),
                perm=[0, 2, 3, 1])
            keys = keys + text_pos_embed

        # start decoding, init accumulators
        decoder_states = []
        outputs = []
        alignments = []
        dones = []

        last_attended = [None] * len(self.conv_attn)
        for idx, monotonic_attn in enumerate(self.force_monotonic_attention):
            if monotonic_attn:
                last_attended[idx] = 0

        t = 0  # decoder time step
        if initial_input is None:
            initial_input = fluid.layers.zeros(
                shape=[B, self.mel_dim * self.r, 1, 1], dtype=keys.dtype)
        current_input = initial_input

        while True:
            frame_pos = fluid.layers.fill_constant(
                shape=[B, 1, 1], value=t + 1, dtype="int64")
            w = self.query_position_rate
            if self.n_speakers > 1:
                w = w * fluid.layers.reshape(
                    self.speaker_proj2(speaker_embed), shape=[B, -1])
            frame_pos_embed = self.embed_query_positions(frame_pos, w)
            frame_pos_embed = fluid.layers.transpose(
                fluid.layers.reshape(
                    frame_pos_embed, shape=frame_pos_embed.shape + [1]),
                perm=[0, 2, 3, 1])

            if test_inputs is not None:
                if t >= test_inputs.shape[3]:
                    break
                current_input = fluid.layers.reshape(
                    test_inputs[:, :, :, t],
                    shape=[B, test_inputs.shape[1], 1, 1])
            else:
                if t > 0:
                    current_input = outputs[-1]

            x = current_input
            x = fluid.layers.dropout(
                x, self.dropout, dropout_implementation="upscale_in_train")

            # Prenet
            for layer in self.prenet:
                x = (layer.add_input(x, speaker_embed_bc11)
                     if isinstance(layer, Conv1DGLU) else layer.add_input(x))

            step_attn_scores = []
            # Casual convolutions + Multi-hop attentions
            for i, (conv, attn) in enumerate(self.conv_attn):
                residual = x
                x = conv.add_input(x, speaker_embed_bc11)
                if attn is not None:
                    if frame_pos_embed is not None:
                        x = x + frame_pos_embed
                    x, attn_scores = attn(x, (keys, values), mask,
                                          last_attended[i])
                    step_attn_scores.append(attn_scores)

                    # update last attended when necessary
                    if self.force_monotonic_attention[i]:
                        last_attended[i] = np.argmax(
                            attn_scores.numpy(), axis=-1)[0][0]
                x = fluid.layers.scale(residual + x, scale=np.sqrt(0.5))
            if len(step_attn_scores):
                average_attn_scores = fluid.layers.reduce_mean(
                    fluid.layers.stack(step_attn_scores), dim=0)
            else:
                average_attn_scores = None

            decoder_state = x
            x = self.last_conv.add_input(x)

            output = fluid.layers.sigmoid(x)  # (B, r * C_mel, 1, 1)
            done = fluid.layers.sigmoid(self.fc(x))  # (B, 1, 1, 1)

            decoder_states.append(decoder_state)
            outputs.append(output)
            if average_attn_scores is not None:
                alignments.append(average_attn_scores)
            dones.append(done)

            t += 1

            if test_inputs is None:
                if (fluid.layers.reduce_min(done).numpy()[0] > 0.5 and
                        t > self.min_decoder_steps):
                    break
                elif t > self.max_decoder_steps:
                    break

        outputs = fluid.layers.concat(outputs, axis=3)
        if len(alignments):
            alignments = fluid.layers.concat(alignments, axis=1)
        else:
            alignments = None
        dones = fluid.layers.concat(dones, axis=3)
        decoder_states = fluid.layers.concat(decoder_states, axis=3)

        return outputs, alignments, dones, decoder_states

    def start_new_sequence(self):
        for layer in self.sublayers():
            if isinstance(layer, conv.Conv1D):
                layer.start_new_sequence()


class Converter(dg.Layer):
    """
    Vocoder that transforms mel spectrogram (or ecoder hidden states) 
    to waveform.
    """

    def __init__(self,
                 name_scope,
                 n_speakers,
                 speaker_dim,
                 in_channels,
                 linear_dim,
                 convolutions=(ConvSpec(256, 5, 1)) * 4,
                 time_upsampling=1,
                 dropout=0.1,
                 dtype="float32"):
        super(Converter, self).__init__(name_scope, dtype=dtype)

        self.n_speakers = n_speakers
        self.speaker_dim = speaker_dim
        self.in_channels = in_channels
        self.linear_dim = linear_dim
        self.time_upsampling = time_upsampling
        self.dropout = dropout

        target_channels = convolutions[0][0]

        # conv proj to target channels
        self.first_conv_proj = Conv1D(
            self.full_name(),
            in_channels,
            target_channels,
            filter_size=1,
            std_mul=1.0,
            dtype=dtype)

        # Idea from nyanko
        # upsampling convolitions
        if time_upsampling == 4:
            self.upsampling_convolutions = [
                Conv1DTranspose(
                    self.full_name(),
                    target_channels,
                    target_channels,
                    filter_size=2,
                    padding=0,
                    stride=2,
                    std_mul=1.0,
                    dtype=dtype),
                Conv1DGLU(
                    self.full_name(),
                    n_speakers,
                    speaker_dim,
                    target_channels,
                    target_channels,
                    filter_size=3,
                    dilation=1,
                    std_mul=1.0,
                    dropout=dropout,
                    causal=False,
                    residual=True,
                    dtype=dtype),
                Conv1DGLU(
                    self.full_name(),
                    n_speakers,
                    speaker_dim,
                    target_channels,
                    target_channels,
                    filter_size=3,
                    dilation=3,
                    std_mul=4.0,
                    dropout=dropout,
                    causal=False,
                    residual=True,
                    dtype=dtype),
                Conv1DTranspose(
                    self.full_name(),
                    target_channels,
                    target_channels,
                    filter_size=2,
                    padding=0,
                    stride=2,
                    std_mul=4.0,
                    dtype=dtype),
                Conv1DGLU(
                    self.full_name(),
                    n_speakers,
                    speaker_dim,
                    target_channels,
                    target_channels,
                    filter_size=3,
                    dilation=1,
                    std_mul=1.0,
                    dropout=dropout,
                    causal=False,
                    residual=True,
                    dtype=dtype),
                Conv1DGLU(
                    self.full_name(),
                    n_speakers,
                    speaker_dim,
                    target_channels,
                    target_channels,
                    filter_size=3,
                    dilation=3,
                    std_mul=4.0,
                    dropout=dropout,
                    causal=False,
                    residual=True,
                    dtype=dtype),
            ]

        elif time_upsampling == 2:
            self.upsampling_convolutions = [
                Conv1DTranspose(
                    self.full_name(),
                    target_channels,
                    target_channels,
                    filter_size=2,
                    padding=0,
                    stride=2,
                    std_mul=1.0,
                    dtype=dtype),
                Conv1DGLU(
                    self.full_name(),
                    n_speakers,
                    speaker_dim,
                    target_channels,
                    target_channels,
                    filter_size=3,
                    dilation=1,
                    std_mul=1.0,
                    dropout=dropout,
                    causal=False,
                    residual=True,
                    dtype=dtype),
                Conv1DGLU(
                    self.full_name(),
                    n_speakers,
                    speaker_dim,
                    target_channels,
                    target_channels,
                    filter_size=3,
                    dilation=3,
                    std_mul=4.0,
                    dropout=dropout,
                    causal=False,
                    residual=True,
                    dtype=dtype),
            ]
        elif time_upsampling == 1:
            self.upsampling_convolutions = [
                Conv1DGLU(
                    self.full_name(),
                    n_speakers,
                    speaker_dim,
                    target_channels,
                    target_channels,
                    filter_size=3,
                    dilation=3,
                    std_mul=4.0,
                    dropout=dropout,
                    causal=False,
                    residual=True,
                    dtype=dtype)
            ]
        else:
            raise ValueError("Not supported.")

        for i, layer in enumerate(self.upsampling_convolutions):
            self.add_sublayer("upsampling_convolutions_{}".format(i), layer)

        # post conv layers
        std_mul = 4.0
        in_channels = target_channels
        self.convolutions = []
        for (out_channels, filter_size, dilation) in convolutions:
            if in_channels != out_channels:
                self.convolutions.append(
                    Conv1D(
                        self.full_name(),
                        in_channels,
                        out_channels,
                        filter_size=1,
                        std_mul=std_mul,
                        act="relu",
                        dtype=dtype))
                in_channels = out_channels
                std_mul = 2.0
            self.convolutions.append(
                Conv1DGLU(
                    self.full_name(),
                    n_speakers,
                    speaker_dim,
                    in_channels,
                    out_channels,
                    filter_size=filter_size,
                    dilation=dilation,
                    std_mul=std_mul,
                    dropout=dropout,
                    causal=False,
                    residual=True,
                    dtype=dtype))
            in_channels = out_channels
            std_mul = 4.0

        for i, layer in enumerate(self.convolutions):
            self.add_sublayer("convolutions_{}".format(i), layer)

        # final conv proj, channel transformed to linear dim
        self.last_conv_proj = Conv1D(
            self.full_name(),
            in_channels,
            linear_dim,
            filter_size=1,
            std_mul=std_mul,
            dropout=dropout,
            act="sigmoid",
            dtype=dtype)

    def forward(self, x, speaker_embed=None):
        """
        Convert mel spectrogram or decoder hidden states to linear spectrogram.
        
        Args:
            x (Variable): Shape(B, C_in, 1, T_mel), converter inputs, where
                C_in means the input channel for the converter. Note that it 
                can be either C_mel (channel of mel spectrogram) or C_dec // r.
                When use mel_spectrogram as the input of converter, C_in = 
                C_mel; and when use decoder states as the input of converter,
                C_in = C_dec // r. In this scenario, decoder hidden states are
                treated as if they were r outputs per decoder step and are
                unpacked before passing to the converter.
            speaker_embed (Variable, optional): shape(B, C_sp), speaker
                embedding, where C_sp means the speaker embedding size.

        Returns:
            out (Variable): Shape(B, C_lin, 1, T_lin), the output linear 
                spectrogram, where C_lin means the channel of linear 
                spectrogram and T_linear means the length(time steps) of linear
                spectrogram. T_line = time_upsampling * T_mel, which depends 
                on the time_upsampling converter.
        """
        speaker_embed_bc1t = None
        if speaker_embed is not None:
            speaker_embed_bc1t = expand_speaker_embed(x, speaker_embed, tdim=-1)
            speaker_embed_bc1t = fluid.layers.dropout(
                speaker_embed_bc1t,
                self.dropout,
                dropout_implementation="upscale_in_train")

        x = self.first_conv_proj(x)

        for layer in chain(self.upsampling_convolutions, self.convolutions):
            # time_steps may change when timt_upsampling > 1
            if (speaker_embed_bc1t is not None and
                    speaker_embed_bc1t.shape[3] != x.shape[3]):
                speaker_embed_bc1t = expand_speaker_embed(
                    x, speaker_embed, tdim=3)
                speaker_embed_bc1t = fluid.layers.dropout(
                    speaker_embed_bc1t,
                    self.dropout,
                    dropout_implementation="upscale_in_train")
            x = (layer(x, speaker_embed_bc1t)
                 if isinstance(layer, Conv1DGLU) else layer(x))

        out = self.last_conv_proj(x)
        return out


class DeepVoiceTTS(dg.Layer):
    def __init__(self, name_scope, n_speakers, speaker_dim,
                 speaker_embedding_weight_std, n_vocab, embed_dim,
                 text_padding_idx, text_embedding_weight_std,
                 freeze_text_embedding, encoder_convolutions, max_positions,
                 position_padding_idx, trainable_positional_encodings, mel_dim,
                 r, prenet_convolutions, attentive_convolutions, attention,
                 use_memory_mask, force_monotonic_attention,
                 query_position_rate, key_position_rate, window_range,
                 key_projection, value_projection, linear_dim,
                 postnet_convolutions, time_upsampling, dropout,
                 use_decoder_state_for_postnet_input, dtype):
        super(DeepVoiceTTS, self).__init__(name_scope, dtype)

        self.n_speakers = n_speakers
        self.speaker_dim = speaker_dim
        if n_speakers > 1:
            self.speaker_embedding = Embedding(
                self.full_name(),
                n_speakers,
                speaker_dim,
                padding_idx=None,
                std=speaker_embedding_weight_std,
                dtype=dtype)

        self.embed_dim = embed_dim
        self.mel_dim = mel_dim
        self.r = r

        self.seq2seq = ConvS2S(
            self.full_name(), n_speakers, speaker_dim,
            speaker_embedding_weight_std, n_vocab, embed_dim, text_padding_idx,
            text_embedding_weight_std, freeze_text_embedding,
            encoder_convolutions, max_positions, position_padding_idx,
            trainable_positional_encodings, mel_dim, r, prenet_convolutions,
            attentive_convolutions, attention, use_memory_mask,
            force_monotonic_attention, query_position_rate, key_position_rate,
            window_range, key_projection, value_projection, dropout, dtype)

        self.use_decoder_state_for_postnet_input = use_decoder_state_for_postnet_input
        if use_decoder_state_for_postnet_input:
            assert (
                attentive_convolutions[-1].out_channels % self.r == 0
            ), "when using decoder states as converter input, you must assure the decoder state channels can be divided by r"
            converter_input_channels = attentive_convolutions[
                -1].out_channels // r
        else:
            converter_input_channels = mel_dim

        self.converter_input_channels = converter_input_channels
        self.linear_dim = linear_dim
        self.converter = Converter(
            self.full_name(),
            n_speakers,
            speaker_dim,
            converter_input_channels,
            linear_dim,
            convolutions=postnet_convolutions,
            time_upsampling=time_upsampling,
            dropout=dropout,
            dtype=dtype)

    def forward(self,
                text_sequences,
                valid_lengths,
                mel_inputs,
                speaker_indices=None,
                text_positions=None,
                frame_positions=None):
        """
        Encode text sequence and decode with ground truth mel spectrogram.
                
        Args:
            text_sequences (Variable): Shape(B, T_enc, 1), dtype: int64. Ihe
                input text indices. T_enc means the timesteps of text_sequences.
            valid_lengths (Variable): shape(batch_size,), dtype: int64,
                valid lengths for each example in text_sequences.
            mel_inputs (Variable): Shape(B, C_mel, 1, T_mel), ground truth
                mel-spectrogram, which is used as decoder inputs when training. 
            speaker_indices (Variable, optional): Shape(Batch_size, 1),
                dtype: int64. Speaker index for each example. This arg is not
                None only when the model is a multispeaker model.
            text_positions (Variable): Shape(B, T_enc, 1), dtype: int64.
                Positions indices for text inputs for the encoder, where 
                T_enc means the encoder timesteps.
            frame_positions (Variable): Shape(B, T_dec // r, 1), dtype: 
                int64. Positions indices for each decoder time steps.

        Returns:
            mel_outputs (Variable): Shape(B, C_mel * r, 1, T_mel // r). Decoder
                outputs, where C_mel means the channels of mel-spectrogram, r 
                means the outputs per decoder step, T_mel means the length(time
                steps) of mel spectrogram. Note that, when r > 1, the decoder
                outputs r frames of mel spectrogram per step.
            linear_outputs (Variable): Shape(B, C_lin, 1, T_lin), the output
                linear spectrogram, where C_lin means the channel of linear 
                spectrogram and T_linear means the length(time steps) of linear
                spectrogram. T_line = time_upsampling * T_mel, which depends 
                on the time_upsampling converter.
            alignments (Variable): Shape(N, B, T_mel // r, T_enc), the alignment
                tensor between the decoder and the encoder, where N means number
                of Attention Layers, T_mel means the length of mel spectrogram,
                r means the outputs per decoder step, T_enc means the encoder
                time steps.
            done (Variable): Shape(B, 1, 1, T_mel // r), probability that the
                outputs should stop.
        """

        batch_size = text_sequences.shape[0]
        if self.n_speakers == 1:
            assert speaker_indices is None, "this model does not support multi-speaker"

        if speaker_indices is not None:
            speaker_embed = self.speaker_embedding(speaker_indices)
        else:
            speaker_embed = None

        mel_outputs, alignments, done, decoder_states = self.seq2seq(
            text_sequences, valid_lengths, mel_inputs, speaker_embed,
            text_positions, frame_positions)

        # unpack multi frames
        if self.r > 1:
            mel_outputs = fluid.layers.transpose(mel_outputs, [0, 3, 2, 1])
            mel_outputs = fluid.layers.reshape(
                mel_outputs, [batch_size, -1, 1, self.mel_dim])
            mel_outputs = fluid.layers.transpose(mel_outputs, [0, 3, 2, 1])

        if self.use_decoder_state_for_postnet_input:
            postnet_input = fluid.layers.transpose(decoder_states, [0, 3, 2, 1])
            postnet_input = fluid.layers.reshape(
                postnet_input,
                [batch_size, -1, 1, self.converter_input_channels])
            postnet_input = fluid.layers.transpose(postnet_input, [0, 3, 2, 1])
        else:
            postnet_input = mel_outputs

        linear_outputs = self.converter(postnet_input, speaker_embed)

        return mel_outputs, linear_outputs, alignments, done

    def transduce(self, text_sequences, text_positions, speaker_indices=None):
        """
        Encode text sequence and decode without ground truth mel spectrogram.
        
        Args:
            text_sequences (Variable): Shape(B, T_enc, 1), dtype: int64. Ihe
                input text indices. T_enc means the timesteps of text_sequences.
            text_positions (Variable): Shape(B, T_enc, 1), dtype: int64.
                Positions indices for text inputs for the encoder, where 
                T_enc means the encoder timesteps.
            speaker_indices (Variable, optional): Shape(Batch_size, 1),
                dtype: int64. Speaker index for each example. This arg is not
                None only when the model is a multispeaker model.

        Returns:
            mel_outputs (Variable): Shape(B, C_mel * r, 1, T_mel // r). Decoder
                outputs, where C_mel means the channels of mel-spectrogram, r 
                means the outputs per decoder step, T_mel means the length(time
                steps) of mel spectrogram. Note that, when r > 1, the decoder
                outputs r frames of mel spectrogram per step.
            linear_outputs (Variable): Shape(B, C_lin, 1, T_lin), the output
                linear spectrogram, where C_lin means the channel of linear 
                spectrogram and T_linear means the length(time steps) of linear
                spectrogram. T_line = time_upsampling * T_mel, which depends 
                on the time_upsampling converter.
            alignments (Variable): Shape(B, T_mel // r, T_enc), the alignment
                tensor between the decoder and the encoder, T_mel means the
                length of mel spectrogram, r means the outputs per decoder
                step, T_enc means the encoder time steps.
            done (Variable): Shape(B, 1, 1, T_mel // r), probability that the
                outputs should stop.
        """
        batch_size = text_sequences.shape[0]

        if speaker_indices is not None:
            speaker_embed = self.speaker_embedding(speaker_indices)
        else:
            speaker_embed = None

        mel_outputs, alignments, done, decoder_states = self.seq2seq.transduce(
            text_sequences, text_positions, speaker_embed)

        if self.r > 1:
            mel_outputs = fluid.layers.transpose(mel_outputs, [0, 3, 2, 1])
            mel_outputs = fluid.layers.reshape(
                mel_outputs, [batch_size, -1, 1, self.mel_dim])
            mel_outputs = fluid.layers.transpose(mel_outputs, [0, 3, 2, 1])

        if self.use_decoder_state_for_postnet_input:
            postnet_input = fluid.layers.transpose(decoder_states, [0, 3, 2, 1])
            postnet_input = fluid.layers.reshape(
                postnet_input,
                [batch_size, -1, 1, self.converter_input_channels])
            postnet_input = fluid.layers.transpose(postnet_input, [0, 3, 2, 1])
        else:
            postnet_input = mel_outputs

        linear_outputs = self.converter(postnet_input, speaker_embed)

        return mel_outputs, linear_outputs, alignments, done


class ConvS2S(dg.Layer):
    def __init__(self, name_scope, n_speakers, speaker_dim,
                 speaker_embedding_weight_std, n_vocab, embed_dim,
                 text_padding_idx, text_embedding_weight_std,
                 freeze_text_embedding, encoder_convolutions, max_positions,
                 position_padding_idx, trainable_positional_encodings, mel_dim,
                 r, prenet_convolutions, attentive_convolutions, attention,
                 use_memory_mask, force_monotonic_attention,
                 query_position_rate, key_position_rate, window_range,
                 key_projection, value_projection, dropout, dtype):
        super(ConvS2S, self).__init__(name_scope, dtype)

        self.freeze_text_embedding = freeze_text_embedding
        self.trainable_positional_encodings = trainable_positional_encodings

        self.n_speakers = n_speakers
        self.speaker_dim = speaker_dim

        self.embed_dim = embed_dim
        self.encoder = Encoder(
            self.full_name(),
            n_vocab,
            embed_dim,
            n_speakers,
            speaker_dim,
            padding_idx=None,
            embedding_weight_std=text_embedding_weight_std,
            convolutions=encoder_convolutions,
            max_positions=max_positions,
            dropout=dropout,
            dtype=dtype)
        if freeze_text_embedding:
            self.encoder.freeze_embedding()

        self.mel_dim = mel_dim
        self.r = r
        self.decoder = Decoder(
            self.full_name(),
            n_speakers,
            speaker_dim,
            embed_dim,
            mel_dim,
            r,
            max_positions,
            position_padding_idx,
            preattention=prenet_convolutions,
            convolutions=attentive_convolutions,
            attention=attention,
            dropout=dropout,
            use_memory_mask=use_memory_mask,
            force_monotonic_attention=force_monotonic_attention,
            query_position_rate=query_position_rate,
            key_position_rate=key_position_rate,
            window_range=window_range,
            key_projection=key_projection,
            value_projection=key_projection,
            dtype=dtype)
        if not trainable_positional_encodings:
            self.decoder.freeze_positional_encoding()

    def forward(self,
                text_sequences,
                valid_lengths,
                mel_inputs,
                speaker_embed=None,
                text_positions=None,
                frame_positions=None):
        """
        Encode text sequence and decode with ground truth mel spectrogram.

        Args:
            text_sequences (Variable): Shape(B, T_enc, 1), dtype: int64. Ihe
                input text indices. T_enc means the timesteps of text_sequences.
            valid_lengths (Variable): shape(batch_size,), dtype: int64,
                valid lengths for each example in text_sequences.
            mel_inputs (Variable): Shape(B, C_mel, 1, T_mel), ground truth
                mel-spectrogram, which is used as decoder inputs when training. 
            speaker_embed (Variable, optional): Shape(Batch_size, speaker_dim),
                dtype: float32. Speaker embeddings. This arg is not None only
                when the model is a multispeaker model.
            text_positions (Variable): Shape(B, T_enc, 1), dtype: int64.
                Positions indices for text inputs for the encoder, where 
                T_enc means the encoder timesteps.
            frame_positions (Variable): Shape(B, T_dec // r, 1), dtype: 
                int64. Positions indices for each decoder time steps.

        Returns:
            mel_outputs (Variable): Shape(B, C_mel * r, 1, T_mel // r). Decoder
                outputs, where C_mel means the channels of mel-spectrogram, r 
                means the outputs per decoder step, T_mel means the length(time
                steps) of mel spectrogram. Note that, when r > 1, the decoder
                outputs r frames of mel spectrogram per step.
            alignments (Variable): Shape(N, B, T_mel // r, T_enc), the alignment
                tensor between the decoder and the encoder, where N means number
                of Attention Layers, T_mel means the length of mel spectrogram,
                r means the outputs per decoder step, T_enc means the encoder
                time steps.
            done (Variable): Shape(B, 1, 1, T_mel // r), probability that the
                outputs should stop.
            decoder_states (Variable): Shape(B, C_dec, 1, T_mel // r), decoder
                hidden states, where C_dec means the channels of decoder states.
        """
        keys, values = self.encoder(text_sequences, speaker_embed)
        mel_outputs, alignments, done, decoder_states = self.decoder(
            (keys, values), valid_lengths, mel_inputs, text_positions,
            frame_positions, speaker_embed)

        return mel_outputs, alignments, done, decoder_states

    def transduce(self, text_sequences, text_positions, speaker_embed=None):
        """
        Encode text sequence and decode without ground truth mel spectrogram.
        
        Args:
            text_sequences (Variable): Shape(B, T_enc, 1), dtype: int64. Ihe
                input text indices. T_enc means the timesteps of text_sequences.
            text_positions (Variable): Shape(B, T_enc, 1), dtype: int64.
                Positions indices for text inputs for the encoder, where 
                T_enc means the encoder timesteps.
            speaker_embed (Variable, optional): Shape(Batch_size, speaker_dim),
                dtype: float32. Speaker embeddings. This arg is not None only
                when the model is a multispeaker model.

        Returns:
            mel_outputs (Variable): Shape(B, C_mel * r, 1, T_mel // r). Decoder
                outputs, where C_mel means the channels of mel-spectrogram, r 
                means the outputs per decoder step, T_mel means the length(time
                steps) of mel spectrogram. Note that, when r > 1, the decoder
                outputs r frames of mel spectrogram per step.
            alignments (Variable): Shape(B, T_mel // r, T_enc), the alignment
                tensor between the decoder and the encoder, T_mel means the
                length of mel spectrogram, r means the outputs per decoder
                step, T_enc means the encoder time steps.
            done (Variable): Shape(B, 1, 1, T_mel // r), probability that the
                outputs should stop.
            decoder_states (Variable): Shape(B, C_dec, 1, T_mel // r), decoder
                hidden states, where C_dec means the channels of decoder states.
        """
        keys, values = self.encoder(text_sequences, speaker_embed)
        mel_outputs, alignments, done, decoder_states = self.decoder.decode(
            (keys, values), text_positions, speaker_embed)

        return mel_outputs, alignments, done, decoder_states
