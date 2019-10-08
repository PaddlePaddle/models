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
Generate a name map from configuration
"""
from collections import OrderedDict

import numpy as np
from argparse import ArgumentParser
from deepvoice3_paddle import frontend
from hparams import hparams, hparams_debug_string


def build_arg_parser():
    parser = ArgumentParser(description="Train deepvoice 3 model.")

    parser.add_argument(
        "--preset",
        type=str,
        required=True,
        help="Path of preset parameters in json format.")
    parser.add_argument(
        "--hparams",
        type=str,
        default="",
        help="Hyper parameters to override preset.")
    return parser


def gen_conv(in_channels, out_channels, kernel_size, torch_state, paddle_state,
             name_map, torch_base_name, paddle_base_name):
    # bias
    torch_bias_name = "{}.bias".format(torch_base_name)
    paddle_bias_name = "{}.b_0".format(paddle_base_name)
    torch_bias_shape = (out_channels, )
    paddle_bias_shape = [out_channels, ]

    torch_state[torch_bias_name] = torch_bias_shape
    paddle_state[paddle_bias_name] = paddle_bias_shape
    name_map[torch_bias_name] = paddle_bias_name

    # wg
    torch_wg_name = "{}.weight_g".format(torch_base_name)
    paddle_wg_name = "{}.w_1".format(paddle_base_name)
    torch_wg_shape = (out_channels, 1, 1)
    paddle_wg_shape = [out_channels, ]

    torch_state[torch_wg_name] = torch_wg_shape
    paddle_state[paddle_wg_name] = paddle_wg_shape
    name_map[torch_wg_name] = paddle_wg_name

    # wv
    torch_wv_name = "{}.weight_v".format(torch_base_name)
    paddle_wv_name = "{}.w_0".format(paddle_base_name)
    torch_wv_shape = (out_channels, in_channels, kernel_size)
    paddle_wv_shape = [out_channels, in_channels, 1, kernel_size]

    torch_state[torch_wv_name] = torch_wv_shape
    paddle_state[paddle_wv_name] = paddle_wv_shape
    name_map[torch_wv_name] = paddle_wv_name


def gen_fc2conv(in_channels, out_channels, torch_state, paddle_state, name_map,
                torch_base_name, paddle_base_name):
    # bias
    torch_bias_name = "{}.bias".format(torch_base_name)
    paddle_bias_name = "{}.b_0".format(paddle_base_name)
    torch_bias_shape = (out_channels, )
    paddle_bias_shape = [out_channels, ]

    torch_state[torch_bias_name] = torch_bias_shape
    paddle_state[paddle_bias_name] = paddle_bias_shape
    name_map[torch_bias_name] = paddle_bias_name

    # wg
    torch_wg_name = "{}.weight_g".format(torch_base_name)
    paddle_wg_name = "{}.w_1".format(paddle_base_name)
    torch_wg_shape = (out_channels, 1)
    paddle_wg_shape = [out_channels, ]

    torch_state[torch_wg_name] = torch_wg_shape
    paddle_state[paddle_wg_name] = paddle_wg_shape
    name_map[torch_wg_name] = paddle_wg_name

    # wv
    torch_wv_name = "{}.weight_v".format(torch_base_name)
    paddle_wv_name = "{}.w_0".format(paddle_base_name)
    torch_wv_shape = (out_channels, in_channels)
    paddle_wv_shape = [out_channels, in_channels, 1, 1]

    torch_state[torch_wv_name] = torch_wv_shape
    paddle_state[paddle_wv_name] = paddle_wv_shape
    name_map[torch_wv_name] = paddle_wv_name


def generate_name_map(name_scope):
    TTS_model_idx = 0
    prefix = "/".join([name_scope, "DeepVoiceTTS_{}".format(TTS_model_idx)])

    _frontend = getattr(frontend, hparams.frontend)

    torch_state = OrderedDict()
    paddle_state = OrderedDict()
    name_map = OrderedDict()

    # text embedding
    torch_name = "seq2seq.encoder.embed_tokens.weight"
    paddle_name = "{}/ConvS2S_0/Encoder_0/Embedding_0.w_0".format(prefix)
    torch_shape = (_frontend.n_vocab, hparams.text_embed_dim)
    paddle_shape = [_frontend.n_vocab, hparams.text_embed_dim]

    torch_state[torch_name] = torch_shape
    paddle_state[paddle_name] = paddle_shape
    name_map[torch_name] = paddle_name

    # encoder

    Conv1D_idx = 0
    Conv1DGLU_idx = 0

    if hparams.n_speakers > 1:
        for i in [1, 2]:
            torch_base_name = "seq2seq.encoder.speaker_fc{}".format(i)
            paddle_base_name = "{}/ConvS2S_0/Encoder_0/Conv1D_{}/Conv2D_0".format(
                prefix, Conv1D_idx)
            Conv1D_idx += 1
            gen_fc2conv(hparams.speaker_embed_dim, hparams.text_embed_dim,
                        torch_state, paddle_state, name_map, torch_base_name,
                        paddle_base_name)

    # encoder convolution specs
    encoder_channels = hparams.encoder_channels
    kernel_size = hparams.kernel_size

    h = encoder_channels
    k = kernel_size
    convolutions = [(h, k, 1), (h, k, 3), (h, k, 9), (h, k, 27), (h, k, 1),
                    (h, k, 3), (h, k, 9), (h, k, 27), (h, k, 1), (h, k, 3)]
    torch_layer_idx = 0

    in_channels = hparams.text_embed_dim

    # first conv1d & conv1dglus
    for out_channels, kernel_size, dilation in convolutions:
        if in_channels != out_channels:
            torch_base_name = "seq2seq.encoder.convolutions.{}".format(
                torch_layer_idx)
            paddle_base_name = "{}/ConvS2S_0/Encoder_0/Conv1D_{}/Conv2D_0".format(
                prefix, Conv1D_idx)
            gen_conv(in_channels, out_channels, 1, torch_state, paddle_state,
                     name_map, torch_base_name, paddle_base_name)
            torch_layer_idx += 2
            Conv1D_idx += 1
            in_channels = out_channels

        torch_base_name = "seq2seq.encoder.convolutions.{}.conv".format(
            torch_layer_idx)
        paddle_base_name = "{}/ConvS2S_0/Encoder_0/Conv1DGLU_{}/Conv1D_0/Conv2D_0".format(
            prefix, Conv1DGLU_idx)
        gen_conv(in_channels, 2 * out_channels, kernel_size, torch_state,
                 paddle_state, name_map, torch_base_name, paddle_base_name)

        if hparams.n_speakers > 1:
            torch_base_name = "seq2seq.encoder.convolutions.{}.speaker_proj".format(
                torch_layer_idx)
            paddle_base_name = "{}/ConvS2S_0/Encoder_0/Conv1DGLU_{}/Conv1D_1/Conv2D_0".format(
                prefix, Conv1DGLU_idx)
            gen_fc2conv(hparams.speaker_embed_dim, out_channels, torch_state,
                        paddle_state, name_map, torch_base_name,
                        paddle_base_name)

        torch_layer_idx += 1
        Conv1DGLU_idx += 1
        in_channels = out_channels

    # last conv1d
    torch_base_name = "seq2seq.encoder.convolutions.{}".format(torch_layer_idx)
    paddle_base_name = "{}/ConvS2S_0/Encoder_0/Conv1D_{}/Conv2D_0".format(
        prefix, Conv1D_idx)
    gen_conv(in_channels, hparams.text_embed_dim, 1, torch_state, paddle_state,
             name_map, torch_base_name, paddle_base_name)
    torch_layer_idx += 2
    Conv1D_idx += 1
    in_channels = out_channels

    # decoder

    # position embeddings
    PositionEmbedding_idx = 0
    torch_name = "seq2seq.decoder.embed_query_positions.weight"
    paddle_name = "{}/ConvS2S_0/Decoder_0/PositionEmbedding_{}/Embedding_0.w_0".format(
        prefix, PositionEmbedding_idx)
    torch_shape = (hparams.max_positions, hparams.decoder_channels)
    paddle_shape = [hparams.max_positions, hparams.decoder_channels]
    PositionEmbedding_idx += 1

    torch_state[torch_name] = torch_shape
    paddle_state[paddle_name] = paddle_shape
    name_map[torch_name] = paddle_name

    torch_name = "seq2seq.decoder.embed_keys_positions.weight"
    paddle_name = "{}/ConvS2S_0/Decoder_0/PositionEmbedding_{}/Embedding_0.w_0".format(
        prefix, PositionEmbedding_idx)
    PositionEmbedding_idx += 1
    torch_shape = (hparams.max_positions, hparams.text_embed_dim)
    paddle_shape = [hparams.max_positions, hparams.text_embed_dim]

    torch_state[torch_name] = torch_shape
    paddle_state[paddle_name] = paddle_shape
    name_map[torch_name] = paddle_name

    Conv1D_idx = 0
    Conv1DGLU_idx = 0

    if hparams.n_speakers > 1:
        for i in [1, 2]:
            torch_base_name = "seq2seq.decoder.speaker_proj{}".format(i)
            paddle_base_name = "{}/ConvS2S_0/Decoder_0/Conv1D_{}/Conv2D_0".format(
                prefix, Conv1D_idx)
            Conv1D_idx += 1
            gen_fc2conv(hparams.speaker_embed_dim, 1, torch_state, paddle_state,
                        name_map, torch_base_name, paddle_base_name)

    # prenet
    torch_layer_idx = 0

    h = hparams.decoder_channels
    k = hparams.kernel_size
    prenet_convolutions = [(h, k, 1), (h, k, 3)]

    in_channels = hparams.num_mels * hparams.outputs_per_step
    for out_channels, kernel_size, dilation in prenet_convolutions:
        if in_channels != out_channels:
            torch_base_name = "seq2seq.decoder.preattention.{}".format(
                torch_layer_idx)
            paddle_base_name = "{}/ConvS2S_0/Decoder_0/Conv1D_{}/Conv2D_0".format(
                prefix, Conv1D_idx)
            gen_conv(in_channels, out_channels, 1, torch_state, paddle_state,
                     name_map, torch_base_name, paddle_base_name)
            torch_layer_idx += 2
            Conv1D_idx += 1
            in_channels = out_channels

        torch_base_name = "seq2seq.decoder.preattention.{}.conv".format(
            torch_layer_idx)
        paddle_base_name = "{}/ConvS2S_0/Decoder_0/Conv1DGLU_{}/Conv1D_0/Conv2D_0".format(
            prefix, Conv1DGLU_idx)
        gen_conv(in_channels, 2 * out_channels, kernel_size, torch_state,
                 paddle_state, name_map, torch_base_name, paddle_base_name)

        if hparams.n_speakers > 1:
            torch_base_name = "seq2seq.decoder.preattention.{}.speaker_proj".format(
                torch_layer_idx)
            paddle_base_name = "{}/ConvS2S_0/Decoder_0/Conv1DGLU_{}/Conv1D_1/Conv2D_0".format(
                prefix, Conv1DGLU_idx)
            gen_fc2conv(hparams.speaker_embed_dim, out_channels, torch_state,
                        paddle_state, name_map, torch_base_name,
                        paddle_base_name)

        torch_layer_idx += 1
        Conv1DGLU_idx += 1
        in_channels = out_channels

    # conv & attn
    torch_layer_idx = 0
    convolutions = [(h, k, 1), (h, k, 3), (h, k, 9), (h, k, 27), (h, k, 1)]
    for out_channels, kernel_size, dilation in convolutions:
        if in_channels != out_channels:
            torch_base_name = "seq2seq.decoder.convolutions.{}".format(
                torch_layer_idx)
            paddle_base_name = "{}/ConvS2S_0/Decoder_0/Conv1D_{}/Conv2D_0".format(
                prefix, Conv1D_idx)
            gen_conv(in_channels, out_channels, kernel_size, in_channels,
                     out_channels, kernel_size, torch_state, paddle_state,
                     name_map, torch_base_name, paddle_base_name)
            torch_layer_idx += 2
            Conv1D_idx += 1
            in_channels = out_channels

        torch_base_name = "seq2seq.decoder.convolutions.{}.conv".format(
            torch_layer_idx)
        paddle_base_name = "{}/ConvS2S_0/Decoder_0/Conv1DGLU_{}/Conv1D_0/Conv2D_0".format(
            prefix, Conv1DGLU_idx)
        gen_conv(in_channels, 2 * out_channels, kernel_size, torch_state,
                 paddle_state, name_map, torch_base_name, paddle_base_name)

        if hparams.n_speakers > 1:
            torch_base_name = "seq2seq.decoder.convolutions.{}.speaker_proj".format(
                torch_layer_idx)
            paddle_base_name = "{}/ConvS2S_0/Decoder_0/Conv1DGLU_{}/Conv1D_1/Conv2D_0".format(
                prefix, Conv1DGLU_idx)
            gen_fc2conv(hparams.speaker_embed_dim, out_channels, torch_state,
                        paddle_state, name_map, torch_base_name,
                        paddle_base_name)

        torch_layer_idx += 1
        Conv1DGLU_idx += 1
        in_channels = out_channels

    # attention
    attention = [True, False, False, False, True]
    AttentionLayer_idx = 0
    parts = [
        "query", "key" if hparams.key_projection else None, "value"
        if hparams.value_projection else None, "out"
    ]
    parts = [x for x in parts if x is not None]
    for (i, (attn, (out_channels, kernel_size,
                    dilation))) in enumerate(zip(attention, convolutions)):
        if attn is False:
            in_channels = out_channels
            continue
        for ipart, part in enumerate(parts):
            torch_base_name = "seq2seq.decoder.attention.{}.{}_projection".format(
                i, part)
            paddle_base_name = "{}/ConvS2S_0/Decoder_0/AttentionLayer_{}/Conv1D_{}/Conv2D_0".format(
                prefix, AttentionLayer_idx, ipart)
            if part == "query":
                C_in = out_channels
                C_out = hparams.text_embed_dim
            elif part == "out":
                C_in = hparams.text_embed_dim
                C_out = out_channels
            else:
                C_in = hparams.text_embed_dim
                C_out = hparams.text_embed_dim
            gen_fc2conv(C_in, C_out, torch_state, paddle_state, name_map,
                        torch_base_name, paddle_base_name)
        in_channels = out_channels
        AttentionLayer_idx += 1

    # last conv
    torch_base_name = "seq2seq.decoder.last_conv"
    paddle_base_name = "{}/ConvS2S_0/Decoder_0/Conv1D_{}/Conv2D_0".format(
        prefix, Conv1D_idx)
    gen_conv(in_channels, hparams.num_mels * hparams.outputs_per_step, 1,
             torch_state, paddle_state, name_map, torch_base_name,
             paddle_base_name)
    Conv1D_idx += 1

    # for done output
    torch_base_name = "seq2seq.decoder.fc"
    paddle_base_name = "{}/ConvS2S_0/Decoder_0/Conv1D_{}/Conv2D_0".format(
        prefix, Conv1D_idx)
    gen_fc2conv(hparams.num_mels * hparams.outputs_per_step, 1, torch_state,
                paddle_state, name_map, torch_base_name, paddle_base_name)

    # converter

    # time_upsampling
    if hparams.use_decoder_state_for_postnet_input:
        in_dim = hparams.decoder_channels // hparams.outputs_per_step
    else:
        in_dim = hparams.num_mels
    h = hparams.converter_channels

    time_upsampling = max(hparams.downsample_step // hparams.outputs_per_step,
                          1)
    assert time_upsampling == hparams.downsample_step, "implementation difference occured"
    assert time_upsampling in [1, 2, 4], "other values not supported yet"

    torch_layer_idx = 0
    Conv1D_idx = 0
    Conv1DGLU_idx = 0
    Conv1DTranspose_idx = 0

    h = hparams.converter_channels
    k = hparams.kernel_size

    postnet_convolutions = [(h, k, 1), (h, k, 3), (2 * h, k, 1), (2 * h, k, 3)]
    in_channels = postnet_convolutions[0][0]

    if time_upsampling == 4:
        torch_base_name = "postnet.convolutions.{}".format(torch_layer_idx)
        paddle_base_name = "{}/Converter_0/Conv1D_{}/Conv2D_0".format(
            prefix, Conv1D_idx)
        gen_conv(in_dim, in_channels, 1, torch_state, paddle_state, name_map,
                 torch_base_name, paddle_base_name)
        torch_layer_idx += 1
        Conv1D_idx += 1

        torch_base_name = "postnet.convolutions.{}".format(torch_layer_idx)
        paddle_base_name = "{}/Converter_0/Conv1DTranspose_{}/Conv2DTranspose_0".format(
            prefix, Conv1DTranspose_idx)
        gen_conv(in_channels, in_channels, 2, torch_state, paddle_state,
                 name_map, torch_base_name, paddle_base_name)
        torch_layer_idx += 1
        Conv1DTranspose_idx += 1

        torch_base_name = "postnet.convolutions.{}.conv".format(torch_layer_idx)
        paddle_base_name = "{}/Converter_0/Conv1DGLU_{}/Conv1D_0/Conv2D_0".format(
            prefix, Conv1DGLU_idx)
        gen_conv(in_channels, 2 * in_channels, 3, torch_state, paddle_state,
                 name_map, torch_base_name, paddle_base_name)

        if hparams.n_speakers > 1:
            torch_base_name = "postnet.convolutions.{}.speaker_proj".format(
                torch_layer_idx)
            paddle_base_name = "{}/Converter_0/Conv1DGLU_{}/Conv1D_1/Conv2D_0".format(
                prefix, Conv1DGLU_idx)
            gen_fc2conv(hparams.speaker_embed_dim, in_channels, torch_state,
                        paddle_state, name_map, torch_base_name,
                        paddle_base_name)
        torch_layer_idx += 1
        Conv1DGLU_idx += 1

        torch_base_name = "postnet.convolutions.{}.conv".format(torch_layer_idx)
        paddle_base_name = "{}/Converter_0/Conv1DGLU_{}/Conv1D_0/Conv2D_0".format(
            prefix, Conv1DTranspose_idx)
        gen_conv(in_channels, 2 * in_channels, 3, torch_state, paddle_state,
                 name_map, torch_base_name, paddle_base_name)

        if hparams.n_speakers > 1:
            torch_base_name = "postnet.convolutions.{}.speaker_proj".format(
                torch_layer_idx)
            paddle_base_name = "{}/Converter_0/Conv1DGLU_{}/Conv1D_1/Conv2D_0".format(
                prefix, Conv1DGLU_idx)
            gen_fc2conv(hparams.speaker_embed_dim, in_channels, torch_state,
                        paddle_state, name_map, torch_base_name,
                        paddle_base_name)
        torch_layer_idx += 1
        Conv1DGLU_idx += 1

        torch_base_name = "postnet.convolutions.{}".format(torch_layer_idx)
        paddle_base_name = "{}/Converter_0/Conv1DTranspose_{}/Conv2DTranspose_0".format(
            prefix, Conv1DTranspose_idx)
        gen_conv(in_channels, in_channels, 2, torch_state, paddle_state,
                 name_map, torch_base_name, paddle_base_name)
        torch_layer_idx += 1
        Conv1DTranspose_idx += 1

        torch_base_name = "postnet.convolutions.{}.conv".format(torch_layer_idx)
        paddle_base_name = "{}/Converter_0/Conv1DGLU_{}/Conv1D_0/Conv2D_0".format(
            prefix, Conv1DGLU_idx)
        gen_conv(in_channels, 2 * in_channels, 3, torch_state, paddle_state,
                 name_map, torch_base_name, paddle_base_name)

        if hparams.n_speakers > 1:
            torch_base_name = "postnet.convolutions.{}.speaker_proj".format(
                torch_layer_idx)
            paddle_base_name = "{}/Converter_0/Conv1DGLU_{}/Conv1D_1/Conv2D_0".format(
                prefix, Conv1DGLU_idx)
            gen_fc2conv(hparams.speaker_embed_dim, in_channels, torch_state,
                        paddle_state, name_map, torch_base_name,
                        paddle_base_name)
        torch_layer_idx += 1
        Conv1DGLU_idx += 1

        torch_base_name = "postnet.convolutions.{}.conv".format(torch_layer_idx)
        paddle_base_name = "{}/Converter_0/Conv1DGLU_{}/Conv1D_0/Conv2D_0".format(
            prefix, Conv1DGLU_idx)
        gen_conv(in_channels, 2 * in_channels, 3, torch_state, paddle_state,
                 name_map, torch_base_name, paddle_base_name)

        if hparams.n_speakers > 1:
            torch_base_name = "postnet.convolutions.{}.speaker_proj".format(
                torch_layer_idx)
            paddle_base_name = "{}/Converter_0/Conv1DGLU_{}/Conv1D_1/Conv2D_0".format(
                prefix, Conv1DGLU_idx)
            gen_fc2conv(hparams.speaker_embed_dim, in_channels, torch_state,
                        paddle_state, name_map, torch_base_name,
                        paddle_base_name)
        torch_layer_idx += 1
        Conv1DGLU_idx += 1

    elif time_upsampling == 2:

        torch_base_name = "postnet.convolutions.{}".format(torch_layer_idx)
        paddle_base_name = "{}/Converter_0/Conv1D_{}/Conv2D_0".format(
            prefix, Conv1D_idx)
        gen_conv(in_dim, in_channels, 1, torch_state, paddle_state, name_map,
                 torch_base_name, paddle_base_name)
        torch_layer_idx += 1
        Conv1D_idx += 1

        torch_base_name = "postnet.convolutions.{}".format(torch_layer_idx)
        paddle_base_name = "{}/Converter_0/Conv1DTranspose_{}/Conv2DTranspose_0".format(
            prefix, Conv1DTranspose_idx)
        gen_conv(in_channels, in_channels, 2, torch_state, paddle_state,
                 name_map, torch_base_name, paddle_base_name)
        torch_layer_idx += 1
        Conv1DTranspose_idx += 1

        torch_base_name = "postnet.convolutions.{}.conv".format(torch_layer_idx)
        paddle_base_name = "{}/Converter_0/Conv1DGLU_{}/Conv1D_0/Conv2D_0".format(
            prefix, Conv1DGLU_idx)
        gen_conv(in_channels, 2 * in_channels, 3, torch_state, paddle_state,
                 name_map, torch_base_name, paddle_base_name)
        if hparams.n_speakers > 1:
            torch_base_name = "postnet.convolutions.{}.speaker_proj".format(
                torch_layer_idx)
            paddle_base_name = "{}/Converter_0/Conv1DGLU_{}/Conv1D_1/Conv2D_0".format(
                prefix, Conv1DGLU_idx)
            gen_fc2conv(hparams.speaker_embed_dim, in_channels, torch_state,
                        paddle_state, name_map, torch_base_name,
                        paddle_base_name)
        torch_layer_idx += 1
        Conv1DGLU_idx += 1

        torch_base_name = "postnet.convolutions.{}.conv".format(torch_layer_idx)
        paddle_base_name = "{}/Converter_0/Conv1DGLU_{}/Conv1D_0/Conv2D_0".format(
            prefix, Conv1DTranspose_idx)
        gen_conv(in_channels, 2 * in_channels, 3, torch_state, paddle_state,
                 name_map, torch_base_name, paddle_base_name)
        if hparams.n_speakers > 1:
            torch_base_name = "postnet.convolutions.{}.speaker_proj".format(
                torch_layer_idx)
            paddle_base_name = "{}/Converter_0/Conv1DGLU_{}/Conv1D_1/Conv2D_0".format(
                prefix, Conv1DGLU_idx)
            gen_fc2conv(hparams.speaker_embed_dim, in_channels, torch_state,
                        paddle_state, name_map, torch_base_name,
                        paddle_base_name)
        torch_layer_idx += 1
        Conv1DGLU_idx += 1

    else:
        assert time_upsampling == 1, "other values are not supported"
        torch_base_name = "postnet.convolutions.{}".format(torch_layer_idx)
        paddle_base_name = "{}/Converter_0/Conv1D_{}/Conv2D_0".format(
            prefix, Conv1D_idx)
        gen_conv(in_dim, in_channels, 1, torch_state, paddle_state, name_map,
                 torch_base_name, paddle_base_name)
        torch_layer_idx += 1
        Conv1D_idx += 1

        torch_base_name = "postnet.convolutions.{}.conv".format(torch_layer_idx)
        paddle_base_name = "{}/Converter_0/Conv1DGLU_{}/Conv1D_0/Conv2D_0".format(
            prefix, Conv1DGLU_idx)
        gen_conv(in_channels, 2 * in_channels, 3, torch_state, paddle_state,
                 name_map, torch_base_name, paddle_base_name)
        if hparams.n_speakers > 1:
            torch_base_name = "postnet.convolutions.{}.speaker_proj".format(
                torch_layer_idx)
            paddle_base_name = "{}/Converter_0/Conv1DGLU_{}/Conv1D_1/Conv2D_0".format(
                prefix, Conv1DGLU_idx)
            gen_fc2conv(hparams.speaker_embed_dim, in_channels, torch_state,
                        paddle_state, name_map, torch_base_name,
                        paddle_base_name)
        torch_layer_idx += 1
        Conv1DGLU_idx += 1

    for (out_channels, kernel_size, dilation) in postnet_convolutions:
        if in_channels != out_channels:
            torch_base_name = "postnet.convolutions.{}".format(torch_layer_idx)
            paddle_base_name = "{}/Converter_0/Conv1D_{}/Conv2D_0".format(
                prefix, Conv1D_idx)
            gen_conv(in_channels, out_channels, 1, torch_state, paddle_state,
                     name_map, torch_base_name, paddle_base_name)
            torch_layer_idx += 2
            Conv1D_idx += 1
            in_channels = out_channels

        torch_base_name = "postnet.convolutions.{}.conv".format(torch_layer_idx)
        paddle_base_name = "{}/Converter_0/Conv1DGLU_{}/Conv1D_0/Conv2D_0".format(
            prefix, Conv1DGLU_idx)
        gen_conv(in_channels, 2 * out_channels, kernel_size, torch_state,
                 paddle_state, name_map, torch_base_name, paddle_base_name)

        if hparams.n_speakers > 1:
            torch_base_name = "postnet.convolutions.{}.speaker_proj".format(
                torch_layer_idx)
            paddle_base_name = "{}/Converter_0/Conv1DGLU_{}/Conv1D_1/Conv2D_0".format(
                prefix, Conv1DGLU_idx)
            gen_fc2conv(hparams.speaker_embed_dim, out_channels, torch_state,
                        paddle_state, name_map, torch_base_name,
                        paddle_base_name)
        torch_layer_idx += 1
        Conv1DGLU_idx += 1
        in_channels = out_channels

    # last conv
    linear_dim = hparams.fft_size // 2 + 1
    torch_base_name = "postnet.convolutions.{}".format(torch_layer_idx)
    paddle_base_name = "{}/Converter_0/Conv1D_{}/Conv2D_0".format(prefix,
                                                                  Conv1D_idx)
    gen_conv(in_channels, linear_dim, 1, torch_state, paddle_state, name_map,
             torch_base_name, paddle_base_name)
    torch_layer_idx += 2
    Conv1D_idx += 1
    in_channels = out_channels

    # speaker_embed
    if hparams.n_speakers > 1:
        torch_name = "embed_speakers.weight"
        torch_shape = (hparams.n_speakers, hparams.speaker_embed_dim)
        paddle_name = "{}/Embedding_0.w_0".format(prefix)
        paddle_shape = [hparams.n_speakers, hparams.speaker_embed_dim]

        torch_state[torch_name] = torch_shape
        paddle_state[paddle_name] = paddle_shape
        name_map[torch_name] = paddle_name

    # for k, v in torch_state.items():
    # print("{}\t{}".format(k, v))

    # for k, v in paddle_state.items():
    # print("{}\t{}".format(k, v))

    for k in name_map:
        assert np.prod(torch_state[k]) == np.prod(paddle_state[name_map[
            k]]), "{} does not match".format(k)
        print("{}\t{}\t{}".format(k, name_map[k], paddle_state[name_map[k]]))


if __name__ == "__main__":
    parser = build_arg_parser()
    args, _ = parser.parse_known_args()

    if args.preset is not None:
        with open(args.preset) as f:
            hparams.parse_json(f.read())
    # Override hyper parameters
    hparams.parse(args.hparams)
    # print(hparams_debug_string())

    generate_name_map("dv3")
