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

from deepvoice3_paddle.deepvoice3 import DeepVoiceTTS, ConvSpec, WindowRange


def deepvoice3(n_vocab,
               embed_dim=256,
               mel_dim=80,
               linear_dim=513,
               r=4,
               downsample_step=1,
               n_speakers=1,
               speaker_dim=16,
               padding_idx=0,
               dropout=(1 - 0.96),
               filter_size=5,
               encoder_channels=128,
               decoder_channels=256,
               converter_channels=256,
               query_position_rate=1.0,
               key_position_rate=1.29,
               use_memory_mask=False,
               trainable_positional_encodings=False,
               force_monotonic_attention=True,
               use_decoder_state_for_postnet_input=True,
               max_positions=512,
               embedding_weight_std=0.1,
               speaker_embedding_weight_std=0.01,
               freeze_embedding=False,
               window_range=WindowRange(-1, 3),
               key_projection=False,
               value_projection=False):
    time_upsampling = max(downsample_step, 1)

    h = encoder_channels
    k = filter_size
    encoder_convolutions = (ConvSpec(h, k, 1), ConvSpec(h, k, 3),
                            ConvSpec(h, k, 9), ConvSpec(h, k, 27),
                            ConvSpec(h, k, 1), ConvSpec(h, k, 3),
                            ConvSpec(h, k, 9), ConvSpec(h, k, 27),
                            ConvSpec(h, k, 1), ConvSpec(h, k, 3))

    h = decoder_channels
    prenet_convolutions = (ConvSpec(h, k, 1), ConvSpec(h, k, 3))
    attentive_convolutions = (ConvSpec(h, k, 1), ConvSpec(h, k, 3),
                              ConvSpec(h, k, 9), ConvSpec(h, k, 27),
                              ConvSpec(h, k, 1))
    attention = [True, False, False, False, True]

    h = converter_channels
    postnet_convolutions = (ConvSpec(h, k, 1), ConvSpec(h, k, 3),
                            ConvSpec(2 * h, k, 1), ConvSpec(2 * h, k, 3))

    model = DeepVoiceTTS(
        "dv3", n_speakers, speaker_dim, speaker_embedding_weight_std, n_vocab,
        embed_dim, padding_idx, embedding_weight_std, freeze_embedding,
        encoder_convolutions, max_positions, padding_idx,
        trainable_positional_encodings, mel_dim, r, prenet_convolutions,
        attentive_convolutions, attention, use_memory_mask,
        force_monotonic_attention, query_position_rate, key_position_rate,
        window_range, key_projection, value_projection, linear_dim,
        postnet_convolutions, time_upsampling, dropout,
        use_decoder_state_for_postnet_input, "float32")
    return model


def deepvoice3_multispeaker(n_vocab,
                            embed_dim=256,
                            mel_dim=80,
                            linear_dim=513,
                            r=4,
                            downsample_step=1,
                            n_speakers=1,
                            speaker_dim=16,
                            padding_idx=0,
                            dropout=(1 - 0.96),
                            filter_size=5,
                            encoder_channels=128,
                            decoder_channels=256,
                            converter_channels=256,
                            query_position_rate=1.0,
                            key_position_rate=1.29,
                            use_memory_mask=False,
                            trainable_positional_encodings=False,
                            force_monotonic_attention=True,
                            use_decoder_state_for_postnet_input=True,
                            max_positions=512,
                            embedding_weight_std=0.1,
                            speaker_embedding_weight_std=0.01,
                            freeze_embedding=False,
                            window_range=WindowRange(-1, 3),
                            key_projection=False,
                            value_projection=False):
    time_upsampling = max(downsample_step, 1)

    h = encoder_channels
    k = filter_size
    encoder_convolutions = (ConvSpec(h, k, 1), ConvSpec(h, k, 3),
                            ConvSpec(h, k, 9), ConvSpec(h, k, 27),
                            ConvSpec(h, k, 1), ConvSpec(h, k, 3),
                            ConvSpec(h, k, 9), ConvSpec(h, k, 27),
                            ConvSpec(h, k, 1), ConvSpec(h, k, 3))

    h = decoder_channels
    prenet_convolutions = (ConvSpec(h, k, 1))
    attentive_convolutions = (ConvSpec(h, k, 1), ConvSpec(h, k, 3),
                              ConvSpec(h, k, 9), ConvSpec(h, k, 27),
                              ConvSpec(h, k, 1))
    attention = [True, False, False, False, False]

    h = converter_channels
    postnet_convolutions = (ConvSpec(h, k, 1), ConvSpec(h, k, 3),
                            ConvSpec(2 * h, k, 1), ConvSpec(2 * h, k, 3))

    model = DeepVoiceTTS(
        "dv3", n_speakers, speaker_dim, speaker_embedding_weight_std, n_vocab,
        embed_dim, padding_idx, embedding_weight_std, freeze_embedding,
        encoder_convolutions, max_positions, padding_idx,
        trainable_positional_encodings, mel_dim, r, prenet_convolutions,
        attentive_convolutions, attention, use_memory_mask,
        force_monotonic_attention, query_position_rate, key_position_rate,
        window_range, key_projection, value_projection, linear_dim,
        postnet_convolutions, time_upsampling, dropout,
        use_decoder_state_for_postnet_input, "float32")
    return model
