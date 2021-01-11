# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import numpy as np
from paddle.nn import Linear, Dropout, LayerNorm, LayerList, Layer
from .encoder import TransformerEncoderLayer, TransformerEncoder
from .decoder import TransformerDecoderLayer, TransformerDecoder


class Transformer(Layer):
    def __init__(self,
                 d_model=512,
                 nhead=8,
                 num_encoder_layers=6,
                 num_decoder_layers=6,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation="relu",
                 attn_dropout=None,
                 act_dropout=None,
                 normalize_before=False,
                 weight_attr=None,
                 bias_attr=None,
                 custom_encoder=None,
                 custom_decoder=None,
                 attention_type="default_attention"):
        super(Transformer, self).__init__()
        if isinstance(bias_attr, (list, tuple)):
            if len(bias_attr) == 1:
                encoder_bias_attr = [bias_attr[0]] * 2
                decoder_bias_attr = [bias_attr[0]] * 3
            elif len(bias_attr) == 2:
                encoder_bias_attr = bias_attr
                decoder_bias_attr = [bias_attr[0], bias_attr[0], bias_attr[-1]]
            elif len(bias_attr) == 3:
                encoder_bias_attr = [bias_attr[0], bias_attr[-1]]
                decoder_bias_attr = bias_attr
            else:
                assert False, (
                    "length of bias_attr should be 1 or 2 or 3 when it is a list/tuple"
                )
        else:
            encoder_bias_attr = bias_attr
            decoder_bias_attr = bias_attr

        if isinstance(weight_attr, (list, tuple)):
            if len(weight_attr) == 1:
                encoder_weight_attr = [weight_attr[0]] * 2
                decoder_weight_attr = [weight_attr[0]] * 3
            elif len(weight_attr) == 2:
                encoder_weight_attr = weight_attr
                decoder_weight_attr = [
                    weight_attr[0], weight_attr[0], weight_attr[-1]
                ]
            elif len(weight_attr) == 3:
                encoder_weight_attr = [weight_attr[0], weight_attr[-1]]
                decoder_weight_attr = weight_attr
            else:
                assert False, (
                    "length of weight_attr should be 1 or 2 or 3 when it is a list/tuple"
                )
        else:
            encoder_weight_attr = weight_attr
            decoder_weight_attr = weight_attr

        if custom_encoder is not None:
            self.encoder = custom_encoder
        else:
            encoder_layer = TransformerEncoderLayer(
                d_model,
                nhead,
                dim_feedforward,
                dropout,
                activation,
                attn_dropout,
                act_dropout,
                normalize_before,
                encoder_weight_attr,
                encoder_bias_attr,
                attention_type=attention_type)
            encoder_norm = LayerNorm(d_model)
            self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers,
                                              encoder_norm)

        if custom_decoder is not None:
            self.decoder = custom_decoder
        else:
            decoder_layer = TransformerDecoderLayer(
                d_model,
                nhead,
                dim_feedforward,
                dropout,
                activation,
                attn_dropout,
                act_dropout,
                normalize_before,
                decoder_weight_attr,
                decoder_bias_attr,
                attention_type=attention_type)
            decoder_norm = LayerNorm(d_model)
            self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers,
                                              decoder_norm)

        self.d_model = d_model
        self.nhead = nhead

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        memory = self.encoder(src, src_mask=src_mask)
        output = self.decoder(
            tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)
        return output

    def generate_square_subsequent_mask(self, length):
        return paddle.tensor.triu(
            (paddle.ones(
                (length, length), dtype=paddle.get_default_dtype()) * -np.inf),
            1)
