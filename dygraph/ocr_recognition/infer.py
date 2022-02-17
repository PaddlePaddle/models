# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
from __future__ import print_function

import os
import numpy as np

import paddle.fluid as fluid
from paddle.fluid.dygraph.base import to_variable

import argparse
import functools
from utility import add_arguments, print_arguments
from PIL import Image
from nets import OCRAttention


parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('image_path',        str,   "",         "image path")
add_arg('pretrained_model',  str,   "",         "pretrained_model.")
add_arg('max_length',        int,   100,     "Max predict length.")
add_arg('encoder_size',      int,   200,     "Encoder size.")
add_arg('decoder_size',      int,   128,     "Decoder size.")
add_arg('word_vector_dim',   int,   128,     "Word vector dim.")
add_arg('num_classes',       int,   95,     "Number classes.")
add_arg('gradient_clip',     float, 5.0,     "Gradient clip value.")


def inference(args):
    img = Image.open(os.path.join(args.image_path)).convert('L')
    with fluid.dygraph.guard():
        ocr_attention = OCRAttention(batch_size=1,
                                     encoder_size=args.encoder_size, decoder_size=args.decoder_size,
                                     num_classes=args.num_classes, word_vector_dim=args.word_vector_dim)
        restore, _ = fluid.load_dygraph(args.pretrained_model)
        ocr_attention.set_dict(restore)
        ocr_attention.eval()
        
        img = img.resize((img.size[0], 48), Image.BILINEAR)
        img = np.array(img).astype('float32') - 127.5
        img = img[np.newaxis, np.newaxis, ...]
        img = to_variable(img)

        gru_backward, encoded_vector, encoded_proj = ocr_attention.encoder_net(img)
        
        backward_first = gru_backward[:, 0]
        decoder_boot = ocr_attention.fc(backward_first)
        label_in = fluid.layers.zeros([1], dtype='int64')
        result = ''
        for i in range(args.max_length):
            trg_embedding = ocr_attention.embedding(label_in)
            trg_embedding = fluid.layers.reshape(
                trg_embedding, [1, -1, trg_embedding.shape[1]],
                inplace=False)

            prediction, decoder_boot = ocr_attention.gru_decoder_with_attention(
                trg_embedding, encoded_vector, encoded_proj, decoder_boot, inference=True)
            prediction = fluid.layers.reshape(prediction, [args.num_classes + 2])
            score, idx = fluid.layers.topk(prediction, 1)

            idx_np = idx.numpy()[0]
            if idx_np == 1:
                print('met end character, predict finish!')
                break

            label_in = fluid.layers.reshape(idx, [1])
            result += chr(int(idx_np + 33))
        print('predict result:', result)


if __name__ == '__main__':
    args = parser.parse_args()
    print_arguments(args)

    inference(args)