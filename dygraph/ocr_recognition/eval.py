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

import argparse
import functools
import numpy as np
import paddle.fluid as fluid
import paddle.fluid.layers as layers
import data_reader
from nets import OCRAttention
from paddle.fluid.dygraph.base import to_variable
from utility import add_arguments, print_arguments, get_attention_feeder_data

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('batch_size',        int,   32,         "Minibatch size.")
add_arg('pretrained_model',  str,   "",         "pretrained_model.")
add_arg('test_images',       str,   None,       "The directory of images to be used for test.")
add_arg('test_list',         str,   None,       "The list file of images to be used for training.")
# model hyper paramters
add_arg('encoder_size',      int,   200,     "Encoder size.")
add_arg('decoder_size',      int,   128,     "Decoder size.")
add_arg('word_vector_dim',   int,   128,     "Word vector dim.")
add_arg('num_classes',       int,   95,     "Number classes.")
add_arg('gradient_clip',     float, 5.0,     "Gradient clip value.")


def evaluate(model, test_reader, batch_size):
    model.eval()

    total_step = 0.0
    equal_size = 0
    for data in test_reader():
        data_dict = get_attention_feeder_data(data)

        label_in = to_variable(data_dict["label_in"])
        label_out = to_variable(data_dict["label_out"])

        label_out.stop_gradient = True

        img = to_variable(data_dict["pixel"])

        prediction = model(img, label_in)
        prediction = fluid.layers.reshape(prediction, [label_out.shape[0] * label_out.shape[1], -1], inplace=False)

        score, topk = layers.topk(prediction, 1)

        seq = topk.numpy()

        seq = seq.reshape((batch_size, -1))

        mask = data_dict['mask'].reshape((batch_size, -1))
        seq_len = np.sum(mask, -1)

        trans_ref = data_dict["label_out"].reshape((batch_size, -1))
        for i in range(batch_size):
            length = int(seq_len[i] - 1)
            trans = seq[i][:length - 1]
            ref = trans_ref[i][: length - 1]
            if np.array_equal(trans, ref):
                equal_size += 1

        total_step += batch_size
    accuracy = equal_size / total_step
    print("eval accuracy:", accuracy)
    return accuracy


def eval(args):
    with fluid.dygraph.guard():
        ocr_attention = OCRAttention(batch_size=args.batch_size,
                                     encoder_size=args.encoder_size, decoder_size=args.decoder_size,
                                     num_classes=args.num_classes, word_vector_dim=args.word_vector_dim)
        restore, _ = fluid.load_dygraph(args.pretrained_model)
        ocr_attention.set_dict(restore)

        test_reader = data_reader.data_reader(
            args.batch_size,
            images_dir=args.test_images,
            list_file=args.test_list,
            data_type="test")
        evaluate(ocr_attention, test_reader, args.batch_size)

if __name__ == '__main__':
    args = parser.parse_args()
    print_arguments(args)

    eval(args)