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

import paddle.fluid.profiler as profiler
import paddle.fluid as fluid

import data_reader

from paddle.fluid.dygraph.base import to_variable
import argparse
import functools
from utility import add_arguments, print_arguments, get_attention_feeder_data

from nets import OCRAttention
from eval import evaluate

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('batch_size',        int,   32,         "Minibatch size.")
add_arg('epoch_num',         int,   30,         "Epoch number.")
add_arg('lr',                float, 0.001,         "Learning rate.")
add_arg('lr_decay_strategy', str,   "", "Learning rate decay strategy.")
add_arg('log_period',        int,   200,       "Log period.")
add_arg('save_model_period', int,   2000,      "Save model period. '-1' means never saving the model.")
add_arg('eval_period',       int,   2000,      "Evaluate period. '-1' means never evaluating the model.")
add_arg('save_model_dir',    str,   "./output", "The directory the model to be saved to.")
add_arg('train_images',      str,   None,       "The directory of images to be used for training.")
add_arg('train_list',        str,   None,       "The list file of images to be used for training.")
add_arg('test_images',       str,   None,       "The directory of images to be used for test.")
add_arg('test_list',         str,   None,       "The list file of images to be used for training.")
add_arg('init_model',        str,   None,       "The init model file of directory.")
add_arg('use_gpu',           bool,  True,      "Whether use GPU to train.")
add_arg('parallel',          bool,  False,     "Whether use parallel training.")
add_arg('profile',           bool,  False,      "Whether to use profiling.")
add_arg('skip_batch_num',    int,   0,          "The number of first minibatches to skip as warm-up for better performance test.")
add_arg('skip_test',         bool,  False,      "Whether to skip test phase.")
# model hyper paramters
add_arg('encoder_size',      int,   200,     "Encoder size.")
add_arg('decoder_size',      int,   128,     "Decoder size.")
add_arg('word_vector_dim',   int,   128,     "Word vector dim.")
add_arg('num_classes',       int,   95,     "Number classes.")
add_arg('gradient_clip',     float, 5.0,     "Gradient clip value.")


def train(args):

    with fluid.dygraph.guard():
        backward_strategy = fluid.dygraph.BackwardStrategy()
        backward_strategy.sort_sum_gradient = True

        ocr_attention = OCRAttention(batch_size=args.batch_size,
                                     encoder_size=args.encoder_size, decoder_size=args.decoder_size,
                                     num_classes=args.num_classes, word_vector_dim=args.word_vector_dim)

        LR = args.lr
        if args.lr_decay_strategy == "piecewise_decay":
            learning_rate = fluid.layers.piecewise_decay([200000, 250000], [LR, LR * 0.1, LR * 0.01])
        else:
            learning_rate = LR

        grad_clip = fluid.clip.GradientClipByGlobalNorm(args.gradient_clip)
        optimizer = fluid.optimizer.Adam(
            learning_rate=learning_rate, parameter_list=ocr_attention.parameters(), grad_clip=grad_clip)


        train_reader = data_reader.data_reader(
            args.batch_size,
            shuffle=True,
            images_dir=args.train_images,
            list_file=args.train_list,
            data_type='train')

        test_reader = data_reader.data_reader(
                args.batch_size,
                images_dir=args.test_images,
                list_file=args.test_list,
                data_type="test")

        if not os.path.exists(args.save_model_dir):
            os.makedirs(args.save_model_dir)
        total_step = 0
        epoch_num = args.epoch_num
        for epoch in range(epoch_num):
            batch_id = 0
            total_loss = 0.0

            for data in train_reader():

                total_step += 1
                data_dict = get_attention_feeder_data(data)

                label_in = to_variable(data_dict["label_in"])
                label_out = to_variable(data_dict["label_out"])

                label_out.stop_gradient = True

                img = to_variable(data_dict["pixel"])

                prediction = ocr_attention(img, label_in)
                prediction = fluid.layers.reshape( prediction, [label_out.shape[0] * label_out.shape[1], -1], inplace=False)
                label_out = fluid.layers.reshape(label_out, [-1, 1], inplace=False)
                loss = fluid.layers.cross_entropy(
                    input=prediction, label=label_out)

                mask = to_variable(data_dict["mask"])

                loss = fluid.layers.elementwise_mul( loss, mask, axis=0)
                avg_loss = fluid.layers.reduce_sum(loss)

                total_loss += avg_loss.numpy()
                avg_loss.backward()
                optimizer.minimize(avg_loss)
                ocr_attention.clear_gradients()

                if batch_id > 0 and batch_id % args.log_period == 0:
                    print("epoch: {}, batch_id: {}, lr: {}, loss {}".format(epoch, batch_id,
                                                                        optimizer._global_learning_rate().numpy(),
                                                                        total_loss / args.batch_size / args.log_period))

                    total_loss = 0.0

                if total_step > 0 and total_step % args.save_model_period == 0:
                    if fluid.dygraph.parallel.Env().dev_id == 0:
                        model_file = os.path.join(args.save_model_dir, 'step_{}'.format(total_step))
                        fluid.save_dygraph(ocr_attention.state_dict(), model_file)
                        print('step_{}.pdparams saved!'.format(total_step))
                if total_step > 0 and total_step % args.eval_period == 0:
                    ocr_attention.eval()
                    evaluate(ocr_attention, test_reader, args.batch_size)
                    ocr_attention.train()

                batch_id += 1


if __name__ == '__main__':
    args = parser.parse_args()
    print_arguments(args)
    if args.profile:
        if args.use_gpu:
            with profiler.cuda_profiler("cuda_profiler.txt", 'csv') as nvprof:
                train(args)
        else:
            with profiler.profiler("CPU", sorted_key='total') as cpuprof:
                train(args)
    else:
        train(args)
