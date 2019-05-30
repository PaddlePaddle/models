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

import sys

import numpy as np
import paddle.fluid.profiler as profiler
import paddle.fluid as fluid
import paddle.fluid.layers as layers
import data_reader
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, FC, BatchNorm, Embedding, GRUUnit
from paddle.fluid.dygraph.base import to_variable
import argparse
import functools
from utility import add_arguments, print_arguments, get_attention_feeder_data
import time

from paddle.fluid import framework

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('batch_size',        int,   32,         "Minibatch size.")
add_arg('total_step',        int,   720000,    "The number of iterations. Zero or less means whole training set. More than 0 means the training set might be looped until # of iterations is reached.")
add_arg('log_period',        int,   1000,       "Log period.")
add_arg('save_model_period', int,   15000,      "Save model period. '-1' means never saving the model.")
add_arg('eval_period',       int,   15000,      "Evaluate period. '-1' means never evaluating the model.")
add_arg('save_model_dir',    str,   "./models", "The directory the model to be saved to.")
add_arg('train_images',      str,   None,       "The directory of images to be used for training.")
add_arg('train_list',        str,   None,       "The list file of images to be used for training.")
add_arg('test_images',       str,   None,       "The directory of images to be used for test.")
add_arg('test_list',         str,   None,       "The list file of images to be used for training.")
add_arg('model',    str,   "attention",           "Which type of network to be used. 'crnn_ctc' or 'attention'")
add_arg('init_model',        str,   None,       "The init model file of directory.")
add_arg('use_gpu',           bool,  True,      "Whether use GPU to train.")
add_arg('min_average_window',int,   10000,     "Min average window.")
add_arg('max_average_window',int,   12500,     "Max average window. It is proposed to be set as the number of minibatch in a pass.")
add_arg('average_window',    float, 0.15,      "Average window.")
add_arg('parallel',          bool,  False,     "Whether use parallel training.")
add_arg('profile',           bool,  False,      "Whether to use profiling.")
add_arg('skip_batch_num',    int,   0,          "The number of first minibatches to skip as warm-up for better performance test.")
add_arg('skip_test',         bool,  False,      "Whether to skip test phase.")


class Config(object):
    '''
    config for training
    '''
    # decoder size for decoder stage
    decoder_size = 128
    # size for word embedding
    word_vector_dim = 128
    # max length for label padding
    max_length = 100
    gradient_clip = 10
    LR = 1.0
    beam_size = 2
    learning_rate_decay = None

    # batch size to train
    batch_size = 32
    # class number to classify
    num_classes = 95

    use_gpu = False
    # special label for start and end
    SOS = 0
    EOS = 1
    # settings for ctc data, not use in unittest
    DATA_DIR_NAME = "./dataset/ctc_data/data"
    TRAIN_DATA_DIR_NAME = "train_images"
    TRAIN_LIST_FILE_NAME = "train.list"

    # data shape for input image
    DATA_SHAPE = [1, 48, 512]


class ConvBNPool(fluid.dygraph.Layer):
    def __init__(self,
                 name_scope,
                 group,
                 out_ch,
                 channels,
                 act="relu",
                 is_test=False,
                 pool=True,
                 use_cudnn=True):
        super(ConvBNPool, self).__init__(name_scope)
        self.group = group
        self.pool = pool

        filter_size = 3
        conv_std_0 = (2.0 / (filter_size**2 * channels[0]))**0.5
        conv_param_0 = fluid.ParamAttr(
            initializer=fluid.initializer.Normal(0.0, conv_std_0))

        conv_std_1 = (2.0 / (filter_size**2 * channels[1]))**0.5
        conv_param_1 = fluid.ParamAttr(
            initializer=fluid.initializer.Normal(0.0, conv_std_1))

        self.conv_0_layer = Conv2D(
            self.full_name(),
            channels[0],
            out_ch[0],
            3,
            padding=1,
            param_attr=conv_param_0,
            bias_attr=False,
            act=None,
            use_cudnn=use_cudnn)
        self.bn_0_layer = BatchNorm(
            self.full_name(), out_ch[0], act=act, is_test=is_test)
        self.conv_1_layer = Conv2D(
            self.full_name(),
            num_channels=channels[1],
            num_filters=out_ch[1],
            filter_size=3,
            padding=1,
            param_attr=conv_param_1,
            bias_attr=False,
            act=None,
            use_cudnn=use_cudnn)
        self.bn_1_layer = BatchNorm(
            self.full_name(), out_ch[1], act=act, is_test=is_test)

        print( "pool", self.pool)
        if self.pool:
            self.pool_layer = Pool2D(
                self.full_name(),
                pool_size=2,
                pool_type='max',
                pool_stride=2,
                use_cudnn=use_cudnn,
                ceil_mode=True)

    def forward(self, inputs):
        conv_0 = self.conv_0_layer(inputs)
        bn_0 = self.bn_0_layer(conv_0)
        conv_1 = self.conv_1_layer(bn_0)
        bn_1 = self.bn_1_layer(conv_1)
        if self.pool:
            bn_pool = self.pool_layer(bn_1)

            return bn_pool
        return bn_1


class OCRConv(fluid.dygraph.Layer):
    def __init__(self, name_scope, is_test=False, use_cudnn=True):
        super(OCRConv, self).__init__(name_scope)
        self.conv_bn_pool_1 = ConvBNPool(
            self.full_name(),
            2, [16, 16], [1, 16],
            is_test=is_test,
            use_cudnn=use_cudnn)
        self.conv_bn_pool_2 = ConvBNPool(
            self.full_name(),
            2, [32, 32], [16, 32],
            is_test=is_test,
            use_cudnn=use_cudnn)
        self.conv_bn_pool_3 = ConvBNPool(
            self.full_name(),
            2, [64, 64], [32, 64],
            is_test=is_test,
            use_cudnn=use_cudnn)
        self.conv_bn_pool_4 = ConvBNPool(
            self.full_name(),
            2, [128, 128], [64, 128],
            is_test=is_test,
            pool=False,
            use_cudnn=use_cudnn)

    def forward(self, inputs):
        inputs_1 = self.conv_bn_pool_1(inputs)
        inputs_2 = self.conv_bn_pool_2(inputs_1)
        inputs_3 = self.conv_bn_pool_3(inputs_2)
        inputs_4 = self.conv_bn_pool_4(inputs_3)

        #print( inputs_4.numpy() )
        return inputs_4


class DynamicGRU(fluid.dygraph.Layer):
    def __init__(self,
                 scope_name,
                 size,
                 param_attr=None,
                 bias_attr=None,
                 is_reverse=False,
                 gate_activation='sigmoid',
                 candidate_activation='tanh',
                 h_0=None,
                 origin_mode=False,
                 init_size = None):
        super(DynamicGRU, self).__init__(scope_name)

        self.gru_unit = GRUUnit(
            self.full_name(),
            size * 3,
            param_attr=param_attr,
            bias_attr=bias_attr,
            activation=candidate_activation,
            gate_activation=gate_activation,
            origin_mode=origin_mode)

        self.size = size
        self.h_0 = h_0
        self.is_reverse = is_reverse


    def forward(self, inputs):
        hidden = self.h_0
        res = []


        for i in range(inputs.shape[1]):
            if self.is_reverse:
                i = inputs.shape[1] - 1 - i

            input_ = inputs[ :, i:i+1, :]

            input_ = fluid.layers.reshape(input_, [-1, input_.shape[2]], inplace=False)
            hidden, reset, gate = self.gru_unit(input_, hidden)

            hidden_ = fluid.layers.reshape(hidden, [-1, 1, hidden.shape[1]], inplace=False)

            res.append(hidden_)

        if self.is_reverse:
            res = res[::-1]
        res = fluid.layers.concat(res, axis=1)
        return res


class EncoderNet(fluid.dygraph.Layer):
    def __init__(self,
                 scope_name,
                 rnn_hidden_size=200,
                 is_test=False,
                 use_cudnn=True):
        super(EncoderNet, self).__init__(scope_name)
        self.rnn_hidden_size = rnn_hidden_size
        para_attr = fluid.ParamAttr(initializer=fluid.initializer.Normal(0.0,
                                                                         0.02))
        bias_attr = fluid.ParamAttr(
            initializer=fluid.initializer.Normal(0.0, 0.02), learning_rate=2.0)
        if fluid.framework.in_dygraph_mode():
            h_0 = np.zeros(
                (Config.batch_size, rnn_hidden_size), dtype="float32")
            h_0 = to_variable(h_0)
        else:
            h_0 = fluid.layers.fill_constant(
                shape=[Config.batch_size, rnn_hidden_size],
                dtype='float32',
                value=0)
        self.ocr_convs = OCRConv(
            self.full_name(), is_test=is_test, use_cudnn=use_cudnn)

        self.fc_1_layer = FC(self.full_name(),
                             rnn_hidden_size * 3,
                             param_attr=para_attr,
                             bias_attr=False,
                             num_flatten_dims=2)
        self.fc_2_layer = FC(self.full_name(),
                             rnn_hidden_size * 3,
                             param_attr=para_attr,
                             bias_attr=False,
                             num_flatten_dims=2)
        self.gru_forward_layer = DynamicGRU(
            self.full_name(),
            size=rnn_hidden_size,
            h_0=h_0,
            param_attr=para_attr,
            bias_attr=bias_attr,
            candidate_activation='relu')
        self.gru_backward_layer = DynamicGRU(
            self.full_name(),
            size=rnn_hidden_size,
            h_0=h_0,
            param_attr=para_attr,
            bias_attr=bias_attr,
            candidate_activation='relu',
            is_reverse=True)

        self.encoded_proj_fc = FC(self.full_name(),
                                  Config.decoder_size,
                                  bias_attr=False,
                                  num_flatten_dims=2)

    def forward(self, inputs):
        conv_features = self.ocr_convs(inputs)
        transpose_conv_features = fluid.layers.transpose(conv_features, perm=[0,3,1,2])

        sliced_feature = fluid.layers.reshape(
            transpose_conv_features, [-1, transpose_conv_features.shape[1] , transpose_conv_features.shape[2]*transpose_conv_features.shape[3]], inplace=False)

        fc_1 = self.fc_1_layer(sliced_feature)

        fc_2 = self.fc_2_layer(sliced_feature)

        gru_forward = self.gru_forward_layer(fc_1)

        gru_backward = self.gru_backward_layer(fc_2)

        encoded_vector = fluid.layers.concat(
            input=[gru_forward, gru_backward], axis=2)

        encoded_proj = self.encoded_proj_fc(encoded_vector)

        return gru_backward, encoded_vector, encoded_proj


class SimpleAttention(fluid.dygraph.Layer):
    def __init__(self, scope_name, decoder_size):
        super(SimpleAttention, self).__init__(scope_name)

        self.fc_1 = FC(self.full_name(),
                       decoder_size,
                       act=None,
                       bias_attr=False)
        self.fc_2 = FC(self.full_name(),
                       1,
                       num_flatten_dims = 2,
                       act=None,
                       bias_attr=False)

    def _build_once(self, encoder_vec, encoder_proj, decoder_state):
        pass

    def forward(self, encoder_vec, encoder_proj, decoder_state):

        decoder_state_fc = self.fc_1(decoder_state)

        decoder_state_proj_reshape = fluid.layers.reshape(
            decoder_state_fc, [-1, 1, decoder_state_fc.shape[1]], inplace=False)
        decoder_state_expand = fluid.layers.expand(
            decoder_state_proj_reshape, [1, encoder_proj.shape[1], 1])
        concated = fluid.layers.elementwise_add(encoder_proj,
                                                decoder_state_expand)
        concated = fluid.layers.tanh(x=concated)
        attention_weight = self.fc_2(concated)
        weights_reshape = fluid.layers.reshape(
            x=attention_weight, shape=[ concated.shape[0], -1], inplace=False)

        weights_reshape = fluid.layers.softmax( weights_reshape )
        scaled = fluid.layers.elementwise_mul(
            x=encoder_vec, y=weights_reshape, axis=0)

        context = fluid.layers.reduce_sum(scaled, dim=1)

        return context


class GRUDecoderWithAttention(fluid.dygraph.Layer):
    def __init__(self, scope_name, decoder_size, num_classes):
        super(GRUDecoderWithAttention, self).__init__(scope_name)
        self.simple_attention = SimpleAttention(self.full_name(), decoder_size)

        self.fc_1_layer = FC(self.full_name(),
                             size=decoder_size * 3,
                             bias_attr=False)
        self.fc_2_layer = FC(self.full_name(),
                             size=decoder_size * 3,
                             bias_attr=False)
        self.gru_unit = GRUUnit(
            self.full_name(),
            size=decoder_size * 3,
            param_attr=None,
            bias_attr=None)
        self.out_layer = FC(self.full_name(),
                            size=num_classes + 2,
                            bias_attr=None,
                            act='softmax')

        self.decoder_size = decoder_size

    def _build_once(self, target_embedding, encoder_vec, encoder_proj,
                    decoder_boot):
        pass

    def forward(self, target_embedding, encoder_vec, encoder_proj,
                decoder_boot):
        res = []
        hidden_mem = decoder_boot
        for i in range(target_embedding.shape[1]):
            current_word = fluid.layers.slice(
                target_embedding, axes=[1], starts=[i], ends=[i + 1])
            current_word = fluid.layers.reshape(
                current_word, [-1, current_word.shape[2]], inplace=False)

            context = self.simple_attention(encoder_vec, encoder_proj,
                                            hidden_mem)
            fc_1 = self.fc_1_layer(context)
            fc_2 = self.fc_2_layer(current_word)
            decoder_inputs = fluid.layers.elementwise_add(x=fc_1, y=fc_2)

            h, _, _ = self.gru_unit(decoder_inputs, hidden_mem)
            hidden_mem = h
            out = self.out_layer(h)

            res.append(out)


        res1 = fluid.layers.concat(res, axis=1)

        batch_size = target_embedding.shape[0]
        seq_len = target_embedding.shape[1]
        res1 = layers.reshape( res1, shape=[batch_size, seq_len, -1])

        return res1


class OCRAttention(fluid.dygraph.Layer):
    def __init__(self, scope_name):
        super(OCRAttention, self).__init__(scope_name)
        self.encoder_net = EncoderNet(self.full_name())
        self.fc = FC(self.full_name(),
                     size=Config.decoder_size,
                     bias_attr=False,
                     act='relu')
        self.embedding = Embedding(
            self.full_name(), [Config.num_classes + 2, Config.word_vector_dim],
            dtype='float32')
        self.gru_decoder_with_attention = GRUDecoderWithAttention(
            self.full_name(), Config.decoder_size, Config.num_classes)

    def _build_once(self, inputs, label_in):
        pass

    def forward(self, inputs, label_in):
        gru_backward, encoded_vector, encoded_proj = self.encoder_net(inputs)
        backward_first = fluid.layers.slice(
            gru_backward, axes=[1], starts=[0], ends=[1])
        backward_first = fluid.layers.reshape(
            backward_first, [-1, backward_first.shape[2]], inplace=False)

        decoder_boot = self.fc(backward_first)

        label_in = fluid.layers.reshape(label_in, [-1, 1], inplace=False)
        trg_embedding = self.embedding(label_in)

        trg_embedding = fluid.layers.reshape(
            trg_embedding, [Config.batch_size, -1, trg_embedding.shape[1]],
            inplace=False)

        prediction = self.gru_decoder_with_attention(
            trg_embedding, encoded_vector, encoded_proj, decoder_boot)

        return prediction


def train(args):

    with fluid.dygraph.guard():
        backward_strategy = fluid.dygraph.BackwardStrategy()
        backward_strategy.sort_sum_gradient = True
        ocr_attention = OCRAttention("ocr_attention")

        if Config.learning_rate_decay == "piecewise_decay":
            learning_rate = fluid.layers.piecewise_decay(
                [50000], [Config.LR, Config.LR * 0.01])
        else:
            learning_rate = Config.LR
        optimizer = fluid.optimizer.Adam(learning_rate=0.001)
        dy_param_init_value = {}

        grad_clip = fluid.dygraph_grad_clip.GradClipByGlobalNorm(5.0 )

        train_reader = data_reader.train(
            Config.batch_size,
            max_length=Config.max_length,
            train_images_dir=args.train_images,
            train_list_file=args.train_list,
            cycle=args.total_step > 0,
            shuffle=True,
            model=args.model)

        infer_image= './data/data/test_images/'
        infer_files = './data/data/test.list'
        test_reader = data_reader.train(
                Config.batch_size,
                1000,
                train_images_dir= infer_image,
                train_list_file= infer_files,
                cycle=False,
                model=args.model)
        def eval():
            ocr_attention.eval()
            total_loss = 0.0
            total_step = 0.0
            equal_size = 0
            for data in test_reader():
                data_dict = get_attention_feeder_data(data)

                label_in = to_variable(data_dict["label_in"])
                label_out = to_variable(data_dict["label_out"])

                label_out._stop_gradient = True
                label_out.trainable = False

                img = to_variable(data_dict["pixel"])

                prediction = ocr_attention(img, label_in)
                prediction = fluid.layers.reshape( prediction, [label_out.shape[0] * label_out.shape[1], -1], inplace=False)

                score, topk = layers.topk( prediction, 1)

                seq = topk.numpy()

                seq = seq.reshape( ( args.batch_size, -1))

                mask = data_dict['mask'].reshape( (args.batch_size, -1))
                seq_len = np.sum( mask, -1)

                trans_ref = data_dict["label_out"].reshape( (args.batch_size, -1))
                for i in range( args.batch_size ):
                    length = int(seq_len[i] -1 )
                    trans = seq[i][:length - 1]
                    ref = trans_ref[i][ : length - 1]
                    if np.array_equal( trans, ref ):
                        equal_size += 1

                total_step += args.batch_size
            print( "eval cost", equal_size / total_step )

        total_step = 0
        epoch_num = 20
        for epoch in range(epoch_num):
            batch_id = 0

            total_loss = 0.0
            for data in train_reader():

                total_step += 1
                data_dict = get_attention_feeder_data(data)

                label_in = to_variable(data_dict["label_in"])
                label_out = to_variable(data_dict["label_out"])

                label_out._stop_gradient = True
                label_out.trainable = False

                img = to_variable(data_dict["pixel"])

                prediction = ocr_attention(img, label_in)
                prediction = fluid.layers.reshape( prediction, [label_out.shape[0] * label_out.shape[1], -1], inplace=False)
                label_out = fluid.layers.reshape(label_out, [-1, 1], inplace=False)
                loss = fluid.layers.cross_entropy(
                    input=prediction, label=label_out)

                mask = to_variable(data_dict["mask"])

                loss = layers.elementwise_mul( loss, mask, axis=0)
                avg_loss = fluid.layers.reduce_sum(loss)

                total_loss += avg_loss.numpy()
                avg_loss.backward()
                optimizer.minimize(avg_loss, grad_clip=grad_clip)
                ocr_attention.clear_gradients()

                framework._dygraph_tracer()._clear_ops()

                if batch_id > 0 and batch_id % 1000 == 0:
                    print("epoch: {}, batch_id: {}, loss {}".format(epoch, batch_id, total_loss / args.batch_size / 1000))

                    total_loss = 0.0

                if total_step > 0 and total_step % 2000 == 0:

                    model_value = ocr_attention.state_dict()
                    np.savez( "model/" + str(total_step), **model_value )

                    ocr_attention.eval()
                    eval()
                    ocr_attention.train()

                batch_id +=1






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
