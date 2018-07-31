#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import argparse
import time
import distutils.util
import os

import paddle
import paddle.fluid as fluid
import paddle.fluid.layers as layers
import paddle.fluid.core as core
import paddle.fluid.framework as framework
from paddle.fluid.executor import Executor
from paddle.fluid.contrib.decoder.beam_search_decoder import *

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--embedding_dim",
    type=int,
    default=512,
    help="The dimension of embedding table. (default: %(default)d)")
parser.add_argument(
    "--encoder_size",
    type=int,
    default=512,
    help="The size of encoder bi-rnn unit. (default: %(default)d)")
parser.add_argument(
    "--decoder_size",
    type=int,
    default=512,
    help="The size of decoder rnn unit. (default: %(default)d)")
parser.add_argument(
    "--batch_size",
    type=int,
    default=32,
    help="The sequence number of a mini-batch data. (default: %(default)d)")
parser.add_argument(
    "--dict_size",
    type=int,
    default=30000,
    help="The dictionary capacity. Dictionaries of source language and "
    "target language have same capacity. (default: %(default)d)")
parser.add_argument(
    "--pass_num",
    type=int,
    default=1,
    help="The pass number to train. (default: %(default)d)")
parser.add_argument(
    "--learning_rate",
    type=float,
    default=0.01,
    help="Learning rate used to train the model. (default: %(default)f)")
parser.add_argument(
    "--infer_only", action='store_true', help="If set, run forward only.")
parser.add_argument(
    "--beam_size",
    type=int,
    default=3,
    help="The width for beam search. (default: %(default)d)")
parser.add_argument(
    "--use_gpu",
    type=distutils.util.strtobool,
    default=True,
    help="Whether to use gpu. (default: %(default)d)")
parser.add_argument(
    "--max_length",
    type=int,
    default=50,
    help="The maximum length of sequence in generation. "
    "(default: %(default)d)")
parser.add_argument(
    "--save_dir",
    type=str,
    default="model",
    help="Specify the path to save trained models.")
parser.add_argument(
    "--save_interval",
    type=int,
    default=1,
    help="Save the trained model every n passes."
    "(default: %(default)d)")


def lstm_step(x_t, hidden_t_prev, cell_t_prev, size):
    def linear(inputs):
        return fluid.layers.fc(input=inputs, size=size, bias_attr=True)

    forget_gate = fluid.layers.sigmoid(x=linear([hidden_t_prev, x_t]))
    input_gate = fluid.layers.sigmoid(x=linear([hidden_t_prev, x_t]))
    output_gate = fluid.layers.sigmoid(x=linear([hidden_t_prev, x_t]))
    cell_tilde = fluid.layers.tanh(x=linear([hidden_t_prev, x_t]))

    cell_t = fluid.layers.sums(input=[
        fluid.layers.elementwise_mul(
            x=forget_gate, y=cell_t_prev), fluid.layers.elementwise_mul(
                x=input_gate, y=cell_tilde)
    ])

    hidden_t = fluid.layers.elementwise_mul(
        x=output_gate, y=fluid.layers.tanh(x=cell_t))

    return hidden_t, cell_t


def seq_to_seq_net(embedding_dim, encoder_size, decoder_size, source_dict_dim,
                   target_dict_dim, is_generating, beam_size, max_length):
    """Construct a seq2seq network."""

    def bi_lstm_encoder(input_seq, gate_size):
        # A bi-directional lstm encoder implementation.
        # Linear transformation part for input gate, output gate, forget gate
        # and cell activation vectors need be done outside of dynamic_lstm.
        # So the output size is 4 times of gate_size.
        input_forward_proj = fluid.layers.fc(input=input_seq,
                                             size=gate_size * 4,
                                             act='tanh',
                                             bias_attr=False)
        forward, _ = fluid.layers.dynamic_lstm(
            input=input_forward_proj, size=gate_size * 4, use_peepholes=False)
        input_reversed_proj = fluid.layers.fc(input=input_seq,
                                              size=gate_size * 4,
                                              act='tanh',
                                              bias_attr=False)
        reversed, _ = fluid.layers.dynamic_lstm(
            input=input_reversed_proj,
            size=gate_size * 4,
            is_reverse=True,
            use_peepholes=False)
        return forward, reversed

    # The encoding process. Encodes the input words into tensors.
    src_word_idx = fluid.layers.data(
        name='source_sequence', shape=[1], dtype='int64', lod_level=1)

    src_embedding = fluid.layers.embedding(
        input=src_word_idx,
        size=[source_dict_dim, embedding_dim],
        dtype='float32')

    src_forward, src_reversed = bi_lstm_encoder(
        input_seq=src_embedding, gate_size=encoder_size)

    encoded_vector = fluid.layers.concat(
        input=[src_forward, src_reversed], axis=1)

    encoded_proj = fluid.layers.fc(input=encoded_vector,
                                   size=decoder_size,
                                   bias_attr=False)

    backward_first = fluid.layers.sequence_pool(
        input=src_reversed, pool_type='first')

    decoder_boot = fluid.layers.fc(input=backward_first,
                                   size=decoder_size,
                                   bias_attr=False,
                                   act='tanh')

    cell_init = fluid.layers.fill_constant_batch_size_like(
        input=decoder_boot,
        value=0.0,
        shape=[-1, decoder_size],
        dtype='float32')
    cell_init.stop_gradient = False

    # Create a RNN state cell by providing the input and hidden states, and
    # specifies the hidden state as output.
    h = InitState(init=decoder_boot, need_reorder=True)
    c = InitState(init=cell_init)

    state_cell = StateCell(
        inputs={'x': None,
                'encoder_vec': None,
                'encoder_proj': None},
        states={'h': h,
                'c': c},
        out_state='h')

    def simple_attention(encoder_vec, encoder_proj, decoder_state):
        # The implementation of simple attention model
        decoder_state_proj = fluid.layers.fc(input=decoder_state,
                                             size=decoder_size,
                                             bias_attr=False)
        decoder_state_expand = fluid.layers.sequence_expand(
            x=decoder_state_proj, y=encoder_proj)
        # concated lod should inherit from encoder_proj
        concated = fluid.layers.concat(
            input=[encoder_proj, decoder_state_expand], axis=1)
        attention_weights = fluid.layers.fc(input=concated,
                                            size=1,
                                            bias_attr=False)
        attention_weights = fluid.layers.sequence_softmax(
            input=attention_weights)
        weigths_reshape = fluid.layers.reshape(x=attention_weights, shape=[-1])
        scaled = fluid.layers.elementwise_mul(
            x=encoder_vec, y=weigths_reshape, axis=0)
        context = fluid.layers.sequence_pool(input=scaled, pool_type='sum')
        return context

    @state_cell.state_updater
    def state_updater(state_cell):
        # Define the updater of RNN state cell
        current_word = state_cell.get_input('x')
        encoder_vec = state_cell.get_input('encoder_vec')
        encoder_proj = state_cell.get_input('encoder_proj')
        prev_h = state_cell.get_state('h')
        prev_c = state_cell.get_state('c')
        context = simple_attention(encoder_vec, encoder_proj, prev_h)
        decoder_inputs = fluid.layers.concat(
            input=[context, current_word], axis=1)
        h, c = lstm_step(decoder_inputs, prev_h, prev_c, decoder_size)
        state_cell.set_state('h', h)
        state_cell.set_state('c', c)

    # Define the decoding process
    if not is_generating:
        # Training process
        trg_word_idx = fluid.layers.data(
            name='target_sequence', shape=[1], dtype='int64', lod_level=1)

        trg_embedding = fluid.layers.embedding(
            input=trg_word_idx,
            size=[target_dict_dim, embedding_dim],
            dtype='float32')

        # A decoder for training
        decoder = TrainingDecoder(state_cell)

        with decoder.block():
            current_word = decoder.step_input(trg_embedding)
            encoder_vec = decoder.static_input(encoded_vector)
            encoder_proj = decoder.static_input(encoded_proj)
            decoder.state_cell.compute_state(inputs={
                'x': current_word,
                'encoder_vec': encoder_vec,
                'encoder_proj': encoder_proj
            })
            h = decoder.state_cell.get_state('h')
            decoder.state_cell.update_states()
            out = fluid.layers.fc(input=h,
                                  size=target_dict_dim,
                                  bias_attr=True,
                                  act='softmax')
            decoder.output(out)

        label = fluid.layers.data(
            name='label_sequence', shape=[1], dtype='int64', lod_level=1)
        cost = fluid.layers.cross_entropy(input=decoder(), label=label)
        avg_cost = fluid.layers.mean(x=cost)
        feeding_list = ["source_sequence", "target_sequence", "label_sequence"]
        return avg_cost, feeding_list

    else:
        # Inference
        init_ids = fluid.layers.data(
            name="init_ids", shape=[1], dtype="int64", lod_level=2)
        init_scores = fluid.layers.data(
            name="init_scores", shape=[1], dtype="float32", lod_level=2)

        # A beam search decoder
        decoder = BeamSearchDecoder(
            state_cell=state_cell,
            init_ids=init_ids,
            init_scores=init_scores,
            target_dict_dim=target_dict_dim,
            word_dim=embedding_dim,
            input_var_dict={
                'encoder_vec': encoded_vector,
                'encoder_proj': encoded_proj
            },
            topk_size=50,
            sparse_emb=True,
            max_len=max_length,
            beam_size=beam_size,
            end_id=1,
            name=None)

        decoder.decode()

        translation_ids, translation_scores = decoder()
        feeding_list = ["source_sequence", "init_ids", "init_scores"]

        return translation_ids, translation_scores, feeding_list


def train():
    # Training process
    avg_cost, feed_order = seq_to_seq_net(
        args.embedding_dim,
        args.encoder_size,
        args.decoder_size,
        args.dict_size,
        args.dict_size,
        False,
        beam_size=args.beam_size,
        max_length=args.max_length)

    # clone from default main program and use it as the validation program
    main_program = fluid.default_main_program()
    inference_program = fluid.default_main_program().clone()

    optimizer = fluid.optimizer.Adam(
        learning_rate=args.learning_rate,
        regularization=fluid.regularizer.L2DecayRegularizer(
            regularization_coeff=1e-5))

    optimizer.minimize(avg_cost)

    train_batch_generator = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.wmt14.train(args.dict_size), buf_size=1000),
        batch_size=args.batch_size,
        drop_last=False)

    test_batch_generator = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.wmt14.test(args.dict_size), buf_size=1000),
        batch_size=args.batch_size,
        drop_last=False)

    place = core.CUDAPlace(0) if args.use_gpu else core.CPUPlace()
    exe = Executor(place)
    exe.run(framework.default_startup_program())

    feed_list = [
        main_program.global_block().var(var_name) for var_name in feed_order
    ]
    feeder = fluid.DataFeeder(feed_list, place)

    def validation():
        # Use test set as validation each pass
        total_loss = 0.0
        count = 0
        val_feed_list = [
            inference_program.global_block().var(var_name)
            for var_name in feed_order
        ]
        val_feeder = fluid.DataFeeder(val_feed_list, place)

        for batch_id, data in enumerate(test_batch_generator()):
            val_fetch_outs = exe.run(inference_program,
                                     feed=val_feeder.feed(data),
                                     fetch_list=[avg_cost],
                                     return_numpy=False)

            total_loss += np.array(val_fetch_outs[0])[0]
            count += 1

        return total_loss / count

    for pass_id in range(1, args.pass_num + 1):
        pass_start_time = time.time()
        words_seen = 0
        for batch_id, data in enumerate(train_batch_generator()):
            words_seen += len(data) * 2

            fetch_outs = exe.run(framework.default_main_program(),
                                 feed=feeder.feed(data),
                                 fetch_list=[avg_cost])

            avg_cost_train = np.array(fetch_outs[0])
            print('pass_id=%d, batch_id=%d, train_loss: %f' %
                  (pass_id, batch_id, avg_cost_train))

        pass_end_time = time.time()
        test_loss = validation()
        time_consumed = pass_end_time - pass_start_time
        words_per_sec = words_seen / time_consumed
        print("pass_id=%d, test_loss: %f, words/s: %f, sec/pass: %f" %
              (pass_id, test_loss, words_per_sec, time_consumed))

        if pass_id % args.save_interval == 0:
            model_path = os.path.join(args.save_dir, str(pass_id))
            if not os.path.isdir(model_path):
                os.makedirs(model_path)

            fluid.io.save_persistables(
                executor=exe,
                dirname=model_path,
                main_program=framework.default_main_program())


def infer():
    # Inference
    translation_ids, translation_scores, feed_order = seq_to_seq_net(
        args.embedding_dim,
        args.encoder_size,
        args.decoder_size,
        args.dict_size,
        args.dict_size,
        True,
        beam_size=args.beam_size,
        max_length=args.max_length)

    test_batch_generator = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.wmt14.test(args.dict_size), buf_size=1000),
        batch_size=args.batch_size,
        drop_last=False)

    place = core.CUDAPlace(0) if args.use_gpu else core.CPUPlace()
    exe = Executor(place)
    exe.run(framework.default_startup_program())

    model_path = os.path.join(args.save_dir, str(args.pass_num))
    fluid.io.load_persistables(
        executor=exe,
        dirname=model_path,
        main_program=framework.default_main_program())

    src_dict, trg_dict = paddle.dataset.wmt14.get_dict(args.dict_size)

    feed_list = [
        framework.default_main_program().global_block().var(var_name)
        for var_name in feed_order[0:1]
    ]
    feeder = fluid.DataFeeder(feed_list, place)

    for batch_id, data in enumerate(test_batch_generator()):
        # The value of batch_size may vary in the last batch
        batch_size = len(data)

        # Setup initial ids and scores lod tensor
        init_ids_data = np.array([0 for _ in range(batch_size)], dtype='int64')
        init_scores_data = np.array(
            [1. for _ in range(batch_size)], dtype='float32')
        init_ids_data = init_ids_data.reshape((batch_size, 1))
        init_scores_data = init_scores_data.reshape((batch_size, 1))
        init_recursive_seq_lens = [1] * batch_size
        init_recursive_seq_lens = [
            init_recursive_seq_lens, init_recursive_seq_lens
        ]
        init_ids = fluid.create_lod_tensor(init_ids_data,
                                           init_recursive_seq_lens, place)
        init_scores = fluid.create_lod_tensor(init_scores_data,
                                              init_recursive_seq_lens, place)

        # Feed dict for inference
        feed_dict = feeder.feed(map(lambda x: [x[0]], data))
        feed_dict[feed_order[1]] = init_ids
        feed_dict[feed_order[2]] = init_scores

        fetch_outs = exe.run(framework.default_main_program(),
                             feed=feed_dict,
                             fetch_list=[translation_ids, translation_scores],
                             return_numpy=False)

        # Split the output words by lod levels
        lod_level_1 = fetch_outs[0].lod()[1]
        token_array = np.array(fetch_outs[0])
        result = []
        for i in xrange(len(lod_level_1) - 1):
            sentence_list = [
                trg_dict[token]
                for token in token_array[lod_level_1[i]:lod_level_1[i + 1]]
            ]
            sentence = " ".join(sentence_list[1:-1])
            result.append(sentence)
        lod_level_0 = fetch_outs[0].lod()[0]
        paragraphs = [
            result[lod_level_0[i]:lod_level_0[i + 1]]
            for i in xrange(len(lod_level_0) - 1)
        ]

        for paragraph in paragraphs:
            print(paragraph)


if __name__ == '__main__':
    args = parser.parse_args()
    if args.infer_only:
        infer()
    else:
        train()
