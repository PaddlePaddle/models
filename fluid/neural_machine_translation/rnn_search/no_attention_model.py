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

import numpy as np
import argparse
import distutils.util
import os

import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
import paddle.fluid.framework as framework
import paddle.fluid.layers as layers
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
    help="The dictionary capacity. Dictionaries of source sequence and "
    "target dictionary have same capacity. (default: %(default)d)")
parser.add_argument(
    "--pass_num",
    type=int,
    default=5,
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
    help="The width for beam searching. (default: %(default)d)")
parser.add_argument(
    "--use_gpu",
    type=distutils.util.strtobool,
    default=True,
    help="Whether to use gpu. (default: %(default)d)")
parser.add_argument(
    "--max_length",
    type=int,
    default=50,
    help="The maximum length of sequence when doing generation. "
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

args = parser.parse_args()
IS_SPARSE = True
topk_size = 50
source_dict_dim = target_dict_dim = args.dict_size
place = core.CUDAPlace(0) if args.use_gpu else core.CPUPlace()


def seq_to_seq_net(is_generating):
    def encoder():
        # Encoder implementation of RNN translation
        src_word = layers.data(
            name="src_word", shape=[1], dtype='int64', lod_level=1)
        src_embedding = layers.embedding(
            input=src_word,
            size=[args.dict_size, args.embedding_dim],
            dtype='float32',
            is_sparse=IS_SPARSE)

        fc1 = layers.fc(
            input=src_embedding, size=args.encoder_size * 4, act='tanh')
        lstm_hidden0, lstm_0 = layers.dynamic_lstm(
            input=fc1, size=args.encoder_size * 4)
        encoder_out = layers.sequence_last_step(input=lstm_hidden0)
        return encoder_out

    def decoder_state_cell(context):
        # Decoder state cell, specifies the hidden state variable and its updater
        h = InitState(init=context, need_reorder=True)
        state_cell = StateCell(
            inputs={'x': None}, states={'h': h}, out_state='h')

        @state_cell.state_updater
        def updater(state_cell):
            current_word = state_cell.get_input('x')
            prev_h = state_cell.get_state('h')
            # make sure lod of h heritted from prev_h
            h = layers.fc(input=[prev_h, current_word],
                          size=args.decoder_size,
                          act='tanh')
            state_cell.set_state('h', h)

        return state_cell

    def decoder_train(state_cell):
        # Decoder for training implementation of RNN translation
        trg_word = layers.data(
            name="target_word", shape=[1], dtype='int64', lod_level=1)
        trg_embedding = layers.embedding(
            input=trg_word,
            size=[args.dict_size, args.embedding_dim],
            dtype='float32',
            is_sparse=IS_SPARSE)

        # A training decoder
        decoder = TrainingDecoder(state_cell)

        # Define the computation in each RNN step done by decoder
        with decoder.block():
            current_word = decoder.step_input(trg_embedding)
            decoder.state_cell.compute_state(inputs={'x': current_word})
            current_score = layers.fc(input=decoder.state_cell.get_state('h'),
                                      size=args.dict_size,
                                      act='softmax')
            decoder.state_cell.update_states()
            decoder.output(current_score)

        return decoder()

    def decoder_infer(state_cell):
        # Decoder for inference implementation
        init_ids = layers.data(
            name="init_ids", shape=[1], dtype="int64", lod_level=2)
        init_scores = layers.data(
            name="init_scores", shape=[1], dtype="float32", lod_level=2)

        # A beam search decoder for inference
        decoder = BeamSearchDecoder(
            state_cell=state_cell,
            init_ids=init_ids,
            init_scores=init_scores,
            target_dict_dim=args.dict_size,
            word_dim=args.embedding_dim,
            input_var_dict={},
            topk_size=topk_size,
            sparse_emb=IS_SPARSE,
            max_len=args.max_length,
            beam_size=args.beam_size,
            end_id=1,
            name=None)
        decoder.decode()
        translation_ids, translation_scores = decoder()

        return translation_ids, translation_scores

    context = encoder()
    state_cell = decoder_state_cell(context)

    if not is_generating:
        rnn_out = decoder_train(state_cell)
        feed_order = ['src_word', 'target_word', 'target_next_word']
        return rnn_out, feed_order
    else:
        translation_ids, translation_scores = decoder_infer(state_cell)
        feed_order = ['src_word']
        return translation_ids, translation_scores, feed_order


def train_main():
    # To train the model from beginning
    rnn_out, feed_order = seq_to_seq_net(is_generating=False)
    label = layers.data(
        name="target_next_word", shape=[1], dtype='int64', lod_level=1)
    cost = layers.cross_entropy(input=rnn_out, label=label)
    avg_cost = layers.mean(x=cost)

    optimizer = fluid.optimizer.Adam(
        learning_rate=args.learning_rate,
        regularization=fluid.regularizer.L2DecayRegularizer(
            regularization_coeff=1e-5))

    optimizer.minimize(avg_cost)

    train_reader = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.wmt14.train(args.dict_size), buf_size=1000),
        batch_size=args.batch_size,
        drop_last=False)

    exe = Executor(place)
    exe.run(framework.default_startup_program())
    program = framework.default_main_program()

    feed_list = [
        program.global_block().var(var_name) for var_name in feed_order
    ]
    feeder = fluid.DataFeeder(feed_list, place)

    for pass_id in range(1, args.pass_num + 1):
        for batch_id, data in enumerate(train_reader()):
            outs = exe.run(program,
                           feed=feeder.feed(data),
                           fetch_list=[avg_cost])
            avg_cost_val = np.array(outs[0])
            if batch_id % 100 == 0:
                print("pass_id=" + str(pass_id) + " batch=" + str(batch_id) +
                      " avg_cost=" + str(avg_cost_val))

        if pass_id % args.save_interval == 0:
            model_path = os.path.join(args.save_dir, str(pass_id))
            if not os.path.isdir(model_path):
                os.makedirs(model_path)

            fluid.io.save_persistables(
                executor=exe,
                dirname=model_path,
                main_program=framework.default_main_program())


def infer_main():
    # Load a trained model and make inference
    translation_ids, translation_scores, feed_order = \
        seq_to_seq_net(is_generating=True)

    exe = Executor(place)
    exe.run(framework.default_startup_program())

    model_path = os.path.join(args.save_dir, str(args.pass_num))
    fluid.io.load_persistables(
        executor=exe,
        dirname=model_path,
        main_program=framework.default_main_program())

    # Set up the inital ids and initial scores lod tensor for beam search
    init_ids_data = np.array([0 for _ in range(args.batch_size)], dtype='int64')
    init_scores_data = np.array(
        [1. for _ in range(args.batch_size)], dtype='float32')
    init_ids_data = init_ids_data.reshape((args.batch_size, 1))
    init_scores_data = init_scores_data.reshape((args.batch_size, 1))
    init_recursive_seq_lens = [1] * args.batch_size
    init_recursive_seq_lens = [init_recursive_seq_lens, init_recursive_seq_lens]
    init_ids = fluid.create_lod_tensor(init_ids_data, init_recursive_seq_lens,
                                       place)
    init_scores = fluid.create_lod_tensor(init_scores_data,
                                          init_recursive_seq_lens, place)

    # In inference process only source word is used as input.
    feed_list = [
        framework.default_main_program().global_block().var(var_name)
        for var_name in feed_order
    ]
    feeder = fluid.DataFeeder(feed_list, place)

    test_reader = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.wmt14.test(args.dict_size), buf_size=1000),
        batch_size=args.batch_size)

    src_dict, trg_dict = paddle.dataset.wmt14.get_dict(args.dict_size)

    for _, data in enumerate(test_reader()):
        # Create a feed dict to feed necessary input variables to the program
        feed_dict = feeder.feed(map(lambda x: [x[0]], data))
        feed_dict['init_ids'] = init_ids
        feed_dict['init_scores'] = init_scores

        # Run the program and fetch its translation result
        result_ids, result_scores = exe.run(
            framework.default_main_program(),
            feed=feed_dict,
            fetch_list=[translation_ids, translation_scores],
            return_numpy=False)

        # Split the sentences by lods, and print the translated sentences
        lod_level_1 = result_ids.lod()[1]
        token_array = np.array(result_ids)
        result = []
        for i in xrange(len(lod_level_1) - 1):
            sentence_list = [
                trg_dict[token]
                for token in token_array[lod_level_1[i]:lod_level_1[i + 1]]
            ]
            sentence = " ".join(sentence_list)
            result.append(sentence)
        lod_level_0 = result_ids.lod()[0]
        paragraphs = [
            result[lod_level_0[i]:lod_level_0[i + 1]]
            for i in xrange(len(lod_level_0) - 1)
        ]
        print "Translation result:"
        for paragraph in paragraphs:
            print paragraph


if __name__ == '__main__':
    if args.infer_only:
        infer_main()
    else:
        train_main()
