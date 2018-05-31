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
import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
import paddle.fluid.framework as framework
import paddle.fluid.layers as pd
from paddle.fluid.executor import Executor
from beam_search_api import *
import os

dict_size = 30000
source_dict_dim = target_dict_dim = dict_size
src_dict, trg_dict = paddle.dataset.wmt14.get_dict(dict_size)
hidden_dim = 1024
word_dim = 1024
IS_SPARSE = True
batch_size = 32
max_length = 100
topk_size = 1
trg_dic_size = 10000
beam_size = 1

decoder_size = hidden_dim

place = core.CUDAPlace(0)

model_save_dir = 'model_wmt14'


def encoder():
    # encoder
    src_word_id = pd.data(
        name="src_word_id", shape=[1], dtype='int64', lod_level=1)
    src_embedding = pd.embedding(
        input=src_word_id,
        size=[dict_size, word_dim],
        dtype='float32',
        is_sparse=IS_SPARSE,
        param_attr=fluid.ParamAttr(name='vemb'))

    fc1 = pd.fc(input=src_embedding, size=hidden_dim * 4, act='tanh')
    lstm_hidden0, lstm_0 = pd.dynamic_lstm(input=fc1, size=hidden_dim * 4)
    encoder_out = pd.sequence_last_step(input=lstm_hidden0)
#    pd.Print(encoder_out, message="encoder_out",
#             print_tensor_name=False,
#             print_tensor_shape=False, print_tensor_lod=False)
    return encoder_out


def decoder_state_cell(context):
    h = InitState(init=context, need_reorder=True)
    state_cell = StateCell(
        cell_size=decoder_size, inputs={'x': None}, states={'h': h})

    @state_cell.state_updater
    def updater(state_cell):
        current_word = state_cell.get_input('x')
        prev_h = state_cell.get_state('h')
        # make sure lod of h heritted from prev_h
        h = pd.fc(input=[prev_h, current_word], size=decoder_size, act='tanh')
        state_cell.set_state('h', h)

    return state_cell


def decoder_train(state_cell):
    # decoder
    trg_language_word = pd.data(
        name="target_language_word", shape=[1], dtype='int64', lod_level=1)
    trg_embedding = pd.embedding(
        input=trg_language_word,
        size=[dict_size, word_dim],
        dtype='float32',
        is_sparse=IS_SPARSE,
        param_attr=fluid.ParamAttr(name='vemb'))

    decoder = TrainingDecoder(state_cell)

    with decoder.block():
        current_word = decoder.step_input(trg_embedding)
        decoder.state_cell.compute_state(inputs={'x': current_word})
        current_score = pd.fc(input=decoder.state_cell.get_state('h'),
                              size=target_dict_dim,
                              act='softmax')
        decoder.state_cell.update_states()
        decoder.output(current_score)

    return decoder()


def decoder_decode(state_cell):
    init_ids = pd.data(name="init_ids", shape=[1], dtype="int64", lod_level=2)
    init_scores = pd.data(
        name="init_scores", shape=[1], dtype="float32", lod_level=2)

    def embedding(input):
        return pd.embedding(
            input=input,
            size=[dict_size, word_dim],
            dtype='float32',
            is_sparse=IS_SPARSE,
            param_attr=fluid.ParamAttr('vemb'))

    decoder = BeamSearchDecoder(state_cell, max_len=max_length)

    with decoder.block():
        prev_ids = decoder.read_array(init=init_ids, is_ids=True)
        prev_scores = decoder.read_array(init=init_scores, is_scores=True)
        prev_ids_embedding = embedding(prev_ids)
        prev_state = decoder.state_cell.get_state('h')
        prev_state_expanded = pd.sequence_expand(prev_state, prev_scores)
        decoder.state_cell.set_state('h', prev_state_expanded)
        decoder.state_cell.compute_state(inputs={'x': prev_ids_embedding})
        current_state = decoder.state_cell.get_state('h')
        current_state_with_lod = pd.lod_reset(x=current_state, y=prev_scores)
        scores = pd.fc(input=current_state_with_lod,
                       size=target_dict_dim,
                       act='softmax')
        topk_scores, topk_indices = pd.topk(scores, k=50)
        accu_scores = pd.elementwise_add(
            x=pd.log(x=pd.softmax(topk_scores)),
            y=pd.reshape(prev_scores, shape=[-1]),
            axis=0)
        selected_ids, selected_scores = pd.beam_search(
            prev_ids, topk_indices, accu_scores, beam_size, end_id=1, level=0)

        with pd.Switch() as switch:
            with switch.case(pd.is_empty(selected_ids)):
                decoder.early_stop()
            with switch.default():
                decoder.state_cell.update_states()
                decoder.update_array(prev_ids, selected_ids)
                decoder.update_array(prev_scores, selected_scores)


    translation_ids, translation_scores = decoder()

    return translation_ids, translation_scores


def set_init_lod(data, lod, place):
    res = core.LoDTensor()
    res.set(data, place)
    res.set_lod(lod)
    return res


def to_lodtensor(data, place):
    seq_lens = [len(seq) for seq in data]
    cur_len = 0
    lod = [cur_len]
    for l in seq_lens:
        cur_len += l
        lod.append(cur_len)
    flattened_data = np.concatenate(data, axis=0).astype("int64")
    flattened_data = flattened_data.reshape([len(flattened_data), 1])
    res = core.LoDTensor()
    res.set(flattened_data, place)
    res.set_lod([lod])
    return res


def train_main():
    context = encoder()
    state_cell = decoder_state_cell(context)
    rnn_out = decoder_train(state_cell)
    label = pd.data(
        name="target_language_next_word", shape=[1], dtype='int64', lod_level=1)
    cost = pd.cross_entropy(input=rnn_out, label=label)
    avg_cost = pd.mean(x=cost)

    optimizer = fluid.optimizer.Adagrad(learning_rate=1e-3)
    optimizer.minimize(avg_cost)

    train_data = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.wmt14.train(dict_size), buf_size=1000),
        batch_size=batch_size)

    exe = Executor(place)

    exe.run(framework.default_startup_program())

    for pass_id in xrange(101):
        batch_id = 0
        for data in train_data():
            word_data = to_lodtensor(map(lambda x: x[0], data), place)
            trg_word = to_lodtensor(map(lambda x: x[1], data), place)
            trg_word_next = to_lodtensor(map(lambda x: x[2], data), place)
            program = framework.default_main_program()
            outs = exe.run(program,
                           feed={
                               'src_word_id': word_data,
                               'target_language_word': trg_word,
                               'target_language_next_word': trg_word_next
                           },
                           fetch_list=[avg_cost])
            avg_cost_val = np.array(outs[0])
            if pass_id % 1 == 0:
                print('pass_id=' + str(pass_id) + ' batch=' + str(batch_id) +
                     " avg_cost=" + str(avg_cost_val))
#            if batch_id > 500:
#                break
            batch_id += 1

        if pass_id % 1 == 0:
            model_path = os.path.join(model_save_dir, str(pass_id))
            if not os.path.isdir(model_path):
                os.makedirs(model_path)

            #fluid.io.save_inference_model(dirname=model_path,
            #                feeded_var_names=['src_word_id'],
            #                target_vars=[rnn_out],
            #                executor=exe,
            #                main_program=framework.default_main_program(),
            #                model_filename='test_save',
            #                params_filename=None)
            fluid.io.save_persistables(executor=exe,
                                       dirname=model_path,
                                       main_program=framework.default_main_program())



def decode_main():
    context = encoder()
    state_cell = decoder_state_cell(context)
    translation_ids, translation_scores = decoder_decode(state_cell)

    exe = Executor(place)
    exe.run(framework.default_startup_program())

#    model_path = os.path.join(model_save_dir, str(0))
#    if not os.path.isdir(model_path):
#        os.makedirs(model_path)
    model_path = os.path.join(model_save_dir, str(100))
#    fluid.io.load_inference_model(dirname=model_path,
#                                  executor=exe,
#                                  model_filename='test_save',
#                                  params_filename=None)

    fluid.io.load_persistables(executor=exe,
                               dirname=model_path,
                               main_program=framework.default_main_program())

    init_ids_data = np.array([0 for _ in range(batch_size)], dtype='int64')
    init_scores_data = np.array(
        [1. for _ in range(batch_size)], dtype='float32')
    init_ids_data = init_ids_data.reshape((batch_size, 1))
    init_scores_data = init_scores_data.reshape((batch_size, 1))
    init_lod = [i for i in range(batch_size)] + [batch_size]
    init_lod = [init_lod, init_lod]

    train_data = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.wmt14.train(dict_size), buf_size=1000),
        batch_size=batch_size)
    for _, data in enumerate(train_data()):
        print "One batch, expected result:"
        for tup in data:
            for i in range(2, len(tup)):
                if i == 0:
                    words = [src_dict[tup[i][j]] for j in xrange(len(tup[i]))]
                else:
                    words = [trg_dict[tup[i][j]] for j in xrange(len(tup[i]))]
                sentence = " ".join(words)
                print sentence
        init_ids = set_init_lod(init_ids_data, init_lod, place)
        init_scores = set_init_lod(init_scores_data, init_lod, place)

        src_word_data = to_lodtensor(map(lambda x: x[0], data), place)
#        print "source lod: ", src_word_data.lod()
        result_ids, result_scores = exe.run(
            framework.default_main_program(),
            feed={
                'src_word_id': src_word_data,
                'init_ids': init_ids,
                'init_scores': init_scores
            },
            fetch_list=[translation_ids, translation_scores],
            return_numpy=False)
#        print(np.array(context))
#        print(np.array(result_scores))
#        print result_scores.lod()
        lod_list_1 = result_ids.lod()[1]
        token_array = np.array(result_ids)
        result = []
        for i in xrange(len(lod_list_1) - 1):
            sentence_list = [trg_dict[token] 
                             for token in 
                             token_array[lod_list_1[i]:lod_list_1[i+1]]]
            sentence = " ".join(sentence_list)
            result.append(sentence)
        lod_list_0 = result_ids.lod()[0]
        final_result = [result[lod_list_0[i]:lod_list_0[i+1]] 
                        for i in xrange(len(lod_list_0) - 1)]
        print "Actual result:"
        for paragraph in final_result:
            print paragraph
        break

if __name__ == '__main__':
    #train_main()
    decode_main()
