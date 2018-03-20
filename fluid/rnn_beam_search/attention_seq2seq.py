"""seq2seq model for fluid."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import argparse
import time
import distutils.util

import paddle.v2
import paddle.fluid as fluid
import paddle.fluid.core as core
import paddle.fluid.framework as framework
from paddle.fluid.executor import Executor
from beam_search_api import *

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
    default=16,
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
    default=2,
    help="The pass number to train. (default: %(default)d)")
parser.add_argument(
    "--learning_rate",
    type=float,
    default=0.0002,
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
    default=False,
    help="Whether to use gpu. (default: %(default)d)")
parser.add_argument(
    "--max_length",
    type=int,
    default=250,
    help="The maximum length of sequence when doing generation. "
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
        # Linear transformation part for input gate, output gate, forget gate
        # and cell activation vectors need be done outside of dynamic_lstm.
        # So the output size is 4 times of gate_size.
        input_forward_proj = fluid.layers.fc(input=input_seq,
                                             size=gate_size * 4,
                                             act=None,
                                             bias_attr=False)
        forward, _ = fluid.layers.dynamic_lstm(
            input=input_forward_proj, size=gate_size * 4, use_peepholes=False)
        input_reversed_proj = fluid.layers.fc(input=input_seq,
                                              size=gate_size * 4,
                                              act=None,
                                              bias_attr=False)
        reversed, _ = fluid.layers.dynamic_lstm(
            input=input_reversed_proj,
            size=gate_size * 4,
            is_reverse=True,
            use_peepholes=False)
        return forward, reversed

    src_word_idx = fluid.layers.data(
        name='source_sequence', shape=[1], dtype='int64', lod_level=1)

    src_embedding = fluid.layers.embedding(
        input=src_word_idx,
        size=[source_dict_dim, embedding_dim],
        dtype='float32',
        param_attr=fluid.ParamAttr(name='src_embedding'))

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

    h = InitState(init=decoder_boot, need_reorder=True)
    c = InitState(init=cell_init)

    state_cell = StateCell(
        cell_size=decoder_size,
        inputs={'x': None,
                'encoder_vec': None,
                'encoder_proj': None},
        states={'h': h,
                'c': c})

    def simple_attention(encoder_vec, encoder_proj, decoder_state):
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
                                            act='tanh',
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

    if not is_generating:
        trg_word_idx = fluid.layers.data(
            name='target_sequence', shape=[1], dtype='int64', lod_level=1)

        trg_embedding = fluid.layers.embedding(
            input=trg_word_idx,
            size=[target_dict_dim, embedding_dim],
            dtype='float32',
            param_attr=fluid.ParamAttr('trg_embedding'))

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
        init_ids = fluid.layers.data(
            name="init_ids", shape=[1], dtype="int64", lod_level=2)
        init_scores = fluid.layers.data(
            name="init_scores", shape=[1], dtype="float32", lod_level=2)

        def embedding(input):
            return fluid.layers.embedding(
                input=input,
                size=[target_dict_dim, embedding_dim],
                dtype='float32',
                param_attr=fluid.ParamAttr('trg_embedding'))

        decoder = BeamSearchDecoder(state_cell, max_len=max_length)

        with decoder.block():
            encoder_vec = decoder.read_array(init=encoded_vector)
            encoder_proj = decoder.read_array(init=encoded_proj)
            prev_ids = decoder.read_array(init=init_ids, is_ids=True)
            prev_scores = decoder.read_array(init=init_scores, is_scores=True)
            prev_ids_embedding = embedding(prev_ids)
            prev_h = decoder.state_cell.get_state('h')
            prev_c = decoder.state_cell.get_state('c')
            prev_h_expanded = fluid.layers.sequence_expand(prev_h, prev_scores)
            prev_c_expanded = fluid.layers.sequence_expand(prev_c, prev_scores)
            encoder_vec_expanded = fluid.layers.sequence_expand(encoder_vec,
                                                                prev_scores)
            encoder_proj_expanded = fluid.layers.sequence_expand(encoder_proj,
                                                                 prev_scores)
            decoder.state_cell.set_state('h', prev_h_expanded)
            decoder.state_cell.set_state('c', prev_c_expanded)
            decoder.state_cell.compute_state(inputs={
                'x': prev_ids_embedding,
                'encoder_vec': encoder_vec_expanded,
                'encoder_proj': encoder_proj_expanded
            })
            current_state = decoder.state_cell.get_state('h')
            current_state_with_lod = fluid.layers.lod_reset(
                x=current_state, y=prev_scores)
            scores = fluid.layers.fc(input=current_state_with_lod,
                                     size=target_dict_dim,
                                     act='softmax')
            topk_scores, topk_indices = fluid.layers.topk(scores, k=beam_size)
            selected_ids, selected_scores = fluid.layers.beam_search(
                prev_ids,
                topk_indices,
                topk_scores,
                beam_size,
                end_id=1,
                level=0)
            decoder.state_cell.update_states()
            decoder.update_array(prev_ids, selected_ids)
            decoder.update_array(prev_scores, selected_scores)
            decoder.update_array(encoder_vec, encoder_vec_expanded)
            decoder.update_array(encoder_proj, encoder_proj_expanded)

        translation_ids, translation_scores = decoder()

        feeding_list = ["source_sequence", "init_ids", "init_scores"]

        return translation_ids, translation_scores, feeding_list


def to_lodtensor(data, place, dtype='int64'):
    seq_lens = [len(seq) for seq in data]
    cur_len = 0
    lod = [cur_len]
    for l in seq_lens:
        cur_len += l
        lod.append(cur_len)
    flattened_data = np.concatenate(data, axis=0).astype(dtype)
    flattened_data = flattened_data.reshape([len(flattened_data), 1])
    lod_t = core.LoDTensor()
    lod_t.set(flattened_data, place)
    lod_t.set_lod([lod])
    return lod_t, lod[-1]


def lodtensor_to_ndarray(lod_tensor):
    dims = lod_tensor.get_dims()
    ndarray = np.zeros(shape=dims).astype('float32')
    for i in xrange(np.product(dims)):
        ndarray.ravel()[i] = lod_tensor.get_float_element(i)
    return ndarray


def train():
    avg_cost, feeding_list = seq_to_seq_net(
        args.embedding_dim,
        args.encoder_size,
        args.decoder_size,
        args.dict_size,
        args.dict_size,
        False,
        beam_size=args.beam_size,
        max_length=args.max_length)

    # clone from default main program
    inference_program = fluid.default_main_program().clone()

    optimizer = fluid.optimizer.Adam(learning_rate=args.learning_rate)
    optimizer.minimize(avg_cost)

    fluid.memory_optimize(fluid.default_main_program(), print_log=False)

    train_batch_generator = paddle.v2.batch(
        paddle.v2.reader.shuffle(
            paddle.v2.dataset.wmt14.train(args.dict_size), buf_size=1000),
        batch_size=args.batch_size)

    test_batch_generator = paddle.v2.batch(
        paddle.v2.reader.shuffle(
            paddle.v2.dataset.wmt14.test(args.dict_size), buf_size=1000),
        batch_size=args.batch_size)

    place = core.CUDAPlace(0) if args.use_gpu else core.CPUPlace()
    exe = Executor(place)
    exe.run(framework.default_startup_program())

    def do_validation():
        total_loss = 0.0
        count = 0
        for batch_id, data in enumerate(test_batch_generator()):
            src_seq = to_lodtensor(map(lambda x: x[0], data), place)[0]
            trg_seq = to_lodtensor(map(lambda x: x[1], data), place)[0]
            lbl_seq = to_lodtensor(map(lambda x: x[2], data), place)[0]

            fetch_outs = exe.run(inference_program,
                                 feed={
                                     feeding_list[0]: src_seq,
                                     feeding_list[1]: trg_seq,
                                     feeding_list[2]: lbl_seq
                                 },
                                 fetch_list=[avg_cost],
                                 return_numpy=False)

            total_loss += lodtensor_to_ndarray(fetch_outs[0])[0]
            count += 1

        return total_loss / count

    for pass_id in xrange(args.pass_num):
        pass_start_time = time.time()
        words_seen = 0
        for batch_id, data in enumerate(train_batch_generator()):
            src_seq, word_num = to_lodtensor(map(lambda x: x[0], data), place)
            words_seen += word_num
            trg_seq, word_num = to_lodtensor(map(lambda x: x[1], data), place)
            words_seen += word_num
            lbl_seq, _ = to_lodtensor(map(lambda x: x[2], data), place)

            fetch_outs = exe.run(framework.default_main_program(),
                                 feed={
                                     feeding_list[0]: src_seq,
                                     feeding_list[1]: trg_seq,
                                     feeding_list[2]: lbl_seq
                                 },
                                 fetch_list=[avg_cost])

            avg_cost_val = np.array(fetch_outs[0])
            print('pass_id=%d, batch_id=%d, train_loss: %f' %
                  (pass_id, batch_id, avg_cost_val))

        pass_end_time = time.time()
        test_loss = do_validation()
        time_consumed = pass_end_time - pass_start_time
        words_per_sec = words_seen / time_consumed
        print("pass_id=%d, test_loss: %f, words/s: %f, sec/pass: %f" %
              (pass_id, test_loss, words_per_sec, time_consumed))


def infer():
    translation_ids, translation_scores, feeding_list = seq_to_seq_net(
        args.embedding_dim,
        args.encoder_size,
        args.decoder_size,
        args.dict_size,
        args.dict_size,
        True,
        beam_size=args.beam_size,
        max_length=args.max_length)

    fluid.memory_optimize(fluid.default_main_program(), print_log=False)

    test_batch_generator = paddle.v2.batch(
        paddle.v2.reader.shuffle(
            paddle.v2.dataset.wmt14.test(args.dict_size), buf_size=1000),
        batch_size=args.batch_size)

    place = core.CUDAPlace(0) if args.use_gpu else core.CPUPlace()
    exe = Executor(place)
    exe.run(framework.default_startup_program())

    for batch_id, data in enumerate(test_batch_generator()):
        batch_size = len(data)
        src_seq, _ = to_lodtensor(map(lambda x: x[0], data), place)
        init_ids, _ = to_lodtensor([[0] for _ in xrange(batch_size)], place)
        init_ids.set_lod(init_ids.lod() + [init_ids.lod()[-1]])
        init_scores, _ = to_lodtensor([[1.0] for _ in xrange(batch_size)],
                                      place, 'float32')
        init_scores.set_lod(init_scores.lod() + [init_scores.lod()[-1]])

        fetch_outs = exe.run(framework.default_main_program(),
                             feed={
                                 feeding_list[0]: src_seq,
                                 feeding_list[1]: init_ids,
                                 feeding_list[2]: init_scores
                             },
                             fetch_list=[translation_ids, translation_scores],
                             return_numpy=False)

        print(fetch_outs[0].lod())
        break


if __name__ == '__main__':
    args = parser.parse_args()
    if args.infer_only:
        infer()
    else:
        train()
