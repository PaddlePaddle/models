#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.fluid as fluid
import six

decoder_size = 128
word_vector_dim = 128
max_length = 100
sos = 0
eos = 1
beam_size = 1


def conv_bn_pool(input,
                 group,
                 out_ch,
                 act="relu",
                 is_test=False,
                 pool=True,
                 use_cudnn=True):
    tmp = input
    for i in six.moves.xrange(group):
        filter_size = 3
        conv_std = (2.0 / (filter_size**2 * tmp.shape[1]))**0.5
        conv_param = fluid.ParamAttr(
            initializer=fluid.initializer.Normal(0.0, conv_std))
        tmp = fluid.layers.conv2d(
            input=tmp,
            num_filters=out_ch[i],
            filter_size=3,
            padding=1,
            bias_attr=False,
            param_attr=conv_param,
            act=None,  # LinearActivation
            use_cudnn=use_cudnn)

        tmp = fluid.layers.batch_norm(input=tmp, act=act, is_test=is_test)
    if pool == True:
        tmp = fluid.layers.pool2d(
            input=tmp,
            pool_size=2,
            pool_type='max',
            pool_stride=2,
            use_cudnn=use_cudnn,
            ceil_mode=True)

    return tmp


def ocr_convs(input, is_test=False, use_cudnn=True):
    tmp = input
    tmp = conv_bn_pool(tmp, 2, [16, 16], is_test=is_test, use_cudnn=use_cudnn)
    tmp = conv_bn_pool(tmp, 2, [32, 32], is_test=is_test, use_cudnn=use_cudnn)
    tmp = conv_bn_pool(tmp, 2, [64, 64], is_test=is_test, use_cudnn=use_cudnn)
    tmp = conv_bn_pool(
        tmp, 2, [128, 128], is_test=is_test, pool=False, use_cudnn=use_cudnn)
    return tmp


def encoder_net(images, rnn_hidden_size=200, is_test=False, use_cudnn=True):

    conv_features = ocr_convs(images, is_test=is_test, use_cudnn=use_cudnn)

    sliced_feature = fluid.layers.im2sequence(
        input=conv_features,
        stride=[1, 1],
        filter_size=[conv_features.shape[2], 1])

    para_attr = fluid.ParamAttr(initializer=fluid.initializer.Normal(0.0, 0.02))
    bias_attr = fluid.ParamAttr(
        initializer=fluid.initializer.Normal(0.0, 0.02), learning_rate=2.0)

    fc_1 = fluid.layers.fc(input=sliced_feature,
                           size=rnn_hidden_size * 3,
                           param_attr=para_attr,
                           bias_attr=False)
    fc_2 = fluid.layers.fc(input=sliced_feature,
                           size=rnn_hidden_size * 3,
                           param_attr=para_attr,
                           bias_attr=False)

    gru_forward = fluid.layers.dynamic_gru(
        input=fc_1,
        size=rnn_hidden_size,
        param_attr=para_attr,
        bias_attr=bias_attr,
        candidate_activation='relu')
    gru_backward = fluid.layers.dynamic_gru(
        input=fc_2,
        size=rnn_hidden_size,
        is_reverse=True,
        param_attr=para_attr,
        bias_attr=bias_attr,
        candidate_activation='relu')

    encoded_vector = fluid.layers.concat(
        input=[gru_forward, gru_backward], axis=1)
    encoded_proj = fluid.layers.fc(input=encoded_vector,
                                   size=decoder_size,
                                   bias_attr=False)

    return gru_backward, encoded_vector, encoded_proj


def gru_decoder_with_attention(target_embedding, encoder_vec, encoder_proj,
                               decoder_boot, decoder_size, num_classes):
    def simple_attention(encoder_vec, encoder_proj, decoder_state):
        decoder_state_proj = fluid.layers.fc(input=decoder_state,
                                             size=decoder_size,
                                             bias_attr=False)
        decoder_state_expand = fluid.layers.sequence_expand(
            x=decoder_state_proj, y=encoder_proj)
        concated = encoder_proj + decoder_state_expand
        concated = fluid.layers.tanh(x=concated)
        attention_weights = fluid.layers.fc(input=concated,
                                            size=1,
                                            act=None,
                                            bias_attr=False)
        attention_weights = fluid.layers.sequence_softmax(
            input=attention_weights)
        weigths_reshape = fluid.layers.reshape(x=attention_weights, shape=[-1])
        scaled = fluid.layers.elementwise_mul(
            x=encoder_vec, y=weigths_reshape, axis=0)
        context = fluid.layers.sequence_pool(input=scaled, pool_type='sum')
        return context

    rnn = fluid.layers.DynamicRNN()

    with rnn.block():
        current_word = rnn.step_input(target_embedding)
        encoder_vec = rnn.static_input(encoder_vec)
        encoder_proj = rnn.static_input(encoder_proj)
        hidden_mem = rnn.memory(init=decoder_boot, need_reorder=True)
        context = simple_attention(encoder_vec, encoder_proj, hidden_mem)
        fc_1 = fluid.layers.fc(input=context,
                               size=decoder_size * 3,
                               bias_attr=False)
        fc_2 = fluid.layers.fc(input=current_word,
                               size=decoder_size * 3,
                               bias_attr=False)
        decoder_inputs = fc_1 + fc_2
        h, _, _ = fluid.layers.gru_unit(
            input=decoder_inputs, hidden=hidden_mem, size=decoder_size * 3)
        rnn.update_memory(hidden_mem, h)
        out = fluid.layers.fc(input=h,
                              size=num_classes + 2,
                              bias_attr=True,
                              act='softmax')
        rnn.output(out)
    return rnn()


def attention_train_net(args, data_shape, num_classes):
    if len(list(data_shape)) == 3:
        data_shape = [None] + list(data_shape)
    images = fluid.data(name='pixel', shape=data_shape, dtype='float32')
    label_in = fluid.data(
        name='label_in', shape=[None, 1], dtype='int32', lod_level=1)
    label_out = fluid.data(
        name='label_out', shape=[None, 1], dtype='int32', lod_level=1)

    gru_backward, encoded_vector, encoded_proj = encoder_net(images)

    backward_first = fluid.layers.sequence_pool(
        input=gru_backward, pool_type='first')
    decoder_boot = fluid.layers.fc(input=backward_first,
                                   size=decoder_size,
                                   bias_attr=False,
                                   act="relu")

    label_in = fluid.layers.cast(x=label_in, dtype='int64')
    trg_embedding = fluid.layers.embedding(
        input=label_in,
        size=[num_classes + 2, word_vector_dim],
        dtype='float32')
    prediction = gru_decoder_with_attention(trg_embedding, encoded_vector,
                                            encoded_proj, decoder_boot,
                                            decoder_size, num_classes)
    fluid.clip.set_gradient_clip(
        fluid.clip.GradientClipByGlobalNorm(args.gradient_clip))
    label_out = fluid.layers.cast(x=label_out, dtype='int64')

    _, maxid = fluid.layers.topk(input=prediction, k=1)
    error_evaluator = fluid.evaluator.EditDistance(
        input=maxid, label=label_out, ignored_tokens=[sos, eos])

    inference_program = fluid.default_main_program().clone(for_test=True)

    cost = fluid.layers.cross_entropy(input=prediction, label=label_out)
    sum_cost = fluid.layers.reduce_sum(cost)
    LR = args.lr
    if args.lr_decay_strategy == "piecewise_decay":
        learning_rate = fluid.layers.piecewise_decay([50000], [LR, LR * 0.01])
    else:
        learning_rate = LR

    optimizer = fluid.optimizer.Adadelta(
        learning_rate=learning_rate, epsilon=1.0e-6, rho=0.9)
    optimizer.minimize(sum_cost)

    model_average = None
    if args.average_window > 0:
        model_average = fluid.optimizer.ModelAverage(
            args.average_window,
            min_average_window=args.min_average_window,
            max_average_window=args.max_average_window)

    return sum_cost, error_evaluator, inference_program, model_average


def simple_attention(encoder_vec, encoder_proj, decoder_state, decoder_size):
    decoder_state_proj = fluid.layers.fc(input=decoder_state,
                                         size=decoder_size,
                                         bias_attr=False)
    decoder_state_expand = fluid.layers.sequence_expand(
        x=decoder_state_proj, y=encoder_proj)
    concated = fluid.layers.elementwise_add(encoder_proj, decoder_state_expand)
    concated = fluid.layers.tanh(x=concated)
    attention_weights = fluid.layers.fc(input=concated,
                                        size=1,
                                        act=None,
                                        bias_attr=False)
    attention_weights = fluid.layers.sequence_softmax(input=attention_weights)
    weigths_reshape = fluid.layers.reshape(x=attention_weights, shape=[-1])
    scaled = fluid.layers.elementwise_mul(
        x=encoder_vec, y=weigths_reshape, axis=0)
    context = fluid.layers.sequence_pool(input=scaled, pool_type='sum')
    return context


def attention_infer(images, num_classes, use_cudnn=True):

    max_length = 20
    gru_backward, encoded_vector, encoded_proj = encoder_net(
        images, is_test=True, use_cudnn=use_cudnn)

    backward_first = fluid.layers.sequence_pool(
        input=gru_backward, pool_type='first')
    decoder_boot = fluid.layers.fc(input=backward_first,
                                   size=decoder_size,
                                   bias_attr=False,
                                   act="relu")
    init_state = decoder_boot
    array_len = fluid.layers.fill_constant(
        shape=[1], dtype='int64', value=max_length)
    counter = fluid.layers.zeros(shape=[1], dtype='int64', force_cpu=True)

    # fill the first element with init_state
    state_array = fluid.layers.create_array('float32')
    fluid.layers.array_write(init_state, array=state_array, i=counter)

    # ids, scores as memory
    ids_array = fluid.layers.create_array('int64')
    scores_array = fluid.layers.create_array('float32')

    init_ids = fluid.data(
        name="init_ids", shape=[None, 1], dtype="int64", lod_level=2)
    init_scores = fluid.data(
        name="init_scores", shape=[None, 1], dtype="float32", lod_level=2)

    fluid.layers.array_write(init_ids, array=ids_array, i=counter)
    fluid.layers.array_write(init_scores, array=scores_array, i=counter)

    cond = fluid.layers.less_than(x=counter, y=array_len)
    while_op = fluid.layers.While(cond=cond)
    with while_op.block():
        pre_ids = fluid.layers.array_read(array=ids_array, i=counter)
        pre_state = fluid.layers.array_read(array=state_array, i=counter)
        pre_score = fluid.layers.array_read(array=scores_array, i=counter)

        pre_ids_emb = fluid.layers.embedding(
            input=pre_ids,
            size=[num_classes + 2, word_vector_dim],
            dtype='float32')

        context = simple_attention(encoded_vector, encoded_proj, pre_state,
                                   decoder_size)

        # expand the recursive_sequence_lengths of pre_state to be the same with pre_score
        pre_state_expanded = fluid.layers.sequence_expand(pre_state, pre_score)
        context_expanded = fluid.layers.sequence_expand(context, pre_score)
        fc_1 = fluid.layers.fc(input=context_expanded,
                               size=decoder_size * 3,
                               bias_attr=False)
        fc_2 = fluid.layers.fc(input=pre_ids_emb,
                               size=decoder_size * 3,
                               bias_attr=False)

        decoder_inputs = fc_1 + fc_2
        current_state, _, _ = fluid.layers.gru_unit(
            input=decoder_inputs,
            hidden=pre_state_expanded,
            size=decoder_size * 3)

        current_state_with_lod = fluid.layers.lod_reset(
            x=current_state, y=pre_score)
        # use score to do beam search
        current_score = fluid.layers.fc(input=current_state_with_lod,
                                        size=num_classes + 2,
                                        bias_attr=True,
                                        act='softmax')
        topk_scores, topk_indices = fluid.layers.topk(
            current_score, k=beam_size)

        # calculate accumulated scores after topk to reduce computation cost
        accu_scores = fluid.layers.elementwise_add(
            x=fluid.layers.log(topk_scores),
            y=fluid.layers.reshape(
                pre_score, shape=[-1]),
            axis=0)
        selected_ids, selected_scores = fluid.layers.beam_search(
            pre_ids,
            pre_score,
            topk_indices,
            accu_scores,
            beam_size,
            eos,  # end_id
            #level=0
        )

        fluid.layers.increment(x=counter, value=1, in_place=True)

        # update the memories
        fluid.layers.array_write(current_state, array=state_array, i=counter)
        fluid.layers.array_write(selected_ids, array=ids_array, i=counter)
        fluid.layers.array_write(selected_scores, array=scores_array, i=counter)

        # update the break condition: up to the max length or all candidates of
        # source sentences have ended.
        length_cond = fluid.layers.less_than(x=counter, y=array_len)
        finish_cond = fluid.layers.logical_not(
            fluid.layers.is_empty(x=selected_ids))
        fluid.layers.logical_and(x=length_cond, y=finish_cond, out=cond)

    ids, scores = fluid.layers.beam_search_decode(ids_array, scores_array,
                                                  beam_size, eos)
    return ids


def attention_eval(data_shape, num_classes, use_cudnn=True):
    if len(list(data_shape)) == 3:
        data_shape = [None] + data_shape
    images = fluid.data(name='pixel', shape=data_shape, dtype='float32')
    label_in = fluid.data(
        name='label_in', shape=[None, 1], dtype='int32', lod_level=1)
    label_out = fluid.data(
        name='label_out', shape=[None, 1], dtype='int32', lod_level=1)
    label_out = fluid.layers.cast(x=label_out, dtype='int64')
    label_in = fluid.layers.cast(x=label_in, dtype='int64')

    gru_backward, encoded_vector, encoded_proj = encoder_net(
        images, is_test=True, use_cudnn=use_cudnn)

    backward_first = fluid.layers.sequence_pool(
        input=gru_backward, pool_type='first')
    decoder_boot = fluid.layers.fc(input=backward_first,
                                   size=decoder_size,
                                   bias_attr=False,
                                   act="relu")
    trg_embedding = fluid.layers.embedding(
        input=label_in,
        size=[num_classes + 2, word_vector_dim],
        dtype='float32')
    prediction = gru_decoder_with_attention(trg_embedding, encoded_vector,
                                            encoded_proj, decoder_boot,
                                            decoder_size, num_classes)
    _, maxid = fluid.layers.topk(input=prediction, k=1)
    error_evaluator = fluid.evaluator.EditDistance(
        input=maxid, label=label_out, ignored_tokens=[sos, eos])
    cost = fluid.layers.cross_entropy(input=prediction, label=label_out)
    sum_cost = fluid.layers.reduce_sum(cost)
    return error_evaluator, sum_cost
