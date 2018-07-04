import paddle.fluid as fluid

decoder_size = 128
word_vector_dim = 128
max_length = 100
sos = 0
eos = 1
gradient_clip = 10
learning_rate = 1.0


def conv_bn_pool(input, group, out_ch, act="relu", is_test=False, pool=True):
    tmp = input
    for i in xrange(group):
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
            use_cudnn=True)

        tmp = fluid.layers.batch_norm(input=tmp, act=act, is_test=is_test)
    if pool == True:
        tmp = fluid.layers.pool2d(
            input=tmp,
            pool_size=2,
            pool_type='max',
            pool_stride=2,
            use_cudnn=True,
            ceil_mode=True)

    return tmp


def ocr_convs(input, is_test=False):
    tmp = input
    tmp = conv_bn_pool(tmp, 2, [16, 16], is_test=is_test)
    tmp = conv_bn_pool(tmp, 2, [32, 32], is_test=is_test)
    tmp = conv_bn_pool(tmp, 2, [64, 64], is_test=is_test)
    tmp = conv_bn_pool(tmp, 2, [128, 128], is_test=is_test, pool=False)
    return tmp


def encoder_net(images, rnn_hidden_size=200, is_test=False):

    conv_features = ocr_convs(images, is_test=is_test)

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

    images = fluid.layers.data(name='pixel', shape=data_shape, dtype='float32')
    label_in = fluid.layers.data(
        name='label_in', shape=[1], dtype='int32', lod_level=1)
    label_out = fluid.layers.data(
        name='label_out', shape=[1], dtype='int32', lod_level=1)

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
    fluid.clip.set_gradient_clip(fluid.clip.GradientClipByValue(gradient_clip))
    label_out = fluid.layers.cast(x=label_out, dtype='int64')

    _, maxid = fluid.layers.topk(input=prediction, k=1)
    error_evaluator = fluid.evaluator.EditDistance(
        input=maxid, label=label_out, ignored_tokens=[sos, eos])

    inference_program = fluid.default_main_program().clone(for_test=True)

    cost = fluid.layers.cross_entropy(input=prediction, label=label_out)
    sum_cost = fluid.layers.reduce_sum(cost)
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


def attention_infer():
    pass


def attention_eval(data_shape, num_classes):
    images = fluid.layers.data(name='pixel', shape=data_shape, dtype='float32')
    label_in = fluid.layers.data(
        name='label_in', shape=[1], dtype='int32', lod_level=1)
    label_out = fluid.layers.data(
        name='label_out', shape=[1], dtype='int32', lod_level=1)
    label_out = fluid.layers.cast(x=label_out, dtype='int64')
    label_in = fluid.layers.cast(x=label_in, dtype='int64')

    gru_backward, encoded_vector, encoded_proj = encoder_net(
        images, is_test=True)

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
