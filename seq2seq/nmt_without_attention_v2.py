#!/usr/bin/env python

import sys
import gzip
import sqlite3
import paddle.v2 as paddle

### Parameters
word_vector_dim = 620
latent_chain_dim = 1000

beam_size = 3
max_length = 50


def seq2seq_net(source_dict_dim, target_dict_dim, generating=False):
    decoder_size = encoder_size = latent_chain_dim

    ### Encoder
    src_word_id = paddle.layer.data(
        name='source_language_word',
        type=paddle.data_type.integer_value_sequence(source_dict_dim))

    src_embedding = paddle.layer.embedding(
        input=src_word_id, size=word_vector_dim)

    encoder_forward = paddle.networks.simple_gru(
        input=src_embedding,
        act=paddle.activation.Tanh(),
        gate_act=paddle.activation.Sigmoid(),
        size=encoder_size,
        reverse=False)
    encoder_backward = paddle.networks.simple_gru(
        input=src_embedding,
        act=paddle.activation.Tanh(),
        gate_act=paddle.activation.Sigmoid(),
        size=encoder_size,
        reverse=True)
    encoded_vector = paddle.layer.concat(
        input=[encoder_forward, encoder_backward])

    #### Decoder
    encoder_last = paddle.layer.last_seq(input=encoded_vector)
    with paddle.layer.mixed(
            size=decoder_size,
            act=paddle.activation.Tanh()) as encoder_last_projected:
        encoder_last_projected += paddle.layer.full_matrix_projection(
            input=encoder_last)

    def gru_decoder_without_attention(enc_vec, current_word):

        decoder_mem = paddle.layer.memory(
            name='gru_decoder',
            size=decoder_size,
            boot_layer=encoder_last_projected)

        context = paddle.layer.last_seq(input=enc_vec)

        with paddle.layer.mixed(size=decoder_size * 3) as decoder_inputs:
            decoder_inputs += paddle.layer.full_matrix_projection(input=context)
            decoder_inputs += paddle.layer.full_matrix_projection(
                input=current_word)

        gru_step = paddle.layer.gru_step(
            name='gru_decoder',
            act=paddle.activation.Tanh(),
            gate_act=paddle.activation.Sigmoid(),
            input=decoder_inputs,
            output_mem=decoder_mem,
            size=decoder_size)

        with paddle.layer.mixed(
                size=target_dict_dim,
                bias_attr=True,
                act=paddle.activation.Softmax()) as out:
            out += paddle.layer.full_matrix_projection(input=gru_step)
        return out

    decoder_group_name = "decoder_group"
    group_input1 = paddle.layer.StaticInputV2(input=encoded_vector, is_seq=True)
    group_inputs = [group_input1]

    if not generating:
        trg_embedding = paddle.layer.embedding(
            input=paddle.layer.data(
                name='target_language_word',
                type=paddle.data_type.integer_value_sequence(target_dict_dim)),
            size=word_vector_dim,
            param_attr=paddle.attr.ParamAttr(name='_target_language_embedding'))
        group_inputs.append(trg_embedding)

        decoder = paddle.layer.recurrent_group(
            name=decoder_group_name,
            step=gru_decoder_without_attention,
            input=group_inputs)

        lbl = paddle.layer.data(
            name='target_language_next_word',
            type=paddle.data_type.integer_value_sequence(target_dict_dim))
        cost = paddle.layer.classification_cost(input=decoder, label=lbl)

        return cost
    else:

        trg_embedding = paddle.layer.GeneratedInputV2(
            size=target_dict_dim,
            embedding_name='_target_language_embedding',
            embedding_size=word_vector_dim)
        group_inputs.append(trg_embedding)

        beam_gen = paddle.layer.beam_search(
            name=decoder_group_name,
            step=gru_decoder_without_attention,
            input=group_inputs,
            bos_id=0,
            eos_id=1,
            beam_size=beam_size,
            max_length=max_length)

        return beam_gen


def train(source_dict_dim, target_dict_dim):
    cost = seq2seq_net(source_dict_dim, target_dict_dim)
    parameters = paddle.parameters.create(cost)

    # define optimize method and trainer
    optimizer = paddle.optimizer.RMSProp(
        learning_rate=1e-3,
        gradient_clipping_threshold=10.0,
        regularization=paddle.optimizer.L2Regularization(rate=8e-4))
    trainer = paddle.trainer.SGD(
        cost=cost, parameters=parameters, update_equation=optimizer)
    # define data reader
    wmt14_reader = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.wmt14.train(source_dict_dim), buf_size=8192),
        batch_size=55)

    # define event_handler callback
    def event_handler(event):
        if isinstance(event, paddle.event.EndIteration):
            if event.batch_id % 100 == 0 and event.batch_id > 0:
                with gzip.open('models/nmt_without_att_params_batch_%d.tar.gz' %
                               event.batch_id, 'w') as f:
                    parameters.to_tar(f)

            if event.batch_id % 10 == 0:
                #                wmt14_test_batch = paddle.batch(
                #                        paddle.reader.shuffle(
                #                            paddle.dataset.wmt14.test(source_dict_dim), 
                #                            buf_size=8192), batch_size=1)

                #test_result = trainer.test(wmt14_test_batch)
                print "\nPass %d, Batch %d, Cost %f, %s" % (
                    event.pass_id,
                    event.batch_id,
                    event.cost,
                    event.metrics,  # test_result.cost, test_result.metrics
                )
            else:
                sys.stdout.write('.')
                sys.stdout.flush()

    # start to train
    trainer.train(
        reader=wmt14_reader, event_handler=event_handler, num_passes=2)


def generate(source_dict_dim, target_dict_dim):
    # use the first 3 samples for generation
    gen_creator = paddle.dataset.wmt14.gen(source_dict_dim)
    gen_data = []
    gen_num = 3
    for item in gen_creator():
        gen_data.append((item[0], ))
        if len(gen_data) == gen_num:
            break

    beam_gen = seq2seq_net(source_dict_dim, target_dict_dim, True)

    with gzip.open('models/nmt_without_att_params_batch_400.tar.gz') as f:
        parameters = paddle.parameters.Parameters.from_tar(f)
    # prob is the prediction probabilities, and id is the prediction word. 
    beam_result = paddle.infer(
        output_layer=beam_gen,
        parameters=parameters,
        input=gen_data,
        field=['prob', 'id'])

    # get the dictionary
    src_dict, trg_dict = paddle.dataset.wmt14.get_dict(source_dict_dim)

    # the delimited element of generated sequences is -1,
    # the first element of each generated sequence is the sequence length
    seq_list = []
    seq = []
    for w in beam_result[1]:
        if w != -1:
            seq.append(w)
        else:
            seq_list.append(' '.join([trg_dict.get(w) for w in seq[1:]]))
            seq = []

    prob = beam_result[0]
    beam_size = 3
    for i in xrange(gen_num):
        print "\n*******************************************************\n"
        print "src:", ' '.join([src_dict.get(w) for w in gen_data[i][0]]), "\n"
        for j in xrange(beam_size):
            print "prob = %f:" % (prob[i][j]), seq_list[i * beam_size + j]


def main():
    paddle.init(use_gpu=False, trainer_count=4)
    source_language_dict_dim = 30000
    target_language_dict_dim = 30000
    generating = True

    if generating:
        generate(source_language_dict_dim, target_language_dict_dim)
    else:
        train(source_language_dict_dim, target_language_dict_dim)


if __name__ == '__main__':
    main()
