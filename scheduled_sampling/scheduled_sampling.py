import sys
import paddle.v2 as paddle
from random_schedule_generator import RandomScheduleGenerator

schedule_generator = RandomScheduleGenerator("linear", 0.75, 1000000)


def gen_schedule_data(reader):
    """
    Creates a data reader for scheduled sampling.

    Output from the iterator that created by original reader will be
    appended with "true_token_flag" to indicate whether to use true token.

    :param reader: the original reader.
    :type reader: callable

    :return: the new reader with the field "true_token_flag".
    :rtype: callable
    """

    def data_reader():
        for src_ids, trg_ids, trg_ids_next in reader():
            yield src_ids, trg_ids, trg_ids_next, \
                  [0] + schedule_generator.processBatch(len(trg_ids) - 1)

    return data_reader


def seqToseq_net(source_dict_dim, target_dict_dim, is_generating=False):
    """
    The definition of the sequence to sequence model
    :param source_dict_dim: the dictionary size of the source language
    :type source_dict_dim: int
    :param target_dict_dim: the dictionary size of the target language
    :type target_dict_dim: int
    :param is_generating: whether in generating mode
    :type is_generating: Bool
    :return: the last layer of the network
    :rtype: LayerOutput
    """
    ### Network Architecture
    word_vector_dim = 512  # dimension of word vector
    decoder_size = 512  # dimension of hidden unit in GRU Decoder network
    encoder_size = 512  # dimension of hidden unit in GRU Encoder network

    beam_size = 3
    max_length = 250

    #### Encoder
    src_word_id = paddle.layer.data(
        name='source_language_word',
        type=paddle.data_type.integer_value_sequence(source_dict_dim))
    src_embedding = paddle.layer.embedding(
        input=src_word_id, size=word_vector_dim)
    src_forward = paddle.networks.simple_gru(
        input=src_embedding, size=encoder_size)
    src_backward = paddle.networks.simple_gru(
        input=src_embedding, size=encoder_size, reverse=True)
    encoded_vector = paddle.layer.concat(input=[src_forward, src_backward])

    #### Decoder
    with paddle.layer.mixed(size=decoder_size) as encoded_proj:
        encoded_proj += paddle.layer.full_matrix_projection(
            input=encoded_vector)

    backward_first = paddle.layer.first_seq(input=src_backward)

    with paddle.layer.mixed(
            size=decoder_size, act=paddle.activation.Tanh()) as decoder_boot:
        decoder_boot += paddle.layer.full_matrix_projection(
            input=backward_first)

    def gru_decoder_with_attention_train(enc_vec, enc_proj, true_word,
                                         true_token_flag):
        """
        The decoder step for training.
        :param enc_vec: the encoder vector for attention
        :type enc_vec: LayerOutput
        :param enc_proj: the encoder projection for attention
        :type enc_proj: LayerOutput
        :param true_word: the ground-truth target word
        :type true_word: LayerOutput
        :param true_token_flag: the flag of using the ground-truth target word
        :type true_token_flag: LayerOutput
        :return: the softmax output layer
        :rtype: LayerOutput
        """

        decoder_mem = paddle.layer.memory(
            name='gru_decoder', size=decoder_size, boot_layer=decoder_boot)

        context = paddle.networks.simple_attention(
            encoded_sequence=enc_vec,
            encoded_proj=enc_proj,
            decoder_state=decoder_mem)

        gru_out_memory = paddle.layer.memory(
            name='gru_out', size=target_dict_dim)

        generated_word = paddle.layer.max_id(input=gru_out_memory)

        generated_word_emb = paddle.layer.embedding(
            input=generated_word,
            size=word_vector_dim,
            param_attr=paddle.attr.ParamAttr(name='_target_language_embedding'))

        current_word = paddle.layer.multiplex(
            input=[true_token_flag, true_word, generated_word_emb])

        with paddle.layer.mixed(size=decoder_size * 3) as decoder_inputs:
            decoder_inputs += paddle.layer.full_matrix_projection(input=context)
            decoder_inputs += paddle.layer.full_matrix_projection(
                input=current_word)

        gru_step = paddle.layer.gru_step(
            name='gru_decoder',
            input=decoder_inputs,
            output_mem=decoder_mem,
            size=decoder_size)

        with paddle.layer.mixed(
                name='gru_out',
                size=target_dict_dim,
                bias_attr=True,
                act=paddle.activation.Softmax()) as out:
            out += paddle.layer.full_matrix_projection(input=gru_step)

        return out

    def gru_decoder_with_attention_test(enc_vec, enc_proj, current_word):
        """
        The decoder step for generating.
        :param enc_vec: the encoder vector for attention
        :type enc_vec: LayerOutput
        :param enc_proj: the encoder projection for attention
        :type enc_proj: LayerOutput
        :param current_word: the previously generated word
        :type current_word: LayerOutput
        :return: the softmax output layer
        :rtype: LayerOutput
        """

        decoder_mem = paddle.layer.memory(
            name='gru_decoder', size=decoder_size, boot_layer=decoder_boot)

        context = paddle.networks.simple_attention(
            encoded_sequence=enc_vec,
            encoded_proj=enc_proj,
            decoder_state=decoder_mem)

        with paddle.layer.mixed(size=decoder_size * 3) as decoder_inputs:
            decoder_inputs += paddle.layer.full_matrix_projection(input=context)
            decoder_inputs += paddle.layer.full_matrix_projection(
                input=current_word)

        gru_step = paddle.layer.gru_step(
            name='gru_decoder',
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
    group_input1 = paddle.layer.StaticInput(input=encoded_vector, is_seq=True)
    group_input2 = paddle.layer.StaticInput(input=encoded_proj, is_seq=True)
    group_inputs = [group_input1, group_input2]

    if not is_generating:
        trg_embedding = paddle.layer.embedding(
            input=paddle.layer.data(
                name='target_language_word',
                type=paddle.data_type.integer_value_sequence(target_dict_dim)),
            size=word_vector_dim,
            param_attr=paddle.attr.ParamAttr(name='_target_language_embedding'))
        group_inputs.append(trg_embedding)

        true_token_flags = paddle.layer.data(
            name='true_token_flag',
            type=paddle.data_type.integer_value_sequence(2))
        group_inputs.append(true_token_flags)

        decoder = paddle.layer.recurrent_group(
            name=decoder_group_name,
            step=gru_decoder_with_attention_train,
            input=group_inputs)

        lbl = paddle.layer.data(
            name='target_language_next_word',
            type=paddle.data_type.integer_value_sequence(target_dict_dim))
        cost = paddle.layer.classification_cost(input=decoder, label=lbl)

        return cost
    else:
        trg_embedding = paddle.layer.GeneratedInput(
            size=target_dict_dim,
            embedding_name='_target_language_embedding',
            embedding_size=word_vector_dim)
        group_inputs.append(trg_embedding)

        beam_gen = paddle.layer.beam_search(
            name=decoder_group_name,
            step=gru_decoder_with_attention_test,
            input=group_inputs,
            bos_id=0,
            eos_id=1,
            beam_size=beam_size,
            max_length=max_length)

        return beam_gen


def main():
    paddle.init(use_gpu=False, trainer_count=1)
    is_generating = False
    model_path_for_generating = 'params_pass_1.tar.gz'

    # source and target dict dim.
    dict_size = 30000
    source_dict_dim = target_dict_dim = dict_size

    # train the network
    if not is_generating:
        cost = seqToseq_net(source_dict_dim, target_dict_dim)
        parameters = paddle.parameters.create(cost)

        # define optimize method and trainer
        optimizer = paddle.optimizer.Adam(
            learning_rate=5e-5,
            regularization=paddle.optimizer.L2Regularization(rate=8e-4))
        trainer = paddle.trainer.SGD(
            cost=cost, parameters=parameters, update_equation=optimizer)
        # define data reader
        wmt14_reader = paddle.batch(
            gen_schedule_data(
                paddle.reader.shuffle(
                    paddle.dataset.wmt14.train(dict_size), buf_size=8192)),
            batch_size=5)

        feeding = {
            'source_language_word': 0,
            'target_language_word': 1,
            'target_language_next_word': 2,
            'true_token_flag': 3
        }

        # define event_handler callback
        def event_handler(event):
            if isinstance(event, paddle.event.EndIteration):
                if event.batch_id % 10 == 0:
                    print "\nPass %d, Batch %d, Cost %f, %s" % (
                        event.pass_id, event.batch_id, event.cost,
                        event.metrics)
                else:
                    sys.stdout.write('.')
                    sys.stdout.flush()
            if isinstance(event, paddle.event.EndPass):
                # save parameters
                with gzip.open('params_pass_%d.tar.gz' % event.pass_id,
                               'w') as f:
                    trainer.save_parameter_to_tar(f)

        # start to train
        trainer.train(
            reader=wmt14_reader,
            event_handler=event_handler,
            feeding=feeding,
            num_passes=2)

    # generate a english sequence to french
    else:
        # use the first 3 samples for generation
        gen_creator = paddle.dataset.wmt14.gen(dict_size)
        gen_data = []
        gen_num = 3
        for item in gen_creator():
            gen_data.append((item[0], ))
            if len(gen_data) == gen_num:
                break

        beam_gen = seqToseq_net(source_dict_dim, target_dict_dim, is_generating)
        # get the trained model
        with gzip.open(model_path_for_generating, 'r') as f:
            parameters = Parameters.from_tar(f)
        # prob is the prediction probabilities, and id is the prediction word.
        beam_result = paddle.infer(
            output_layer=beam_gen,
            parameters=parameters,
            input=gen_data,
            field=['prob', 'id'])

        # get the dictionary
        src_dict, trg_dict = paddle.dataset.wmt14.get_dict(dict_size)

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
            print "src:", ' '.join(
                [src_dict.get(w) for w in gen_data[i][0]]), "\n"
            for j in xrange(beam_size):
                print "prob = %f:" % (prob[i][j]), seq_list[i * beam_size + j]


if __name__ == '__main__':
    main()
