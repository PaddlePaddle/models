"""
    This python script is an example model configuration for neural machine
    translation with external memory, based on PaddlePaddle V2 APIs.

    The "external memory" refers to two types of memories.
    - Unbounded memory: i.e. vanilla attention mechanism in Seq2Seq.
    - Bounded memory: i.e. external memory in NTM.
    Both types of external memories are exploited to enhance the vanilla
    Seq2Seq neural machine translation.

    The implementation primarily follows the paper
    `Memory-enhanced Decoder for Neural Machine Translation
    <https://arxiv.org/abs/1606.02003>`_,
    with some minor differences (will be listed in README.md).

    For details about "external memory", please also refer to
    `Neural Turing Machines <https://arxiv.org/abs/1410.5401>`_.
"""

import paddle.v2 as paddle
import sys
import gzip

dict_size = 30000
word_vec_dim = 512
hidden_size = 1024
batch_size = 5
memory_slot_num = 8
beam_size = 3
infer_data_num = 3


class ExternalMemory(object):
    """
    External neural memory class.

    A simplified Neural Turing Machines (NTM) with only content-based
    addressing (including content addressing and interpolation, but excluding
    convolutional shift and sharpening). It serves as an external differential
    memory bank, with differential write/read head controllers to store
    and read information dynamically as needed. Simple feedforward networks are
    used as the write/read head controllers.

    The ExternalMemory class could be utilized by many neural network structures
    to easily expand their memory bandwidth and accomplish a long-term memory
    handling. Besides, some existing mechanism can be realized directly with
    the ExternalMemory class, e.g. the attention mechanism in Seq2Seq (i.e. an
    unbounded external memory).
    
    For more details, please refer to
    `Neural Turing Machines <https://arxiv.org/abs/1410.5401>`_.

    :param name: Memory name.
    :type name: basestring
    :param mem_slot_size: Size of memory slot/vector.
    :type mem_slot_size: int
    :param boot_layer: Boot layer for initializing the external memory. The
                       sequence layer has sequence length indicating the number
                       of memory slots, and size as memory slot size.
    :type boot_layer: LayerOutput
    :param readonly: If true, the memory is read-only, and write function cannot
                     be called. Default is false.
    :type readonly: bool
    :param enable_interpolation: If set true, the read/write addressing weights
                                 will be interpolated with the weights in the
                                 last step, with the affine coefficients being
                                 a learnable gate function.
    :type enable_interpolation: bool
    """

    def __init__(self,
                 name,
                 mem_slot_size,
                 boot_layer,
                 readonly=False,
                 enable_interpolation=True):
        self.name = name
        self.mem_slot_size = mem_slot_size
        self.readonly = readonly
        self.enable_interpolation = enable_interpolation
        self.external_memory = paddle.layer.memory(
            name=self.name,
            size=self.mem_slot_size,
            is_seq=True,
            boot_layer=boot_layer)
        # prepare a constant (zero) intializer for addressing weights 
        self.zero_addressing_init = paddle.layer.slope_intercept(
            input=paddle.layer.fc(input=boot_layer, size=1),
            slope=0.0,
            intercept=0.0)
        # set memory to constant when readonly=True
        if self.readonly:
            self.updated_external_memory = paddle.layer.mixed(
                name=self.name,
                input=[
                    paddle.layer.identity_projection(input=self.external_memory)
                ],
                size=self.mem_slot_size)

    def __content_addressing__(self, key_vector):
        """
        Get write/read head's addressing weights via content-based addressing.
        """
        # content-based addressing: a=tanh(W*M + U*key)
        key_projection = paddle.layer.fc(
            input=key_vector,
            size=self.mem_slot_size,
            act=paddle.activation.Linear(),
            bias_attr=False)
        key_proj_expanded = paddle.layer.expand(
            input=key_projection, expand_as=self.external_memory)
        memory_projection = paddle.layer.fc(
            input=self.external_memory,
            size=self.mem_slot_size,
            act=paddle.activation.Linear(),
            bias_attr=False)
        merged_projection = paddle.layer.addto(
            input=[key_proj_expanded, memory_projection],
            act=paddle.activation.Tanh())
        # softmax addressing weight: w=softmax(v^T a)
        addressing_weight = paddle.layer.fc(
            input=merged_projection,
            size=1,
            act=paddle.activation.SequenceSoftmax(),
            bias_attr=False)
        return addressing_weight

    def __interpolation__(self, head_name, key_vector, addressing_weight):
        """
        Interpolate between previous and current addressing weights.
        """
        # prepare interpolation scalar gate: g=sigmoid(W*key)
        gate = paddle.layer.fc(
            input=key_vector,
            size=1,
            act=paddle.activation.Sigmoid(),
            bias_attr=False)
        # interpolation: w_t = g*w_t+(1-g)*w_{t-1}
        last_addressing_weight = paddle.layer.memory(
            name=self.name + "_addressing_weight_" + head_name,
            size=1,
            is_seq=True,
            boot_layer=self.zero_addressing_init)
        interpolated_weight = paddle.layer.interpolation(
            name=self.name + "_addressing_weight_" + head_name,
            input=[addressing_weight, addressing_weight],
            weight=paddle.layer.expand(input=gate, expand_as=addressing_weight))
        return interpolated_weight

    def __get_addressing_weight__(self, head_name, key_vector):
        """
        Get final addressing weights for read/write heads, including content
        addressing and interpolation.
        """
        # current content-based addressing
        addressing_weight = self.__content_addressing__(key_vector)
        # interpolation with previous addresing weight
        if self.enable_interpolation:
            return self.__interpolation__(head_name, key_vector,
                                          addressing_weight)
        else:
            return addressing_weight

    def write(self, write_key):
        """
        Write onto the external memory.
        It cannot be called if "readonly" set True.

        :param write_key: Key vector for write heads to generate writing
                          content and addressing signals.
        :type write_key: LayerOutput
        """
        # check readonly
        if self.readonly:
            raise ValueError("ExternalMemory with readonly=True cannot write.")
        # get addressing weight for write head
        write_weight = self.__get_addressing_weight__("write_head", write_key)
        # prepare add_vector and erase_vector
        erase_vector = paddle.layer.fc(
            input=write_key,
            size=self.mem_slot_size,
            act=paddle.activation.Sigmoid(),
            bias_attr=False)
        add_vector = paddle.layer.fc(
            input=write_key,
            size=self.mem_slot_size,
            act=paddle.activation.Sigmoid(),
            bias_attr=False)
        erase_vector_expand = paddle.layer.expand(
            input=erase_vector, expand_as=self.external_memory)
        add_vector_expand = paddle.layer.expand(
            input=add_vector, expand_as=self.external_memory)
        # prepare scaled add part and erase part
        scaled_erase_vector_expand = paddle.layer.scaling(
            weight=write_weight, input=erase_vector_expand)
        erase_memory_part = paddle.layer.mixed(
            input=paddle.layer.dotmul_operator(
                a=self.external_memory,
                b=scaled_erase_vector_expand,
                scale=-1.0))
        add_memory_part = paddle.layer.scaling(
            weight=write_weight, input=add_vector_expand)
        # update external memory
        self.updated_external_memory = paddle.layer.addto(
            input=[self.external_memory, add_memory_part, erase_memory_part],
            name=self.name)

    def read(self, read_key):
        """
        Read from the external memory.

        :param write_key: Key vector for read head to generate addressing
                          signals.
        :type write_key: LayerOutput
        :return: Content (vector) read from external memory.
        :rtype: LayerOutput
        """
        # get addressing weight for write head
        read_weight = self.__get_addressing_weight__("read_head", read_key)
        # read content from external memory
        scaled = paddle.layer.scaling(
            weight=read_weight, input=self.updated_external_memory)
        return paddle.layer.pooling(
            input=scaled, pooling_type=paddle.pooling.Sum())


def bidirectional_gru_encoder(input, size, word_vec_dim):
    """
    Bidirectional GRU encoder.
    """
    # token embedding
    embeddings = paddle.layer.embedding(input=input, size=word_vec_dim)
    # token-level forward and backard encoding for attentions
    forward = paddle.networks.simple_gru(
        input=embeddings, size=size, reverse=False)
    backward = paddle.networks.simple_gru(
        input=embeddings, size=size, reverse=True)
    forward_backward = paddle.layer.concat(input=[forward, backward])
    # sequence-level encoding
    backward_first = paddle.layer.first_seq(input=backward)
    return forward_backward, backward_first


def memory_enhanced_decoder(input, target, initial_state, source_context, size,
                            word_vec_dim, dict_size, is_generating, beam_size):
    """
    GRU sequence decoder enhanced with external memory.

    The "external memory" refers to two types of memories.
    - Unbounded memory: i.e. attention mechanism in Seq2Seq.
    - Bounded memory: i.e. external memory in NTM.
    Both types of external memories can be implemented with
    ExternalMemory class, and are both exploited in this enhanced RNN decoder.

    The vanilla RNN/LSTM/GRU also has a narrow memory mechanism, namely the
    hidden state vector (or cell state in LSTM) carrying information through
    a span of sequence time, which is a successful design enriching the model
    with the capability to "remember" things in the long run. However, such a
    vector state is somewhat limited to a very narrow memory bandwidth. External
    memory introduced here could easily increase the memory capacity with linear
    complexity cost (rather than quadratic for vector state).

    This enhanced decoder expands its "memory passage" through two
    ExternalMemory objects:
    - Bounded memory for handling long-term information exchange within decoder
      itself. A direct expansion of traditional "vector" state.
    - Unbounded memory for handling source language's token-wise information.
      Exactly the attention mechanism over Seq2Seq.
    
    Notice that we take the attention mechanism as a particular form of external
    memory, with read-only memory bank initialized with encoder states, and a
    read head with content-based addressing (attention). From this view point,
    we arrive at a better understanding of attention mechanism itself and other
    external memory, and a concise and unified implementation for them.

    For more details about external memory, please refer to
    `Neural Turing Machines <https://arxiv.org/abs/1410.5401>`_.

    For more details about this memory-enhanced decoder, please
    refer to `Memory-enhanced Decoder for Neural Machine Translation 
    <https://arxiv.org/abs/1606.02003>`_. This implementation is highly
    correlated to this paper, but with minor differences (e.g. put "write"
    before "read" to bypass a potential bug in V2 APIs. See
    (`issue <https://github.com/PaddlePaddle/Paddle/issues/2061>`_).
    """
    # prepare initial bounded and unbounded memory
    bounded_memory_slot_init = paddle.layer.fc(
        input=paddle.layer.pooling(
            input=source_context, pooling_type=paddle.pooling.Avg()),
        size=size,
        act=paddle.activation.Sigmoid())
    bounded_memory_init = paddle.layer.expand(
        input=bounded_memory_slot_init,
        expand_as=paddle.layer.data(
            name='bounded_memory_template',
            type=paddle.data_type.integer_value_sequence(0)))
    unbounded_memory_init = source_context

    # prepare step function for reccurent group
    def recurrent_decoder_step(cur_embedding):
        # create hidden state, bounded and unbounded memory.
        state = paddle.layer.memory(
            name="gru_decoder", size=size, boot_layer=initial_state)
        bounded_memory = ExternalMemory(
            name="bounded_memory",
            mem_slot_size=size,
            boot_layer=bounded_memory_init,
            readonly=False,
            enable_interpolation=True)
        unbounded_memory = ExternalMemory(
            name="unbounded_memory",
            mem_slot_size=size * 2,
            boot_layer=unbounded_memory_init,
            readonly=True,
            enable_interpolation=False)
        # write bounded memory
        bounded_memory.write(state)
        # read bounded memory
        bounded_memory_read = bounded_memory.read(state)
        # prepare key for unbounded memory
        key_for_unbounded_memory = paddle.layer.fc(
            input=[bounded_memory_read, cur_embedding],
            size=size,
            act=paddle.activation.Tanh(),
            bias_attr=False)
        # read unbounded memory (i.e. attention mechanism) 
        context = unbounded_memory.read(key_for_unbounded_memory)
        # gated recurrent unit
        gru_inputs = paddle.layer.fc(
            input=[context, cur_embedding, bounded_memory_read],
            size=size * 3,
            act=paddle.activation.Linear(),
            bias_attr=False)
        gru_output = paddle.layer.gru_step(
            name="gru_decoder", input=gru_inputs, output_mem=state, size=size)
        # step output
        return paddle.layer.fc(
            input=[gru_output, context, cur_embedding],
            size=dict_size,
            act=paddle.activation.Softmax(),
            bias_attr=True)

    if not is_generating:
        target_embeddings = paddle.layer.embedding(
            input=input,
            size=word_vec_dim,
            param_attr=paddle.attr.ParamAttr(name="_decoder_word_embedding"))
        decoder_result = paddle.layer.recurrent_group(
            name="decoder_group",
            step=recurrent_decoder_step,
            input=[target_embeddings])
        cost = paddle.layer.classification_cost(
            input=decoder_result, label=target)
        return cost
    else:
        target_embeddings = paddle.layer.GeneratedInputV2(
            size=dict_size,
            embedding_name="_decoder_word_embedding",
            embedding_size=word_vec_dim)
        beam_gen = paddle.layer.beam_search(
            name="decoder_group",
            step=recurrent_decoder_step,
            input=[target_embeddings],
            bos_id=0,
            eos_id=1,
            beam_size=beam_size,
            max_length=100)
        return beam_gen


def memory_enhanced_seq2seq(encoder_input, decoder_input, decoder_target,
                            hidden_size, word_vec_dim, dict_size, is_generating,
                            beam_size):
    """
    Seq2Seq Model enhanced with external memory.

    The "external memory" refers to two types of memories.
    - Unbounded memory: i.e. attention mechanism in Seq2Seq.
    - Bounded memory: i.e. external memory in NTM.
    Both types of external memories can be implemented with
    ExternalMemory class, and are both exploited in this Seq2Seq model.

    Please refer to the function comments of memory_enhanced_decoder(...).

    For more details about external memory, please refer to
    `Neural Turing Machines <https://arxiv.org/abs/1410.5401>`_.

    For more details about this memory-enhanced Seq2Seq, please
    refer to `Memory-enhanced Decoder for Neural Machine Translation 
    <https://arxiv.org/abs/1606.02003>`_.
    """
    # encoder
    context_encodings, sequence_encoding = bidirectional_gru_encoder(
        input=encoder_input, size=hidden_size, word_vec_dim=word_vec_dim)
    # decoder
    return memory_enhanced_decoder(
        input=decoder_input,
        target=decoder_target,
        initial_state=sequence_encoding,
        source_context=context_encodings,
        size=hidden_size,
        word_vec_dim=word_vec_dim,
        dict_size=dict_size,
        is_generating=is_generating,
        beam_size=beam_size)


def parse_beam_search_result(beam_result, dictionary):
    """
    Beam search result parser.
    """
    sentence_list = []
    sentence = []
    for word in beam_result[1]:
        if word != -1:
            sentence.append(word)
        else:
            sentence_list.append(
                ' '.join([dictionary.get(word) for word in sentence[1:]]))
            sentence = []
    beam_probs = beam_result[0]
    beam_size = len(beam_probs[0])
    beam_sentences = [
        sentence_list[i:i + beam_size]
        for i in range(0, len(sentence_list), beam_size)
    ]
    return beam_probs, beam_sentences


def reader_append_wrapper(reader, append_tuple):
    """
    Data reader wrapper for appending extra data to exisiting reader.
    """

    def new_reader():
        for ins in reader():
            yield ins + append_tuple

    return new_reader


def train(num_passes):
    """
    For training.
    """
    # create network config
    source_words = paddle.layer.data(
        name="source_words",
        type=paddle.data_type.integer_value_sequence(dict_size))
    target_words = paddle.layer.data(
        name="target_words",
        type=paddle.data_type.integer_value_sequence(dict_size))
    target_next_words = paddle.layer.data(
        name='target_next_words',
        type=paddle.data_type.integer_value_sequence(dict_size))
    cost = memory_enhanced_seq2seq(
        encoder_input=source_words,
        decoder_input=target_words,
        decoder_target=target_next_words,
        hidden_size=hidden_size,
        word_vec_dim=word_vec_dim,
        dict_size=dict_size,
        is_generating=False,
        beam_size=beam_size)

    # create parameters and optimizer
    parameters = paddle.parameters.create(cost)
    optimizer = paddle.optimizer.Adam(
        learning_rate=5e-5,
        gradient_clipping_threshold=5,
        regularization=paddle.optimizer.L2Regularization(rate=8e-4))
    trainer = paddle.trainer.SGD(
        cost=cost, parameters=parameters, update_equation=optimizer)

    # create data readers
    feeding = {
        "source_words": 0,
        "target_words": 1,
        "target_next_words": 2,
        "bounded_memory_template": 3
    }
    train_append_reader = reader_append_wrapper(
        reader=paddle.dataset.wmt14.train(dict_size),
        append_tuple=([0] * memory_slot_num, ))
    train_batch_reader = paddle.batch(
        reader=paddle.reader.shuffle(reader=train_append_reader, buf_size=8192),
        batch_size=batch_size)
    test_append_reader = reader_append_wrapper(
        reader=paddle.dataset.wmt14.test(dict_size),
        append_tuple=([0] * memory_slot_num, ))
    test_batch_reader = paddle.batch(
        reader=paddle.reader.shuffle(reader=test_append_reader, buf_size=8192),
        batch_size=batch_size)

    # create event handler
    def event_handler(event):
        if isinstance(event, paddle.event.EndIteration):
            if event.batch_id % 10 == 0:
                print "Pass: %d, Batch: %d, TrainCost: %f, %s" % (
                    event.pass_id, event.batch_id, event.cost, event.metrics)
            else:
                sys.stdout.write('.')
                sys.stdout.flush()
        if isinstance(event, paddle.event.EndPass):
            result = trainer.test(reader=test_batch_reader, feeding=feeding)
            print "Pass: %d, TestCost: %f, %s" % (event.pass_id, event.cost,
                                                  result.metrics)
            with gzip.open("params.tar.gz", 'w') as f:
                parameters.to_tar(f)

    # run train
    trainer.train(
        reader=train_batch_reader,
        event_handler=event_handler,
        num_passes=num_passes,
        feeding=feeding)


def infer():
    """
    For inferencing.
    """
    # create network config
    source_words = paddle.layer.data(
        name="source_words",
        type=paddle.data_type.integer_value_sequence(dict_size))
    beam_gen = seq2seq(
        encoder_input=source_words,
        decoder_input=None,
        decoder_target=None,
        hidden_size=hidden_size,
        word_vec_dim=word_vec_dim,
        dict_size=dict_size,
        is_generating=True,
        beam_size=beam_size)

    # load parameters
    parameters = paddle.parameters.Parameters.from_tar(
        gzip.open("params.tar.gz"))

    # prepare infer data
    infer_data = []
    test_append_reader = reader_append_wrapper(
        reader=paddle.dataset.wmt14.test(dict_size),
        append_tuple=([0] * memory_slot_num, ))
    for i, item in enumerate(test_append_reader()):
        if i < infer_data_num:
            infer_data.append((item[0], item[3], ))

    # run inference
    beam_result = paddle.infer(
        output_layer=beam_gen,
        parameters=parameters,
        input=infer_data,
        field=['prob', 'id'])

    # parse beam result and print 
    source_dict, target_dict = paddle.dataset.wmt14.get_dict(dict_size)
    beam_probs, beam_sentences = parse_beam_search_result(beam_result,
                                                          target_dict)
    for i in xrange(infer_data_num):
        print "\n***************************************************\n"
        print "src:", ' '.join(
            [source_dict.get(word) for word in infer_data[i][0]]), "\n"
        for j in xrange(beam_size):
            print "prob = %f : %s" % (beam_probs[i][j], beam_sentences[i][j])


def main():
    paddle.init(use_gpu=False, trainer_count=1)
    train(num_passes=1)
    infer()


if __name__ == '__main__':
    main()
