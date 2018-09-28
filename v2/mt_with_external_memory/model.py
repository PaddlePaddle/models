"""
    Contains model configuration for external-memory-enhanced seq2seq.

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
from external_memory import ExternalMemory


def bidirectional_gru_encoder(input, size, word_vec_dim):
    """Bidirectional GRU encoder.

    :params size: Hidden cell number in decoder rnn.
    :type size: int
    :params word_vec_dim: Word embedding size.
    :type word_vec_dim: int
    :return: Tuple of 1. concatenated forward and backward hidden sequence.
             2. last state of backward rnn.
    :rtype: tuple of LayerOutput
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
    """GRU sequence decoder enhanced with external memory.

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

    :params input: Decoder input.
    :type input: LayerOutput
    :params target: Decoder target.
    :type target: LayerOutput
    :params initial_state: Initial hidden state.
    :type initial_state: LayerOutput
    :params source_context: Group of context hidden states for each token in the
                            source sentence, for attention mechanisim.
    :type source_context: LayerOutput
    :params size: Hidden cell number in decoder rnn.
    :type size: int
    :params word_vec_dim: Word embedding size.
    :type word_vec_dim: int
    :param dict_size: Vocabulary size.
    :type dict_size: int
    :params is_generating: Whether for beam search inferencing (True) or
                           for training (False).
    :type is_generating: bool
    :params beam_size: Beam search width.
    :type beam_size: int
    :return: Cost layer if is_generating=False; Beam search layer if
             is_generating = True.
    :rtype: LayerOutput
    """
    # prepare initial bounded and unbounded memory
    bounded_memory_slot_init = paddle.layer.fc(input=paddle.layer.pooling(
        input=source_context, pooling_type=paddle.pooling.Avg()),
                                               size=size,
                                               act=paddle.activation.Sigmoid())
    bounded_memory_perturbation = paddle.layer.data(
        name='bounded_memory_perturbation',
        type=paddle.data_type.dense_vector_sequence(size))
    bounded_memory_init = paddle.layer.addto(
        input=[
            paddle.layer.expand(
                input=bounded_memory_slot_init,
                expand_as=bounded_memory_perturbation),
            bounded_memory_perturbation
        ],
        act=paddle.activation.Linear())
    bounded_memory_weight_init = paddle.layer.slope_intercept(
        input=paddle.layer.fc(input=bounded_memory_init, size=1),
        slope=0.0,
        intercept=0.0)
    unbounded_memory_init = source_context
    unbounded_memory_weight_init = paddle.layer.slope_intercept(
        input=paddle.layer.fc(input=unbounded_memory_init, size=1),
        slope=0.0,
        intercept=0.0)

    # prepare step function for reccurent group
    def recurrent_decoder_step(cur_embedding):
        # create hidden state, bounded and unbounded memory.
        state = paddle.layer.memory(
            name="gru_decoder", size=size, boot_layer=initial_state)
        bounded_memory = ExternalMemory(
            name="bounded_memory",
            mem_slot_size=size,
            boot_layer=bounded_memory_init,
            initial_weight=bounded_memory_weight_init,
            readonly=False,
            enable_interpolation=True)
        unbounded_memory = ExternalMemory(
            name="unbounded_memory",
            mem_slot_size=size * 2,
            boot_layer=unbounded_memory_init,
            initial_weight=unbounded_memory_weight_init,
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
        return paddle.layer.fc(input=[gru_output, context, cur_embedding],
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
        target_embeddings = paddle.layer.GeneratedInput(
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
    """Seq2Seq Model enhanced with external memory.

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

    :params encoder_input: Encoder input.
    :type encoder_input: LayerOutput
    :params decoder_input: Decoder input.
    :type decoder_input: LayerOutput
    :params decoder_target: Decoder target.
    :type decoder_target: LayerOutput
    :params hidden_size: Hidden cell number, both in encoder and decoder rnn.
    :type hidden_size: int
    :params word_vec_dim: Word embedding size.
    :type word_vec_dim: int
    :param dict_size: Vocabulary size.
    :type dict_size: int
    :params is_generating: Whether for beam search inferencing (True) or
                           for training (False).
    :type is_generating: bool
    :params beam_size: Beam search width.
    :type beam_size: int
    :return: Cost layer if is_generating=False; Beam search layer if
             is_generating = True.
    :rtype: LayerOutput
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
