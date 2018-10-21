#!/usr/bin/env python
#coding=utf-8

import paddle.v2 as paddle
from paddle.v2.layer import parse_network
import basic_modules
from config import ModelConfig

__all__ = ["GNR"]


def build_pretrained_embedding(name, data_type, emb_dim, emb_drop=0.):
    """create word a embedding layer which loads pre-trained embeddings.

    Arguments:
        - name:       The name of the data layer which accepts one-hot input.
        - data_type:  PaddlePaddle's data type for data layer.
        - emb_dim:    The path to the data files.
    """

    return paddle.layer.embedding(
        input=paddle.layer.data(
            name=name, type=data_type),
        size=emb_dim,
        param_attr=paddle.attr.Param(
            name="GloveVectors", is_static=True),
        layer_attr=paddle.attr.ExtraLayerAttribute(drop_rate=emb_drop), )


def encode_question(input_embedding,
                    lstm_hidden_dim,
                    depth,
                    passage_indep_embedding_dim,
                    prefix=""):
    """build question encoding by using bidirectional LSTM.

    Each question word is encoded by runing a stack of bidirectional LSTM over
    word embedding in question, producing hidden states. The hidden states are
    used to compute a passage-independent question embedding.

    The final question encoding is constructed by concatenating the final
    hidden states of the forward and backward LSTMs and the passage-independent
    embedding.

    Arguments:
        - input_embedding:    The question word embeddings.
        - lstm_hidden_dim:  The dimension of bi-directional LSTM.
        - depth:  The depth of stacked bi-directional LSTM.
        - passage_indep_embedding_dim:  The dimension of passage-independent
                                        embedding.
        - prefix:    A string which will be appended to name of each layer
                     created in this function. Each layer in a network should
                     has a unique name. The prefix makes this fucntion can be
                     called multiple times.
    """
    # stacked bi-directional LSTM to process question embeddings.
    lstm_final, lstm_outs = basic_modules.stacked_bidirectional_lstm(
        input_embedding, lstm_hidden_dim, depth, 0., prefix)

    # compute passage-independent embeddings.
    candidates = paddle.layer.fc(input=lstm_outs,
                                 bias_attr=False,
                                 size=passage_indep_embedding_dim,
                                 act=paddle.activation.Linear())
    weights = paddle.layer.fc(input=lstm_outs,
                              size=1,
                              bias_attr=False,
                              act=paddle.activation.SequenceSoftmax())
    weighted_candidates = paddle.layer.scaling(input=candidates, weight=weights)
    passage_indep_embedding = paddle.layer.pooling(
        input=weighted_candidates, pooling_type=paddle.pooling.Sum())

    return paddle.layer.concat(
        input=[lstm_final, passage_indep_embedding]), lstm_outs


def question_aligned_passage_embedding(question_lstm_outs, document_embeddings,
                                       passage_aligned_embedding_dim):
    """create question aligned passage embedding.

    Arguments:
        - question_lstm_outs:    The dimension of output of LSTM that process
                                 question word embedding.
        - document_embeddings:   The document embeddings.
        - passage_aligned_embedding_dim:    The dimension of passage aligned
                                            embedding.
    """

    def outer_sentence_step(document_embeddings, question_lstm_outs,
                            passage_aligned_embedding_dim):
        """step function for PaddlePaddle's recurrent_group.

        In this function, the original input document_embeddings are scattered
        from nested sequence into sequence by recurrent_group in PaddlePaddle.
        The step function iterates over each sentence in the document.

        Arguments:
            - document_embeddings:   The word embeddings of the document.
            - question_lstm_outs:    The dimension of output of LSTM that
                                     process question word embedding.
            - passage_aligned_embedding_dim:    The dimension of passage aligned
                                                embedding.
        """

        def inner_word_step(word_embedding, question_lstm_outs,
                            question_outs_proj, passage_aligned_embedding_dim):
            """
            In this recurrent_group, sentence embedding has been scattered into
            word embeddings. The step function iterates over each word in one
            sentence in the document.

            Arguments:
                - word_embedding: The word embeddings of documents.
                - question_lstm_outs:    The dimension of output of LSTM that
                                         process question word embedding.
                - question_outs_proj:    The projection of question_lstm_outs
                                         into a new hidden space.
                - passage_aligned_embedding_dim:    The dimension of passage
                                                    aligned embedding.
            """

            doc_word_expand = paddle.layer.expand(
                input=word_embedding,
                expand_as=question_lstm_outs,
                expand_level=paddle.layer.ExpandLevel.FROM_NO_SEQUENCE)

            weights = paddle.layer.fc(
                input=[question_lstm_outs, doc_word_expand],
                size=1,
                bias_attr=False,
                act=paddle.activation.SequenceSoftmax())
            weighted_candidates = paddle.layer.scaling(
                input=question_outs_proj, weight=weights)
            return paddle.layer.pooling(
                input=weighted_candidates, pooling_type=paddle.pooling.Sum())

        question_outs_proj = paddle.layer.fc(input=question_lstm_outs,
                                             bias_attr=False,
                                             size=passage_aligned_embedding_dim)
        return paddle.layer.recurrent_group(
            input=[
                paddle.layer.SubsequenceInput(document_embeddings),
                paddle.layer.StaticInput(question_lstm_outs),
                paddle.layer.StaticInput(question_outs_proj),
                passage_aligned_embedding_dim,
            ],
            step=inner_word_step,
            name="iter_over_word")

    return paddle.layer.recurrent_group(
        input=[
            paddle.layer.SubsequenceInput(document_embeddings),
            paddle.layer.StaticInput(question_lstm_outs),
            passage_aligned_embedding_dim
        ],
        step=outer_sentence_step,
        name="iter_over_sen")


def encode_documents(input_embedding, same_as_question, question_vector,
                     question_lstm_outs, passage_indep_embedding_dim, prefix):
    """Build the final question-aware document embeddings.

    Each word in the document is represented as concatenation of its word
    vector, the question vector, boolean features indicating if a word appers
    in the question or is repeated, and a question aligned embedding.


    Arguments:
        - input_embedding:   The word embeddings of the document.
        - same_as_question:  The boolean features indicating if a word appears
                             in the question or is repeated.
        - question_lstm_outs: The final question encoding.
        - passage_indep_embedding_dim:  The dimension of passage independent
                                        embedding.
        - prefix:    The prefix which will be appended to name of each layer in
                     This function.
    """

    question_expanded = paddle.layer.expand(
        input=question_vector,
        expand_as=input_embedding,
        expand_level=paddle.layer.ExpandLevel.FROM_NO_SEQUENCE)
    question_aligned_embedding = question_aligned_passage_embedding(
        question_lstm_outs, input_embedding, passage_indep_embedding_dim)
    return paddle.layer.concat(input=[
        input_embedding, question_expanded, same_as_question,
        question_aligned_embedding
    ])


def search_answer(doc_lstm_outs, sentence_idx, start_idx, end_idx, config,
                  is_infer):
    """Search the answer from the document.

    The search process for this layer begins with searching a target sequence
    from a nested sequence by using paddle.layer.kmax_seq_score and
    paddle.layer.sub_nested_seq_layer. In the first search step, top beam size
    sequences with highest scores, indices of these top k sequences in the
    original nested sequence, and the ground truth (also called gold)
    altogether (a triple) make up of the first beam.

    Then, start and end positions are searched. In these searches, top k
    positions with highest scores are selected, and then sequence, starting
    from the selected starts till ends of the sequences are taken to search
    next by using paddle.layer.seq_slice.

    Finally, the layer paddle.layer.cross_entropy_over_beam takes all the beam
    expansions which contain several candidate targets found along the
    three-step search. cross_entropy_over_beam calculates cross entropy over
    the expanded beams which all the candidates in the beam as the normalized
    factor.

    Note that, if gold falls off the beam at search step t, then the cost is
    calculated over the beam at step t.

    Arguments:
        - doc_lstm_outs:    The output of LSTM that process each document words.
        - sentence_idx:    Ground-truth indicating sentence index of the answer
                           in the document.
        - start_idx:    Ground-truth indicating start span index of the answer
                        in the sentence.
        - end_idx:    Ground-truth indicating end span index of the answer
                      in the sentence.
        - is_infer:    The boolean parameter indicating inferring or training.
    """

    last_state_of_sentence = paddle.layer.last_seq(
        input=doc_lstm_outs, agg_level=paddle.layer.AggregateLevel.TO_SEQUENCE)
    sentence_scores = paddle.layer.fc(input=last_state_of_sentence,
                                      size=1,
                                      bias_attr=False,
                                      act=paddle.activation.Linear())
    topk_sentence_ids = paddle.layer.kmax_seq_score(
        input=sentence_scores, beam_size=config.beam_size)
    topk_sen = paddle.layer.sub_nested_seq(
        input=doc_lstm_outs, selected_indices=topk_sentence_ids)

    # expand beam to search start positions on selected sentences
    start_pos_scores = paddle.layer.fc(
        input=topk_sen,
        size=1,
        layer_attr=paddle.attr.ExtraLayerAttribute(
            error_clipping_threshold=5.0),
        bias_attr=False,
        act=paddle.activation.Linear())
    topk_start_pos_ids = paddle.layer.kmax_seq_score(
        input=start_pos_scores, beam_size=config.beam_size)
    topk_start_spans = paddle.layer.seq_slice(
        input=topk_sen, starts=topk_start_pos_ids, ends=None)

    # expand beam to search end positions on selected start spans
    _, end_span_embedding = basic_modules.stacked_bidirectional_lstm(
        topk_start_spans, config.lstm_hidden_dim, config.lstm_depth,
        config.lstm_hidden_droprate, "__end_span_embeddings__")
    end_pos_scores = paddle.layer.fc(input=end_span_embedding,
                                     size=1,
                                     bias_attr=False,
                                     act=paddle.activation.Linear())
    topk_end_pos_ids = paddle.layer.kmax_seq_score(
        input=end_pos_scores, beam_size=config.beam_size)

    if is_infer:
        return [
            sentence_scores, topk_sentence_ids, start_pos_scores,
            topk_start_pos_ids, end_pos_scores, topk_end_pos_ids
        ]
    else:
        return paddle.layer.cross_entropy_over_beam(input=[
            paddle.layer.BeamInput(sentence_scores, topk_sentence_ids,
                                   sentence_idx),
            paddle.layer.BeamInput(start_pos_scores, topk_start_pos_ids,
                                   start_idx),
            paddle.layer.BeamInput(end_pos_scores, topk_end_pos_ids, end_idx)
        ])


def GNR(config, is_infer=False):
    """Build the globally normalized reader model.

    Arguments:
        - config:    The model configuration.
        - is_infer:    The boolean parameter indicating inferring or training.
    """

    # encode question words
    question_embeddings = build_pretrained_embedding(
        "question",
        paddle.data_type.integer_value_sequence(config.vocab_size),
        config.embedding_dim, config.embedding_droprate)
    question_vector, question_lstm_outs = encode_question(
        question_embeddings, config.lstm_hidden_dim, config.lstm_depth,
        config.passage_indep_embedding_dim, "__ques")

    # encode document words
    document_embeddings = build_pretrained_embedding(
        "documents",
        paddle.data_type.integer_value_sub_sequence(config.vocab_size),
        config.embedding_dim, config.embedding_droprate)
    same_as_question = paddle.layer.data(
        name="same_as_question",
        type=paddle.data_type.dense_vector_sub_sequence(1))

    document_words_ecoding = encode_documents(
        document_embeddings, same_as_question, question_vector,
        question_lstm_outs, config.passage_indep_embedding_dim, "__doc")

    doc_lstm_outs = basic_modules.stacked_bidirectional_lstm_by_nested_seq(
        document_words_ecoding, config.lstm_depth, config.lstm_hidden_dim,
        "__doc_lstm")

    # search the answer.
    sentence_idx = paddle.layer.data(
        name="sen_idx", type=paddle.data_type.integer_value(1))
    start_idx = paddle.layer.data(
        name="start_idx", type=paddle.data_type.integer_value(1))
    end_idx = paddle.layer.data(
        name="end_idx", type=paddle.data_type.integer_value(1))
    return search_answer(doc_lstm_outs, sentence_idx, start_idx, end_idx,
                         config, is_infer)


if __name__ == "__main__":
    print(parse_network(GNR(ModelConfig)))
