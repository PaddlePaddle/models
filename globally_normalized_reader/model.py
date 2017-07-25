#!/usr/bin/env python
#coding=utf-8
import pdb

import paddle.v2 as paddle
from paddle.v2.layer import parse_network
import basic_modules
from config import ModelConfig

__all__ = ["GNR"]


def build_pretrained_embedding(name,
                               data_type,
                               vocab_size,
                               emb_dim,
                               emb_drop=0.):
    one_hot_input = paddle.layer.data(
        name=name, type=paddle.data_type.integer_value_sequence(vocab_size))
    return paddle.layer.embedding(
        input=one_hot_input,
        size=emb_dim,
        param_attr=paddle.attr.Param(
            name="GloveVectors", is_static=True),
        layer_attr=paddle.attr.ExtraLayerAttribute(drop_rate=emb_drop), )


def encode_question(input_embedding, config, prefix):
    lstm_final, lstm_outs = basic_modules.stacked_bidirectional_lstm(
        inputs=input_embedding,
        size=config.lstm_hidden_dim,
        depth=config.lstm_depth,
        drop_rate=config.lstm_hidden_droprate,
        prefix=prefix)

    # passage-independent embeddings
    candidates = paddle.layer.fc(input=lstm_outs,
                                 bias_attr=False,
                                 size=config.passage_indep_embedding_dim,
                                 act=paddle.activation.Linear())
    weights = paddle.layer.fc(input=lstm_outs,
                              size=1,
                              act=paddle.activation.SequenceSoftmax())
    weighted_candidates = paddle.layer.scaling(input=candidates, weight=weights)
    passage_indep_embedding = paddle.layer.pooling(
        input=weighted_candidates, pooling_type=paddle.pooling.Sum())
    return paddle.layer.concat(
        input=[lstm_final, passage_indep_embedding]), lstm_outs


def question_aligned_passage_embedding(question_lstm_outs, document_embeddings,
                                       config):
    def outer_sentence_step(document_embeddings, question_lstm_outs, config):
        '''
        in this recurrent_group, document_embeddings has scattered into sequence,
        '''

        def inner_word_step(word_embedding, question_lstm_outs,
                            question_outs_proj, config):
            '''
            in this recurrent_group, sentence embedding has scattered into word
            embeddings.
            '''
            doc_word_expand = paddle.layer.expand(
                input=word_embedding,
                expand_as=question_lstm_outs,
                expand_level=paddle.layer.ExpandLevel.FROM_NO_SEQUENCE)

            weights = paddle.layer.fc(
                input=[question_lstm_outs, doc_word_expand],
                size=1,
                act=paddle.activation.SequenceSoftmax())
            weighted_candidates = paddle.layer.scaling(
                input=question_outs_proj, weight=weights)
            return paddle.layer.pooling(
                input=weighted_candidates, pooling_type=paddle.pooling.Sum())

        question_outs_proj = paddle.layer.fc(
            input=question_lstm_outs,
            bias_attr=False,
            size=config.passage_aligned_embedding_dim)
        return paddle.layer.recurrent_group(
            input=[
                paddle.layer.SubsequenceInput(document_embeddings),
                paddle.layer.StaticInput(question_lstm_outs),
                paddle.layer.StaticInput(question_outs_proj),
                config,
            ],
            step=inner_word_step,
            name="iter_over_word")

    return paddle.layer.recurrent_group(
        input=[
            paddle.layer.SubsequenceInput(document_embeddings),
            paddle.layer.StaticInput(question_lstm_outs), config
        ],
        step=outer_sentence_step,
        name="iter_over_sen")


def encode_documents(input_embedding, same_as_question, question_vector,
                     question_lstm_outs, config, prefix):
    question_expanded = paddle.layer.expand(
        input=question_vector,
        expand_as=input_embedding,
        expand_level=paddle.layer.ExpandLevel.FROM_NO_SEQUENCE)
    question_aligned_embedding = question_aligned_passage_embedding(
        question_lstm_outs, input_embedding, config)
    return paddle.layer.concat(input=[
        input_embedding, question_expanded, same_as_question,
        question_aligned_embedding
    ])


def search_answer(doc_lstm_outs, sentence_idx, start_idx, end_idx, config):
    last_state_of_sentence = paddle.layer.last_seq(
        input=doc_lstm_outs, agg_level=paddle.layer.AggregateLevel.TO_SEQUENCE)

    # HERE do not use sequence softmax activition.
    sentence_scores = paddle.layer.fc(input=last_state_of_sentence,
                                      size=1,
                                      act=paddle.activation.Exp())
    topk_sentence_ids = paddle.layer.kmax_sequence_score(
        input=sentence_scores, beam_size=config.beam_size)
    topk_sen = paddle.layer.sub_nested_seq(
        input=last_state_of_sentence, selected_indices=topk_sentence_ids)

    # expand beam to search start positions on selected sentences
    start_pos_scores = paddle.layer.fc(input=topk_sen,
                                       size=1,
                                       act=paddle.activation.Exp())
    topk_start_pos_ids = paddle.layer.kmax_sequence_score(
        input=sentence_scores, beam_size=config.beam_size)
    topk_start_spans = paddle.layer.seq_slice(
        input=topk_sen, starts=topk_start_pos_ids, ends=None)

    # expand beam to search end positions on selected start spans
    _, end_span_embedding = basic_modules.stacked_bidirectional_lstm(
        inputs=topk_start_spans,
        size=config.lstm_hidden_dim,
        depth=config.lstm_depth,
        drop_rate=config.lstm_hidden_droprate,
        prefix="__end_span_embeddings__")
    end_pos_scores = paddle.layer.fc(input=end_span_embedding,
                                     size=1,
                                     act=paddle.activation.Exp())
    topk_end_pos_ids = paddle.layer.kmax_sequence_score(
        input=end_pos_scores, beam_size=config.beam_size)
    cost = paddle.layer.cross_entropy_over_beam(
        input=[
            sentence_scores, topk_sentence_ids, start_pos_scores,
            topk_start_pos_ids, end_pos_scores, topk_end_pos_ids
        ],
        label=[sentence_idx, start_idx, end_idx])
    return cost


def GNR(config):
    # encoding question words
    question_embeddings = build_pretrained_embedding(
        "question", paddle.data_type.integer_value_sequence, config.vocab_size,
        config.embedding_dim, config.embedding_droprate)
    question_vector, question_lstm_outs = encode_question(
        input_embedding=question_embeddings, config=config, prefix="__ques")

    # encoding document words
    document_embeddings = build_pretrained_embedding(
        "documents", paddle.data_type.integer_value_sub_sequence,
        config.vocab_size, config.embedding_dim, config.embedding_droprate)
    same_as_question = paddle.layer.data(
        name="same_as_question",
        type=paddle.data_type.integer_value_sub_sequence(2))
    document_words_ecoding = encode_documents(
        input_embedding=document_embeddings,
        question_vector=question_vector,
        question_lstm_outs=question_lstm_outs,
        same_as_question=same_as_question,
        config=config,
        prefix="__doc")

    doc_lstm_outs = basic_modules.stacked_bi_lstm_by_nested_seq(
        input_layer=document_words_ecoding,
        hidden_dim=config.lstm_hidden_dim,
        depth=config.lstm_depth,
        prefix="__doc_lstm")

    # define labels
    sentence_idx = paddle.layer.data(
        name="sen_idx", type=paddle.data_type.integer_value(1))
    start_idx = paddle.layer.data(
        name="start_idx", type=paddle.data_type.integer_value(1))
    end_idx = paddle.layer.data(
        name="end_idx", type=paddle.data_type.integer_value(1))
    return search_answer(doc_lstm_outs, sentence_idx, start_idx, end_idx,
                         config)


if __name__ == "__main__":
    print(parse_network(GNR(ModelConfig)))
