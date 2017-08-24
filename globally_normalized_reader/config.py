#!/usr/bin/env python
#coding=utf-8

__all__ = ["ModelConfig"]


class ModelConfig(object):
    beam_size = 3
    vocab_size = 104808
    embedding_dim = 300
    embedding_droprate = 0.3

    lstm_depth = 3
    lstm_hidden_dim = 300
    lstm_hidden_droprate = 0.3

    passage_indep_embedding_dim = 300
    passage_aligned_embedding_dim = 128

    beam_size = 32

    dict_path = "data/featurized/vocab.txt"
    pretrained_emb_path = "data/featurized/embeddings.npy"


class TrainerConfig(object):
    learning_rate = 1e-3
    data_dir = "data/featurized"
    save_dir = "models"

    batch_size = 12 * 4

    epochs = 100
