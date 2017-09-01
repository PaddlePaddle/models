#!/usr/bin/env python
#coding=utf-8

__all__ = ["ModelConfig", "TrainerConfig"]


class ModelConfig(object):
    vocab_size = 104808
    embedding_dim = 300
    embedding_droprate = 0.3

    lstm_depth = 3
    lstm_hidden_dim = 300
    lstm_hidden_droprate = 0.3

    passage_indep_embedding_dim = 300
    passage_aligned_embedding_dim = 300

    beam_size = 32

    dict_path = "data/featurized/vocab.txt"
    pretrained_emb_path = "data/featurized/embeddings.npy"


class TrainerConfig(object):
    learning_rate = 1e-3
    l2_decay_rate = 5e-4
    gradient_clipping_threshold = 20

    data_dir = "data/featurized"
    save_dir = "models"

    use_gpu = False
    trainer_count = 1
    train_batch_size = trainer_count * 8

    epochs = 20

    # for debug print, if set to 0, no information will be printed.
    show_parameter_status_period = 0
    checkpoint_period = 100
    log_period = 1

    # this is used to resume training, this path can set to previously
    # trained model.
    init_model_path = None
