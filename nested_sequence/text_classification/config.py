__all__ = ["TrainerConfig", "ModelConfig"]


class TrainerConfig(object):

    # whether to use GPU for training
    use_gpu = False
    # the number of threads used in one machine
    trainer_count = 1

    # train batch size
    batch_size = 32

    # number of pass during training
    num_passes = 10

    # learning rate for optimizer
    learning_rate = 1e-3

    # learning rate for L2Regularization
    l2_learning_rate = 1e-3

    # average_window for ModelAverage
    average_window = 0.5

    # buffer size for shuffling
    buf_size = 1000

    # log progress every log_period batches
    log_period = 100


class ModelConfig(object):

    # embedding vector dimension
    emb_size = 28

    # size of sentence vector representation and fc layer in cnn
    hidden_size = 128
