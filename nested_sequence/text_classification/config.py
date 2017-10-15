__all__ = ["TrainerConfig", "ModelConfig"]


class TrainerConfig(object):

    # Whether to use GPU in training or not.
    use_gpu = False
    # The number of computing threads.
    trainer_count = 1

    # The training batch size.
    batch_size = 32

    # The epoch number.
    num_passes = 10

    # The global learning rate.
    learning_rate = 1e-3

    # The decay rate for L2Regularization
    l2_learning_rate = 1e-3

    # This parameter is used for the averaged SGD.
    # About the average_window * (number of the processed batch) parameters
    # are used for average.
    # To be accurate, between average_window *(number of the processed batch)
    # and 2 * average_window * (number of the processed batch) parameters
    # are used for average.
    average_window = 0.5

    # The buffer size of the data reader.
    # The number of buffer size samples will be shuffled in training.
    buf_size = 1000

    # The parameter is used to control logging period.
    # Training log will be printed every log_period.
    log_period = 100


class ModelConfig(object):

    # The dimension of embedding vector.
    emb_size = 28

    # The hidden size of sentence vectors.
    hidden_size = 128
