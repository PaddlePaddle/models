class TrainConfig(object):
    use_gpu = False
    infer_only = False
    parallel = False
    batch_size = 16
    pass_num = 5
    learning_rate = 0.0002
    buf_size = 100000


class ModelConfig(object):
    embedding_dim = 512
    encoder_size = 512
    decoder_size = 512
    source_dict_dim = 10000
    target_dict_dim = 10000
    is_generating = False
    beam_size = 3
    max_length = 250
