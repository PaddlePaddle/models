source_dict_dim = 10000
target_dict_dim = 10000


class TrainConfig(object):
    source_dict_dim = source_dict_dim
    target_dict_dim = target_dict_dim
    use_gpu = False
    infer_only = False
    parallel = False
    batch_size = 16
    pass_num = 2
    learning_rate = 0.0002


class ModelConfig(object):
    dict_size = 10000
    embedding_dim = 512
    encoder_size = 512
    decoder_size = 512
    source_dict_dim = source_dict_dim
    target_dict_dim = target_dict_dim
    is_generating = False
    beam_size = 3
    max_length = 250
