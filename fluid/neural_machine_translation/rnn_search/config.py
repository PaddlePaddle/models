class TrainConfig(object):
    dict_size = 10000
    use_gpu = True
    infer_only = False
    parallel = True
    batch_size = 16
    pass_num = 2
    learning_rate = 0.0002


class ModelConfig(object):
    dict_size = 10000
    embedding_dim = 512
    encoder_size = 512
    decoder_size = 512
    source_dict_dim = dict_size
    target_dict_dim = dict_size
    is_generating = False
    beam_size = 3
    max_length = 250
