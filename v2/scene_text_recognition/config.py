__all__ = ["TrainerConfig", "ModelConfig"]


class TrainerConfig(object):

    # Whether to use GPU in training or not.
    use_gpu = True

    # The number of computing threads.
    trainer_count = 1

    # The training batch size.
    batch_size = 10

    # The epoch number.
    num_passes = 10

    # Parameter updates momentum.
    momentum = 0

    # The shape of images.
    image_shape = (173, 46)

    # The buffer size of the data reader.
    # The number of buffer size samples will be shuffled in training.
    buf_size = 1000

    # The parameter is used to control logging period.
    # Training log will be printed every log_period.
    log_period = 50


class ModelConfig(object):

    # Number of the filters for convolution group.
    filter_num = 8

    # Use batch normalization or not in image convolution group.
    with_bn = True

    # The number of channels for block expand layer.
    num_channels = 128

    # The parameter stride_x  in block expand layer.
    stride_x = 1

    # The parameter stride_y  in block expand layer.
    stride_y = 1

    # The parameter block_x  in block expand layer.
    block_x = 1

    # The parameter block_y  in block expand layer.
    block_y = 11

    # The hidden size for gru.
    hidden_size = num_channels

    # Use norm_by_times or not in warp ctc layer.
    norm_by_times = True

    # The list for number of filter in image convolution group layer.
    filter_num_list = [16, 32, 64, 128]

    # The parameter conv_padding in image convolution group layer.
    conv_padding = 1

    # The parameter conv_filter_size in image convolution group layer.
    conv_filter_size = 3

    # The parameter pool_size in image convolution group layer.
    pool_size = 2

    # The parameter pool_stride in image convolution group layer.
    pool_stride = 2
