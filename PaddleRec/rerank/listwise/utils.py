from paddle import fluid
import numpy as np

def fluid_sequence_pad(input, pad_value, maxlen=None):
    """
    args:
        input: (batch*seq_len, dim)
    returns:
        (batch, max_seq_len, dim)
    """
    pad_value = fluid.layers.cast(fluid.layers.assign(input=np.array([pad_value], 'float32')), input.dtype)
    input_padded, _ = fluid.layers.sequence_pad(input, pad_value, maxlen=maxlen)    # (batch, max_seq_len, 1), (batch, 1)
    # TODO, maxlen=300, used to solve issues: https://github.com/PaddlePaddle/Paddle/issues/14164
    return input_padded

def fluid_sequence_get_pos(lodtensor):
    """
    args:
        lodtensor: lod = [[0,4,7]]
    return:
        pos: lod = [[0,4,7]]
             data = [0,1,2,3,0,1,3]
             shape = [-1, 1]
    """
    lodtensor = fluid.layers.reduce_sum(lodtensor, dim=1, keep_dim=True) 
    assert lodtensor.shape == (-1, 1), (lodtensor.shape())
    ones = fluid.layers.cast(lodtensor * 0 + 1, 'float32')        # (batch*seq_len, 1)
    ones_padded = fluid_sequence_pad(ones, 0)               # (batch, max_seq_len, 1)
    ones_padded = fluid.layers.squeeze(ones_padded, [2])          # (batch, max_seq_len)
    seq_len = fluid.layers.cast(fluid.layers.reduce_sum(ones_padded, 1, keep_dim=True), 'int64')    # (batch, 1)
    seq_len = fluid.layers.squeeze(seq_len, [1])

    pos = fluid.layers.cast(fluid.layers.cumsum(ones_padded, 1, exclusive=True), 'int64')
    pos = fluid.layers.sequence_unpad(pos, seq_len)               # (batch*seq_len, 1)
    pos.stop_gradient = True
    return pos