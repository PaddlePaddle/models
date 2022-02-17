from paddle import fluid
from paddle.fluid import layers
from pytracking.libs.Fconv2d import FConv2D
from pytracking.libs.tensorlist import tensor_operation, TensorList
from paddle.fluid.framework import Variable as PTensor


@tensor_operation
def conv2d(input: PTensor,
           weight: PTensor,
           bias: PTensor=None,
           stride=1,
           padding=0,
           dilation=1,
           groups=1,
           mode=None):
    """Standard conv2d. Returns the input if weight=None."""

    if weight is None:
        return input

    ind = None
    if mode is not None:
        if padding != 0:
            raise ValueError('Cannot input both padding and mode.')
        if mode == 'same':
            padding = (weight.shape[2] // 2, weight.shape[3] // 2)
            if weight.shape[2] % 2 == 0 or weight.shape[3] % 2 == 0:
                ind = (slice(-1)
                       if weight.shape[2] % 2 == 0 else slice(None), slice(-1)
                       if weight.shape[3] % 2 == 0 else slice(None))
        elif mode == 'valid':
            padding = (0, 0)
        elif mode == 'full':
            padding = (weight.shape[2] - 1, weight.shape[3] - 1)
        else:
            raise ValueError('Unknown mode for padding.')

    assert bias is None
    out = FConv2D(
        input,
        weight,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups)
    if ind is None:
        return out
    return out[:, :, ind[0], ind[1]]


@tensor_operation
def conv1x1(input: PTensor, weight: PTensor):
    """Do a convolution with a 1x1 kernel weights. Implemented with matmul, which can be faster than using conv."""

    if weight is None:
        return input

    return FConv2D(input, weight)
