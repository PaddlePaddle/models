import numpy as np
import paddle
import paddle.fluid as fluid
from paddle.fluid import dygraph
from paddle.fluid import layers
from paddle.fluid.framework import Variable
import cv2 as cv
PTensor = Variable


def broadcast_op(a, b, op='mul'):
    a_expand_factors = []
    b_expand_factors = []
    assert len(a.shape) == len(
        b.shape), 'a.shape = {} while b.shape = {}'.format(a.shape, b.shape)
    for a_s, b_s in zip(a.shape, b.shape):
        if a_s != b_s:
            if a_s == 1:
                a_expand_factors.append(b_s)
                b_expand_factors.append(1)
            elif b_s == 1:
                a_expand_factors.append(1)
                b_expand_factors.append(a_s)
            else:
                raise NotImplementedError
        else:
            a_expand_factors.append(1)
            b_expand_factors.append(1)
    if op == 'mul':
        op = layers.elementwise_mul
    elif op == 'add':
        op = layers.elementwise_add
    elif op == 'sub':
        op = layers.elementwise_sub
    elif op == 'div':
        op = layers.elementwise_div
    else:
        raise NotImplementedError
    return op(
        layers.expand(a, a_expand_factors), layers.expand(b, b_expand_factors))


def paddle_prod(x):
    prod = 1
    num_elems = x.shape[0]
    for idx in range(num_elems):
        prod *= x[idx]
    return prod


def n2p(x, dtype=None):
    if dtype is None:
        x = np.array(x)
        if x.dtype == np.float64:
            x = x.astype('float32')
    else:
        x = np.array(x, dtype=dtype)
    return dygraph.to_variable(x)


def p2n(x):
    return x.numpy()


def clone(x):
    v = dygraph.to_variable(x.numpy())
    v.stop_gradient = x.stop_gradient
    return v


def static_identity(x):
    x = layers.reshape(x, x.shape)
    return x


def static_clone(x):
    x1 = static_identity(x)
    x1.stop_gradient = True
    x2 = static_identity(x1)
    x2.stop_gradient = x.stop_gradient
    return x2


def detach(x):
    v = dygraph.to_variable(x.numpy())
    v.stop_gradient = True
    return v


def squeeze(input, axes):
    new_shape = []
    for i, s in enumerate(input.shape):
        if i in axes:
            assert s == 1
        else:
            new_shape.append(s)
    return layers.reshape(input, new_shape)


def unsqueeze(input, axes):
    new_shape = []
    for i, s in enumerate(input.shape):
        for a in axes:
            if i == a:
                new_shape.append(1)
        new_shape.append(s)
    return layers.reshape(input, new_shape)


def crop(x, crops):
    slices = []
    for c in crops:
        c1 = None if c[1] == 0 else -c[1]
        slices.append(slice(c[0], c1))
    return x[tuple(slices)]


def _padding(x, pads, mode='constant'):
    return_tensor = False
    if isinstance(x, PTensor):
        x = x.numpy()
        return_tensor = True

    assert len(pads) % 2 == 0
    pads = list(pads) + [0] * (len(x.shape) * 2 - len(pads))

    # convert to numpy pad format
    pads_np, pad_per_dim = [], []
    for i, p in enumerate(pads):
        if i % 2 == 0:
            pad_per_dim = [p]
        else:
            pad_per_dim.append(p)
            pads_np.insert(0, pad_per_dim)

    # handle negative pads (cropping)
    pads_np_pos, pads_np_neg = [], []
    for pad_per_dim in pads_np:
        pad_per_dim_pos, pad_per_dim_neg = [], []
        for p in pad_per_dim:
            if p < 0:
                pad_per_dim_pos.append(0)
                pad_per_dim_neg.append(-p)
            else:
                pad_per_dim_pos.append(p)
                pad_per_dim_neg.append(0)
        pads_np_pos.append(pad_per_dim_pos)
        pads_np_neg.append(pad_per_dim_neg)

    # cropping
    x = crop(x, pads_np_neg)

    # padding
    # if x is an image
    if len(x.shape) == 3 and pads_np_pos[-1][0] == 0 and pads_np_pos[-1][
            1] == 0:
        if mode == 'replicate':
            pad_mode = cv.BORDER_REPLICATE
        else:
            pad_mode = cv.BORDER_CONSTANT
        y1_pad, y2_pad = pads_np_pos[0]
        x1_pad, x2_pad = pads_np_pos[1]
        x = cv.copyMakeBorder(x, y1_pad, y2_pad, x1_pad, x2_pad, pad_mode)
    else:
        np_mode = 'edge' if mode == 'replicate' else 'constant'
        x = np.pad(x, pads_np_pos, mode=np_mode)

    out = dygraph.to_variable(x) if return_tensor else x
    return out


def mod(a, b):
    arg_list, new_arg_list = [a, b], []
    return_PTensor = False
    for x in arg_list:
        if isinstance(x, PTensor):
            x = p2n(x)
            return_PTensor = True
        new_arg_list.append(x)

    out = new_arg_list[0] % new_arg_list[1]
    return n2p(out) if return_PTensor else out


def floordiv(a, b):
    arg_list, new_arg_list = [a, b], []
    return_PTensor = False
    for x in arg_list:
        if isinstance(x, PTensor):
            x = p2n(x)
            return_PTensor = True
        new_arg_list.append(x)

    out = new_arg_list[0] // new_arg_list[1]
    return n2p(out) if return_PTensor else out


def stack_sum(x):
    return layers.reduce_sum(layers.stack(x))


def leaky_relu(x, alpha):
    return layers.relu(x) + alpha * (-1 * layers.relu(-1 * x))


def elu(x, alpha):
    return layers.relu(x) + alpha * (layers.exp(-1 * layers.relu(-1 * x)) - 1)


def dropout2d(input, prob, is_train=False):
    if not is_train:
        return input
    channels = input.shape[1]
    keep_prob = 1.0 - prob
    random_tensor = np.random.uniform(0, 1, [input.shape[0], channels, 1, 1]).astype(np.float32)
    random_tensor = keep_prob + dygraph.to_variable(random_tensor)
    binary_tensor = layers.floor(random_tensor)
    output = input / keep_prob * binary_tensor
    return output


def create_var_list(scope, var_lists, shape):
    vars = []
    for idx, v in enumerate(var_lists):
        name = "{}_{}".format(scope, idx)
        if shape is None:
            var = fluid.data(name, shape=v.shape)
        else:
            var = fluid.data(name, shape=shape + list(v[0].shape))
        var.stop_gradient = False
        vars.append(var)
    return vars

