import numpy as np
import logging
import paddle.v2 as paddle
import paddle.fluid as fluid
import paddle.fluid.layers as pd
import paddle.fluid.nets as nets
import paddle.fluid.framework as framework
import paddle.fluid.debuger as debuger
from paddle.fluid.framework import framework_pb2
'''
Reference the paper: Convolutional Sequence to Sequence Learning



Some difference with the pytorch implementation:

- conv2d padding will change output, need a special strategy
'''

is_test = False


class Op:
    @staticmethod
    def reshape(x, shape):
        return pd.reshape(x=x, shape=shape)

    @staticmethod
    def transpose(x, *offsets):
        ndims = len(list(get_tensor(x).dims))
        dims = [i for i in range(ndims)]
        assert len(dims) >= 2
        assert len(offsets) == 2
        l = offsets[0]
        r = offsets[1]
        dims[l], dims[r] = dims[r], dims[l]
        return pd.transpose(x, dims)

    @staticmethod
    def dropout(x, prob, is_test=is_test):
        return pd.dropout(x, dropout_prob=prob, is_test=is_test)


class Embedding:
    def __init__(self, num_embeddings, embed_dim, padding_idx=-1):
        self.atom = Atom(
            pd.embedding,
            "embedding",
            size=[num_embeddings, embed_dim],
            dtype="float32",
            padding_idx=padding_idx)

    def __call__(self, x):
        # need some reshape here
        # x should be a Tensor, not LoDTensor or others
        dims = list(get_tensor(x).dims)
        # pd.embedding can only accept 2D tensor, need a reshape
        if len(dims) > 2:
            dims1 = [np.prod(dims), 1]
            x = Op.reshape(x, dims1)
        x = self.atom(x)
        # restore to original shape
        if len(dims) > 2:
            dims[-1] = -1
            x = Op.reshape(x, dims)
        return x


class Linear:
    def __init__(self, size, dropout=None):
        assert size > 0
        self.atom = Atom(pd.fc, "fc", size=size, dropout=dropout)

    def __call__(self, x):
        # pd.fc take dims[1:] as projecton's input, need reshape to avoid that
        dims = list(get_tensor(x).dims)
        assert len(dims) > 1
        if len(dims) > 2:
            new_dims = (np.prod(dims[:-1]), dims[-1])
            x = Op.reshape(x, new_dims)
        x = self.atom(x)

        if len(dims) > 2:
            dims[-1] = -1
            x = Op.reshape(x, dims)
        return x


# def Linear(size, dropout=None):
#     return Atom(pd.fc, "fc", size=size, dropout=dropout)


def conv2d(num_filters, filter_size, padding):
    return Atom(
        pd.conv2d,
        "conv",
        num_filters=num_filters,
        filter_size=filter_size,
        padding=padding)


class Conv1D:
    '''
    a convolution for sequence.
    '''

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dropout=None):
        # for 1D conv
        filter_size = [1, kernel_size]
        padding = [1, padding]
        self.atom = conv2d(out_channels, filter_size, padding=padding)
        self.recover_proj = None

    def __call__(self, x):
        dims = get_dims(x)
        # format: batch_size, seq, word_vec
        # NOTE here is different from fairseq
        assert len(dims) == 3, "format shoud be BTC, get shape: %s" % str(dims)
        # conv2d's input should be format NCHW, which is
        # - batch_size, channels, height, width
        B, T, C = dims
        x = Op.transpose(x, 1, 2)
        x = Op.reshape(x, (B, C, 1, T))

        x = self.atom(x)

        # here something bug, conv2d will change the original width and height by padding.
        # just use a fc to map to the original size, need to change latter
        N, C, H, W = get_tensor(x).dims
        if H != 1 or W != T:
            if self.recover_proj is None:
                self.recover_proj = Linear(1 * T)
            x = Op.reshape(x, (N * C, -1))
            x = self.recover_proj(x)
        x = Op.reshape(x, (B, T, -1))
        return x


class Atom(object):
    counter = 0

    def __init__(self, op, prefix="atom", **kwargs):
        self.op = op
        assert 'param_attr' not in kwargs
        self.kwargs = kwargs
        self.name = '%s-%d' % (prefix, Atom.counter)
        Atom.counter += 1

    def __call__(self, x):
        if 'dropout' in self.kwargs:
            dropout = self.kwargs['dropout']
            del self.kwargs['dropout']
            if dropout is not None:
                x = self.op(
                    x,
                    param_attr=fluid.ParamAttr(name=self.name),
                    **self.kwargs)
                return Op.dropout(x, dropout, is_test=is_test)

        return self.op(
            x, param_attr=fluid.ParamAttr(name=self.name), **self.kwargs)


def get_var_desc(var):
    protostr = var.desc.serialize_to_string()
    proto = framework_pb2.VarDesc.FromString(str(protostr))
    return proto


def get_tensor(var):
    proto = get_var_desc(var)
    return proto.type.lod_tensor.tensor


def get_dims(var):
    return get_tensor(var).dims


def log(*args):
    res = ""
    for a in args:
        res += " " + str(a)
    logging.warning(res)
