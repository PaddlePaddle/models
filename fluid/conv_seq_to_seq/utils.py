import numpy as np
import logging
import paddle.v2 as paddle
import paddle.fluid as fluid
import paddle.fluid.layers as pd
import paddle.fluid.nets as nets
import paddle.fluid.framework as framework
import paddle.fluid.debuger as debuger
from paddle.fluid.framework import framework_pb2

is_test = False


class Op:
    @staticmethod
    def reshape(x, shape):
        return pd.reshape(x=x, shape=shape)

    @staticmethod
    def transpose(x, offsets):
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

    @staticmethod
    def sigmoid(x):
        return pd.sigmoid(x=x)

    @staticmethod
    def softmax(x):
        '''
        Softmax the last dim.
        '''
        dims = get_dims(x)
        if len(dims) > 2:
            first_dim = np.prod(dims[:-1])
            last_dim = dims[-1]
            x = Op.reshape(x, [first_dim, last_dim])
        x = pd.softmax(x=x)
        if len(dims) > 2:
            x = Op.reshape(x, dims)
        return x


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
                 dropout=None):
        self.stride = stride
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels

        # for 1D conv
        self.atom = None  #conv2d(out_channels, filter_size, padding=padding)
        self.recover_proj = None

    def __call__(self, x):
        dims = get_dims(x)
        # format: batch_size, seq, word_vec
        # NOTE here is different from fairseq
        assert len(dims) == 3, "format shoud be BTC, get shape: %s" % str(dims)
        # conv2d's input should be format NCHW, which is
        # - batch_size, channels, height, width
        B, T, C = dims
        x = Op.transpose(x, [1, 2])
        x = Op.reshape(x, (B, C, 1, T))

        # here something bug, conv2d will change the original width and height by padding.
        # just use a trick to map to the original size
        N, C, H, W = get_tensor(x).dims

        #H = (H + 2 * padding - kernel_size) / stride + 1
        if self.atom is None:
            size = [H, W]
            self.kernel_size = [1, self.kernel_size]
            padding = [
                ((size[i] - 1) * self.stride + self.kernel_size[i] - size[i]) /
                2 for i in range(2)
            ]
            self.atom = conv2d(
                self.out_channels, self.kernel_size, padding=padding)

        x = self.atom(x)

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


def set_init_lod(data, lod, place):
    res = fluid.LoDTensor()
    res.set(data, place)
    res.set_lod(lod)
    return res


def pad(ids, max_len):
    if len(ids) < max_len:
        for i in xrange(max_len - len(ids)):
            ids.append(0)
    return ids


def to_tensor(data, max_len):
    datas = []
    for inst in data:
        inst = inst[:max_len]
        inst = pad(inst, max_len)
        datas.append(inst)
    return np.array(datas, dtype='int64')


def pad_batch_data(data, max_len):
    '''
    data: a batch of input
    '''
    for sent in data:
        if len(sent) < max_len:
            sent += [0 for i in xrange(max_len - len(sent))]
    return data
