import re
import numbers
from collections import namedtuple

import custom_layers
from .shapes import *

LAYER_DESCRIPTORS = {

    # Caffe Types
    'AbsVal': shape_identity,
    'Accuracy': shape_scalar,
    'ArgMax': shape_not_implemented,
    'BatchNorm': shape_identity,
    'BNLL': shape_not_implemented,
    'Concat': shape_concat,
    'ContrastiveLoss': shape_scalar,
    'Convolution': shape_convolution,
    'Deconvolution': shape_deconvolution,
    'Data': shape_data,
    'Dropout': shape_identity,
    'DummyData': shape_data,
    'Crop': shape_crop,
    'EuclideanLoss': shape_scalar,
    'Eltwise': shape_identity,
    'Exp': shape_identity,
    'Flatten': shape_not_implemented,
    'HDF5Data': shape_data,
    'HDF5Output': shape_identity,
    'HingeLoss': shape_scalar,
    'Im2col': shape_not_implemented,
    'ImageData': shape_data,
    'InfogainLoss': shape_scalar,
    'InnerProduct': shape_inner_product,
    'Input': shape_data,
    'LRN': shape_identity,
    'MemoryData': shape_mem_data,
    'MultinomialLogisticLoss': shape_scalar,
    'MVN': shape_not_implemented,
    'Pooling': shape_pool,
    'Power': shape_power,
    'ReLU': shape_identity,
    'PReLU': shape_identity,
    'Scale': shape_identity,
    'Sigmoid': shape_identity,
    'SigmoidCrossEntropyLoss': shape_scalar,
    'Silence': shape_not_implemented,
    'Softmax': shape_identity,
    'SoftmaxWithLoss': shape_scalar,
    'Split': shape_not_implemented,
    'Slice': shape_not_implemented,
    'TanH': shape_identity,
    'WindowData': shape_not_implemented,
    'Threshold': shape_identity,
}

# layer types in 'V1LayerParameter'
# (v1layertype name, enum value, mapped to layer type)
v1_layertypes = [
    ('ABSVAL', 35),
    ('ACCURACY', 1),
    ('ARGMAX', 30),
    ('BNLL', 2),
    ('CONCAT', 3),
    ('CONVOLUTION', 4),
    ('DATA', 5),
    ('DECONVOLUTION', 39),
    ('DROPOUT', 6),
    ('ELTWISE', 25),
    ('EXP', 38),
    ('FLATTEN', 8),
    ('IM2COL', 11),
    ('INNERPRODUCT', 14),
    ('LRN', 15),
    ('MEMORYDATA', 29),
    ('MULTINOMIALLOGISTICLOSS', 16),
    ('MVN', 34),
    ('POOLING', 17),
    ('POWER', 26),
    ('RELU', 18),
    ('SIGMOID', 19),
    ('SIGMOIDCROSSENTROPYLOSS', 27),
    ('SILENCE', 36),
    ('SOFTMAX', 20),
    ('SPLIT', 22),
    ('SLICE', 33),
    ('TANH', 23),
    ('WINDOWDATA', 24),
    ('THRESHOLD', 31),
]

LAYER_TYPES = LAYER_DESCRIPTORS.keys()
LayerType = type('LayerType', (), {t: t for t in LAYER_TYPES})

#map the layer name in V1 to standard name
V1_LAYER_MAP = {'_not_init_': True}


def get_v1_layer_map():
    global V1_LAYER_MAP
    if '_not_init_' not in V1_LAYER_MAP:
        return V1_LAYER_MAP
    else:
        del V1_LAYER_MAP['_not_init_']

    name2layer = {}
    for n in LAYER_TYPES:
        name2layer[n.upper()] = n

    for l in v1_layertypes:
        n, v = l
        if n in name2layer and v not in V1_LAYER_MAP:
            V1_LAYER_MAP[v] = name2layer[n]
        else:
            raise KaffeError('not found v1 layer type %s' % n)
    return V1_LAYER_MAP


class NodeKind(LayerType):
    @staticmethod
    def map_raw_kind(kind):
        if custom_layers.has_layer(kind):
            return kind

        if kind in LAYER_TYPES:
            return kind

        v1_layers = get_v1_layer_map()
        if kind in v1_layers:
            return v1_layers[kind]
        else:
            return None

    @staticmethod
    def compute_output_shape(node):
        if custom_layers.has_layer(node.kind):
            return custom_layers.compute_output_shape(node.kind, node)

        try:
            val = LAYER_DESCRIPTORS[node.kind](node)
            return val
        except NotImplementedError:
            raise KaffeError(
                'Output shape computation not implemented for type: %s' %
                node.kind)


class NodeDispatchError(KaffeError):
    pass


class NodeDispatch(object):
    @staticmethod
    def get_handler_name(node_kind):
        if len(node_kind) <= 6:
            # A catch-all for things like ReLU and tanh
            return node_kind.lower()
        # Convert from CamelCase to under_scored
        name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', node_kind)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()

    def get_handler(self, node_kind, prefix):
        if custom_layers.has_layer(node_kind):
            return getattr(self, 'map_custom')

        name = self.get_handler_name(node_kind)
        name = '_'.join((prefix, name))
        try:
            return getattr(self, name)
        except AttributeError:
            raise NodeDispatchError(
                'No handler found for node kind: %s (expected: %s)' %
                (node_kind, name))


class LayerAdapter(object):
    def __init__(self, layer, kind):
        self.layer = layer
        self.kind = kind

    @property
    def parameters(self):
        name = NodeDispatch.get_handler_name(self.kind)
        if self.kind.lower() == "normalize":
            name = "norm"
        elif self.kind.lower() == "deconvolution":
            name = "convolution"

        name = '_'.join((name, 'param'))
        try:
            return getattr(self.layer, name)
        except AttributeError:
            print(dir(self.layer))
            raise NodeDispatchError(
                'Caffe parameters not found attr[%s] for layer kind[%s]' %
                (name, self.kind))

    @staticmethod
    def get_kernel_value(scalar, repeated, idx, default=None):
        if scalar:
            return scalar
        if repeated:
            if isinstance(repeated, numbers.Number):
                return repeated
            if len(repeated) == 1:
                # Same value applies to all spatial dimensions
                return int(repeated[0])
            assert idx < len(repeated)
            # Extract the value for the given spatial dimension
            return repeated[idx]
        if default is None:
            raise ValueError('Unable to determine kernel parameter!')
        return default

    @property
    def kernel_parameters(self):
        assert self.kind in (NodeKind.Convolution, NodeKind.Pooling,\
                    NodeKind.Deconvolution)

        params = self.parameters
        k_h = self.get_kernel_value(params.kernel_h, params.kernel_size, 0)
        k_w = self.get_kernel_value(params.kernel_w, params.kernel_size, 1)
        s_h = self.get_kernel_value(
            params.stride_h, params.stride, 0, default=1)
        s_w = self.get_kernel_value(
            params.stride_w, params.stride, 1, default=1)
        p_h = self.get_kernel_value(params.pad_h, params.pad, 0, default=0)
        p_w = self.get_kernel_value(params.pad_w, params.pad, 1, default=0)

        dila_h = dila_w = 1
        if self.kind in (NodeKind.Convolution, NodeKind.Deconvolution):
            dila_len = len(params.dilation)
            if dila_len == 2:
                dila_h = params.dilation[0]
                dila_w = params.dilation[1]
            elif dila_len == 1:
                dila_h = dila_w = params.dilation[0]
            else:
                assert dila_len == 0, "invalid length[%s] of dilation in convolution" % (
                    dila_len)

        return KernelParameters(k_h, k_w, s_h, s_w, p_h, p_w, dila_h, dila_w)


KernelParameters = namedtuple(
    'KernelParameters',
    [
        'kernel_h', 'kernel_w', 'stride_h', 'stride_w', 'pad_h', 'pad_w',
        'dila_h', 'dila_w'
    ], )
