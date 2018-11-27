'''
A collection of graph transforms.

A transformer is a callable that accepts a graph and returns a transformed version.
'''
import os
import numpy as np

from .caffe import get_caffe_resolver, has_pycaffe
from .errors import KaffeError, debug, notice, warn
from .layers import NodeKind


class DataInjector(object):
    '''
    Associates parameters loaded from a .caffemodel file with their corresponding nodes.
    '''

    def __init__(self, def_path, data_path):
        # The .prototxt file defining the graph
        self.def_path = def_path
        # The .caffemodel file containing the learned parameters
        self.data_path = data_path
        # Set to true if the fallback protocol-buffer based backend was used
        self.did_use_pb = False
        # A list containing (layer name, parameters) tuples
        self.params = None
        # Load the parameters
        self.load()

    def load(self):
        if has_pycaffe():
            self.load_using_caffe()
        else:
            self.load_using_pb()

    def load_using_caffe(self):
        caffe = get_caffe_resolver().caffe
        net = caffe.Net(self.def_path, self.data_path, caffe.TEST)
        data = lambda blob: blob.data
        self.params = [(k, map(data, v)) for k, v in net.params.items()]

    def load_using_pb(self):
        data = get_caffe_resolver().NetParameter()
        data.MergeFromString(open(self.data_path, 'rb').read())
        pair = lambda layer: (layer.name, self.normalize_pb_data(layer))
        layers = data.layers or data.layer
        self.params = [pair(layer) for layer in layers if layer.blobs]
        self.did_use_pb = True

    def normalize_pb_data(self, layer):
        transformed = []
        for blob in layer.blobs:
            if len(blob.shape.dim):
                dims = blob.shape.dim
                c_o, c_i, h, w = map(int, [1] * (4 - len(dims)) + list(dims))
            else:
                c_o = blob.num
                c_i = blob.channels
                h = blob.height
                w = blob.width
            data = np.array(blob.data, dtype=np.float32).reshape(c_o, c_i, h, w)
            transformed.append(data)
        return transformed

    def adjust_parameters(self, node, data):
        if not self.did_use_pb:
            return data

        # When using the protobuf-backend, each parameter initially has four dimensions.
        # In certain cases (like FC layers), we want to eliminate the singleton dimensions.
        # This implementation takes care of the common cases. However, it does leave the
        # potential for future issues.
        # The Caffe-backend does not suffer from this problem.
        data = list(data)

        squeeze_indices = [1]  # Squeeze biases.
        if node.kind == NodeKind.InnerProduct:
            squeeze_indices.append(0)  # Squeeze FC.

        for idx in squeeze_indices:
            if idx >= len(data):
                continue

            d = data[idx]
            assert len(
                d.shape
            ) == 4, 'invalid shape[%s] from caffe when adjust_parameters' % (
                str(d.shape))

            shape_old = d.shape
            sq_axis = None
            if idx == 0:
                sq_axis = (0, 1)
            elif idx == 1:
                sq_axis = (0, 1, 2)
            else:
                continue

            data[idx] = np.squeeze(d, axis=sq_axis)
            shape_new = data[idx].shape
            if len(shape_old) != shape_new:
                debug('squeeze idx:%d, with kind:%s,name:%s' % \
                        (idx, node.kind, node.name))
        return data

    def __call__(self, graph):
        for layer_name, data in self.params:
            if layer_name in graph:
                node = graph.get_node(layer_name)
                node.data = self.adjust_parameters(node, data)
            else:
                notice('Ignoring parameters for non-existent layer: %s' % \
                        layer_name)
        return graph


class DataReshaper(object):
    def __init__(self, mapping, replace=True):
        # A dictionary mapping NodeKind to the transposed order.
        self.mapping = mapping
        # The node kinds eligible for reshaping
        self.reshaped_node_types = self.mapping.keys()
        # If true, the reshaped data will replace the old one.
        # Otherwise, it's set to the reshaped_data attribute.
        self.replace = replace

    def has_spatial_parent(self, node):
        try:
            parent = node.get_only_parent()
            s = parent.output_shape
            if len(s) == 4:
                return s.height > 1 or s.width > 1
            else:
                return False
        except KaffeError:
            return False

    def map(self, node_kind):
        try:
            return self.mapping[node_kind]
        except KeyError:
            raise KaffeError('Ordering not found for node kind: {}'.format(
                node_kind))

    def __call__(self, graph):
        for node in graph.nodes:
            if node.data is None:
                continue

            if node.kind not in self.reshaped_node_types:
                # Check for 2+ dimensional data
                #if any(len(tensor.shape) > 1 for tensor in node.data):
                #    notice('parmaters not reshaped for node: {}'.format(node))
                continue

            transpose_order = self.map(node.kind)
            weights = node.data[0]
            if node.kind == NodeKind.InnerProduct:
                # The FC layer connected to the spatial layer needs to be
                # re-wired to match the new spatial ordering.
                #in_shape = node.get_only_parent().output_shape
                fc_shape = weights.shape
                output_channels = fc_shape[0]
                weights = weights.reshape((output_channels, -1))
                weights = weights.transpose(transpose_order)
                node.reshaped_data = weights
            else:
                node.reshaped_data = weights.transpose(transpose_order)

        if self.replace:
            for node in graph.nodes:
                if hasattr(node, 'reshaped_data'):
                    # Set the weights
                    node.data[0] = node.reshaped_data
                    del node.reshaped_data
        return graph


class CropFuser(object):
    '''
    Crop is to return a scalar output Blob for an input Blob of arbitrary size.
    When one of the input Blob is "input" or "DummyData", we can remove the input Blob
    and put the shape into the reduction layer.
    '''
    _traced_names = {}

    @classmethod
    def traced_names(cls):
        return cls._traced_names

    @classmethod
    def trace(cls, fname, tname):
        """ recording the names mapping,
            the value of 'fname' will be replaced by value of 'tname'
        """
        if fname not in cls._traced_names:
            cls._traced_names[fname] = []
        cls._traced_names[fname].append(tname)

    def __init__(self,
                 allowed_parent_types=[NodeKind.Input, NodeKind.DummyData]):
        self.allowed_parent_types = allowed_parent_types

    def __call__(self, graph):
        nodes = graph.nodes
        fused_nodes = []
        for node in nodes:
            if len(node.parents) != 2:
                # reduction layer must has two parent layers.
                continue
            parent = node.parents[1]
            if not self.is_eligible_pair(parent, node):
                continue
            # Change the graph structure.
            parent.children.remove(node)
            node.parents.remove(parent)
            # Let the sub-class merge the fused node in any arbitrary way.
            if not len(parent.children):
                fused_nodes.append(parent)
            #fused_nodes.append(parent)
            self.merge(parent, node)
        # rebuild the graph
        transformed_nodes = [node for node in nodes if node not in fused_nodes]
        return graph.replaced(transformed_nodes)

    def is_eligible_pair(self, parent, child):
        '''Returns true if this parent/child pair is eligible for fusion.'''
        return child.kind == NodeKind.Crop
        #return (self.allowed_parent_types is not None and \
        #        len(parent.children) == 1 and \
        #        parent.kind in self.allowed_parent_types and \
        #        child.kind == NodeKind.Crop)

    def merge(self, parent, child):
        '''Merge the parent node into the child.'''
        child.metadata['shape'] = [
            parent.output_shape.batch_size, parent.output_shape.channels,
            parent.output_shape.height, parent.output_shape.width
        ]


class SubNodeFuser(object):
    '''
    An abstract helper for merging a single-child with its single-parent.
    '''
    _traced_names = {}

    @classmethod
    def traced_names(cls):
        return cls._traced_names

    @classmethod
    def trace(cls, fname, tname):
        """ recording the names mapping,
            the value of 'fname' will be replaced by value of 'tname'
        """
        if fname not in cls._traced_names:
            cls._traced_names[fname] = []
        cls._traced_names[fname].append(tname)

    def __call__(self, graph):
        nodes = graph.nodes
        fused_nodes = []
        for node in nodes:
            if len(node.parents) != 1:
                # We're only fusing nodes with single parents
                continue
            parent = node.get_only_parent()
            if len(parent.children) != 1:
                # We can only fuse a node if its parent's
                # value isn't used by any other node.
                continue
            if not self.is_eligible_pair(parent, node):
                continue
            # Rewrite the fused node's children to its parent.
            for child in node.children:
                pos = child.parents.index(node)
                child.parents[pos] = parent
                parent.add_child(child)
            # Disconnect the fused node from the graph.
            parent.children.remove(node)
            fused_nodes.append(node)
            # Let the sub-class merge the fused node in any arbitrary way.
            self.merge(parent, node)
        transformed_nodes = [node for node in nodes if node not in fused_nodes]
        return graph.replaced(transformed_nodes)

    def is_eligible_pair(self, parent, child):
        '''Returns true if this parent/child pair is eligible for fusion.'''
        raise NotImplementedError('Must be implemented by subclass.')

    def merge(self, parent, child):
        '''Merge the child node into the parent.'''
        raise NotImplementedError('Must be implemented by subclass')


class ReLUFuser(SubNodeFuser):
    '''
    Fuses rectified linear units with their parent nodes.
    '''

    def __init__(self, allowed_parent_types=None):
        # Fuse ReLUs when the parent node is one of the given types.
        # If None, all node types are eligible.
        self.allowed_parent_types = allowed_parent_types

    def is_eligible_pair(self, parent, child):
        return ((self.allowed_parent_types is None or \
                parent.kind in self.allowed_parent_types) and \
                child.kind == NodeKind.ReLU)

    def merge(self, parent, child):
        SubNodeFuser.trace(parent.name, child.name)
        parent.metadata['relu'] = True
        parent.metadata['relu_negative_slope'] = child.parameters.negative_slope


class BatchNormScaleBiasFuser(SubNodeFuser):
    '''
    The original batch normalization paper includes two learned
    parameters: a scaling factor \gamma and a bias \beta.
    Caffe's implementation does not include these two. However, it is commonly
    replicated by adding a scaling+bias layer immidiately after the batch norm.

    This fuser merges the scaling+bias layer with the batch norm.
    '''

    def is_eligible_pair(self, parent, child):
        return (parent.kind == NodeKind.BatchNorm and \
                child.kind == NodeKind.Scale and \
                child.parameters.axis == 1 and \
                child.parameters.bias_term == True)

    def merge(self, parent, child):
        SubNodeFuser.trace(parent.name, child.name)
        parent.scale_bias_node = child


class BatchNormPreprocessor(object):
    '''
    Prescale batch normalization parameters.
    Concatenate gamma (scale) and beta (bias) terms if set.
    '''

    def __call__(self, graph):
        for node in graph.nodes:
            if node.kind != NodeKind.BatchNorm:
                continue
            assert node.data is not None
            assert len(node.data) == 3
            node.data = [np.squeeze(i) for i in node.data]
            mean, variance, scale = node.data
            # Prescale the stats
            scaling_factor = 1.0 / scale if scale != 0 else 0
            mean *= scaling_factor
            variance *= scaling_factor
            # Replace with the updated values
            node.data = [mean, variance]
            if hasattr(node, 'scale_bias_node'):
                # Include the scale and bias terms
                gamma, beta = node.scale_bias_node.data
                node.data += [np.squeeze(i) for i in [gamma, beta]]
        return graph


class NodeRenamer(object):
    '''
    Renames nodes in the graph using a given unary function that
    accepts a node and returns its new name.
    '''

    def __init__(self, renamer):
        self.renamer = renamer

    def __call__(self, graph):
        for node in graph.nodes:
            node.name = self.renamer(node)
        return graph


class ParameterNamer(object):
    '''
    Convert layer data arrays to a dictionary mapping parameter names to their values.
    '''

    def __call__(self, graph):
        for node in graph.nodes:
            if node.data is None:
                continue
            if node.kind in (NodeKind.Convolution, NodeKind.InnerProduct,\
                    NodeKind.Deconvolution):
                names = ('weights', )
                if node.parameters.bias_term:
                    names += ('biases', )
            elif node.kind == NodeKind.BatchNorm:
                names = ('mean', 'variance')
                if len(node.data) == 4:
                    names += ('scale', 'offset')
            elif node.kind == NodeKind.Scale:
                names = ('scale', )
                if getattr(node.parameters, 'bias_term', False):
                    names = ('scale', 'offset')
            elif node.kind == NodeKind.PReLU:
                names = ('negslope', )
            elif node.kind == "Normalize":
                names = ('scale', )
            else:
                warn('Unhandled parameters when naming this it[%s]' %
                     (node.kind))
                continue
            assert len(names) == len(node.data)
            node.data = dict(zip(names, node.data))
        return graph
