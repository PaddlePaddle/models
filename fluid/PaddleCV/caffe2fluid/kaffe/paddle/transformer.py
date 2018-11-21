import numpy as np

from ..errors import KaffeError, print_stderr
from ..graph import GraphBuilder, NodeMapper
from ..layers import NodeKind
from ..transformers import (DataInjector, DataReshaper, NodeRenamer,
                            SubNodeFuser, ReLUFuser, BatchNormScaleBiasFuser,
                            BatchNormPreprocessor, ParameterNamer, CropFuser)
from . import network


class PaddleNode(object):
    '''An intermediate representation for Paddle operations.'''

    def __init__(self, op, *args, **kwargs):
        # A string corresponding to the Paddle operation
        self.op = op
        # Positional arguments for the operation
        self.args = args
        # Keyword arguments for the operation
        self.kwargs = list(kwargs.items())
        # The source Caffe node
        self.node = None

    def format(self, arg):
        '''Returns a string representation for the given value.'''
        return "'%s'" % arg if isinstance(arg, basestring) else str(arg)

    def pair(self, key, value):
        '''Returns key=formatted(value).'''
        return '%s=%s' % (key, self.format(value))

    def emit(self):
        '''Emits the Python source for this node.'''
        # Format positional arguments
        args = map(self.format, self.args)
        # Format any keyword arguments
        if self.kwargs:
            args += [self.pair(k, v) for k, v in self.kwargs]
        # Set the node name
        args.append(self.pair('name', self.node.name))
        args = ', '.join(args)
        return '%s(%s)' % (self.op, args)


class MaybeActivated(object):
    def __init__(self, node, default=True):
        self.inject_kwargs = {}
        if node.metadata.get('relu', False) != default:
            self.inject_kwargs['relu'] = not default

        default_slope = 0.0
        slope = node.metadata.get('relu_negative_slope', default_slope)
        if slope != default_slope:
            self.inject_kwargs['relu_negative_slope'] = slope

    def __call__(self, *args, **kwargs):
        kwargs.update(self.inject_kwargs)
        return PaddleNode(*args, **kwargs)


class PaddleMapper(NodeMapper):
    def get_kernel_params(self, node):
        kernel_params = node.layer.kernel_parameters
        input_shape = node.get_only_parent().output_shape
        padding = [kernel_params.pad_h, kernel_params.pad_w]
        if padding[0] == 0 and padding[1] == 0:
            padding = {}
        else:
            padding = {'padding': padding}
        return (kernel_params, padding)

    def map_convolution(self, node):
        (kernel_params, kwargs) = self.get_kernel_params(node)
        h = kernel_params.kernel_h
        w = kernel_params.kernel_w
        c_o = node.output_shape[1]
        c_i = node.parents[0].output_shape[1]
        group = node.parameters.group
        if group != 1:
            kwargs['group'] = group
        if not node.parameters.bias_term:
            kwargs['biased'] = False

        if kernel_params.dila_h != 1 or kernel_params.dila_w != 1:
            kwargs['dilation'] = (kernel_params.dila_h, kernel_params.dila_w)

        assert kernel_params.kernel_h == h
        assert kernel_params.kernel_w == w
        return MaybeActivated(node)(
            'conv', kernel_params.kernel_h, kernel_params.kernel_w, c_o,
            kernel_params.stride_h, kernel_params.stride_w, **kwargs)

    def map_deconvolution(self, node):
        (kernel_params, kwargs) = self.get_kernel_params(node)
        h = kernel_params.kernel_h
        w = kernel_params.kernel_w
        c_o = node.output_shape[1]
        c_i = node.parents[0].output_shape[1]
        if not node.parameters.bias_term:
            kwargs['biased'] = False

        if kernel_params.dila_h != 1 or kernel_params.dila_w != 1:
            kwargs['dilation'] = (kernel_params.dila_h, kernel_params.dila_w)

        assert kernel_params.kernel_h == h
        assert kernel_params.kernel_w == w
        return MaybeActivated(node)(
            'deconv', kernel_params.kernel_h, kernel_params.kernel_w, c_o,
            kernel_params.stride_h, kernel_params.stride_w, **kwargs)

    def map_relu(self, node):
        return PaddleNode('relu')

    def map_prelu(self, node):
        channel_shared = getattr(node.parameters, 'channel_shared', False)
        return PaddleNode('prelu', channel_shared)

    def map_tanh(self, node):
        return PaddleNode('tanh')

    def map_pooling(self, node):
        pool_type = node.parameters.pool
        if pool_type == 0:
            pool_op = 'max_pool'
        elif pool_type == 1:
            pool_op = 'avg_pool'
        else:
            # Stochastic pooling, for instance.
            raise KaffeError('Unsupported pooling type.')

        ceil_mode = getattr(node.layer.parameters, 'ceil_mode', True)
        global_pool = getattr(node.layer.parameters, 'global_pooling', False)
        if global_pool:
            input_shape = node.get_only_parent().output_shape
            return PaddleNode(pool_op, input_shape.height, input_shape.width, 1,
                              1, ceil_mode)
        else:
            (kernel_params, padding) = self.get_kernel_params(node)
            return PaddleNode(pool_op, kernel_params.kernel_h,
                              kernel_params.kernel_w, kernel_params.stride_h,
                              kernel_params.stride_w, ceil_mode, **padding)

    def map_sigmoid(self, node):
        return PaddleNode('sigmoid')

    def map_custom(self, node):
        from .. import custom_layers
        return custom_layers.make_node(PaddleNode, node.kind, node)

    def map_inner_product(self, node):
        #TODO: Axis
        assert node.parameters.axis == 1
        #TODO: Unbiased
        assert node.parameters.bias_term == True
        return MaybeActivated(node)('fc', node.parameters.num_output)

    def map_softmax(self, node):
        return PaddleNode('softmax', node.parameters.axis)

    def map_lrn(self, node):
        params = node.parameters
        # The window size must be an odd value. For a window
        # size of (2*n+1), Paddle defines depth_radius = n.
        assert params.local_size % 2 == 1
        # Caffe scales by (alpha/(2*n+1)), whereas Paddle
        # just scales by alpha (as does Krizhevsky's paper).
        # We'll account for that here.
        alpha = params.alpha / float(params.local_size)
        return PaddleNode('lrn', params.local_size, alpha, params.beta)

    def map_concat(self, node):
        return PaddleNode('concat', node.parameters.axis)

    def map_dropout(self, node):
        return PaddleNode('dropout', node.parameters.dropout_ratio)

    def map_batch_norm(self, node):
        scale_offset = len(node.data) == 4

        #this default value comes from caffe's param in batch_norm
        default_eps = 1e-5
        kwargs = {'scale_offset': scale_offset}
        if node.parameters.eps != default_eps:
            kwargs['eps'] = node.parameters.eps

        return MaybeActivated(
            node, default=False)('batch_normalization', **kwargs)

    def map_eltwise(self, node):
        operations = {0: 'multiply', 1: 'add', 2: 'max'}
        op_code = node.parameters.operation
        try:
            return PaddleNode(operations[op_code])
        except KeyError:
            raise KaffeError('Unknown elementwise operation: {}'.format(
                op_code))

    def map_scale(self, node):
        params = node.parameters
        return PaddleNode('scale', axis=params.axis, num_axes=params.num_axes)

    def commit(self, chains):
        return chains


class PaddleEmitter(object):
    def __init__(self, tab=None):
        self.tab = tab or ' ' * 4
        self.prefix = ''
        self.net_name = ''

    def indent(self):
        self.prefix += self.tab

    def outdent(self):
        self.prefix = self.prefix[:-len(self.tab)]

    def statement(self, s):
        return self.prefix + s + '\n'

    def emit_imports(self):
        import inspect
        codes = []
        codes.append(
            '### generated by caffe2fluid, your net is in class "%s" ###\n' %
            (self.net_name))
        network_source = inspect.getsource(network)
        codes.append(network_source + '\n')
        return self.statement('\n'.join(codes))

    def emit_setup_def(self):
        return self.statement('def setup(self):')

    def get_inputs_info(self, input_nodes):
        input_shapes = {}
        for n in input_nodes:
            name = n.name
            output_shape = n.output_shape
            shape = [str(s) for s in output_shape[1:]]
            input_shapes[name] = ', '.join(shape)
        input_shapes = ['"%s": [%s]' % (n, l) for n, l in input_shapes.items()]
        shape_str = ','.join(input_shapes)
        return '{%s}' % (shape_str)

    def emit_main_def(self, name):
        if name is None:
            return ''

        self.prefix = ''
        main_def = self.statement('if __name__ == "__main__":')
        self.indent()
        main_def += self.statement('exit(main())')
        return '\n\n' + main_def

    def emit_parents(self, chain):
        assert len(chain)
        s = 'self.feed('
        sep = ', \n' + self.prefix + (' ' * len(s))
        s += sep.join(
            ["'%s'" % parent.name for parent in chain[0].node.parents])
        return self.statement(s + ')')

    def emit_node(self, node):
        return self.statement('self.' + node.emit())

    def emit(self, name, chains, input_nodes=None):
        from ..net_template import generate_net_code
        from ..net_template import generate_main_code

        self.net_name = name
        inputs_info = self.get_inputs_info(input_nodes)

        s = self.emit_imports()
        s += generate_net_code(name, inputs_info) + '\n'
        self.indent()

        # define the net using api
        s += self.emit_setup_def()
        self.indent()
        blocks = []
        for chain in chains:
            b = ''
            b += self.emit_parents(chain)
            for node in chain:
                b += self.emit_node(node)
            blocks.append(b[:-1])
        s = s + '\n\n'.join(blocks)

        # define the main function
        s += '\n\n\n' + generate_main_code(name)
        s += self.emit_main_def(name)
        return s


class Transformer(object):
    def __init__(self, def_path, data_path, verbose=True, phase='test'):
        self.verbose = verbose
        self.phase = phase
        self.load(def_path, data_path, phase)
        self.params = None
        self.source = None

    def load(self, def_path, data_path, phase):
        # Build the graph
        graph = GraphBuilder(def_path, phase).build()

        if data_path is not None:
            # Load and associate learned parameters
            graph = DataInjector(def_path, data_path)(graph)

        # Transform the graph
        transformers = [
            # Fuse split batch normalization layers
            BatchNormScaleBiasFuser(),

            # Fuse ReLUs
            # TODO: Move non-linearity application to layer wrapper, allowing
            # any arbitrary operation to be optionally activated.
            ReLUFuser(allowed_parent_types=[
                NodeKind.Convolution, NodeKind.InnerProduct, NodeKind.BatchNorm
            ]),

            # Rename nodes
            # Slashes are used for scoping in Paddle. Replace slashes
            # in node names with underscores.
            # (Caffe's GoogLeNet implementation uses slashes)
            NodeRenamer(lambda node: node.name.replace('/', '_')),

            # Fuse Crop
            # Crop is to return a scalar output Blob for an input Blob of arbitrary size.
            # When one of the input Blob is "input" or "DummyData", we can remove this input Blob
            # and put the shape into the reduction layer.
            CropFuser()
        ]

        self.graph = graph.transformed(transformers)

        #for the purpose of recording name mapping because of fused nodes
        trace = SubNodeFuser.traced_names()
        chg2real = {}
        deleted = {}
        for k, v in trace.items():
            chg2real[k] = v[-1]  #mapping from changed-name to real-name
            for n in v:
                if n in chg2real:
                    continue
                if n not in deleted:
                    deleted[n] = '%s.%s' % (k, v[-1])

        self.graph.add_name_trace({
            'chg2real': chg2real,
            'deleted': deleted
        }, 'paddle')

        # Display the graph
        if self.verbose:
            print_stderr(self.graph)

    def transform_data(self):
        if self.params is None:
            transformers = [
                # Reshape the parameters to Paddle's ordering
                DataReshaper({
                    # (c_o, c_i) -> (c_i, c_o)
                    NodeKind.InnerProduct: (1, 0)
                }),

                # Pre-process batch normalization data
                BatchNormPreprocessor(),

                # Convert parameters to dictionaries
                ParameterNamer(),
            ]
            self.graph = self.graph.transformed(transformers)
            self.params = {
                node.name: node.data
                for node in self.graph.nodes if node.data
            }
            self.params['caffe2fluid_name_trace'] = self.graph.get_name_trace()

        return self.params

    def transform_source(self):
        if self.source is None:
            mapper = PaddleMapper(self.graph)
            chains = mapper.map()
            emitter = PaddleEmitter()
            input_nodes = self.graph.get_input_nodes()
            self.source = emitter.emit(self.graph.name, chains, input_nodes)
        return self.source
