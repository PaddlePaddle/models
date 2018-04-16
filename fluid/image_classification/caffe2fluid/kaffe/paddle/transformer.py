import numpy as np

from ..errors import KaffeError, print_stderr
from ..graph import GraphBuilder, NodeMapper
from ..layers import NodeKind
from ..transformers import (DataInjector, DataReshaper, NodeRenamer, ReLUFuser,
                            BatchNormScaleBiasFuser, BatchNormPreprocessor,
                            ParameterNamer)
from . import network


def get_padding_type(kernel_params, input_shape, output_shape):
    '''Translates Caffe's numeric padding to one of ('SAME', 'VALID').
    Caffe supports arbitrary padding values, while TensorFlow only
    supports 'SAME' and 'VALID' modes. So, not all Caffe paddings
    can be translated to TensorFlow. There are some subtleties to
    how the padding edge-cases are handled. These are described here:
    https://github.com/Yangqing/caffe2/blob/master/caffe2/proto/caffe2_legacy.proto
    '''
    k_h, k_w, s_h, s_w, p_h, p_w = kernel_params
    if p_h * p_w > 0:
        return [p_h, p_w]
    else:
        return None


class TensorFlowNode(object):
    '''An intermediate representation for TensorFlow operations.'''

    def __init__(self, op, *args, **kwargs):
        # A string corresponding to the TensorFlow operation
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

    def __call__(self, *args, **kwargs):
        kwargs.update(self.inject_kwargs)
        return TensorFlowNode(*args, **kwargs)


class TensorFlowMapper(NodeMapper):
    def get_kernel_params(self, node):
        kernel_params = node.layer.kernel_parameters
        input_shape = node.get_only_parent().output_shape
        padding = get_padding_type(kernel_params, input_shape,
                                   node.output_shape)
        # Only emit the padding if it's not the default value.
        padding = {'padding': padding} if padding is not None else {}
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
        assert kernel_params.kernel_h == h
        assert kernel_params.kernel_w == w
        return MaybeActivated(node)(
            'conv', kernel_params.kernel_h, kernel_params.kernel_w, c_o,
            kernel_params.stride_h, kernel_params.stride_w, **kwargs)

    def map_relu(self, node):
        return TensorFlowNode('relu')

    def map_pooling(self, node):
        pool_type = node.parameters.pool
        if pool_type == 0:
            pool_op = 'max_pool'
        elif pool_type == 1:
            pool_op = 'avg_pool'
        else:
            # Stochastic pooling, for instance.
            raise KaffeError('Unsupported pooling type.')
        (kernel_params, padding) = self.get_kernel_params(node)
        return TensorFlowNode(pool_op, kernel_params.kernel_h,
                              kernel_params.kernel_w, kernel_params.stride_h,
                              kernel_params.stride_w, **padding)

    def map_inner_product(self, node):
        #TODO: Axis
        assert node.parameters.axis == 1
        #TODO: Unbiased
        assert node.parameters.bias_term == True
        return MaybeActivated(node)('fc', node.parameters.num_output)

    def map_softmax(self, node):
        return TensorFlowNode('softmax')

    def map_lrn(self, node):
        params = node.parameters
        # The window size must be an odd value. For a window
        # size of (2*n+1), TensorFlow defines depth_radius = n.
        assert params.local_size % 2 == 1
        # Caffe scales by (alpha/(2*n+1)), whereas TensorFlow
        # just scales by alpha (as does Krizhevsky's paper).
        # We'll account for that here.
        alpha = params.alpha / float(params.local_size)
        return TensorFlowNode('lrn', params.local_size, alpha, params.beta)

    def map_concat(self, node):
        return TensorFlowNode('concat', node.parameters.axis)

    def map_dropout(self, node):
        return TensorFlowNode('dropout', node.parameters.dropout_ratio)

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
            return TensorFlowNode(operations[op_code])
        except KeyError:
            raise KaffeError('Unknown elementwise operation: {}'.format(
                op_code))

    def commit(self, chains):
        return chains


class TensorFlowEmitter(object):
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

    def emit_class_def(self, name):
        return self.statement('class %s(Network):' % (name))

    def emit_setup_def(self):
        return self.statement('def setup(self):')

    def emit_shape_def(self, input_nodes):
        self.outdent()
        func_def = self.statement('@classmethod')
        func_def += self.statement('def input_shapes(cls):')
        self.indent()

        input_shapes = {}
        for n in input_nodes:
            name = n.name
            output_shape = n.output_shape
            shape = [str(s) for s in output_shape[1:]]
            input_shapes[name] = ', '.join(shape)
        input_shapes = ['"%s": [%s]' % (n, l) for n, l in input_shapes.items()]
        shape_str = ','.join(input_shapes)
        func_def += self.statement('return {%s}' % (shape_str))
        return '\n\n' + func_def

    def emit_convert_def(self, input_nodes):
        codes = []
        inputs = {}
        #codes.append('shapes = cls.input_shapes()')
        codes.append('shapes = cls.input_shapes()')
        codes.append('input_name = shapes.keys()[0]')
        codes.append('input_shape = shapes[input_name]')
        for n in input_nodes:
            name = n.name
            layer_var = name + '_layer'
            layer_def = '%s = fluid.layers.data(name="%s", shape=shapes["%s"],'\
                    ' dtype="float32")' % (layer_var, name, name)
            #layer_var, layer_def = data_layer_def(n.name, n.output_shape)
            codes.append(layer_def)
            inputs[name] = layer_var

        input_dict = ','.join(['"%s": %s' % (n, l) for n, l in inputs.items()])

        codes.append('feed_data = {' + input_dict + '}')
        codes.append('net = cls(feed_data)')

        codes.append("place = fluid.CPUPlace()")
        codes.append("exe = fluid.Executor(place)")
        codes.append("exe.run(fluid.default_startup_program())")
        codes.append("net.load(data_path=npy_model, exe=exe, place=place)")
        codes.append("output_vars = [net.get_output()]")
        codes.append("fluid.io.save_inference_model(" \
                "fluid_path, [input_name],output_vars," \
                "exe, main_program=None, model_filename='model'," \
                "params_filename='params')")
        codes.append(
            "print('save fluid model as [model] and [params] in directory [%s]' % (fluid_path))"
        )

        self.outdent()
        func_def = self.statement('@classmethod')
        func_def += self.statement('def convert(cls, npy_model, fluid_path):')
        self.indent()
        func_def += self.statement('fluid = import_fluid()')
        for l in codes:
            func_def += self.statement(l)
        return '\n' + func_def

    def emit_main_def(self, name):
        if name is None:
            return ''

        self.prefix = ''
        main_def = self.statement('if __name__ == "__main__":')
        self.indent()
        main_def += self.statement(
            "#usage: save as an inference model for online service\n")
        main_def += self.statement("import sys")
        main_def += self.statement("if len(sys.argv) != 3:")
        self.indent()
        main_def += self.statement("print('usage:')")
        main_def += self.statement(
            "print('\tpython %s [xxxnet.npy] [save_dir]' % (sys.argv[0]))")
        main_def += self.statement("exit(1)")

        self.outdent()
        main_def += self.statement("npy_weight = sys.argv[1]")
        main_def += self.statement("fluid_model = sys.argv[2]")
        main_def += self.statement("%s.convert(npy_weight, fluid_model)" %
                                   (name))
        main_def += self.statement("exit(0)")
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
        self.net_name = name
        s = self.emit_imports()
        s += self.emit_class_def(name)
        self.indent()
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
        s += self.emit_shape_def(input_nodes)
        s += self.emit_convert_def(input_nodes)
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
            # Slashes are used for scoping in TensorFlow. Replace slashes
            # in node names with underscores.
            # (Caffe's GoogLeNet implementation uses slashes)
            NodeRenamer(lambda node: node.name.replace('/', '_'))
        ]
        self.graph = graph.transformed(transformers)

        # Display the graph
        if self.verbose:
            print_stderr(self.graph)

    def transform_data(self):
        if self.params is None:
            transformers = [
                # Reshape the parameters to TensorFlow's ordering
                DataReshaper({
                    # (c_o, c_i, h, w) -> (h, w, c_i, c_o) for TF
                    NodeKind.Convolution: (0, 1, 2, 3),

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
        return self.params

    def transform_source(self):
        if self.source is None:
            mapper = TensorFlowMapper(self.graph)
            chains = mapper.map()
            emitter = TensorFlowEmitter()
            input_nodes = self.graph.get_input_nodes()
            self.source = emitter.emit(self.graph.name, chains, input_nodes)
        return self.source
