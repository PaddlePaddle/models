from google.protobuf import text_format

from .caffe import get_caffe_resolver
from .errors import KaffeError, print_stderr
from .layers import LayerAdapter, LayerType, NodeKind, NodeDispatch
from .shapes import make_tensor


class Node(object):
    def __init__(self, name, kind, layer=None):
        self.name = name
        self.kind = kind
        self.layer = LayerAdapter(layer, kind) if layer else None
        self.parents = []
        self.children = []
        self.data = None  #parameters of this node
        self.output_shape = None  #output shape of this node
        self.metadata = {}

    def add_parent(self, parent_node):
        assert parent_node not in self.parents
        self.parents.append(parent_node)
        if self not in parent_node.children:
            parent_node.children.append(self)

    def add_child(self, child_node):
        assert child_node not in self.children
        self.children.append(child_node)
        if self not in child_node.parents:
            child_node.parents.append(self)

    def get_only_parent(self):
        if len(self.parents) != 1:
            raise KaffeError('Node (%s) expected to have 1 parent. Found %s.' %
                             (self, len(self.parents)))
        return self.parents[0]

    @property
    def parameters(self):
        """ get parameters stored in a protobuf object
        """
        if self.layer is not None:
            return self.layer.parameters
        return None

    @property
    def params(self):
        """ get parameters stored in a dict
        """
        from .protobuf_to_dict import protobuf_to_dict

        p = self.parameters
        if p is not None:
            return protobuf_to_dict(p)
        else:
            return None

    def __str__(self):
        return '[%s] %s' % (self.kind, self.name)

    def __repr__(self):
        return '%s (0x%x)' % (self.name, id(self))


class Graph(object):
    def __init__(self, nodes=None, name=None, trace={}):
        self.nodes = nodes or []
        self.node_lut = {node.name: node for node in self.nodes}
        self.output_trace = trace
        if name is None or name == '':
            self.name = 'MyNet'
        else:
            self.name = name

    def add_node(self, node):
        self.nodes.append(node)
        self.node_lut[node.name] = node

    def get_node(self, name):
        try:
            return self.node_lut[name]
        except KeyError:
            raise KaffeError('Layer not found: %s' % name)

    def add_name_trace(self, trace, which='caffe'):
        self.output_trace[which] = trace

    def get_name_trace(self, which=None):
        if which is not None:
            return self.output_trace[which]
        else:
            return self.output_trace

    def get_input_nodes(self):
        return [node for node in self.nodes if len(node.parents) == 0]

    def get_output_nodes(self):
        return [node for node in self.nodes if len(node.children) == 0]

    def topologically_sorted(self):
        sorted_nodes = []
        unsorted_nodes = list(self.nodes)
        temp_marked = set()
        perm_marked = set()

        def visit(node):
            if node in temp_marked:
                raise KaffeError('Graph is not a DAG.')
            if node in perm_marked:
                return
            temp_marked.add(node)
            for child in node.children:
                visit(child)
            perm_marked.add(node)
            temp_marked.remove(node)
            sorted_nodes.insert(0, node)

        while len(unsorted_nodes):
            visit(unsorted_nodes.pop())
        return sorted_nodes

    def compute_output_shapes(self):
        sorted_nodes = self.topologically_sorted()
        for node in sorted_nodes:
            node.output_shape = make_tensor(
                *NodeKind.compute_output_shape(node))

    def replaced(self, new_nodes):
        return Graph(nodes=new_nodes, name=self.name, trace=self.output_trace)

    def transformed(self, transformers):
        graph = self
        for transformer in transformers:
            graph = transformer(graph)
            if graph is None:
                raise KaffeError('Transformer failed: {}'.format(transformer))
            assert isinstance(graph, Graph)

        return graph

    def __contains__(self, key):
        return key in self.node_lut

    def __str__(self):
        hdr = '{:<20} {:<30} {:>20} {:>20}'.format('Type', 'Name', 'Param',
                                                   'Output')
        s = [hdr, '-' * 94]
        for node in self.topologically_sorted():
            # If the node has learned parameters, display the first one's shape.
            # In case of convolutions, this corresponds to the weights.
            if node.data is None:
                data_shape = '--'
                out_shape = node.output_shape or '--'
                s.append('{:<20} {:<30} {:>20} {:>20}'.format(
                    node.kind, node.name, data_shape, tuple(out_shape)))
            else:
                for d in node.data:
                    #data_shape = node.data[0].shape if node.data else '--'
                    data_shape = d.shape
                    out_shape = node.output_shape or '--'
                    s.append('{:<20} {:<30} {:>20} {:>20}'.format(
                        node.kind, node.name, data_shape, tuple(out_shape)))
        return '\n'.join(s)


class GraphBuilder(object):
    '''Constructs a model graph from a Caffe protocol buffer definition.'''

    def __init__(self, def_path, phase='test'):
        '''
        def_path: Path to the model definition (.prototxt)
        data_path: Path to the model data (.caffemodel)
        phase: Either 'test' or 'train'. Used for filtering phase-specific nodes.
        '''
        self.def_path = def_path
        self.phase = phase
        self.load()

    def load(self):
        '''Load the layer definitions from the prototxt.'''
        self.params = get_caffe_resolver().NetParameter()
        with open(self.def_path, 'rb') as def_file:
            text_format.Merge(def_file.read(), self.params)

    def filter_layers(self, layers):
        '''Filter out layers based on the current phase.'''
        phase_map = {0: 'train', 1: 'test'}
        filtered_layer_names = set()
        filtered_layers = []
        for layer in layers:
            phase = self.phase
            if len(layer.include):
                phase = phase_map[layer.include[0].phase]
            if len(layer.exclude):
                phase = phase_map[1 - layer.include[0].phase]
            exclude = (phase != self.phase)
            # Dropout layers appear in a fair number of Caffe
            # test-time networks. These are just ignored. We'll
            # filter them out here.
            if (not exclude) and (phase == 'test'):
                exclude = (layer.type == LayerType.Dropout)
            if not exclude:
                filtered_layers.append(layer)
                # Guard against dupes.
                assert layer.name not in filtered_layer_names
                filtered_layer_names.add(layer.name)
        return filtered_layers

    def make_node(self, layer):
        '''Create a graph node for the given layer.'''
        kind = NodeKind.map_raw_kind(layer.type)
        if kind is None:
            raise KaffeError('Unknown layer type encountered: %s' % layer.type)

        # We want to use the layer's top names (the "output" names), rather than the
        # name attribute, which is more of readability thing than a functional one.
        # Other layers will refer to a node by its "top name".
        return Node(layer.name, kind, layer=layer)

    def make_input_nodes(self):
        '''
        Create data input nodes.

        This method is for old-style inputs, where the input specification
        was not treated as a first-class layer in the prototext.
        Newer models use the "Input layer" type.
        '''
        nodes = [Node(name, NodeKind.Data) for name in self.params.input]
        inputs_num = len(nodes)
        if inputs_num > 0:
            input_dims_num = len(self.params.input_dim)
            if input_dims_num > 0 and input_dims_num != inputs_num * 4:
                raise KaffeError('invalid input_dim[%d] param in prototxt' %
                                 (input_dims_num))

            input_dims = [[]] * inputs_num
            for i in range(input_dims_num):
                dim = self.params.input_dim[i]
                which = int(i / 4)
                input_dims[which].append(int(dim))

            for i in range(inputs_num):
                if len(self.params.input_shape) == inputs_num:
                    input_dim = map(int, self.params.input_shape[i].dim)
                    input_dims[i] = input_dim

                nodes[i].output_shape = tuple(input_dims[i])
        return nodes

    def build(self):
        '''
        Builds the graph from the Caffe layer definitions.
        '''
        # Get the layers
        layers = self.params.layers or self.params.layer
        # Filter out phase-excluded layers
        layers = self.filter_layers(layers)
        # Get any separately-specified input layers
        nodes = self.make_input_nodes()
        nodes += [self.make_node(layer) for layer in layers]
        # Initialize the graph
        graph = Graph(nodes=nodes, name=self.params.name)
        # Connect the nodes
        #
        # A note on layers and outputs:
        # In Caffe, each layer can produce multiple outputs ("tops") from a set of inputs
        # ("bottoms"). The bottoms refer to other layers' tops. The top can rewrite a bottom
        # (in case of in-place operations). Note that the layer's name is not used for establishing
        # any connectivity. It's only used for data association. By convention, a layer with a
        # single top will often use the same name (although this is not required).
        #
        # The current implementation only supports single-output nodes (note that a node can still
        # have multiple children, since multiple child nodes can refer to the single top's name).
        node_outputs = {}
        output_trace = {}
        for layer in layers:
            node = graph.get_node(layer.name)
            for input_name in layer.bottom:
                assert input_name != layer.name
                parent_node = node_outputs.get(input_name)
                if (parent_node is None) or (parent_node == node):
                    parent_node = graph.get_node(input_name)
                node.add_parent(parent_node)

            if len(layer.top) > 1:
                raise KaffeError('Multiple top nodes are not supported.')

            for output_name in layer.top:
                if output_name == layer.name:
                    # Output is named the same as the node. No further action required.
                    continue
                # There are two possibilities here:
                #
                # Case 1: output_name refers to another node in the graph.
                # This is an "in-place operation" that overwrites an existing node.
                # This would create a cycle in the graph. We'll undo the in-placing
                # by substituting this node wherever the overwritten node is referenced.
                #
                # Case 2: output_name violates the convention layer.name == output_name.
                # Since we are working in the single-output regime, we will can rename it to
                # match the layer name.
                #
                # For both cases, future references to this top re-routes to this node.
                node_outputs[output_name] = node
                if output_name in output_trace:
                    output_trace[output_name].append(node.name)
                else:
                    output_trace[output_name] = [output_name, node.name]

        #build a mapping from real-name to changed-name(for caffe's INPLACE inference)
        real2chg = {}
        deleted = {}
        for k, v in output_trace.items():
            real2chg[v[-1]] = k
            for n in v:
                if n in real2chg:
                    continue
                if n not in deleted:
                    deleted[n] = '%s.%s' % (k, v[-1])

        graph.add_name_trace({
            'real2chg': real2chg,
            'deleted': deleted
        }, 'caffe')
        graph.compute_output_shapes()
        return graph


class NodeMapper(NodeDispatch):
    def __init__(self, graph):
        self.graph = graph

    def map(self):
        nodes = self.graph.topologically_sorted()
        # Remove input nodes - we'll handle them separately.
        input_nodes = self.graph.get_input_nodes()
        nodes = [t for t in nodes if t not in input_nodes]
        # Decompose DAG into chains.
        chains = []
        for node in nodes:
            attach_to_chain = None
            if len(node.parents) == 1:
                parent = node.get_only_parent()
                for chain in chains:
                    if chain[-1] == parent:
                        # Node is part of an existing chain.
                        attach_to_chain = chain
                        break
            if attach_to_chain is None:
                # Start a new chain for this node.
                attach_to_chain = []
                chains.append(attach_to_chain)
            attach_to_chain.append(node)
        # Map each chain.
        mapped_chains = []
        for chain in chains:
            mapped_chains.append(self.map_chain(chain))
        return self.commit(mapped_chains)

    def map_chain(self, chain):
        return [self.map_node(node) for node in chain]

    def map_node(self, node):
        map_func = self.get_handler(node.kind, 'map')
        mapped_node = map_func(node)
        assert mapped_node is not None
        mapped_node.node = node
        return mapped_node

    def commit(self, mapped_chains):
        raise NotImplementedError('Must be implemented by subclass.')
