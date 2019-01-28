import sys
import os
import math
import numpy as np


def import_fluid():
    import paddle.fluid as fluid
    return fluid


def layer(op):
    '''Decorator for composable network layers.'''

    def layer_decorated(self, *args, **kwargs):
        # Automatically set a name if not provided.
        name = kwargs.setdefault('name', self.get_unique_name(op.__name__))
        # Figure out the layer inputs.
        if len(self.terminals) == 0:
            raise RuntimeError('No input variables found for layer %s.' % name)
        elif len(self.terminals) == 1:
            layer_input = self.terminals[0]
        else:
            layer_input = list(self.terminals)

        self.layer_reverse_trace[name] = layer_input
        # Perform the operation and get the output.
        layer_output = op(self, layer_input, *args, **kwargs)
        # Add to layer LUT.
        self.layers[name] = layer_output
        self.var2name[layer_output.name] = (name, layer_output)

        # This output is now the input for the next layer.
        self.feed(layer_output)
        # Return self for chained calls.
        return self

    return layer_decorated


class Network(object):
    def __init__(self, inputs, trainable=True):
        # The input nodes for this network
        self.inputs = inputs
        # The current list of terminal nodes
        self.terminals = []
        # Mapping from layer names to layers
        self.layers = dict(inputs)
        # If true, the resulting variables are set as trainable
        self.trainable = trainable
        # Switch variable for dropout
        self.paddle_env = None
        self.output_names = []
        self.name_trace = None

        self.layer_reverse_trace = {}
        self.var2name = {}
        self.setup()

    def setup(self):
        '''Construct the network. '''
        raise NotImplementedError('Must be implemented by the subclass.')

    def locate_ancestor(self, v, which=[0], ancestor_level=1):
        """ find a ancestor for a node 'v' which is a fluid variable
        """
        ancestor = None
        which = which * ancestor_level
        name = self.var2name[v.name][0]

        for i in range(ancestor_level):
            v = self.layer_reverse_trace[name]
            if type(v) is list:
                ancestor = self.var2name[v[which[i]].name]
            else:
                ancestor = self.var2name[v.name]
            name = ancestor[0]
        return ancestor

    def load(self, data_path, exe=None, place=None, ignore_missing=False):
        '''Load network weights.
        data_path: The path to the numpy-serialized network weights
        ignore_missing: If true, serialized weights for missing layers are ignored.
        '''
        fluid = import_fluid()
        #load fluid mode directly
        if os.path.isdir(data_path):
            assert (exe is not None), \
                'must provide a executor to load fluid model'
            fluid.io.load_persistables(executor=exe, dirname=data_path)
            return True

        #load model from a npy file
        if exe is None or place is None:
            if self.paddle_env is None:
                place = fluid.CPUPlace()
                exe = fluid.Executor(place)
                self.paddle_env = {'place': place, 'exe': exe}
                exe = exe.run(fluid.default_startup_program())
            else:
                place = self.paddle_env['place']
                exe = self.paddle_env['exe']

        data_dict = np.load(data_path).item()
        for op_name in data_dict:
            if op_name == 'caffe2fluid_name_trace':
                self.name_trace = data_dict[op_name]
                continue

            layer = self.layers[op_name]
            for param_name, data in data_dict[op_name].iteritems():
                try:
                    name = '%s_%s' % (op_name, param_name)
                    v = fluid.global_scope().find_var(name)
                    w = v.get_tensor()
                    w.set(data.reshape(w.shape()), place)
                except ValueError:
                    if not ignore_missing:
                        raise
        return True

    def feed(self, *args):
        '''Set the input(s) for the next operation by replacing the terminal nodes.
        The arguments can be either layer names or the actual layers.
        '''
        assert len(args) != 0
        self.terminals = []
        for fed_layer in args:
            if isinstance(fed_layer, basestring):
                try:
                    fed_layer = self.layers[fed_layer]
                except KeyError:
                    raise KeyError('Unknown layer name fed: %s' % fed_layer)
            self.terminals.append(fed_layer)
        return self

    def get_output(self):
        '''Returns the current network output.'''
        return self.terminals[-1]

    def get_unique_name(self, prefix):
        '''Returns an index-suffixed unique name for the given prefix.
        This is used for auto-generating layer names based on the type-prefix.
        '''
        ident = sum(t.startswith(prefix) for t, _ in self.layers.items()) + 1
        return '%s_%d' % (prefix, ident)

    def get_unique_output_name(self, prefix, layertype):
        '''Returns an index-suffixed unique name for the given prefix.
            This is used for auto-generating layer names based on the type-prefix.
        '''
        ident = sum(t.startswith(prefix) for t in self.output_names) + 1
        unique_name = '%s.%s.output.%d' % (prefix, layertype, ident)
        self.output_names.append(unique_name)
        return unique_name

    @layer
    def conv(self,
             input,
             k_h,
             k_w,
             c_o,
             s_h,
             s_w,
             name,
             relu=True,
             relu_negative_slope=0.0,
             padding=None,
             dilation=1,
             group=1,
             biased=True):
        if padding is None:
            padding = [0, 0]

        # Get the number of channels in the input
        c_i, h_i, w_i = input.shape[1:]

        # Verify that the grouping parameter is valid
        assert c_i % group == 0
        assert c_o % group == 0

        fluid = import_fluid()
        prefix = name + '_'
        leaky_relu = False
        act = 'relu'
        if relu is False:
            act = None
        elif relu_negative_slope != 0.0:
            leaky_relu = True
            act = None

        output = fluid.layers.conv2d(
            name=self.get_unique_output_name(name, 'conv2d'),
            input=input,
            filter_size=[k_h, k_w],
            num_filters=c_o,
            stride=[s_h, s_w],
            padding=padding,
            dilation=dilation,
            groups=group,
            param_attr=fluid.ParamAttr(name=prefix + "weights"),
            bias_attr=fluid.ParamAttr(name=prefix + "biases"),
            act=act)

        if leaky_relu:
            output = fluid.layers.leaky_relu(output, alpha=relu_negative_slope)

        return output

    @layer
    def deconv(self,
               input,
               k_h,
               k_w,
               c_o,
               s_h,
               s_w,
               name,
               relu=True,
               relu_negative_slope=0.0,
               padding=None,
               dilation=1,
               biased=True):
        if padding is None:
            padding = [0, 0]

        # Get the number of channels in the input
        c_i, h_i, w_i = input.shape[1:]

        fluid = import_fluid()
        prefix = name + '_'
        leaky_relu = False
        act = 'relu'
        if relu is False:
            act = None
        elif relu_negative_slope != 0.0:
            leaky_relu = True
            act = None

        p_h = padding[0]
        p_w = padding[1]
        h_o = (h_i - 1) * s_h - 2 * p_h + dilation * (k_h - 1) + 1
        w_o = (w_i - 1) * s_w - 2 * p_w + dilation * (k_w - 1) + 1
        output = fluid.layers.conv2d_transpose(
            name=self.get_unique_output_name(name, 'conv2d_transpose'),
            input=input,
            num_filters=c_o,
            output_size=[h_o, w_o],
            filter_size=[k_h, k_w],
            padding=padding,
            stride=[s_h, s_w],
            dilation=dilation,
            param_attr=fluid.ParamAttr(name=prefix + "weights"),
            bias_attr=fluid.ParamAttr(name=prefix + "biases"),
            act=act)

        if leaky_relu:
            output = fluid.layers.leaky_relu(output, alpha=relu_negative_slope)

        return output

    @layer
    def relu(self, input, name):
        fluid = import_fluid()
        output = fluid.layers.relu(input)
        return output

    @layer
    def prelu(self, input, channel_shared, name):
        fluid = import_fluid()
        if channel_shared:
            mode = 'all'
        else:
            mode = 'channel'

        prefix = name + '_'
        output = fluid.layers.prelu(
            input,
            mode=mode,
            param_attr=fluid.ParamAttr(name=prefix + 'negslope'))
        return output

    def pool(self,
             pool_type,
             input,
             k_h,
             k_w,
             s_h,
             s_w,
             ceil_mode,
             padding,
             name,
             exclusive=True):
        # Get the number of channels in the input
        in_hw = input.shape[2:]
        k_hw = [k_h, k_w]
        s_hw = [s_h, s_w]

        fluid = import_fluid()
        output = fluid.layers.pool2d(
            name=name,
            input=input,
            pool_size=k_hw,
            pool_stride=s_hw,
            pool_padding=padding,
            ceil_mode=ceil_mode,
            pool_type=pool_type,
            exclusive=exclusive)
        return output

    @layer
    def max_pool(self,
                 input,
                 k_h,
                 k_w,
                 s_h,
                 s_w,
                 ceil_mode,
                 padding=[0, 0],
                 name=None):
        return self.pool(
            'max',
            input,
            k_h,
            k_w,
            s_h,
            s_w,
            ceil_mode,
            padding,
            name=self.get_unique_output_name(name, 'max_pool'))

    @layer
    def avg_pool(self,
                 input,
                 k_h,
                 k_w,
                 s_h,
                 s_w,
                 ceil_mode,
                 padding=[0, 0],
                 name=None):
        return self.pool(
            'avg',
            input,
            k_h,
            k_w,
            s_h,
            s_w,
            ceil_mode,
            padding,
            name=self.get_unique_output_name(name, 'avg_pool'),
            exclusive=False)

    @layer
    def sigmoid(self, input, name):
        fluid = import_fluid()
        return fluid.layers.sigmoid(
            input, name=self.get_unique_output_name(name, 'sigmoid'))

    @layer
    def tanh(self, input, name):
        fluid = import_fluid()
        return fluid.layers.tanh(
            input, name=self.get_unique_output_name(name, 'tanh'))

    @layer
    def lrn(self, input, radius, alpha, beta, name, bias=1.0):
        fluid = import_fluid()
        output = fluid.layers.lrn(input=input,
                                  n=radius,
                                  k=bias,
                                  alpha=alpha,
                                  beta=beta,
                                  name=self.get_unique_output_name(name, 'lrn'))
        return output

    @layer
    def concat(self, inputs, axis, name):
        fluid = import_fluid()
        output = fluid.layers.concat(
            input=inputs,
            axis=axis,
            name=self.get_unique_output_name(name, 'concat'))
        return output

    @layer
    def add(self, inputs, name):
        fluid = import_fluid()
        output = inputs[0]
        for i in inputs[1:]:
            output = fluid.layers.elementwise_add(
                x=output, y=i, name=self.get_unique_output_name(name, 'add'))
        return output

    @layer
    def max(self, inputs, name):
        fluid = import_fluid()
        output = inputs[0]
        for i in inputs[1:]:
            output = fluid.layers.elementwise_max(
                x=output, y=i, name=self.get_unique_output_name(name, 'max'))
        return output

    @layer
    def multiply(self, inputs, name):
        fluid = import_fluid()
        output = inputs[0]
        for i in inputs[1:]:
            output = fluid.layers.elementwise_mul(
                x=output, y=i, name=self.get_unique_output_name(name, 'mul'))
        return output

    @layer
    def fc(self, input, num_out, name, relu=True, act=None):
        fluid = import_fluid()

        if act is None:
            act = 'relu' if relu is True else None

        prefix = name + '_'
        output = fluid.layers.fc(
            name=self.get_unique_output_name(name, 'fc'),
            input=input,
            size=num_out,
            act=act,
            param_attr=fluid.ParamAttr(name=prefix + 'weights'),
            bias_attr=fluid.ParamAttr(name=prefix + 'biases'))
        return output

    @layer
    def softmax(self, input, axis=2, name=None):
        fluid = import_fluid()
        shape = input.shape
        dims = len(shape)
        axis = axis + dims if axis < 0 else axis

        need_transpose = False
        if axis + 1 != dims:
            need_transpose = True

        if need_transpose:
            order = range(dims)
            order.remove(axis)
            order.append(axis)
            input = fluid.layers.transpose(
                input,
                perm=order,
                name=self.get_unique_output_name(name, 'transpose'))

        output = fluid.layers.softmax(
            input, name=self.get_unique_output_name(name, 'softmax'))

        if need_transpose:
            order = range(len(shape))
            order[axis] = dims - 1
            order[-1] = axis
            output = fluid.layers.transpose(
                output,
                perm=order,
                name=self.get_unique_output_name(name, 'transpose'))
        return output

    @layer
    def batch_normalization(self,
                            input,
                            name,
                            scale_offset=True,
                            eps=1e-5,
                            relu=False,
                            relu_negative_slope=0.0):
        # NOTE: Currently, only inference is supported
        fluid = import_fluid()
        prefix = name + '_'
        param_attr = None if scale_offset is False else fluid.ParamAttr(
            name=prefix + 'scale')
        bias_attr = None if scale_offset is False else fluid.ParamAttr(
            name=prefix + 'offset')
        mean_name = prefix + 'mean'
        variance_name = prefix + 'variance'

        leaky_relu = False
        act = 'relu'
        if relu is False:
            act = None
        elif relu_negative_slope != 0.0:
            leaky_relu = True
            act = None

        output = fluid.layers.batch_norm(
            name=self.get_unique_output_name(name, 'batch_norm'),
            input=input,
            is_test=True,
            param_attr=param_attr,
            bias_attr=bias_attr,
            moving_mean_name=mean_name,
            moving_variance_name=variance_name,
            epsilon=eps,
            act=act)

        if leaky_relu:
            output = fluid.layers.leaky_relu(output, alpha=relu_negative_slope)

        return output

    @layer
    def dropout(self, input, drop_prob, name, is_test=True):
        fluid = import_fluid()
        if is_test:
            output = input
        else:
            output = fluid.layers.dropout(
                input,
                dropout_prob=drop_prob,
                is_test=is_test,
                name=self.get_unique_output_name(name, 'dropout'))
        return output

    @layer
    def scale(self, input, axis=1, num_axes=1, name=None):
        fluid = import_fluid()

        assert num_axes == 1, "layer scale not support this num_axes[%d] now" % (
            num_axes)

        prefix = name + '_'
        scale_shape = input.shape[axis:axis + num_axes]
        param_attr = fluid.ParamAttr(name=prefix + 'scale')
        scale_param = fluid.layers.create_parameter(
            shape=scale_shape,
            dtype=input.dtype,
            name=name,
            attr=param_attr,
            is_bias=True,
            default_initializer=fluid.initializer.Constant(value=1.0))

        offset_attr = fluid.ParamAttr(name=prefix + 'offset')
        offset_param = fluid.layers.create_parameter(
            shape=scale_shape,
            dtype=input.dtype,
            name=name,
            attr=offset_attr,
            is_bias=True,
            default_initializer=fluid.initializer.Constant(value=0.0))

        output = fluid.layers.elementwise_mul(
            input,
            scale_param,
            axis=axis,
            name=self.get_unique_output_name(name, 'scale_mul'))
        output = fluid.layers.elementwise_add(
            output,
            offset_param,
            axis=axis,
            name=self.get_unique_output_name(name, 'scale_add'))
        return output

    def custom_layer_factory(self):
        """ get a custom layer maker provided by subclass
        """
        raise NotImplementedError(
            '[custom_layer_factory] must be implemented by the subclass.')

    @layer
    def custom_layer(self, inputs, kind, name, *args, **kwargs):
        """ make custom layer
        """
        #FIX ME:
        #   there is a trick for different API between caffe and paddle
        if kind == "DetectionOutput":
            conf_var = inputs[1]
            real_conf_var = self.locate_ancestor(conf_var, ancestor_level=2)
            inputs[1] = real_conf_var[1]

        name = self.get_unique_output_name(name, kind)
        layer_factory = self.custom_layer_factory()
        return layer_factory(kind, inputs, name, *args, **kwargs)
