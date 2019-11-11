from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import paddle.fluid as fluid
import collections
import re
import math
import copy
from .layers import conv2d, init_batch_norm_layer, init_fc_layer


__all__ = ['EfficientNet', 'EfficientNetB0', 'EfficientNetB1', 'EfficientNetB2', 'EfficientNetB3', 'EfficientNetB4',
           'EfficientNetB5', 'EfficientNetB6', 'EfficientNetB7']

GlobalParams = collections.namedtuple('GlobalParams', [
    'batch_norm_momentum', 'batch_norm_epsilon', 'dropout_rate',
    'num_classes', 'width_coefficient', 'depth_coefficient',
    'depth_divisor', 'min_depth', 'drop_connect_rate', ])

BlockArgs = collections.namedtuple('BlockArgs', [
    'kernel_size', 'num_repeat', 'input_filters', 'output_filters',
    'expand_ratio', 'id_skip', 'stride', 'se_ratio'])

GlobalParams.__new__.__defaults__ = (None,) * len(GlobalParams._fields)
BlockArgs.__new__.__defaults__ = (None,) * len(BlockArgs._fields)


def efficientnet_params(model_name):
    """ Map EfficientNet model name to parameter coefficients. """
    params_dict = {
        # Coefficients:   width,depth,resolution,dropout
        'efficientnet-b0': (1.0, 1.0, 224, 0.2),
        'efficientnet-b1': (1.0, 1.1, 240, 0.2),
        'efficientnet-b2': (1.1, 1.2, 260, 0.3),
        'efficientnet-b3': (1.2, 1.4, 300, 0.3),
        'efficientnet-b4': (1.4, 1.8, 380, 0.4),
        'efficientnet-b5': (1.6, 2.2, 456, 0.4),
        'efficientnet-b6': (1.8, 2.6, 528, 0.5),
        'efficientnet-b7': (2.0, 3.1, 600, 0.5),
    }
    return params_dict[model_name]


def efficientnet(width_coefficient=None, depth_coefficient=None,
                 dropout_rate=0.2, drop_connect_rate=0.2):
    """ Get block arguments according to parameter and coefficients. """
    blocks_args = [
        'r1_k3_s11_e1_i32_o16_se0.25', 'r2_k3_s22_e6_i16_o24_se0.25',
        'r2_k5_s22_e6_i24_o40_se0.25', 'r3_k3_s22_e6_i40_o80_se0.25',
        'r3_k5_s11_e6_i80_o112_se0.25', 'r4_k5_s22_e6_i112_o192_se0.25',
        'r1_k3_s11_e6_i192_o320_se0.25',
    ]
    blocks_args = BlockDecoder.decode(blocks_args)

    global_params = GlobalParams(
        batch_norm_momentum=0.99,
        batch_norm_epsilon=1e-3,
        dropout_rate=dropout_rate,
        drop_connect_rate=drop_connect_rate,
        num_classes=1000,
        width_coefficient=width_coefficient,
        depth_coefficient=depth_coefficient,
        depth_divisor=8,
        min_depth=None
    )

    return blocks_args, global_params


def get_model_params(model_name, override_params):
    """ Get the block args and global params for a given model """
    if model_name.startswith('efficientnet'):
        w, d, _, p = efficientnet_params(model_name)
        blocks_args, global_params = efficientnet(width_coefficient=w, depth_coefficient=d, dropout_rate=p)
    else:
        raise NotImplementedError('model name is not pre-defined: %s' % model_name)
    if override_params:
        global_params = global_params._replace(**override_params)
    return blocks_args, global_params


def round_filters(filters, global_params):
    """ Calculate and round number of filters based on depth multiplier. """
    multiplier = global_params.width_coefficient
    if not multiplier:
        return filters
    divisor = global_params.depth_divisor
    min_depth = global_params.min_depth
    filters *= multiplier
    min_depth = min_depth or divisor
    new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
    if new_filters < 0.9 * filters:  # prevent rounding by more than 10%
        new_filters += divisor
    return int(new_filters)


def round_repeats(repeats, global_params):
    """ Round number of filters based on depth multiplier. """
    multiplier = global_params.depth_coefficient
    if not multiplier:
        return repeats
    return int(math.ceil(multiplier * repeats))


class EfficientNet():
    def __init__(self, name='b0', padding_type='SAME', override_params=None, is_test=False, use_se=True):
        valid_names = ['b' + str(i) for i in range(8)]
        assert name in valid_names, 'efficient name should be in b0~b7'
        model_name = 'efficientnet-' + name
        self._blocks_args, self._global_params = get_model_params(model_name, override_params)
        self._bn_mom = self._global_params.batch_norm_momentum
        self._bn_eps = self._global_params.batch_norm_epsilon
        self.is_test = is_test
        self.padding_type = padding_type
        self.use_se = use_se

    def net(self, input, class_dim=1000, is_test=False):

        conv = self.extract_features(input, is_test=is_test)

        out_channels = round_filters(1280, self._global_params)
        conv = self.conv_bn_layer(conv,
                                  num_filters=out_channels,
                                  filter_size=1,
                                  bn_act='swish',
                                  bn_mom=self._bn_mom,
                                  bn_eps=self._bn_eps,
                                  padding_type=self.padding_type,
                                  name='',
                                  conv_name='_conv_head',
                                  bn_name='_bn1')

        pool = fluid.layers.pool2d(input=conv, pool_type='avg', global_pooling=True, use_cudnn=False)

        if self._global_params.dropout_rate:
            pool = fluid.layers.dropout(pool, self._global_params.dropout_rate, dropout_implementation='upscale_in_train')

        param_attr, bias_attr = init_fc_layer(class_dim, '_fc')
        out = fluid.layers.fc(pool, class_dim, name='_fc', param_attr=param_attr, bias_attr=bias_attr)
        return out

    def _drop_connect(self, inputs, prob, is_test):
        if is_test:
            return inputs
        keep_prob = 1.0 - prob
        random_tensor = keep_prob + fluid.layers.uniform_random_batch_size_like(inputs, [-1, 1, 1, 1], min=0., max=1.)
        binary_tensor = fluid.layers.floor(random_tensor)
        output = inputs / keep_prob * binary_tensor
        return output

    def _expand_conv_norm(self, inputs, block_args, is_test, name=None):
        # Expansion phase
        oup = block_args.input_filters * block_args.expand_ratio  # number of output channels

        if block_args.expand_ratio != 1:
            conv = self.conv_bn_layer(inputs,
                                      num_filters=oup,
                                      filter_size=1,
                                      bn_act=None,
                                      bn_mom=self._bn_mom,
                                      bn_eps=self._bn_eps,
                                      padding_type=self.padding_type,
                                      name=name,
                                      conv_name=name + '_expand_conv',
                                      bn_name='_bn0')

        return conv

    def _depthwise_conv_norm(self, inputs, block_args, is_test, name=None):
        k = block_args.kernel_size
        s = block_args.stride
        if isinstance(s, list) or isinstance(s, tuple):
            s = s[0]
        oup = block_args.input_filters * block_args.expand_ratio  # number of output channels

        conv = self.conv_bn_layer(inputs,
                                  num_filters=oup,
                                  filter_size=k,
                                  stride=s,
                                  num_groups=oup,
                                  bn_act=None,
                                  padding_type=self.padding_type,
                                  bn_mom=self._bn_mom,
                                  bn_eps=self._bn_eps,
                                  name=name,
                                  use_cudnn=False,
                                  conv_name=name + '_depthwise_conv',
                                  bn_name='_bn1')

        return conv

    def _project_conv_norm(self, inputs, block_args, is_test, name=None):
        final_oup = block_args.output_filters
        conv = self.conv_bn_layer(inputs,
                                  num_filters=final_oup,
                                  filter_size=1,
                                  bn_act=None,
                                  padding_type=self.padding_type,
                                  bn_mom=self._bn_mom,
                                  bn_eps=self._bn_eps,
                                  name=name,
                                  conv_name=name + '_project_conv',
                                  bn_name='_bn2')
        return conv

    def conv_bn_layer(self, input, filter_size, num_filters, stride=1, num_groups=1, padding_type="SAME", conv_act=None,
                      bn_act='swish', use_cudnn=True, use_bn=True, bn_mom=0.9, bn_eps=1e-05, use_bias=False, name=None,
                      conv_name=None, bn_name=None):
        conv = conv2d(
            input=input,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            groups=num_groups,
            act=conv_act,
            padding_type=padding_type,
            use_cudnn=use_cudnn,
            name=conv_name,
            use_bias=use_bias)

        if use_bn == False:
            return conv
        else:
            bn_name = name + bn_name
            param_attr, bias_attr = init_batch_norm_layer(bn_name)
            return fluid.layers.batch_norm(input=conv,
                                           act=bn_act,
                                           momentum=bn_mom,
                                           epsilon=bn_eps,
                                           name=bn_name,
                                           moving_mean_name=bn_name + '_mean',
                                           moving_variance_name=bn_name + '_variance',
                                           param_attr=param_attr,
                                           bias_attr=bias_attr)

    def _conv_stem_norm(self, inputs, is_test):
        out_channels = round_filters(32, self._global_params)
        bn = self.conv_bn_layer(inputs, num_filters=out_channels, filter_size=3, stride=2, bn_act=None,
                                bn_mom=self._bn_mom, padding_type=self.padding_type,
                                bn_eps=self._bn_eps, name='', conv_name='_conv_stem', bn_name='_bn0')

        return bn

    def mb_conv_block(self, inputs, block_args, is_test=False, drop_connect_rate=None, name=None):
        # Expansion and Depthwise Convolution
        oup = block_args.input_filters * block_args.expand_ratio  # number of output channels
        has_se = self.use_se and (block_args.se_ratio is not None) and (0 < block_args.se_ratio <= 1)
        id_skip = block_args.id_skip  # skip connection and drop connect
        conv = inputs
        if block_args.expand_ratio != 1:
            conv = fluid.layers.swish(self._expand_conv_norm(conv, block_args, is_test, name))

        conv = fluid.layers.swish(self._depthwise_conv_norm(conv, block_args, is_test, name))

        # Squeeze and Excitation
        if has_se:
            num_squeezed_channels = max(1, int(block_args.input_filters * block_args.se_ratio))
            conv = self.se_block(conv, num_squeezed_channels, oup, name)

        conv = self._project_conv_norm(conv, block_args, is_test, name)

        # Skip connection and drop connect
        input_filters, output_filters = block_args.input_filters, block_args.output_filters
        if id_skip and block_args.stride == 1 and input_filters == output_filters:
            if drop_connect_rate:
                conv = self._drop_connect(conv, drop_connect_rate, self.is_test)
            conv = fluid.layers.elementwise_add(conv, inputs)

        return conv

    def se_block(self, inputs, num_squeezed_channels, oup, name):
        x_squeezed = fluid.layers.pool2d(
            input=inputs,
            pool_type='avg',
            global_pooling=True,
            use_cudnn=False)
        x_squeezed = conv2d(x_squeezed,
                            num_filters=num_squeezed_channels,
                            filter_size=1,
                            use_bias=True,
                            padding_type=self.padding_type,
                            act='swish',
                            name=name + '_se_reduce')
        x_squeezed = conv2d(x_squeezed,
                            num_filters=oup,
                            filter_size=1,
                            use_bias=True,
                            padding_type=self.padding_type,
                            name=name + '_se_expand')
        se_out = inputs * fluid.layers.sigmoid(x_squeezed)
        return se_out

    def extract_features(self, inputs, is_test):
        """ Returns output of the final convolution layer """

        conv = fluid.layers.swish(self._conv_stem_norm(inputs, is_test=is_test))

        block_args_copy = copy.deepcopy(self._blocks_args)
        idx = 0
        block_size = 0
        for block_arg in block_args_copy:
            block_arg = block_arg._replace(
                input_filters=round_filters(block_arg.input_filters, self._global_params),
                output_filters=round_filters(block_arg.output_filters, self._global_params),
                num_repeat=round_repeats(block_arg.num_repeat, self._global_params)
            )
            block_size += 1
            for _ in range(block_arg.num_repeat - 1):
                block_size += 1

        for block_args in self._blocks_args:

            # Update block input and output filters based on depth multiplier.
            block_args = block_args._replace(
                input_filters=round_filters(block_args.input_filters, self._global_params),
                output_filters=round_filters(block_args.output_filters, self._global_params),
                num_repeat=round_repeats(block_args.num_repeat, self._global_params)
            )

            # The first block needs to take care of stride and filter size increase.
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / block_size
            conv = self.mb_conv_block(conv, block_args, is_test, drop_connect_rate, '_blocks.' + str(idx) + '.')

            idx += 1
            if block_args.num_repeat > 1:
                block_args = block_args._replace(input_filters=block_args.output_filters, stride=1)
            for _ in range(block_args.num_repeat - 1):
                drop_connect_rate = self._global_params.drop_connect_rate
                if drop_connect_rate:
                    drop_connect_rate *= float(idx) / block_size
                conv = self.mb_conv_block(conv, block_args, is_test, drop_connect_rate, '_blocks.' + str(idx) + '.')
                idx += 1

        return conv

    def shortcut(self, input, data_residual):
        return fluid.layers.elementwise_add(input, data_residual)


class BlockDecoder(object):
    """ Block Decoder for readability, straight from the official TensorFlow repository """

    @staticmethod
    def _decode_block_string(block_string):
        """ Gets a block through a string notation of arguments. """
        assert isinstance(block_string, str)

        ops = block_string.split('_')
        options = {}
        for op in ops:
            splits = re.split(r'(\d.*)', op)
            if len(splits) >= 2:
                key, value = splits[:2]
                options[key] = value

        # Check stride
        assert (('s' in options and len(options['s']) == 1) or
                (len(options['s']) == 2 and options['s'][0] == options['s'][1]))

        return BlockArgs(
            kernel_size=int(options['k']),
            num_repeat=int(options['r']),
            input_filters=int(options['i']),
            output_filters=int(options['o']),
            expand_ratio=int(options['e']),
            id_skip=('noskip' not in block_string),
            se_ratio=float(options['se']) if 'se' in options else None,
            stride=[int(options['s'][0])])

    @staticmethod
    def _encode_block_string(block):
        """Encodes a block to a string."""
        args = [
            'r%d' % block.num_repeat,
            'k%d' % block.kernel_size,
            's%d%d' % (block.strides[0], block.strides[1]),
            'e%s' % block.expand_ratio,
            'i%d' % block.input_filters,
            'o%d' % block.output_filters
        ]
        if 0 < block.se_ratio <= 1:
            args.append('se%s' % block.se_ratio)
        if block.id_skip is False:
            args.append('noskip')
        return '_'.join(args)

    @staticmethod
    def decode(string_list):
        """
        Decodes a list of string notations to specify blocks inside the network.

        :param string_list: a list of strings, each string is a notation of block
        :return: a list of BlockArgs namedtuples of block args
        """
        assert isinstance(string_list, list)
        blocks_args = []
        for block_string in string_list:
            blocks_args.append(BlockDecoder._decode_block_string(block_string))
        return blocks_args

    @staticmethod
    def encode(blocks_args):
        """
        Encodes a list of BlockArgs to a list of strings.

        :param blocks_args: a list of BlockArgs namedtuples of block args
        :return: a list of strings, each string is a notation of block
        """
        block_strings = []
        for block in blocks_args:
            block_strings.append(BlockDecoder._encode_block_string(block))
        return block_strings


def EfficientNetB0(is_test=False, padding_type='SAME', override_params=None, use_se=True):
    model = EfficientNet(name='b0', is_test=is_test, padding_type=padding_type, override_params=override_params, use_se=use_se)
    return model


def EfficientNetB1(is_test=False, padding_type='SAME', override_params=None, use_se=True):
    model = EfficientNet(name='b1', is_test=is_test, padding_type=padding_type, override_params=override_params, use_se=use_se)
    return model


def EfficientNetB2(is_test=False, padding_type='SAME', override_params=None, use_se=True):
    model = EfficientNet(name='b2', is_test=is_test, padding_type=padding_type, override_params=override_params, use_se=use_se)
    return model


def EfficientNetB3(is_test=False, padding_type='SAME', override_params=None, use_se=True):
    model = EfficientNet(name='b3', is_test=is_test, padding_type=padding_type, override_params=override_params, use_se=use_se)
    return model


def EfficientNetB4(is_test=False, padding_type='SAME', override_params=None, use_se=True):
    model = EfficientNet(name='b4', is_test=is_test, padding_type=padding_type, override_params=override_params, use_se=use_se)
    return model


def EfficientNetB5(is_test=False, padding_type='SAME', override_params=None, use_se=True):
    model = EfficientNet(name='b5', is_test=is_test, padding_type=padding_type, override_params=override_params, use_se=use_se)
    return model


def EfficientNetB6(is_test=False, padding_type='SAME', override_params=None, use_se=True):
    model = EfficientNet(name='b6', is_test=is_test, padding_type=padding_type, override_params=override_params, use_se=use_se)
    return model


def EfficientNetB7(is_test=False, padding_type='SAME', override_params=None, use_se=True):
    model = EfficientNet(name='b7', is_test=is_test, padding_type=padding_type, override_params=override_params, use_se=use_se)
    return model