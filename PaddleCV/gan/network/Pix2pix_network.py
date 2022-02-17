#copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .base_network import conv2d, deconv2d, norm_layer
import paddle.fluid as fluid


class Pix2pix_model(object):
    def __init__(self):
        pass

    def network_G(self, input, name, cfg):
        if cfg.net_G == 'resnet_9block':
            net = build_generator_resnet_blocks(
                input,
                name=name + "_resnet9block",
                n_gen_res=9,
                g_base_dims=cfg.g_base_dims,
                use_dropout=cfg.dropout,
                norm_type=cfg.norm_type)
        elif cfg.net_G == 'resnet_6block':
            net = build_generator_resnet_blocks(
                input,
                name=name + "_resnet6block",
                n_gen_res=6,
                g_base_dims=cfg.g_base_dims,
                use_dropout=cfg.dropout,
                norm_type=cfg.norm_type)
        elif cfg.net_G == 'unet_128':
            net = build_generator_Unet(
                input,
                name=name + "_unet128",
                num_downsample=7,
                g_base_dims=cfg.g_base_dims,
                use_dropout=cfg.dropout,
                norm_type=cfg.norm_type)
        elif cfg.net_G == 'unet_256':
            net = build_generator_Unet(
                input,
                name=name + "_unet256",
                num_downsample=8,
                g_base_dims=cfg.g_base_dims,
                use_dropout=cfg.dropout,
                norm_type=cfg.norm_type)
        else:
            raise NotImplementedError(
                'network G: [%s] is wrong format, please check it' % cfg.net_G)
        return net

    def network_D(self, input, name, cfg):
        if cfg.net_D == 'basic':
            net = build_discriminator_Nlayers(
                input,
                name=name + '_basic',
                d_nlayers=3,
                d_base_dims=cfg.d_base_dims,
                norm_type=cfg.norm_type)
        elif cfg.net_D == 'nlayers':
            net = build_discriminator_Nlayers(
                input,
                name=name + '_nlayers',
                d_nlayers=cfg.d_nlayers,
                d_base_dims=cfg.d_base_dims,
                norm_type=cfg.norm_type)
        elif cfg.net_D == 'pixel':
            net = build_discriminator_Pixel(
                input,
                name=name + '_pixel',
                d_base_dims=cfg.d_base_dims,
                norm_type=cfg.norm_type)
        else:
            raise NotImplementedError(
                'network D: [%s] is wrong format, please check it' % cfg.net_D)
        return net


def build_resnet_block(inputres,
                       dim,
                       name="resnet",
                       use_bias=False,
                       use_dropout=False,
                       norm_type='batch_norm'):
    out_res = fluid.layers.pad2d(inputres, [1, 1, 1, 1], mode="reflect")
    out_res = conv2d(
        input=out_res,
        num_filters=dim,
        filter_size=3,
        stride=1,
        stddev=0.02,
        name=name + "_c1",
        norm=norm_type,
        activation_fn='relu',
        use_bias=use_bias)

    if use_dropout:
        out_res = fluid.layers.dropout(out_res, dropout_prob=0.5)

    out_res = fluid.layers.pad2d(out_res, [1, 1, 1, 1], mode="reflect")
    out_res = conv2d(
        input=out_res,
        num_filters=dim,
        filter_size=3,
        stride=1,
        stddev=0.02,
        name=name + "_c2",
        norm=norm_type,
        use_bias=use_bias)
    return out_res + inputres


def build_generator_resnet_blocks(inputgen,
                                  name="generator",
                                  n_gen_res=9,
                                  g_base_dims=64,
                                  use_dropout=False,
                                  norm_type='batch_norm'):
    ''' generator use resnet block'''
    '''The shape of input should be equal to the shape of output.'''
    use_bias = norm_type == 'instance_norm'
    pad_input = fluid.layers.pad2d(inputgen, [3, 3, 3, 3], mode="reflect")
    o_c1 = conv2d(
        input=pad_input,
        num_filters=g_base_dims,
        filter_size=7,
        stride=1,
        stddev=0.02,
        name=name + "_c1",
        norm=norm_type,
        activation_fn='relu')
    o_c2 = conv2d(
        input=o_c1,
        num_filters=g_base_dims * 2,
        filter_size=3,
        stride=2,
        stddev=0.02,
        padding=1,
        name=name + "_c2",
        norm=norm_type,
        activation_fn='relu')
    res_input = conv2d(
        input=o_c2,
        num_filters=g_base_dims * 4,
        filter_size=3,
        stride=2,
        stddev=0.02,
        padding=1,
        name=name + "_c3",
        norm=norm_type,
        activation_fn='relu')
    for i in range(n_gen_res):
        conv_name = name + "_r{}".format(i + 1)
        res_output = build_resnet_block(
            res_input,
            g_base_dims * 4,
            name=conv_name,
            use_bias=use_bias,
            use_dropout=use_dropout)
        res_input = res_output

    o_c4 = deconv2d(
        input=res_output,
        num_filters=g_base_dims * 2,
        filter_size=3,
        stride=2,
        stddev=0.02,
        padding=[1, 1],
        outpadding=[0, 1, 0, 1],
        name=name + "_c4",
        norm=norm_type,
        activation_fn='relu')
    o_c5 = deconv2d(
        input=o_c4,
        num_filters=g_base_dims,
        filter_size=3,
        stride=2,
        stddev=0.02,
        padding=[1, 1],
        outpadding=[0, 1, 0, 1],
        name=name + "_c5",
        norm=norm_type,
        activation_fn='relu')
    o_p2 = fluid.layers.pad2d(o_c5, [3, 3, 3, 3], mode="reflect")
    o_c6 = conv2d(
        input=o_p2,
        num_filters=3,
        filter_size=7,
        stride=1,
        stddev=0.02,
        name=name + "_c6",
        activation_fn='tanh',
        use_bias=True)

    return o_c6


def Unet_block(inputunet,
               i,
               outer_dim,
               inner_dim,
               num_downsample,
               innermost=False,
               outermost=False,
               norm_type='batch_norm',
               use_bias=False,
               use_dropout=False,
               name=None):
    if outermost == True:
        downconv = conv2d(
            input=inputunet,
            num_filters=inner_dim,
            filter_size=4,
            stride=2,
            stddev=0.02,
            padding=1,
            name=name + '_outermost_dc1',
            use_bias=True)
        i += 1
        mid_block = Unet_block(
            downconv,
            i,
            inner_dim,
            inner_dim * 2,
            num_downsample,
            norm_type=norm_type,
            use_bias=use_bias,
            use_dropout=use_dropout,
            name=name)
        uprelu = fluid.layers.relu(mid_block, name=name + '_outermost_relu')
        updeconv = deconv2d(
            input=uprelu,
            num_filters=outer_dim,
            filter_size=4,
            stride=2,
            stddev=0.02,
            padding=1,
            name=name + '_outermost_uc1',
            activation_fn='tanh',
            use_bias=use_bias)
        return updeconv
    elif innermost == True:
        downrelu = fluid.layers.leaky_relu(
            inputunet, 0.2, name=name + '_innermost_leaky_relu')
        upconv = conv2d(
            input=downrelu,
            num_filters=inner_dim,
            filter_size=4,
            stride=2,
            stddev=0.02,
            padding=1,
            name=name + '_innermost_dc1',
            activation_fn='relu',
            use_bias=use_bias)
        updeconv = deconv2d(
            input=upconv,
            num_filters=outer_dim,
            filter_size=4,
            stride=2,
            stddev=0.02,
            padding=1,
            name=name + '_innermost_uc1',
            norm=norm_type,
            use_bias=use_bias)
        return fluid.layers.concat([inputunet, updeconv], 1)
    else:
        downrelu = fluid.layers.leaky_relu(
            inputunet, 0.2, name=name + '_leaky_relu')
        downnorm = conv2d(
            input=downrelu,
            num_filters=inner_dim,
            filter_size=4,
            stride=2,
            stddev=0.02,
            padding=1,
            name=name + 'dc1',
            norm=norm_type,
            use_bias=use_bias)
        i += 1
        if i < 4:
            mid_block = Unet_block(
                downnorm,
                i,
                inner_dim,
                inner_dim * 2,
                num_downsample,
                norm_type=norm_type,
                use_bias=use_bias,
                name=name + '_mid{}'.format(i))
        elif i < num_downsample - 1:
            mid_block = Unet_block(
                downnorm,
                i,
                inner_dim,
                inner_dim,
                num_downsample,
                norm_type=norm_type,
                use_bias=use_bias,
                use_dropout=use_dropout,
                name=name + '_mid{}'.format(i))
        else:
            mid_block = Unet_block(
                downnorm,
                i,
                inner_dim,
                inner_dim,
                num_downsample,
                innermost=True,
                norm_type=norm_type,
                use_bias=use_bias,
                name=name + '_innermost')
        uprelu = fluid.layers.relu(mid_block, name=name + '_relu')
        updeconv = deconv2d(
            input=uprelu,
            num_filters=outer_dim,
            filter_size=4,
            stride=2,
            stddev=0.02,
            padding=1,
            name=name + '_uc1',
            norm=norm_type,
            use_bias=use_bias)

        if use_dropout:
            updeconv = fluid.layers.dropout(updeconv, dropout_prob=0.5)
        return fluid.layers.concat([inputunet, updeconv], 1)


def UnetSkipConnectionBlock(input,
                            i,
                            num_downs,
                            outer_nc,
                            inner_nc,
                            outermost=False,
                            innermost=False,
                            norm='batch_norm',
                            use_dropout=False,
                            name=""):
    use_bias = norm == "instance"
    if outermost:
        downconv = conv2d(
            input,
            num_filters=inner_nc,
            filter_size=4,
            stride=2,
            padding=1,
            use_bias=use_bias,
            name=name + '_down_conv')
        i += 1
        ngf = inner_nc
        sub_res = UnetSkipConnectionBlock(
            downconv,
            i,
            num_downs,
            outer_nc=ngf,
            inner_nc=ngf * 2,
            norm=norm,
            name=name + '_u%d' % i)
        uprelu = fluid.layers.relu(sub_res)
        upconv = deconv2d(
            input=uprelu,
            num_filters=outer_nc,
            filter_size=4,
            stride=2,
            padding=1,
            activation_fn='tanh',
            name=name + '_up_conv')
        return upconv
    elif innermost:
        downrelu = fluid.layers.leaky_relu(input, 0.2)
        downconv = conv2d(
            input=downrelu,
            num_filters=inner_nc,
            filter_size=4,
            stride=2,
            padding=1,
            use_bias=use_bias,
            name=name + '_down_conv')
        uprelu = fluid.layers.relu(downconv)
        upconv = deconv2d(
            input=uprelu,
            num_filters=outer_nc,
            filter_size=4,
            stride=2,
            padding=1,
            use_bias=use_bias,
            norm=norm,
            name=name + '_up_conv')
        return fluid.layers.concat([input, upconv], 1)
    else:
        downrelu = fluid.layers.leaky_relu(input, 0.2)
        downconv = conv2d(
            input=downrelu,
            num_filters=inner_nc,
            filter_size=4,
            stride=2,
            padding=1,
            use_bias=use_bias,
            norm=norm,
            name=name + '_down_conv')
        i += 1
        ngf = inner_nc
        if i < 4:
            sub_res = UnetSkipConnectionBlock(
                downconv,
                i,
                num_downs,
                outer_nc=ngf,
                inner_nc=ngf * 2,
                norm=norm,
                name=name + '_u%d' % i)
        elif i < num_downs - 1:
            sub_res = UnetSkipConnectionBlock(
                downconv,
                i,
                num_downs,
                outer_nc=ngf,
                inner_nc=ngf,
                norm=norm,
                name=name + '_u%d' % i)

        else:
            sub_res = UnetSkipConnectionBlock(
                downconv,
                i,
                num_downs,
                outer_nc=ngf,
                inner_nc=ngf,
                innermost=True,
                norm=norm,
                name=name + '_u%d' % i)

        uprelu = fluid.layers.relu(sub_res)
        upconv = deconv2d(
            input=uprelu,
            num_filters=outer_nc,
            filter_size=4,
            stride=2,
            padding=1,
            use_bias=use_bias,
            norm=norm,
            name=name + '_up_conv')
        out = upconv
        if use_dropout:
            out = fluid.layers.dropout(out, 0.5)
        return fluid.layers.concat([input, out], 1)


def build_generator_Unet(input,
                         name="",
                         num_downsample=8,
                         g_base_dims=64,
                         use_dropout=False,
                         norm_type='batch_norm'):
    ''' generator use Unet'''
    i = 0
    output = UnetSkipConnectionBlock(
        input,
        i,
        num_downsample,
        3,
        g_base_dims,
        outermost=True,
        norm=norm_type,
        name=name + '_u%d' % i)

    return output


def build_discriminator_Nlayers(inputdisc,
                                name="discriminator",
                                d_nlayers=3,
                                d_base_dims=64,
                                norm_type='batch_norm'):
    use_bias = norm_type != 'batch_norm'
    dis_input = conv2d(
        input=inputdisc,
        num_filters=d_base_dims,
        filter_size=4,
        stride=2,
        stddev=0.02,
        padding=1,
        name=name + "_c1",
        activation_fn='leaky_relu',
        relufactor=0.2,
        use_bias=True)
    d_dims = d_base_dims
    for i in range(d_nlayers - 1):
        conv_name = name + "_c{}".format(i + 2)
        d_dims *= 2
        dis_output = conv2d(
            input=dis_input,
            num_filters=d_dims,
            filter_size=4,
            stride=2,
            stddev=0.02,
            padding=1,
            name=conv_name,
            norm=norm_type,
            activation_fn='leaky_relu',
            relufactor=0.2,
            use_bias=use_bias)
        dis_input = dis_output
    last_dims = min(2**d_nlayers, 8)
    o_c4 = conv2d(
        input=dis_output,
        num_filters=d_base_dims * last_dims,
        filter_size=4,
        stride=1,
        stddev=0.02,
        padding=1,
        name=name + "_c{}".format(d_nlayers + 1),
        norm=norm_type,
        activation_fn='leaky_relu',
        relufactor=0.2,
        use_bias=use_bias)
    o_c5 = conv2d(
        input=o_c4,
        num_filters=1,
        filter_size=4,
        stride=1,
        stddev=0.02,
        padding=1,
        name=name + "_c{}".format(d_nlayers + 2),
        use_bias=True)
    return o_c5


def build_discriminator_Pixel(inputdisc,
                              name="discriminator",
                              d_base_dims=64,
                              norm_type='batch_norm'):
    use_bias = norm_type != 'instance_norm'
    o_c1 = conv2d(
        input=inputdisc,
        num_filters=d_base_dims,
        filter_size=1,
        stride=1,
        stddev=0.02,
        name=name + '_c1',
        activation_fn='leaky_relu',
        relufactor=0.2,
        use_bias=True)
    o_c2 = conv2d(
        input=o_c1,
        num_filters=d_base_dims * 2,
        filter_size=1,
        stride=1,
        stddev=0.02,
        name=name + '_c2',
        norm=norm_type,
        activation_fn='leaky_relu',
        relufactor=0.2,
        use_bias=use_bias)
    o_c3 = conv2d(
        o_c2,
        num_filters=1,
        filter_size=1,
        stride=1,
        stddev=0.02,
        name=name + '_c3',
        use_bias=use_bias)
    return o_c3
