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

from .base_network import conv2d, deconv2d, norm_layer, conv2d_spectral_norm
import paddle.fluid as fluid
import numpy as np


class SPADE_model(object):
    def __init__(self):
        pass

    def network_G(self, input, name, cfg, is_test=False):
        nf = cfg.ngf
        num_up_layers = 5
        sw = cfg.crop_width // (2**num_up_layers)
        sh = cfg.crop_height // (2**num_up_layers)
        seg = input
        x = fluid.layers.resize_nearest(
            seg, out_shape=(sh, sw), align_corners=False)
        x = conv2d(
            x,
            16 * nf,
            3,
            padding=1,
            name=name + "_fc",
            use_bias=True,
            initial="kaiming",
            is_test=is_test)
        x = self.SPADEResnetBlock(
            x,
            seg,
            16 * nf,
            16 * nf,
            cfg,
            name=name + "_head_0",
            is_test=is_test)
        x = fluid.layers.resize_nearest(x, scale=2.0, align_corners=False)
        x = self.SPADEResnetBlock(
            x,
            seg,
            16 * nf,
            16 * nf,
            cfg,
            name=name + "_G_middle_0",
            is_test=is_test)
        x = self.SPADEResnetBlock(
            x,
            seg,
            16 * nf,
            16 * nf,
            cfg,
            name=name + "_G_middle_1",
            is_test=is_test)
        x = fluid.layers.resize_nearest(x, scale=2.0, align_corners=False)

        x = self.SPADEResnetBlock(
            x, seg, 16 * nf, 8 * nf, cfg, name=name + "_up_0", is_test=is_test)
        x = fluid.layers.resize_nearest(x, scale=2.0, align_corners=False)
        x = self.SPADEResnetBlock(
            x, seg, 8 * nf, 4 * nf, cfg, name=name + "_up_1", is_test=is_test)
        x = fluid.layers.resize_nearest(x, scale=2.0, align_corners=False)
        x = self.SPADEResnetBlock(
            x, seg, 4 * nf, 2 * nf, cfg, name=name + "_up_2", is_test=is_test)
        x = fluid.layers.resize_nearest(x, scale=2.0, align_corners=False)
        x = self.SPADEResnetBlock(
            x, seg, 2 * nf, 1 * nf, cfg, name=name + "_up_3", is_test=is_test)
        x = fluid.layers.leaky_relu(
            x, alpha=0.2, name=name + '_conv_img_leaky_relu')
        x = conv2d(
            x,
            3,
            3,
            padding=1,
            name=name + "_conv_img",
            use_bias=True,
            initial="kaiming",
            is_test=is_test)
        x = fluid.layers.tanh(x)

        return x

    def SPADEResnetBlock(self, x, seg, fin, fout, opt, name, is_test=False):
        learn_shortcut = (fin != fout)
        fmiddle = min(fin, fout)
        semantic_nc = opt.label_nc + (0 if opt.no_instance else 1)
        if learn_shortcut:
            x_s = self.SPADE(
                x, seg, fin, name=name + ".norm_s", is_test=is_test)
            x_s = conv2d_spectral_norm(
                x_s,
                fout,
                1,
                use_bias=False,
                name=name + ".conv_s",
                is_test=is_test)
        else:
            x_s = x
        dx = self.SPADE(x, seg, fin, name=name + ".norm_0", is_test=is_test)
        dx = fluid.layers.leaky_relu(dx, alpha=0.2, name=name + '_leaky_relu0')
        dx = conv2d_spectral_norm(
            dx,
            fmiddle,
            3,
            padding=1,
            name=name + ".conv_0",
            use_bias=True,
            is_test=is_test)

        dx = self.SPADE(
            dx, seg, fmiddle, name=name + ".norm_1", is_test=is_test)
        dx = fluid.layers.leaky_relu(dx, alpha=0.2, name=name + '_leaky_relu1')
        dx = conv2d_spectral_norm(
            dx,
            fout,
            3,
            padding=1,
            name=name + ".conv_1",
            use_bias=True,
            is_test=is_test)

        output = dx + x_s
        return output

    def SPADE(self, input, seg_map, norm_nc, name, is_test=False):
        nhidden = 128
        ks = 3
        pw = ks // 2
        seg_map = fluid.layers.resize_nearest(
            seg_map, out_shape=input.shape[2:], align_corners=False)
        actv = conv2d(
            seg_map,
            nhidden,
            ks,
            padding=pw,
            activation_fn='relu',
            name=name + ".mlp_shared.0",
            initial="kaiming",
            use_bias=True)
        gamma = conv2d(
            actv,
            norm_nc,
            ks,
            padding=pw,
            name=name + ".mlp_gamma",
            initial="kaiming",
            use_bias=True)
        beta = conv2d(
            actv,
            norm_nc,
            ks,
            padding=pw,
            name=name + ".mlp_beta",
            initial="kaiming",
            use_bias=True)
        param_attr = fluid.ParamAttr(
            name=name + ".param_free_norm.weight",
            initializer=fluid.initializer.Constant(value=1.0),
            trainable=False)
        bias_attr = fluid.ParamAttr(
            name=name + ".param_free_norm.bias",
            initializer=fluid.initializer.Constant(0.0),
            trainable=False)

        norm = fluid.layers.batch_norm(
            input=input,
            name=name,
            param_attr=param_attr,
            bias_attr=bias_attr,
            moving_mean_name=name + ".param_free_norm.running_mean",
            moving_variance_name=name + ".param_free_norm.running_var",
            is_test=is_test)
        out = norm * (1 + gamma) + beta
        return out

    def network_D(self, input, name, cfg):
        num_D = 2
        result = []
        for i in range(num_D):
            out = build_discriminator_Nlayers(input, name=name + "_%d" % i)
            result.append(out)
            input = fluid.layers.pool2d(
                input,
                pool_size=3,
                pool_type="avg",
                pool_stride=2,
                pool_padding=1,
                name=name + "_pool%d" % i)

        return result


def build_discriminator_Nlayers(input,
                                name="discriminator",
                                d_nlayers=4,
                                d_base_dims=64,
                                norm_type='instance_norm'):
    kw = 4
    padw = int(np.ceil((kw - 1.0) / 2))
    nf = d_base_dims
    res_list = []
    res1 = conv2d(
        input,
        nf,
        kw,
        2,
        0.02,
        1,
        name=name + ".model0.0",
        activation_fn='leaky_relu',
        relufactor=0.2,
        initial="kaiming",
        use_bias=True)
    d_dims = d_base_dims
    res_list.append(res1)
    for i in range(1, d_nlayers):
        conv_name = name + ".model{}.0.0".format(i)
        nf = min(nf * 2, 512)
        stride = 1 if i == d_nlayers - 1 else 2
        dis_output = conv2d_spectral_norm(
            res_list[-1],
            nf,
            kw,
            stride,
            0.02,
            1,
            name=conv_name,
            norm=norm_type,
            activation_fn='leaky_relu',
            relufactor=0.2,
            use_bias=False,
            norm_affine=False)
        res_list.append(dis_output)
    o_c4 = conv2d(
        res_list[-1],
        1,
        4,
        1,
        0.02,
        1,
        name + ".model{}.0".format(d_nlayers),
        initial="kaiming",
        use_bias=True)
    res_list.append(o_c4)
    return res_list
