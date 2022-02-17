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

from .base_network import conv2d, deconv2d, norm_layer, linear
import paddle.fluid as fluid
import numpy as np

MAX_DIM = 64 * 16


class AttGAN_model(object):
    def __init__(self):
        pass

    def network_G(self,
                  input,
                  label_org,
                  label_trg,
                  cfg,
                  name="generator",
                  is_test=False):
        _a = label_org
        _b = label_trg
        z = self.Genc(
            input,
            name=name + '_Genc',
            dim=cfg.g_base_dims,
            n_layers=cfg.n_layers,
            is_test=is_test)
        fake_image = self.Gdec(
            z, _b, name=name + '_Gdec', dim=cfg.g_base_dims, is_test=is_test)

        rec_image = self.Gdec(
            z, _a, name=name + '_Gdec', dim=cfg.g_base_dims, is_test=is_test)
        return fake_image, rec_image

    def network_D(self, input, cfg, name="discriminator"):
        return self.D(input,
                      n_atts=cfg.c_dim,
                      name=name,
                      dim=cfg.d_base_dims,
                      fc_dim=cfg.d_fc_dim,
                      norm=cfg.dis_norm,
                      n_layers=cfg.n_layers)

    def concat(self, z, a):
        """Concatenate attribute vector on feature map axis."""
        batch = fluid.layers.shape(z)[0]
        ones = fluid.layers.fill_constant(
            shape=[batch, a.shape[1], z.shape[2], z.shape[3]],
            dtype="float32",
            value=1.0)
        return fluid.layers.concat(
            [z, fluid.layers.elementwise_mul(
                ones, a, axis=0)], axis=1)

    def Genc(self, input, dim=64, n_layers=5, name='G_enc_', is_test=False):
        z = input
        zs = []
        for i in range(n_layers):
            d = min(dim * 2**i, MAX_DIM)
            #SAME padding
            z = conv2d(
                input=z,
                num_filters=d,
                filter_size=4,
                stride=2,
                padding_type='SAME',
                norm='batch_norm',
                activation_fn='leaky_relu',
                name=name + str(i),
                use_bias=False,
                relufactor=0.01,
                initial='kaiming',
                is_test=is_test)
            zs.append(z)

        return zs

    def Gdec(self,
             zs,
             a,
             dim=64,
             n_layers=5,
             shortcut_layers=1,
             inject_layers=1,
             name='G_dec_',
             is_test=False):
        shortcut_layers = min(shortcut_layers, n_layers - 1)
        inject_layers = min(inject_layers, n_layers - 1)

        z = self.concat(zs[-1], a)
        for i in range(n_layers):
            if i < n_layers - 1:
                d = min(dim * 2**(n_layers - 1 - i), MAX_DIM)
                z = deconv2d(
                    input=z,
                    num_filters=d,
                    filter_size=4,
                    stride=2,
                    padding_type='SAME',
                    name=name + str(i),
                    norm='batch_norm',
                    activation_fn='relu',
                    use_bias=False,
                    initial='kaiming',
                    is_test=is_test)
                if shortcut_layers > i:
                    z = fluid.layers.concat([z, zs[n_layers - 2 - i]], axis=1)
                if inject_layers > i:
                    z = self.concat(z, a)
            else:
                x = z = deconv2d(
                    input=z,
                    num_filters=3,
                    filter_size=4,
                    stride=2,
                    padding_type='SAME',
                    name=name + str(i),
                    activation_fn='tanh',
                    use_bias=True,
                    initial='kaiming',
                    is_test=is_test)
        return x

    def D(self,
          x,
          n_atts=13,
          dim=64,
          fc_dim=1024,
          n_layers=5,
          norm='instance_norm',
          name='D_'):

        y = x
        for i in range(n_layers):
            d = min(dim * 2**i, MAX_DIM)
            y = conv2d(
                input=y,
                num_filters=d,
                filter_size=4,
                stride=2,
                norm=norm,
                padding=1,
                activation_fn='leaky_relu',
                name=name + str(i),
                use_bias=(norm == None),
                relufactor=0.01,
                initial='kaiming')

        logit_gan = linear(
            input=y,
            output_size=fc_dim,
            activation_fn='relu',
            name=name + 'fc_adv_1',
            initial='kaiming')
        logit_gan = linear(
            logit_gan, 1, name=name + 'fc_adv_2', initial='kaiming')

        logit_att = linear(
            input=y,
            output_size=fc_dim,
            activation_fn='relu',
            name=name + 'fc_cls_1',
            initial='kaiming')
        logit_att = linear(logit_att, n_atts, name=name + 'fc_cls_2')

        return logit_gan, logit_att
