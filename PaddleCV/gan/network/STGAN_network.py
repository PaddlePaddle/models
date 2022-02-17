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


class STGAN_model(object):
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
            n_layers=cfg.n_layers,
            dim=cfg.g_base_dims,
            is_test=is_test)
        zb = self.GRU(z,
                      fluid.layers.elementwise_sub(_b, _a),
                      name=name + '_GRU',
                      dim=cfg.g_base_dims,
                      n_layers=cfg.gru_n_layers,
                      is_test=is_test) if cfg.use_gru else z
        fake_image = self.Gdec(
            zb,
            fluid.layers.elementwise_sub(_b, _a),
            name=name + '_Gdec',
            dim=cfg.g_base_dims,
            n_layers=cfg.n_layers,
            is_test=is_test)

        za = self.GRU(z,
                      fluid.layers.elementwise_sub(_a, _a),
                      name=name + '_GRU',
                      dim=cfg.g_base_dims,
                      n_layers=cfg.gru_n_layers,
                      is_test=is_test) if cfg.use_gru else z
        rec_image = self.Gdec(
            za,
            fluid.layers.elementwise_sub(_a, _a),
            name=name + '_Gdec',
            dim=cfg.g_base_dims,
            n_layers=cfg.n_layers,
            is_test=is_test)
        return fake_image, rec_image

    def network_D(self, input, cfg, name="discriminator"):
        return self.D(input,
                      n_atts=cfg.c_dim,
                      dim=cfg.d_base_dims,
                      fc_dim=cfg.d_fc_dim,
                      norm=cfg.dis_norm,
                      n_layers=cfg.n_layers,
                      name=name)

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
            z = conv2d(
                input=z,
                num_filters=d,
                filter_size=4,
                stride=2,
                padding_type='SAME',
                norm="batch_norm",
                activation_fn='leaky_relu',
                name=name + str(i),
                use_bias=False,
                relufactor=0.2,
                initial='kaiming',
                is_test=is_test)
            zs.append(z)

        return zs

    def GRU(self,
            zs,
            a,
            dim=64,
            n_layers=4,
            inject_layers=4,
            kernel_size=3,
            norm=None,
            pass_state='lstate',
            name='G_gru_',
            is_test=False):

        zs_ = [zs[-1]]
        state = self.concat(zs[-1], a)
        for i in range(n_layers):
            d = min(dim * 2**(n_layers - 1 - i), MAX_DIM)
            output = self.gru_cell(
                in_data=zs[n_layers - 1 - i],
                state=state,
                out_channel=d,
                kernel_size=kernel_size,
                norm=norm,
                pass_state=pass_state,
                name=name + str(i),
                is_test=is_test)
            zs_.insert(0, output[0])
            if inject_layers > i:
                state = self.concat(output[1], a)
            else:
                state = output[1]
        return zs_

    def Gdec(self,
             zs,
             a,
             dim=64,
             n_layers=5,
             shortcut_layers=4,
             inject_layers=4,
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
                padding_type="SAME",
                activation_fn='leaky_relu',
                name=name + str(i),
                use_bias=(norm == None),
                relufactor=0.2,
                initial='kaiming')

        logit_gan = linear(
            input=y,
            output_size=fc_dim,
            activation_fn='leaky_relu',
            name=name + 'fc_adv_1',
            initial='kaiming')
        logit_gan = linear(
            logit_gan, output_size=1, name=name + 'fc_adv_2', initial='kaiming')

        logit_att = linear(
            input=y,
            output_size=fc_dim,
            activation_fn='leaky_relu',
            name=name + 'fc_cls_1',
            initial='kaiming')
        logit_att = linear(
            logit_att,
            output_size=n_atts,
            name=name + 'fc_cls_2',
            initial='kaiming')

        return logit_gan, logit_att

    def gru_cell(self,
                 in_data,
                 state,
                 out_channel,
                 kernel_size=3,
                 norm=None,
                 pass_state='lstate',
                 name='gru',
                 is_test=False):
        state_ = deconv2d(
            input=state,
            num_filters=out_channel,
            filter_size=4,
            stride=2,
            padding_type='SAME',
            name=name + '_deconv2d',
            use_bias=True,
            initial='kaiming',
            is_test=is_test,
        )  # upsample and make `channel` identical to `out_channel`
        reset_gate = conv2d(
            input=fluid.layers.concat(
                [in_data, state_], axis=1),
            num_filters=out_channel,
            filter_size=kernel_size,
            norm=norm,
            activation_fn='sigmoid',
            padding_type='SAME',
            use_bias=True,
            name=name + '_reset_gate',
            initial='kaiming',
            is_test=is_test)
        update_gate = conv2d(
            input=fluid.layers.concat(
                [in_data, state_], axis=1),
            num_filters=out_channel,
            filter_size=kernel_size,
            norm=norm,
            activation_fn='sigmoid',
            padding_type='SAME',
            use_bias=True,
            name=name + '_update_gate',
            initial='kaiming',
            is_test=is_test)
        left_state = reset_gate * state_
        new_info = conv2d(
            input=fluid.layers.concat(
                [in_data, left_state], axis=1),
            num_filters=out_channel,
            filter_size=kernel_size,
            norm=norm,
            activation_fn='tanh',
            name=name + '_info',
            padding_type='SAME',
            use_bias=True,
            initial='kaiming',
            is_test=is_test)
        output = (1 - update_gate) * state_ + update_gate * new_info
        if pass_state == 'output':
            return output, output
        elif pass_state == 'state':
            return output, state_
        else:
            return output, left_state
