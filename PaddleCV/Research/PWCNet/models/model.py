# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import paddle.fluid as fluid
from paddle.fluid.dygraph import Conv2D, Conv2DTranspose
from correlation_op.correlation import correlation


class PWCDCNet(fluid.dygraph.Layer):
    def __init__(self, name_scope, md=4):
        super(PWCDCNet, self).__init__(name_scope)
        self.param_attr = fluid.ParamAttr(
        name='conv_weights',
        regularizer=fluid.regularizer.L2DecayRegularizer(
            regularization_coeff=0.0004),
        initializer=fluid.initializer.MSRAInitializer(uniform=True, fan_in=None, seed=0))
        self.md = md
        self.conv1a = Conv2D("conv1a", 16, filter_size=3, stride=2, padding=1, param_attr=self.param_attr)
        self.conv1aa = Conv2D("conv1aa", 16, filter_size=3, stride=1, padding=1, param_attr=self.param_attr)
        self.conv1b = Conv2D("conv1b", 16, filter_size=3, stride=1, padding=1, param_attr=self.param_attr)
        self.conv2a = Conv2D("conv2a", 32, filter_size=3, stride=2, padding=1, param_attr=self.param_attr)
        self.conv2aa = Conv2D("conv2aa", 32, filter_size=3, stride=1, padding=1, param_attr=self.param_attr)
        self.conv2b = Conv2D("conv2b", 32, filter_size=3, stride=1, padding=1, param_attr=self.param_attr)
        self.conv3a = Conv2D("conv3a", 64, filter_size=3, stride=2, padding=1, param_attr=self.param_attr)
        self.conv3aa = Conv2D("conv3aa", 64, filter_size=3, stride=1, padding=1, param_attr=self.param_attr)
        self.conv3b = Conv2D("conv3b", 64, filter_size=3, stride=1, padding=1, param_attr=self.param_attr)
        self.conv4a = Conv2D("conv4a", 96, filter_size=3, stride=2, padding=1, param_attr=self.param_attr)
        self.conv4aa = Conv2D("conv4aa", 96, filter_size=3, stride=1, padding=1, param_attr=self.param_attr)
        self.conv4b = Conv2D("conv4b", 96, filter_size=3, stride=1, padding=1, param_attr=self.param_attr)
        self.conv5a = Conv2D("conv5a", 128, filter_size=3, stride=2, padding=1, param_attr=self.param_attr)
        self.conv5aa = Conv2D("conv5aa", 128, filter_size=3, stride=1, padding=1, param_attr=self.param_attr)
        self.conv5b = Conv2D("conv5b", 128, filter_size=3, stride=1, padding=1, param_attr=self.param_attr)
        self.conv6aa = Conv2D("conv6aa", 196, filter_size=3, stride=2, padding=1, param_attr=self.param_attr)
        self.conv6a = Conv2D("conv6a", 196, filter_size=3, stride=1, padding=1, param_attr=self.param_attr)
        self.conv6b = Conv2D("conv6b", 196, filter_size=3, stride=1, padding=1, param_attr=self.param_attr)

        self.conv6_0 = Conv2D("conv6_0", 128, filter_size=3, stride=1, padding=1, param_attr=self.param_attr)
        self.conv6_1 = Conv2D("conv6_1", 128, filter_size=3, stride=1, padding=1, param_attr=self.param_attr)
        self.conv6_2 = Conv2D("conv6_2", 96, filter_size=3, stride=1, padding=1, param_attr=self.param_attr)
        self.conv6_3 = Conv2D("conv6_3", 64, filter_size=3, stride=1, padding=1, param_attr=self.param_attr)
        self.conv6_4 = Conv2D("conv6_4", 32, filter_size=3, stride=1, padding=1, param_attr=self.param_attr)
        self.predict_flow6 = Conv2D("predict_flow6", 2, filter_size=3,stride=1,padding=1, param_attr=self.param_attr)
        self.deconv6 = Conv2DTranspose("deconv6", 2, filter_size=4, stride=2, padding=1, param_attr=self.param_attr)
        self.upfeat6 = Conv2DTranspose("upfeat6", 2, filter_size=4, stride=2, padding=1, param_attr=self.param_attr)

        self.conv5_0 = Conv2D("conv5_0", 128, filter_size=3, stride=1, padding=1, param_attr=self.param_attr)
        self.conv5_1 = Conv2D("conv5_1", 128, filter_size=3, stride=1, padding=1, param_attr=self.param_attr)
        self.conv5_2 = Conv2D("conv5_2", 96, filter_size=3, stride=1, padding=1, param_attr=self.param_attr)
        self.conv5_3 = Conv2D("conv5_3", 64, filter_size=3, stride=1, padding=1, param_attr=self.param_attr)
        self.conv5_4 = Conv2D("conv5_4", 32, filter_size=3, stride=1, padding=1, param_attr=self.param_attr)
        self.predict_flow5 = Conv2D("predict_flow5", 2, filter_size=3,stride=1,padding=1, param_attr=self.param_attr)
        self.deconv5 = Conv2DTranspose("deconv5", 2, filter_size=4, stride=2, padding=1, param_attr=self.param_attr)
        self.upfeat5 = Conv2DTranspose("upfeat5", 2, filter_size=4, stride=2, padding=1, param_attr=self.param_attr)

        self.conv4_0 = Conv2D("conv4_0", 128, filter_size=3, stride=1, padding=1, param_attr=self.param_attr)
        self.conv4_1 = Conv2D("conv4_1", 128, filter_size=3, stride=1, padding=1, param_attr=self.param_attr)
        self.conv4_2 = Conv2D("conv4_2", 96, filter_size=3, stride=1, padding=1, param_attr=self.param_attr)
        self.conv4_3 = Conv2D("conv4_3", 64, filter_size=3, stride=1, padding=1, param_attr=self.param_attr)
        self.conv4_4 = Conv2D("conv4_4", 32, filter_size=3, stride=1, padding=1, param_attr=self.param_attr)
        self.predict_flow4 = Conv2D("predict_flow4", 2, filter_size=3,stride=1,padding=1, param_attr=self.param_attr)
        self.deconv4 = Conv2DTranspose("deconv4", 2, filter_size=4, stride=2, padding=1, param_attr=self.param_attr)
        self.upfeat4 = Conv2DTranspose("upfeat4", 2, filter_size=4, stride=2, padding=1, param_attr=self.param_attr)

        self.conv3_0 = Conv2D("conv3_0", 128, filter_size=3, stride=1, padding=1, param_attr=self.param_attr)
        self.conv3_1 = Conv2D("conv3_1", 128, filter_size=3, stride=1, padding=1 ,param_attr=self.param_attr)
        self.conv3_2 = Conv2D("conv3_2", 96, filter_size=3, stride=1, padding=1, param_attr=self.param_attr)
        self.conv3_3 = Conv2D("conv3_3", 64, filter_size=3, stride=1, padding=1, param_attr=self.param_attr)
        self.conv3_4 = Conv2D("conv3_4", 32, filter_size=3, stride=1, padding=1, param_attr=self.param_attr)
        self.predict_flow3 = Conv2D("predict_flow3", 2, filter_size=3,stride=1,padding=1, param_attr=self.param_attr)
        self.deconv3 = Conv2DTranspose("deconv3", 2, filter_size=4, stride=2, padding=1, param_attr=self.param_attr)
        self.upfeat3 = Conv2DTranspose("upfeat3", 2, filter_size=4, stride=2, padding=1, param_attr=self.param_attr)

        self.conv2_0 = Conv2D("conv2_0", 128, filter_size=3, stride=1, padding=1, param_attr=self.param_attr)
        self.conv2_1 = Conv2D("conv2_1", 128, filter_size=3, stride=1, padding=1, param_attr=self.param_attr)
        self.conv2_2 = Conv2D("conv2_2", 96, filter_size=3, stride=1, padding=1, param_attr=self.param_attr)
        self.conv2_3 = Conv2D("conv2_3", 64, filter_size=3, stride=1, padding=1, param_attr=self.param_attr)
        self.conv2_4 = Conv2D("conv2_4", 32, filter_size=3, stride=1, padding=1, param_attr=self.param_attr)
        self.predict_flow2 = Conv2D("predict_flow2", 2, filter_size=3,stride=1,padding=1, param_attr=self.param_attr)
        self.deconv2 = Conv2DTranspose("deconv2", 2, filter_size=4, stride=2, padding=1, param_attr=self.param_attr)

        self.dc_conv1 = Conv2D("dc_conv1", 128, filter_size=3, stride=1, padding=1, dilation=1, param_attr=self.param_attr)
        self.dc_conv2 = Conv2D("dc_conv2", 128, filter_size=3, stride=1, padding=2, dilation=2, param_attr=self.param_attr)
        self.dc_conv3 = Conv2D("dc_conv3", 128, filter_size=3, stride=1, padding=4, dilation=4, param_attr=self.param_attr)
        self.dc_conv4 = Conv2D("dc_conv4", 96, filter_size=3, stride=1, padding=8, dilation=8, param_attr=self.param_attr)
        self.dc_conv5 = Conv2D("dc_conv5", 64, filter_size=3, stride=1, padding=16, dilation=16, param_attr=self.param_attr)
        self.dc_conv6 = Conv2D("dc_conv6", 32, filter_size=3, stride=1, padding=1, dilation=1, param_attr=self.param_attr)
        self.dc_conv7 = Conv2D("dc_conv7", 2, filter_size=3,stride=1,padding=1, param_attr=self.param_attr)

    def warp(self, x, flo):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow

        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow

        """

        B, C, H, W = x.shape
        # mesh grid
        xx_pd = fluid.layers.range(0, W, 1, 'float32')
        xx_pd = fluid.layers.reshape(xx_pd, shape=[1, -1])
        xx_pd = fluid.layers.expand(x=xx_pd, expand_times=[H, 1])
        xx_pd = fluid.layers.reshape(xx_pd, shape=[1, 1, H, W])
        xx_pd = fluid.layers.expand(x=xx_pd, expand_times=[B, 1, 1, 1])

        yy_pd = fluid.layers.range(0, H, 1, 'float32')
        yy_pd = fluid.layers.reshape(yy_pd, shape=[-1, 1])
        yy_pd = fluid.layers.expand(x=yy_pd, expand_times=[1, W])
        yy_pd = fluid.layers.reshape(x=yy_pd, shape=[1, 1, H, W])
        yy_pd = fluid.layers.expand(x=yy_pd, expand_times=[B, 1, 1, 1])
        grid_pd = fluid.layers.concat(input=[xx_pd, yy_pd], axis=1)
        flo_pd = flo
        vgrid_pd = fluid.layers.elementwise_add(grid_pd, flo_pd)
        vgrid_pd_0 = 2.0 * fluid.layers.slice(vgrid_pd, axes=[1], starts=[0], ends=[1]) / max(W - 1, 1) - 1.0
        vgrid_pd_1 = 2.0 * fluid.layers.slice(vgrid_pd, axes=[1], starts=[1], ends=[2]) / max(H - 1, 1) - 1.0
        vgrid_pd = fluid.layers.concat(input=[vgrid_pd_0, vgrid_pd_1], axis=1)
        vgrid_pd = fluid.layers.transpose(vgrid_pd, [0, 2, 3, 1])
        output = fluid.layers.grid_sampler(name='grid_sample', x=x, grid=vgrid_pd)

        mask = fluid.layers.zeros_like(x)
        mask = mask + 1.0
        mask = fluid.layers.grid_sampler(name='grid_sample', x=mask, grid=vgrid_pd)
        mask_temp1 = fluid.layers.cast(mask < 0.9990, 'float32')
        mask = mask * (1 - mask_temp1)
        mask = fluid.layers.cast(mask > 0, 'float32')
        outwarp = fluid.layers.elementwise_mul(output, mask)

        return outwarp

    def corr(self, x_1, x_2):
        out = correlation(x_1, x_2, pad_size=self.md, kernel_size=1, max_displacement=self.md,
                          stride1=1, stride2=1, corr_type_multiply=1)
        return out

    def forward(self, x, output_more=False):
        im1 = fluid.layers.slice(x, axes=[1], starts=[0], ends=[3])
        im2 = fluid.layers.slice(x, axes=[1], starts=[3], ends=[6])
        # print("\n\n***************************PWC Net details *************** \n\n")
        c11 = fluid.layers.leaky_relu(self.conv1a(im1), 0.1)
        c11 = fluid.layers.leaky_relu(self.conv1aa(c11), 0.1)
        c11 = fluid.layers.leaky_relu(self.conv1b(c11), 0.1)

        c21 = fluid.layers.leaky_relu(self.conv1a(im2), 0.1)
        c21 = fluid.layers.leaky_relu(self.conv1aa(c21), 0.1)
        c21 = fluid.layers.leaky_relu(self.conv1b(c21), 0.1)

        c12 = fluid.layers.leaky_relu(self.conv2a(c11), 0.1)
        c12 = fluid.layers.leaky_relu(self.conv2aa(c12), 0.1)
        c12 = fluid.layers.leaky_relu(self.conv2b(c12), 0.1)

        c22 = fluid.layers.leaky_relu(self.conv2a(c21), 0.1)
        c22 = fluid.layers.leaky_relu(self.conv2aa(c22), 0.1)
        c22 = fluid.layers.leaky_relu(self.conv2b(c22), 0.1)

        c13 = fluid.layers.leaky_relu(self.conv3a(c12), 0.1)
        c13 = fluid.layers.leaky_relu(self.conv3aa(c13), 0.1)
        c13 = fluid.layers.leaky_relu(self.conv3b(c13), 0.1)

        c23 = fluid.layers.leaky_relu(self.conv3a(c22), 0.1)
        c23 = fluid.layers.leaky_relu(self.conv3aa(c23), 0.1)
        c23 = fluid.layers.leaky_relu(self.conv3b(c23), 0.1)

        c14 = fluid.layers.leaky_relu(self.conv4a(c13), 0.1)
        c14 = fluid.layers.leaky_relu(self.conv4aa(c14), 0.1)
        c14 = fluid.layers.leaky_relu(self.conv4b(c14), 0.1)

        c24 = fluid.layers.leaky_relu(self.conv4a(c23), 0.1)
        c24 = fluid.layers.leaky_relu(self.conv4aa(c24), 0.1)
        c24 = fluid.layers.leaky_relu(self.conv4b(c24), 0.1)

        c15 = fluid.layers.leaky_relu(self.conv5a(c14), 0.1)
        c15 = fluid.layers.leaky_relu(self.conv5aa(c15), 0.1)
        c15 = fluid.layers.leaky_relu(self.conv5b(c15), 0.1)

        c25 = fluid.layers.leaky_relu(self.conv5a(c24), 0.1)
        c25 = fluid.layers.leaky_relu(self.conv5aa(c25), 0.1)
        c25 = fluid.layers.leaky_relu(self.conv5b(c25), 0.1)

        c16 = fluid.layers.leaky_relu(self.conv6aa(c15), 0.1)
        c16 = fluid.layers.leaky_relu(self.conv6a(c16), 0.1)
        c16 = fluid.layers.leaky_relu(self.conv6b(c16), 0.1)

        c26 = fluid.layers.leaky_relu(self.conv6aa(c25), 0.1)
        c26 = fluid.layers.leaky_relu(self.conv6a(c26), 0.1)
        c26 = fluid.layers.leaky_relu(self.conv6b(c26), 0.1)

        corr6 = self.corr(c16, c26)
        corr6 = fluid.layers.leaky_relu(corr6, alpha=0.1)

        x = fluid.layers.concat(input=[fluid.layers.leaky_relu(self.conv6_0(corr6), 0.1), corr6], axis=1)
        x = fluid.layers.concat(input=[fluid.layers.leaky_relu(self.conv6_1(x), 0.1), x], axis=1)
        x = fluid.layers.concat(input=[fluid.layers.leaky_relu(self.conv6_2(x), 0.1), x], axis=1)
        x = fluid.layers.concat(input=[fluid.layers.leaky_relu(self.conv6_3(x), 0.1), x], axis=1)
        x = fluid.layers.concat(input=[fluid.layers.leaky_relu(self.conv6_4(x), 0.1), x], axis=1)

        flow6 = self.predict_flow6(x)
        up_flow6 = self.deconv6(flow6)
        up_feat6 = self.upfeat6(x)

        warp5 = self.warp(c25, up_flow6 * 0.625)
        corr5 = self.corr(c15, warp5)
        corr5 = fluid.layers.leaky_relu(corr5, alpha=0.1)

        x = fluid.layers.concat(input=[corr5, c15, up_flow6, up_feat6], axis=1)
        x = fluid.layers.concat(input=[fluid.layers.leaky_relu(self.conv5_0(x), 0.1), x], axis=1)
        x = fluid.layers.concat(input=[fluid.layers.leaky_relu(self.conv5_1(x), 0.1), x], axis=1)
        x = fluid.layers.concat(input=[fluid.layers.leaky_relu(self.conv5_2(x), 0.1), x], axis=1)
        x = fluid.layers.concat(input=[fluid.layers.leaky_relu(self.conv5_3(x), 0.1), x], axis=1)
        x = fluid.layers.concat(input=[fluid.layers.leaky_relu(self.conv5_4(x), 0.1), x], axis=1)

        flow5 = self.predict_flow5(x)
        up_flow5 = self.deconv5(flow5)
        up_feat5 = self.upfeat5(x)

        warp4 = self.warp(c24, up_flow5 * 1.25)
        corr4 = self.corr(c14, warp4)
        corr4 = fluid.layers.leaky_relu(corr4, alpha=0.1)

        x = fluid.layers.concat(input=[corr4, c14, up_flow5, up_feat5], axis=1)
        x = fluid.layers.concat(input=[fluid.layers.leaky_relu(self.conv4_0(x), 0.1), x], axis=1)
        x = fluid.layers.concat(input=[fluid.layers.leaky_relu(self.conv4_1(x), 0.1), x], axis=1)
        x = fluid.layers.concat(input=[fluid.layers.leaky_relu(self.conv4_2(x), 0.1), x], axis=1)
        x = fluid.layers.concat(input=[fluid.layers.leaky_relu(self.conv4_3(x), 0.1), x], axis=1)
        x = fluid.layers.concat(input=[fluid.layers.leaky_relu(self.conv4_4(x), 0.1), x], axis=1)

        flow4 = self.predict_flow4(x)
        up_flow4 = self.deconv4(flow4)
        up_feat4 = self.upfeat4(x)

        warp3 = self.warp(c23, up_flow4 * 2.5)
        corr3 = self.corr(c13, warp3)
        corr3 = fluid.layers.leaky_relu(corr3, alpha=0.1)

        x = fluid.layers.concat(input=[corr3, c13, up_flow4, up_feat4], axis=1)
        x = fluid.layers.concat(input=[fluid.layers.leaky_relu(self.conv3_0(x), 0.1), x], axis=1)
        x = fluid.layers.concat(input=[fluid.layers.leaky_relu(self.conv3_1(x), 0.1), x], axis=1)
        x = fluid.layers.concat(input=[fluid.layers.leaky_relu(self.conv3_2(x), 0.1), x], axis=1)
        x = fluid.layers.concat(input=[fluid.layers.leaky_relu(self.conv3_3(x), 0.1), x], axis=1)
        x = fluid.layers.concat(input=[fluid.layers.leaky_relu(self.conv3_4(x), 0.1), x], axis=1)

        flow3 = self.predict_flow3(x)
        up_flow3 = self.deconv3(flow3)
        up_feat3 = self.upfeat3(x)

        warp2 = self.warp(c22, up_flow3 * 5.0)
        corr2 = self.corr(c12, warp2)
        corr2 = fluid.layers.leaky_relu(corr2, alpha=0.1)

        x = fluid.layers.concat(input=[corr2, c12, up_flow3, up_feat3], axis=1)
        x = fluid.layers.concat(input=[fluid.layers.leaky_relu(self.conv2_0(x), 0.1), x], axis=1)
        x = fluid.layers.concat(input=[fluid.layers.leaky_relu(self.conv2_1(x), 0.1), x], axis=1)
        x = fluid.layers.concat(input=[fluid.layers.leaky_relu(self.conv2_2(x), 0.1), x], axis=1)
        x = fluid.layers.concat(input=[fluid.layers.leaky_relu(self.conv2_3(x), 0.1), x], axis=1)
        x = fluid.layers.concat(input=[fluid.layers.leaky_relu(self.conv2_4(x), 0.1), x], axis=1)

        flow2 = self.predict_flow2(x)

        x = fluid.layers.leaky_relu(self.dc_conv4(fluid.layers.leaky_relu(
            self.dc_conv3(fluid.layers.leaky_relu(self.dc_conv2(fluid.layers.leaky_relu(self.dc_conv1(x), 0.1)), 0.1)),
            0.1)), 0.1)
        flow2 += self.dc_conv7(
            fluid.layers.leaky_relu(self.dc_conv6(fluid.layers.leaky_relu(self.dc_conv5(x), 0.1)), 0.1))
        if not output_more:
            return flow2
        else:
            return [flow2, flow3, flow4, flow5, flow6]

