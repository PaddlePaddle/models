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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from model import *
import paddle.fluid as fluid

step_per_epoch = 2974
lambda_A = 10.0
lambda_B = 10.0
lambda_identity = 0.5


class Cycle_Gan(fluid.dygraph.Layer):
    def __init__(self, input_channel, istrain=True):
        super (Cycle_Gan, self).__init__()

        self.build_generator_resnet_9blocks_a = build_generator_resnet_9blocks(input_channel)
        self.build_generator_resnet_9blocks_b = build_generator_resnet_9blocks(input_channel)
        if istrain:
            self.build_gen_discriminator_a = build_gen_discriminator(input_channel)
            self.build_gen_discriminator_b = build_gen_discriminator(input_channel)

    def forward(self,input_A,input_B,is_G,is_DA,is_DB):

        if is_G:
            fake_B = self.build_generator_resnet_9blocks_a(input_A)
            fake_A = self.build_generator_resnet_9blocks_b(input_B)
            cyc_A = self.build_generator_resnet_9blocks_b(fake_B)
            cyc_B = self.build_generator_resnet_9blocks_a(fake_A)

            diff_A = fluid.layers.abs(
                fluid.layers.elementwise_sub(
                    x=input_A,y=cyc_A))
            diff_B = fluid.layers.abs(
                fluid.layers.elementwise_sub(
                    x=input_B, y=cyc_B))
            cyc_A_loss = fluid.layers.reduce_mean(diff_A) * lambda_A
            cyc_B_loss = fluid.layers.reduce_mean(diff_B) * lambda_B
            cyc_loss = cyc_A_loss + cyc_B_loss

            fake_rec_A = self.build_gen_discriminator_a(fake_B)
            g_A_loss = fluid.layers.reduce_mean(fluid.layers.square(fake_rec_A-1))
         
            fake_rec_B = self.build_gen_discriminator_b(fake_A)
            g_B_loss = fluid.layers.reduce_mean(fluid.layers.square(fake_rec_B-1))
            G = g_A_loss + g_B_loss
            idt_A = self.build_generator_resnet_9blocks_a(input_B)
            idt_loss_A = fluid.layers.reduce_mean(fluid.layers.abs(fluid.layers.elementwise_sub(x = input_B , y = idt_A))) * lambda_B * lambda_identity

            idt_B = self.build_generator_resnet_9blocks_b(input_A)
            idt_loss_B = fluid.layers.reduce_mean(fluid.layers.abs(fluid.layers.elementwise_sub(x = input_A , y = idt_B))) * lambda_A * lambda_identity
            idt_loss = fluid.layers.elementwise_add(idt_loss_A,idt_loss_B)
            g_loss = cyc_loss + G + idt_loss
            return fake_A,fake_B,cyc_A,cyc_B,g_A_loss,g_B_loss,idt_loss_A,idt_loss_B,cyc_A_loss,cyc_B_loss,g_loss


        if is_DA:

            ### D
            rec_B = self.build_gen_discriminator_a(input_A)
            fake_pool_rec_B = self.build_gen_discriminator_a(input_B)
            
            return rec_B, fake_pool_rec_B

        if is_DB:

            rec_A = self.build_gen_discriminator_b(input_A)

            fake_pool_rec_A = self.build_gen_discriminator_b(input_B)


        return rec_A, fake_pool_rec_A

