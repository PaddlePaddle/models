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
import paddle
import paddle.fluid as fluid


def EPE(input_flow, target_flow, loss_type, sparse=False, mean=True):
    if loss_type == 'l1':
        EPE_map = fluid.layers.abs(input_flow - target_flow)
    else:
        EPE_map = fluid.layers.square(input_flow - target_flow)
    if sparse: #TODO mask = (target_flow[:,0] == 0) & (target_flow[:,1] == 0) EPE_map = EPE_map[~mask]
        mask_temp1 = fluid.layers.cast(target_flow[:, 0] == 0, 'float32')
        mask_temp2 = fluid.layers.cast(target_flow[:, 1] == 0, 'float32')
        mask = 1 - fluid.layers.elementwise_mul(mask_temp1, mask_temp2)
        mask = fluid.layers.reshape(mask, [mask.shape[0], 1, mask.shape[1], mask.shape[2]])
        mask = fluid.layers.concat([mask, mask], 1)
        EPE_map = EPE_map * mask

    if mean:
        return fluid.layers.mean(EPE_map)
    else:
        batch_size = EPE_map.shape[0]
        res_sum = fluid.layers.reduce_sum(EPE_map)
        res = res_sum / batch_size
        return res


def sparse_max_pool(input, size):
    '''Downsample the input by considering 0 values as invalid.

    Unfortunately, no generic interpolation mode can resize a sparse map correctly,
    the strategy here is to use max pooling for positive values and "min pooling"
    for negative values, the two results are then summed.
    This technique allows sparsity to be minized, contrary to nearest interpolation,
    which could potentially lose information for isolated data points.'''

    positive = fluid.layers.cast(input > 0, 'float32')
    negative = fluid.layers.cast(input < 0, 'float32')
    output = fluid.layers.adaptive_pool2d(input * positive, size) - fluid.layers.adaptive_pool2d(-input * negative,
                                                                                                 size)
    return output


def multiscaleEPE(network_output, target_flow, loss_type, weights=None, sparse=False):
    def one_scale(output, target, sparse, loss_type):
        if sparse:
            h = output.shape[2]
            w = output.shape[3]
            target_scaled = sparse_max_pool(target, [h, w])
        else:
            target_scaled = fluid.layers.resize_bilinear(target, out_shape=[output.shape[2],
                                                                               output.shape[3]],
                                                            align_corners=False, align_mode=False)
        return EPE(output, target_scaled, loss_type=loss_type, sparse=sparse, mean=False)

    if type(network_output) not in [tuple, list]:
        network_output = [network_output]
    if weights is None:
        weights = [0.005, 0.01, 0.02, 0.08, 0.32]  # as in original article
    assert(len(weights) == len(network_output))

    loss = 0
    for output, weight in zip(network_output, weights):
        loss += weight * one_scale(output, target_flow, sparse, loss_type)
    return loss


def realEPE(output, target, sparse=False):
    upsampled_output = fluid.layers.resize_bilinear(output, out_shape=[target.shape[2],
                                                                       target.shape[3]],
                                               align_corners=False, align_mode=False)
    return EPE(upsampled_output, target, sparse, mean=True)

