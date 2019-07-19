#  Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
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

from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr

from . import resnet_helper
import logging

logger = logging.getLogger(__name__)

# For more depths, add the block config here
BLOCK_CONFIG = {
    50: (3, 4, 6, 3),
    101: (3, 4, 23, 3),
}


# ------------------------------------------------------------------------
# obtain_arc defines the temporal kernel radius and temporal strides for
# each layers residual blocks in a resnet.
# e.g. use_temp_convs = 1 means a temporal kernel of 3 is used.
# In ResNet50, it has (3, 4, 6, 3) blocks in conv2, 3, 4, 5, 
# so the lengths of the corresponding lists are (3, 4, 6, 3).
# ------------------------------------------------------------------------
def obtain_arc(arc_type, video_length):

    pool_stride = 1

    # c2d, ResNet50
    if arc_type == 1:
        use_temp_convs_1 = [0]
        temp_strides_1 = [1]
        use_temp_convs_2 = [0, 0, 0]
        temp_strides_2 = [1, 1, 1]
        use_temp_convs_3 = [0, 0, 0, 0]
        temp_strides_3 = [1, 1, 1, 1]
        use_temp_convs_4 = [0, ] * 6
        temp_strides_4 = [1, ] * 6
        use_temp_convs_5 = [0, 0, 0]
        temp_strides_5 = [1, 1, 1]

        pool_stride = int(video_length / 2)

    # i3d, ResNet50
    if arc_type == 2:
        use_temp_convs_1 = [2]
        temp_strides_1 = [1]
        use_temp_convs_2 = [1, 1, 1]
        temp_strides_2 = [1, 1, 1]
        use_temp_convs_3 = [1, 0, 1, 0]
        temp_strides_3 = [1, 1, 1, 1]
        use_temp_convs_4 = [1, 0, 1, 0, 1, 0]
        temp_strides_4 = [1, 1, 1, 1, 1, 1]
        use_temp_convs_5 = [0, 1, 0]
        temp_strides_5 = [1, 1, 1]

        pool_stride = int(video_length / 2)

    # c2d, ResNet101
    if arc_type == 3:
        use_temp_convs_1 = [0]
        temp_strides_1 = [1]
        use_temp_convs_2 = [0, 0, 0]
        temp_strides_2 = [1, 1, 1]
        use_temp_convs_3 = [0, 0, 0, 0]
        temp_strides_3 = [1, 1, 1, 1]
        use_temp_convs_4 = [0, ] * 23
        temp_strides_4 = [1, ] * 23
        use_temp_convs_5 = [0, 0, 0]
        temp_strides_5 = [1, 1, 1]

        pool_stride = int(video_length / 2)

    # i3d, ResNet101
    if arc_type == 4:
        use_temp_convs_1 = [2]
        temp_strides_1 = [1]
        use_temp_convs_2 = [1, 1, 1]
        temp_strides_2 = [1, 1, 1]
        use_temp_convs_3 = [1, 0, 1, 0]
        temp_strides_3 = [1, 1, 1, 1]
        use_temp_convs_4 = []
        for i in range(23):
            if i % 2 == 0:
                use_temp_convs_4.append(1)
            else:
                use_temp_convs_4.append(0)

        temp_strides_4 = [1] * 23
        use_temp_convs_5 = [0, 1, 0]
        temp_strides_5 = [1, 1, 1]

        pool_stride = int(video_length / 2)

    use_temp_convs_set = [
        use_temp_convs_1, use_temp_convs_2, use_temp_convs_3, use_temp_convs_4,
        use_temp_convs_5
    ]
    temp_strides_set = [
        temp_strides_1, temp_strides_2, temp_strides_3, temp_strides_4,
        temp_strides_5
    ]

    return use_temp_convs_set, temp_strides_set, pool_stride


def create_model(data, label, cfg, is_training=True, mode='train'):
    group = cfg.RESNETS.num_groups
    width_per_group = cfg.RESNETS.width_per_group
    batch_size = int(cfg.TRAIN.batch_size / cfg.TRAIN.num_gpus)

    logger.info('--------------- ResNet-{} {}x{}d-{}, {} ---------------'.
                format(cfg.MODEL.depth, group, width_per_group,
                       cfg.RESNETS.trans_func, cfg.MODEL.dataset))

    assert cfg.MODEL.depth in BLOCK_CONFIG.keys(), \
        "Block config is not defined for specified model depth."
    (n1, n2, n3, n4) = BLOCK_CONFIG[cfg.MODEL.depth]

    res_block = resnet_helper._generic_residual_block_3d
    dim_inner = group * width_per_group

    use_temp_convs_set, temp_strides_set, pool_stride = obtain_arc(
        cfg.MODEL.video_arc_choice, cfg[mode.upper()]['video_length'])
    logger.info(use_temp_convs_set)
    logger.info(temp_strides_set)
    conv_blob = fluid.layers.conv3d(
        input=data,
        num_filters=64,
        filter_size=[1 + use_temp_convs_set[0][0] * 2, 7, 7],
        stride=[temp_strides_set[0][0], 2, 2],
        padding=[use_temp_convs_set[0][0], 3, 3],
        param_attr=ParamAttr(
            name='conv1' + "_weights", initializer=fluid.initializer.MSRA()),
        bias_attr=False,
        name='conv1')

    test_mode = False if (mode == 'train') else True
    if cfg.MODEL.use_affine is False:
        # use bn
        bn_name = 'bn_conv1'
        bn_blob = fluid.layers.batch_norm(
            conv_blob,
            is_test=test_mode,
            momentum=cfg.MODEL.bn_momentum,
            epsilon=cfg.MODEL.bn_epsilon,
            name=bn_name,
            param_attr=ParamAttr(
                name=bn_name + "_scale",
                regularizer=fluid.regularizer.L2Decay(
                    cfg.TRAIN.weight_decay_bn)),
            bias_attr=ParamAttr(
                name=bn_name + "_offset",
                regularizer=fluid.regularizer.L2Decay(
                    cfg.TRAIN.weight_decay_bn)),
            moving_mean_name=bn_name + "_mean",
            moving_variance_name=bn_name + "_variance")
    else:
        # use affine
        affine_name = 'bn_conv1'
        conv_blob_shape = conv_blob.shape
        affine_scale = fluid.layers.create_parameter(
            shape=[conv_blob_shape[1]],
            dtype=conv_blob.dtype,
            attr=ParamAttr(name=affine_name + '_scale'),
            default_initializer=fluid.initializer.Constant(value=1.))
        affine_bias = fluid.layers.create_parameter(
            shape=[conv_blob_shape[1]],
            dtype=conv_blob.dtype,
            attr=ParamAttr(name=affine_name + '_offset'),
            default_initializer=fluid.initializer.Constant(value=0.))
        bn_blob = fluid.layers.affine_channel(
            conv_blob, scale=affine_scale, bias=affine_bias, name=affine_name)

    # relu
    relu_blob = fluid.layers.relu(bn_blob, name='res_conv1_bn_relu')
    # max pool
    max_pool = fluid.layers.pool3d(
        input=relu_blob,
        pool_size=[1, 3, 3],
        pool_type='max',
        pool_stride=[1, 2, 2],
        pool_padding=[0, 0, 0],
        name='pool1')

    # building res block
    if cfg.MODEL.depth in [50, 101]:
        blob_in, dim_in = resnet_helper.res_stage_nonlocal(
            res_block,
            max_pool,
            64,
            256,
            stride=1,
            num_blocks=n1,
            prefix='res2',
            cfg=cfg,
            dim_inner=dim_inner,
            group=group,
            use_temp_convs=use_temp_convs_set[1],
            temp_strides=temp_strides_set[1],
            test_mode=test_mode)

        layer_mod = cfg.NONLOCAL.layer_mod
        if cfg.MODEL.depth == 101:
            layer_mod = 2
        if cfg.NONLOCAL.conv3_nonlocal is False:
            layer_mod = 1000

        blob_in = fluid.layers.pool3d(
            blob_in,
            pool_size=[2, 1, 1],
            pool_type='max',
            pool_stride=[2, 1, 1],
            pool_padding=[0, 0, 0],
            name='pool2')

        if cfg.MODEL.use_affine is False:
            blob_in, dim_in = resnet_helper.res_stage_nonlocal(
                res_block,
                blob_in,
                dim_in,
                512,
                stride=2,
                num_blocks=n2,
                prefix='res3',
                cfg=cfg,
                dim_inner=dim_inner * 2,
                group=group,
                use_temp_convs=use_temp_convs_set[2],
                temp_strides=temp_strides_set[2],
                batch_size=batch_size,
                nonlocal_name="nonlocal_conv3",
                nonlocal_mod=layer_mod,
                test_mode=test_mode)
        else:
            crop_size = cfg[mode.upper()]['crop_size']
            blob_in, dim_in = resnet_helper.res_stage_nonlocal_group(
                res_block,
                blob_in,
                dim_in,
                512,
                stride=2,
                num_blocks=n2,
                prefix='res3',
                cfg=cfg,
                dim_inner=dim_inner * 2,
                group=group,
                use_temp_convs=use_temp_convs_set[2],
                temp_strides=temp_strides_set[2],
                batch_size=batch_size,
                pool_stride=pool_stride,
                spatial_dim=int(crop_size / 8),
                group_size=4,
                nonlocal_name="nonlocal_conv3_group",
                nonlocal_mod=layer_mod,
                test_mode=test_mode)

        layer_mod = cfg.NONLOCAL.layer_mod
        if cfg.MODEL.depth == 101:
            layer_mod = layer_mod * 4 - 1
        if cfg.NONLOCAL.conv4_nonlocal is False:
            layer_mod = 1000

        blob_in, dim_in = resnet_helper.res_stage_nonlocal(
            res_block,
            blob_in,
            dim_in,
            1024,
            stride=2,
            num_blocks=n3,
            prefix='res4',
            cfg=cfg,
            dim_inner=dim_inner * 4,
            group=group,
            use_temp_convs=use_temp_convs_set[3],
            temp_strides=temp_strides_set[3],
            batch_size=batch_size,
            nonlocal_name="nonlocal_conv4",
            nonlocal_mod=layer_mod,
            test_mode=test_mode)

        blob_in, dim_in = resnet_helper.res_stage_nonlocal(
            res_block,
            blob_in,
            dim_in,
            2048,
            stride=2,
            num_blocks=n4,
            prefix='res5',
            cfg=cfg,
            dim_inner=dim_inner * 8,
            group=group,
            use_temp_convs=use_temp_convs_set[4],
            temp_strides=temp_strides_set[4],
            test_mode=test_mode)

    else:
        raise Exception("Unsupported network settings.")

    blob_out = fluid.layers.pool3d(
        blob_in,
        pool_size=[pool_stride, 7, 7],
        pool_type='avg',
        pool_stride=[1, 1, 1],
        pool_padding=[0, 0, 0],
        name='pool5')

    if (cfg.TRAIN.dropout_rate > 0):
        blob_out = fluid.layers.dropout(
            blob_out, cfg.TRAIN.dropout_rate, is_test=test_mode)

    if mode in ['train', 'valid']:
        blob_out = fluid.layers.fc(
            blob_out,
            cfg.MODEL.num_classes,
            param_attr=ParamAttr(
                name='pred' + "_w",
                initializer=fluid.initializer.Normal(
                    loc=0.0, scale=cfg.MODEL.fc_init_std)),
            bias_attr=ParamAttr(
                name='pred' + "_b",
                initializer=fluid.initializer.Constant(value=0.)),
            name='pred')
    elif mode in ['test', 'infer']:
        blob_out = fluid.layers.conv3d(
            input=blob_out,
            num_filters=cfg.MODEL.num_classes,
            filter_size=[1, 1, 1],
            stride=[1, 1, 1],
            padding=[0, 0, 0],
            param_attr=ParamAttr(
                name='pred' + "_w", initializer=fluid.initializer.MSRA()),
            bias_attr=ParamAttr(
                name='pred' + "_b",
                initializer=fluid.initializer.Constant(value=0.)),
            name='pred')

    if (mode == 'train') or (mode == 'valid'):
        softmax = fluid.layers.softmax(blob_out)
        loss = fluid.layers.cross_entropy(
            softmax, label, soft_label=False, ignore_index=-100)

    elif (mode == 'test') or (mode == 'infer'):
        # fully convolutional testing, when loading test model, 
        # params should be copied from train_prog fc layer named pred
        blob_out = fluid.layers.transpose(
            blob_out, [0, 2, 3, 4, 1], name='pred_tr')
        blob_out = fluid.layers.softmax(blob_out, name='softmax_conv')
        softmax = fluid.layers.reduce_mean(
            blob_out, dim=[1, 2, 3], keep_dim=False, name='softmax')
        loss = None
    else:
        raise 'Not implemented Error'

    return softmax, loss
