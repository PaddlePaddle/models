import numpy as np

import paddle.fluid as fluid
from paddle.fluid.initializer import Constant


def conv_bn(input, filter, ksize, stride, padding, act='relu', bias_attr=False):
    conv = fluid.layers.conv2d(
        input=input,
        filter_size=ksize,
        num_filters=filter,
        stride=stride,
        padding=padding,
        act=None,
        bias_attr=bias_attr)
    return fluid.layers.batch_norm(input=conv, act=act)


def conv_group(input, groups, filters, ksizes, strides=None, with_pool=True):
    assert len(filters) == groups
    assert len(ksizes) == groups
    strides = [1] * groups if strides is None else strides
    conv = input
    for i in xrange(groups):
        conv = fluid.layers.conv2d(
            input=conv,
            num_filters=filters[i],
            filter_size=ksizes[i],
            stride=strides[i],
            padding=(ksizes[i] - 1) / 2,
            act='relu')
    if with_pool:
        pool = fluid.layers.pool2d(
            input=conv, pool_size=2, pool_type='max', pool_stride=2)
        return pool
    else:
        return conv


class PyramidBox(object):
    def __init__(self, data_shape, is_infer=False):
        self.data_shape = data_shape
        self.min_sizes = [16., 32., 64., 128., 256., 512.]
        self.steps = [4., 8., 16., 32., 64., 128.]
        self.is_infer = is_infer

        # the base network is VGG with atrus layers
        self._input()
        self._vgg()
        self._low_level_fpn()
        self._cpm_module()
        self._pyramidbox()

    def _input(self):
        self.image = fluid.layers.data(
            name='image', shape=self.data_shape, dtype='float32')
        if not self.is_infer:
            self.gt_box = fluid.layers.data(
                name='gt_box', shape=[4], dtype='float32', lod_level=1)
            self.gt_label = fluid.layers.data(
                name='gt_label', shape=[1], dtype='int32', lod_level=1)

    def _vgg(self):
        self.conv1 = conv_group(self.image, 2, [64] * 2, [3] * 2)
        self.conv2 = conv_group(self.conv1, 2, [128] * 2, [3] * 2)

        #priorbox min_size is 16
        self.conv3 = conv_group(self.conv2, 3, [256] * 3, [3] * 3)
        #priorbox min_size is 32
        self.conv4 = conv_group(self.conv3, 3, [512] * 3, [3] * 3)
        #priorbox min_size is 64
        self.conv5 = conv_group(self.conv4, 3, [512] * 3, [3] * 3)

        # fc6 and fc7 in paper, priorbox min_size is 128
        self.conv6 = conv_group(
            self.conv5, 2, [1024, 1024], [3, 1], with_pool=False)
        # conv6_1 and conv6_2 in paper, priorbox min_size is 256
        self.conv7 = conv_group(
            self.conv6, 2, [256, 512], [1, 3], [1, 2], with_pool=False)
        # conv7_1 and conv7_2 in paper, priorbox mini_size is 512
        self.conv8 = conv_group(
            self.conv7, 2, [128, 256], [1, 3], [1, 2], with_pool=False)

    def _low_level_fpn(self):
        """
        Low-level feature pyramid network.
        """

        def fpn(up_from, up_to):
            ch = up_to.shape[1]
            conv1 = fluid.layers.conv2d(up_from, ch, 1, act='relu')
            # TODO: add group
            conv_trans = fluid.layers.conv2d_transpose(
                conv1, ch, None, 4, 1, 2, bias_attr=False)
            conv2 = fluid.layers.conv2d(up_to, ch, 1, act='relu')
            # eltwise mul
            conv_fuse = conv_trans * conv2
            return conv_fuse

        self.lfpn2_on_conv5 = fpn(self.conv6, self.conv5)
        self.lfpn1_on_conv4 = fpn(self.lfpn2_on_conv5, self.conv4)
        self.lfpn0_on_conv3 = fpn(self.lfpn1_on_conv4, self.conv3)

    def _cpm_module(self):
        """
        Context-sensitive Prediction Module 
        """

        def cpm(input):
            # residual
            branch1 = conv_bn(input, 1024, 1, 1, 0, None)
            branch2a = conv_bn(input, 256, 1, 1, 0, act='relu')
            branch2b = conv_bn(branch2a, 256, 3, 1, 1, act='relu')
            branch2c = conv_bn(branch2b, 1024, 1, 1, 0, None)
            sum = branch1 + branch2c
            rescomb = fluid.layers.relu(x=sum)

            # ssh
            ssh_1 = fluid.layers.conv2d(rescomb, 256, 3, 1, 1)
            ssh_dimred = fluid.layers.conv2d(rescomb, 128, 3, 1, 1, act='relu')
            ssh_2 = fluid.layers.conv2d(ssh_dimred, 128, 3, 1, 1)
            ssh_3a = fluid.layers.conv2d(ssh_dimred, 128, 3, 1, 1, act='relu')
            ssh_3b = fluid.layers.conv2d(ssh_3a, 128, 3, 1, 1)

            ssh_concat = fluid.layers.concat([ssh_1, ssh_2, ssh_3b], axis=1)
            ssh_out = fluid.layers.relu(x=ssh_concat)
            return ssh_out

        self.ssh_conv3 = cpm(self.lfpn0_on_conv3)
        self.ssh_conv4 = cpm(self.lfpn1_on_conv4)
        self.ssh_conv5 = cpm(self.lfpn2_on_conv5)
        self.ssh_conv6 = cpm(self.conv6)
        self.ssh_conv7 = cpm(self.conv7)
        self.ssh_conv8 = cpm(self.conv8)

    def _l2_norm_scale(self, input, init_scale=1.0, channel_shared=False):
        from paddle.fluid.layer_helper import LayerHelper
        helper = LayerHelper("Scale")
        l2_norm = fluid.layers.l2_normalize(
            input, axis=1)  # l2 norm along channel
        shape = [1] if channel_shared else [input.shape[1]]
        scale = helper.create_parameter(
            attr=helper.param_attr,
            shape=shape,
            dtype=input.dtype,
            default_initializer=Constant(init_scale))
        out = fluid.layers.elementwise_mul(
            x=l2_norm, y=scale, axis=-1 if channel_shared else 1)
        return out

    def _pyramidbox(self):
        """
        Get prior-boxes and pyramid-box
        """
        self.ssh_conv3_norm = self._l2_norm_scale(self.ssh_conv3)
        self.ssh_conv4_norm = self._l2_norm_scale(self.ssh_conv4)
        self.ssh_conv5_norm = self._l2_norm_scale(self.ssh_conv5)

        def permute_and_reshape(input, last_dim):
            trans = fluid.layers.transpose(input, perm=[0, 2, 3, 1])
            new_shape = [
                trans.shape[0], np.prod(trans.shape[1:]) / last_dim, last_dim
            ]
            return fluid.layers.reshape(trans, shape=new_shape)

        face_locs = []
        face_confs = []
        head_locs = []
        head_confs = []
        boxes = []
        vars = []
        inputs = [
            self.ssh_conv3_norm, self.ssh_conv4_norm, self.ssh_conv5_norm,
            self.ssh_conv6, self.ssh_conv7, self.ssh_conv8
        ]
        for i, input in enumerate(inputs):
            mbox_loc = fluid.layers.conv2d(input, 8, 3, 1, 1, bias_attr=True)
            face_loc, head_loc = fluid.layers.split(
                mbox_loc, num_or_sections=2, dim=1)
            face_loc = permute_and_reshape(face_loc, 4)
            head_loc = permute_and_reshape(head_loc, 4)

            mbox_conf = fluid.layers.conv2d(input, 6, 3, 1, 1, bias_attr=True)
            face_conf1, face_conf3, head_conf = fluid.layers.split(
                mbox_conf, num_or_sections=[1, 3, 2], dim=1)
            face_conf3_maxin = fluid.layers.reduce_max(
                face_conf3, dim=1, keep_dim=True)
            face_conf = fluid.layers.concat(
                [face_conf1, face_conf3_maxin], axis=1)

            face_conf = permute_and_reshape(face_conf, 2)
            head_conf = permute_and_reshape(head_conf, 2)

            face_locs.append(face_loc)
            face_confs.append(face_conf)

            head_locs.append(head_loc)
            head_confs.append(head_conf)

            box, var = fluid.layers.prior_box(
                input,
                self.image,
                min_sizes=[self.min_sizes[1]],
                steps=[self.steps[i]] * 2,
                aspect_ratios=[1.],
                offset=0.5)
            box = fluid.layers.reshape(box, shape=[-1, 4])
            var = fluid.layers.reshape(var, shape=[-1, 4])

            boxes.append(box)
            vars.append(var)

        self.face_mbox_loc = fluid.layers.concat(face_locs, axis=1)
        self.face_mbox_conf = fluid.layers.concat(face_confs, axis=1)

        self.head_mbox_loc = fluid.layers.concat(head_locs, axis=1)
        self.head_mbox_conf = fluid.layers.concat(head_confs, axis=1)

        self.prior_boxes = fluid.layers.concat(boxes)
        self.box_vars = fluid.layers.concat(vars)

    def train(self):
        face_loss = fluid.layers.ssd_loss(
            self.face_mbox_loc, self.face_mbox_conf, self.gt_box, self.gt_label,
            self.prior_boxes, self.box_vars)
        head_loss = fluid.layers.ssd_loss(
            self.head_mbox_loc, self.head_mbox_conf, self.gt_box, self.gt_label,
            self.prior_boxes, self.box_vars)
        return face_loss, head_loss

    def test(self):
        test_program = fluid.default_main_program().clone(for_test=True)
        with fluid.program_guard(test_program):
            face_nmsed_out = fluid.layers.detection_output(
                self.face_mbox_loc,
                self.face_mbox_conf,
                self.prior_boxes,
                self.box_vars,
                nms_threshold=0.45)
            head_nmsed_out = fluid.layers.detection_output(
                self.head_mbox_loc,
                self.head_mbox_conf,
                self.prior_boxes,
                self.box_vars,
                nms_threshold=0.45)
            face_map_eval = fluid.evaluator.DetectionMAP(
                face_nmsed_out,
                self.gt_label,
                self.gt_box,
                class_num=2,
                overlap_threshold=0.5,
                ap_version='11point')
            head_map_eval = fluid.evaluator.DetectionMAP(
                head_nmsed_out,
                self.gt_label,
                self.gt_box,
                class_num=2,
                overlap_threshold=0.5,
                ap_version='11point')
        return test_program, face_map_eval, head_map_eval
