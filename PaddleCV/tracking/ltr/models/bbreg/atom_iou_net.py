"""
the implementation of ATOM iou net
"""
import paddle
import paddle.fluid as fluid
import paddle.fluid.dygraph.nn as nn
import numpy as np
import os.path as osp
import sys

CURRENT_DIR = osp.dirname(__file__)
sys.path.append(osp.join(CURRENT_DIR, '..', '..', '..'))


def weight_init():
    init = fluid.initializer.MSRAInitializer(uniform=False)
    param = fluid.ParamAttr(initializer=init)
    return param


def bias_init():
    init = fluid.initializer.ConstantInitializer(value=0.)
    param = fluid.ParamAttr(initializer=init)
    return param


def norm_weight_init():
    # init = fluid.initializer.ConstantInitializer(1.0)
    init = fluid.initializer.Uniform(low=0., high=1.)
    param = fluid.ParamAttr(initializer=init)
    return param


def norm_bias_init():
    init = fluid.initializer.ConstantInitializer(value=0.)
    param = fluid.ParamAttr(initializer=init)
    return param


class ConvBNReluLayer(fluid.dygraph.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 filter_size,
                 stride=1,
                 groups=1,
                 padding=1,
                 is_test=False):
        super(ConvBNReluLayer, self).__init__()

        self.conv = nn.Conv2D(
            num_channels=in_channels,
            filter_size=filter_size,
            num_filters=out_channels,
            stride=stride,
            padding=padding,
            groups=groups,
            bias_attr=bias_init(),
            param_attr=weight_init())
        self.bn = nn.BatchNorm(
            out_channels,
            param_attr=norm_weight_init(),
            bias_attr=norm_bias_init(),
            act=None,
            momentum=0.9,
            use_global_stats=is_test)

    def forward(self, inputs):
        res = self.conv(inputs)
        self.conv_res = res
        res = self.bn(res)
        res = fluid.layers.relu(res)
        return res


class FCBNReluLayer(fluid.dygraph.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 in_size,
                 is_bias=True,
                 is_bn=True,
                 is_relu=True,
                 is_test=False):
        super(FCBNReluLayer, self).__init__()
        self.is_bn = is_bn
        self.is_relu = is_relu

        if is_bias:
            bias_init = fluid.ParamAttr(
                initializer=fluid.initializer.ConstantInitializer(0.))
        else:
            bias_init = False
        self.linear = nn.Linear(
            in_channels * in_size * in_size, out_channels, bias_attr=bias_init)
        self.bn = nn.BatchNorm(
            out_channels,
            param_attr=norm_weight_init(),
            bias_attr=norm_bias_init(),
            act=None,
            momentum=0.9,
            use_global_stats=is_test)

    def forward(self, x):
        x = fluid.layers.reshape(x, [x.shape[0], -1])

        x = self.linear(x)
        if self.is_bn:
            x = self.bn(x)
        if self.is_relu:
            x = fluid.layers.relu(x)
        return x


class AtomIouNet(fluid.dygraph.Layer):
    def __init__(self,
                 name,
                 input_dim=(128, 256),
                 pred_input_dim=(256, 256),
                 pred_inter_dim=(256, 256),
                 is_test=False):
        super(AtomIouNet, self).__init__(name)
        self.name = self.full_name()
        self.conv3_1r = ConvBNReluLayer(
            input_dim[0], 128, filter_size=3, stride=1, is_test=is_test)
        self.conv3_1t = ConvBNReluLayer(
            input_dim[0], 256, filter_size=3, stride=1, is_test=is_test)

        self.conv3_2t = ConvBNReluLayer(
            256, pred_input_dim[0], filter_size=3, stride=1, is_test=is_test)

        self.fc3_1r = ConvBNReluLayer(
            128, 256, filter_size=3, stride=1, padding=0, is_test=is_test)

        self.conv4_1r = ConvBNReluLayer(
            input_dim[1], 256, filter_size=3, stride=1, is_test=is_test)
        self.conv4_1t = ConvBNReluLayer(
            input_dim[1], 256, filter_size=3, stride=1, is_test=is_test)

        self.conv4_2t = ConvBNReluLayer(
            256, pred_input_dim[1], filter_size=3, stride=1, is_test=is_test)

        self.fc34_3r = ConvBNReluLayer(
            512,
            pred_input_dim[0],
            filter_size=1,
            stride=1,
            padding=0,
            is_test=is_test)
        self.fc34_4r = ConvBNReluLayer(
            512,
            pred_input_dim[1],
            filter_size=1,
            stride=1,
            padding=0,
            is_test=is_test)

        self.fc3_rt = FCBNReluLayer(
            pred_input_dim[0], pred_inter_dim[0], in_size=5, is_test=is_test)
        self.fc4_rt = FCBNReluLayer(
            pred_input_dim[1], pred_inter_dim[1], in_size=3, is_test=is_test)

        bias_init = fluid.initializer.ConstantInitializer(0.)
        self.iou_predictor = nn.Linear(
            pred_inter_dim[0] + pred_inter_dim[1], 1, bias_attr=bias_init)

        self.outs = {}

    def predict_iou(self, filter, feat2, proposals):
        """
        predicts IOU for the given proposals
        :param modulation: Modulation vectors for the targets. Dims (batch, feature_dim).
        :param feat: IoU features (from get_iou_feat) for test images. Dims (batch, feature_dim, H, W).
        :param proposals: Proposal boxes for which the IoU will be predicted (batch, num_proposals, 4).
        :return:
        """
        fc34_3_r, fc34_4_r = filter
        c3_t, c4_t = feat2

        batch_size = c3_t.shape[0]

        # Modulation
        c3_t_att = c3_t * fluid.layers.reshape(fc34_3_r, [batch_size, -1, 1, 1])
        c4_t_att = c4_t * fluid.layers.reshape(fc34_4_r, [batch_size, -1, 1, 1])

        # add batch roi nums
        num_proposals_per_batch = proposals.shape[1]
        batch_roi_nums = np.array([num_proposals_per_batch] *
                                  batch_size).astype(np.int64)
        batch_roi_nums = fluid.dygraph.to_variable(batch_roi_nums)

        # input proposals2 is in format xywh, convert it to x0y0x1y1 format
        proposals_xyxy = fluid.layers.concat(
            [
                proposals[:, :, 0:2],
                proposals[:, :, 0:2] + proposals[:, :, 2:4]
            ],
            axis=2)

        roi2 = fluid.layers.reshape(proposals_xyxy, [-1, 4])
        roi2.stop_gradient = False

        roi3t = fluid.layers.prroi_pool(
            c3_t_att, roi2, 1 / 8., 5, 5, batch_roi_nums=batch_roi_nums)
        roi4t = fluid.layers.prroi_pool(
            c4_t_att, roi2, 1 / 16., 3, 3, batch_roi_nums=batch_roi_nums)

        fc3_rt = self.fc3_rt(roi3t)
        fc4_rt = self.fc4_rt(roi4t)

        fc34_rt_cat = fluid.layers.concat([fc3_rt, fc4_rt], axis=1)

        iou_pred = self.iou_predictor(fc34_rt_cat)
        iou_pred = fluid.layers.reshape(iou_pred,
                                        [batch_size, num_proposals_per_batch])

        return iou_pred

    def forward(self, feat1, feat2, bb1, proposals2):
        """Runs the ATOM IoUNet during training operation.
        This forward pass is mainly used for training. Call the individual functions during tracking instead.
        args:
            feat1:  Variable, Features from the reference frames (4 or 5 dims).
            feat2:  Variable, Features from the test frames (4 or 5 dims).
            bb1:  Target boxes (x,y,x2,y2) in image coords in the reference samples. Dims (images, sequences, 4).
            proposals2:  Proposal boxes for which the IoU will be predicted (images, sequences, num_proposals, 4)."""
        assert len(feat1[0].shape) == 5, 'Expect 5  dimensional feat1'
        num_test_images = feat2[0].shape[0]
        batch_size = feat2[0].shape[1]

        # Extract first train sample
        feat1 = [f[0] for f in feat1]
        bb1 = bb1[0]

        # Get modulation vector
        modulation = self.get_filter(feat1, bb1)

        feat2 = [
            fluid.layers.reshape(f,
                                 (batch_size * num_test_images, *f.shape[-3:]))
            for f in feat2
        ]
        iou_feat = self.get_iou_feat(feat2)

        new_modulation = []
        for i in range(0, len(modulation)):
            tmp = modulation[i]
            tmp = fluid.layers.reshape(tmp, [1, batch_size, -1])
            tmp = fluid.layers.expand(tmp, [num_test_images, 1, 1])
            tmp = fluid.layers.reshape(tmp, [batch_size * num_test_images, -1])
            new_modulation.append(tmp)

        proposals2 = fluid.layers.reshape(
            proposals2, [batch_size * num_test_images, -1, 4])

        pred_iou = self.predict_iou(new_modulation, iou_feat, proposals2)
        pred_iou = fluid.layers.reshape(pred_iou,
                                        [num_test_images, batch_size, -1])
        return pred_iou

    def get_filter(self, feat1, bb1):
        """
        get modulation feature [feature1, feature2] for the targets
        :param feat1: variable, Backbone features from reference images. shapes (batch, feature_dim, H, W).
        :param bb1: variable, Target boxes (x,y,w,h) in image coords in the reference samples. shapes (batch, 4).
        :return:
        """
        feat3_r, feat4_r = feat1

        c3_r = self.conv3_1r(feat3_r)

        # Add batch_index to rois
        batch_size = bb1.shape[0]
        batch_roi_nums = np.array([1] * batch_size).astype(np.int64)
        batch_roi_nums = fluid.dygraph.to_variable(batch_roi_nums)

        # input bb is in format xywh, convert it to x0y0x1y1 format
        roi1 = fluid.layers.concat(
            [bb1[:, 0:2], bb1[:, 0:2] + bb1[:, 2:4]], axis=1)
        roi1.stop_gradient = False

        roi3r = fluid.layers.prroi_pool(c3_r, roi1, 1 / 8., 3, 3,
                                        batch_roi_nums)

        c4_r = self.conv4_1r(feat4_r)
        roi4r = fluid.layers.prroi_pool(c4_r, roi1, 1 / 16., 1, 1,
                                        batch_roi_nums)

        fc3_r = self.fc3_1r(roi3r)

        # Concatenate
        fc34_r = fluid.layers.concat([fc3_r, roi4r], axis=1)

        fc34_3_r = self.fc34_3r(fc34_r)
        fc34_4_r = self.fc34_4r(fc34_r)

        return fc34_3_r, fc34_4_r

    def get_iou_feat(self, feat2):
        """
        Get IoU prediction features from a 4 or 5 dimensional backbone input.
        :param feat2: variable, Backbone features from reference images. [feature1, feature2]
        :return: features, variable
        """
        feat3_t, feat4_t = feat2
        c3_t = self.conv3_2t(self.conv3_1t(feat3_t))
        c4_t = self.conv4_2t(self.conv4_1t(feat4_t))

        return c3_t, c4_t


def atom_iounet(name,
                input_dim=(128, 256),
                pred_input_dim=(256, 256),
                pred_inter_dim=(256, 256)):
    return AtomIouNet(
        name,
        input_dim=input_dim,
        pred_input_dim=pred_input_dim,
        pred_inter_dim=pred_inter_dim)


def test_paddle_iounet():
    a = np.random.uniform(-1, 1, [1, 1, 512, 18, 18]).astype(np.float32)
    b = np.random.uniform(-1, 1, [1, 1, 1024, 9, 9]).astype(np.float32)
    bbox = [[3, 4, 10, 11]]
    proposal_bbox = [[4, 5, 11, 12] * 16]
    bbox = np.reshape(np.array(bbox), [1, 1, 4]).astype(np.float32)
    proposal_bbox = np.reshape(np.array(proposal_bbox),
                               [1, 16, 4]).astype(np.float32)
    with fluid.dygraph.guard():
        a_pd = fluid.dygraph.to_variable(a)
        b_pd = fluid.dygraph.to_variable(b)
        bbox_pd = fluid.dygraph.to_variable(bbox)
        proposal_bbox_pd = fluid.dygraph.to_variable(proposal_bbox)
        feat1 = [a_pd, b_pd]
        feat2 = [a_pd, b_pd]

        model = AtomIouNet('IOUNet', input_dim=(512, 1024))
        res = model(feat1, feat2, bbox_pd, proposal_bbox_pd)
        print(res.shape)
        params = model.state_dict()

        for v in params:
            print(v, '\t', params[v].shape)
        print(len(params))


if __name__ == '__main__':
    test_paddle_iounet()
