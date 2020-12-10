#  Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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
"""loss"""
import sys
from functools import reduce

sys.dont_write_bytecode = True

# -----------------------------------------
# custom modules
# -----------------------------------------
import paddle.fluid as fluid
import paddle
from paddle.fluid.dygraph import to_variable
sys.path.append("../../")
from lib.rpn_util import *


class RPN_3D_loss(fluid.dygraph.Layer):
    def __init__(self, conf):

        super(RPN_3D_loss, self).__init__()

        self.num_classes = len(conf.lbls) + 1
        self.num_anchors = conf.anchors.shape[0]
        self.anchors = conf.anchors
        self.bbox_means = conf.bbox_means
        self.bbox_stds = conf.bbox_stds
        self.feat_stride = conf.feat_stride
        self.fg_fraction = conf.fg_fraction
        self.box_samples = conf.box_samples
        self.ign_thresh = conf.ign_thresh
        self.nms_thres = conf.nms_thres
        self.fg_thresh = conf.fg_thresh
        self.bg_thresh_lo = conf.bg_thresh_lo
        self.bg_thresh_hi = conf.bg_thresh_hi
        self.best_thresh = conf.best_thresh
        self.hard_negatives = conf.hard_negatives
        self.focal_loss = conf.focal_loss

        self.crop_size = conf.crop_size

        self.cls_2d_lambda = conf.cls_2d_lambda
        self.iou_2d_lambda = conf.iou_2d_lambda
        self.bbox_2d_lambda = conf.bbox_2d_lambda
        self.bbox_3d_lambda = conf.bbox_3d_lambda
        self.bbox_3d_proj_lambda = conf.bbox_3d_proj_lambda

        self.lbls = conf.lbls
        self.ilbls = conf.ilbls

        self.min_gt_vis = conf.min_gt_vis
        self.min_gt_h = conf.min_gt_h
        self.max_gt_h = conf.max_gt_h

    def forward(self, cls, prob, bbox_2d, bbox_3d, imobjs, feat_size):

        stats = []
        loss = np.array([0]).astype('float32')
        loss = to_variable(loss)

        FG_ENC = 1000
        BG_ENC = 2000

        IGN_FLAG = 3000

        batch_size = cls.shape[0]

        prob_detach = prob.detach().numpy()

        bbox_x = bbox_2d[:, :, 0]
        bbox_y = bbox_2d[:, :, 1]
        bbox_w = bbox_2d[:, :, 2]
        bbox_h = bbox_2d[:, :, 3]

        bbox_x3d = bbox_3d[:, :, 0]
        bbox_y3d = bbox_3d[:, :, 1]
        bbox_z3d = bbox_3d[:, :, 2]
        bbox_w3d = bbox_3d[:, :, 3]
        bbox_h3d = bbox_3d[:, :, 4]
        bbox_l3d = bbox_3d[:, :, 5]
        bbox_ry3d = bbox_3d[:, :, 6]

        bbox_x3d_proj = np.zeros(bbox_x3d.shape)
        bbox_y3d_proj = np.zeros(bbox_x3d.shape)
        bbox_z3d_proj = np.zeros(bbox_x3d.shape)

        labels = np.zeros(cls.shape[0:2])
        labels_weight = np.zeros(cls.shape[0:2])

        labels_scores = np.zeros(cls.shape[0:2])

        bbox_x_tar = np.zeros(cls.shape[0:2])
        bbox_y_tar = np.zeros(cls.shape[0:2])
        bbox_w_tar = np.zeros(cls.shape[0:2])
        bbox_h_tar = np.zeros(cls.shape[0:2])

        bbox_x3d_tar = np.zeros(cls.shape[0:2])
        bbox_y3d_tar = np.zeros(cls.shape[0:2])
        bbox_z3d_tar = np.zeros(cls.shape[0:2])
        bbox_w3d_tar = np.zeros(cls.shape[0:2])
        bbox_h3d_tar = np.zeros(cls.shape[0:2])
        bbox_l3d_tar = np.zeros(cls.shape[0:2])
        bbox_ry3d_tar = np.zeros(cls.shape[0:2])

        bbox_x3d_proj_tar = np.zeros(cls.shape[0:2])
        bbox_y3d_proj_tar = np.zeros(cls.shape[0:2])
        bbox_z3d_proj_tar = np.zeros(cls.shape[0:2])

        bbox_weights = np.zeros(cls.shape[0:2])

        ious_2d = np.zeros(cls.shape[0:2])
        ious_3d = np.zeros(cls.shape[0:2])
        coords_abs_z = np.zeros(cls.shape[0:2])
        coords_abs_ry = np.zeros(cls.shape[0:2])

        # get all rois
        # rois' type now is nparray
        rois = locate_anchors(self.anchors, feat_size, self.feat_stride)
        rois = rois.astype('float32')

        #bbox_3d dtype is Variable, so bbox_3d_dn is
        bbox_x3d_dn = bbox_x3d * self.bbox_stds[:, 4][0] + self.bbox_means[:,
                                                                           4][0]
        bbox_y3d_dn = bbox_y3d * self.bbox_stds[:, 5][0] + self.bbox_means[:,
                                                                           5][0]
        bbox_z3d_dn = bbox_z3d * self.bbox_stds[:, 6][0] + self.bbox_means[:,
                                                                           6][0]
        bbox_w3d_dn = bbox_w3d * self.bbox_stds[:, 7][0] + self.bbox_means[:,
                                                                           7][0]
        bbox_h3d_dn = bbox_h3d * self.bbox_stds[:, 8][0] + self.bbox_means[:,
                                                                           8][0]
        bbox_l3d_dn = bbox_l3d * self.bbox_stds[:, 9][0] + self.bbox_means[:,
                                                                           9][0]
        bbox_ry3d_dn = bbox_ry3d * self.bbox_stds[:, 10][
            0] + self.bbox_means[:, 10][0]

        src_anchors = self.anchors[rois[:, 4].astype('int64'), :]  #nparray
        src_anchors = src_anchors.astype('float32')
        src_anchors = to_variable(src_anchors)  #Variable
        src_anchors.stop_gradient = True
        if len(src_anchors.shape) == 1:
            src_anchors = fluid.layers.unsqueeze(input=src_anchors, axis=0)

        # compute 3d transform
        #the following four all are nparrays
        widths = rois[:, 2] - rois[:, 0] + 1.0
        heights = rois[:, 3] - rois[:, 1] + 1.0
        ctr_x = rois[:, 0] + 0.5 * widths
        ctr_y = rois[:, 1] + 0.5 * heights

        ctr_x_unsqueeze = fluid.layers.unsqueeze(
            input=to_variable(ctr_x), axes=0)
        ctr_y_unsqueeze = fluid.layers.unsqueeze(
            input=to_variable(ctr_y), axes=0)
        widths_unsqueeze = fluid.layers.unsqueeze(
            input=to_variable(widths), axes=0)
        heights_unsqueeze = fluid.layers.unsqueeze(
            input=to_variable(heights), axes=0)
        bbox_z3d_unsqueeze = fluid.layers.unsqueeze(
            input=src_anchors[:, 4], axes=0)
        bbox_w3d_unsqueeze = fluid.layers.unsqueeze(
            input=src_anchors[:, 5], axes=0)
        bbox_h3d_unsqueeze = fluid.layers.unsqueeze(
            input=src_anchors[:, 6], axes=0)
        bbox_l3d_unsqueeze = fluid.layers.unsqueeze(
            input=src_anchors[:, 7], axes=0)
        bbox_ry3d_unsqueeze = fluid.layers.unsqueeze(
            input=src_anchors[:, 8], axes=0)
        bbox_x3d_dn = bbox_x3d_dn * widths_unsqueeze + ctr_x_unsqueeze
        bbox_y3d_dn = bbox_y3d_dn * heights_unsqueeze + ctr_y_unsqueeze
        bbox_z3d_dn = bbox_z3d_unsqueeze + bbox_z3d_dn
        bbox_w3d_dn = fluid.layers.exp(bbox_w3d_dn) * bbox_w3d_unsqueeze
        bbox_h3d_dn = fluid.layers.exp(bbox_h3d_dn) * bbox_h3d_unsqueeze
        bbox_l3d_dn = fluid.layers.exp(bbox_l3d_dn) * bbox_l3d_unsqueeze
        bbox_ry3d_dn = bbox_ry3d_unsqueeze + bbox_ry3d_dn

        ious_2d_var_list = []
        for bind in range(0, batch_size):

            imobj = imobjs[bind]
            gts = imobj.gts

            p2_inv = to_variable(imobj.p2_inv).astype('float32')

            # filter gts
            igns, rmvs = determine_ignores(gts, self.lbls, self.ilbls,
                                           self.min_gt_vis, self.min_gt_h)

            # accumulate boxes
            gts_all = bbXYWH2Coords(np.array([gt.bbox_full for gt in gts]))
            gts_3d = np.array([gt.bbox_3d for gt in gts])

            if not ((rmvs == False) & (igns == False)).any():
                continue

            # filter out irrelevant cls, and ignore cls
            gts_val = gts_all[(rmvs == False) & (igns == False), :]
            gts_ign = gts_all[(rmvs == False) & (igns == True), :]
            gts_3d = gts_3d[(rmvs == False) & (igns == False), :]

            # accumulate labels
            box_lbls = np.array([gt.cls for gt in gts])
            box_lbls = box_lbls[(rmvs == False) & (igns == False)]
            box_lbls = np.array(
                [clsName2Ind(self.lbls, cls) for cls in box_lbls])

            if gts_val.shape[0] > 0 or gts_ign.shape[0] > 0:

                # bbox regression
                transforms, ols, raw_gt = compute_targets(
                    gts_val,
                    gts_ign,
                    box_lbls,
                    rois,
                    self.fg_thresh,
                    self.ign_thresh,
                    self.bg_thresh_lo,
                    self.bg_thresh_hi,
                    self.best_thresh,
                    anchors=self.anchors,
                    gts_3d=gts_3d,
                    tracker=rois[:, 4])

                # normalize 2d
                transforms[:, 0:4] -= self.bbox_means[:, 0:4]
                transforms[:, 0:4] /= self.bbox_stds[:, 0:4]

                # normalize 3d
                transforms[:, 5:12] -= self.bbox_means[:, 4:]
                transforms[:, 5:12] /= self.bbox_stds[:, 4:]

                labels_fg = transforms[:, 4] > 0
                labels_bg = transforms[:, 4] < 0
                labels_ign = transforms[:, 4] == 0

                fg_inds = np.flatnonzero(labels_fg)
                bg_inds = np.flatnonzero(labels_bg)
                ign_inds = np.flatnonzero(labels_ign)

                labels[bind, fg_inds] = transforms[fg_inds, 4]
                labels[bind, ign_inds] = IGN_FLAG
                labels[bind, bg_inds] = 0

                bbox_x_tar[bind, :] = transforms[:, 0]
                bbox_y_tar[bind, :] = transforms[:, 1]
                bbox_w_tar[bind, :] = transforms[:, 2]
                bbox_h_tar[bind, :] = transforms[:, 3]

                bbox_x3d_tar[bind, :] = transforms[:, 5]
                bbox_y3d_tar[bind, :] = transforms[:, 6]
                bbox_z3d_tar[bind, :] = transforms[:, 7]
                bbox_w3d_tar[bind, :] = transforms[:, 8]
                bbox_h3d_tar[bind, :] = transforms[:, 9]
                bbox_l3d_tar[bind, :] = transforms[:, 10]
                bbox_ry3d_tar[bind, :] = transforms[:, 11]

                bbox_x3d_proj_tar[bind, :] = raw_gt[:, 12]
                bbox_y3d_proj_tar[bind, :] = raw_gt[:, 13]
                bbox_z3d_proj_tar[bind, :] = raw_gt[:, 14]

                transforms = to_variable(transforms)
                # ----------------------------------------
                # box sampling
                # ----------------------------------------

                if self.box_samples == np.inf:
                    fg_num = len(fg_inds)
                    bg_num = len(bg_inds)

                else:
                    fg_num = min(
                        round(rois.shape[0] * self.box_samples *
                              self.fg_fraction), len(fg_inds))
                    bg_num = min(
                        round(rois.shape[0] * self.box_samples - fg_num),
                        len(bg_inds))

                if self.hard_negatives:

                    if fg_num > 0 and fg_num != fg_inds.shape[0]:
                        scores = prob_detach[bind, fg_inds, labels[
                            bind, fg_inds].astype(int)]
                        fg_score_ascend = (scores).argsort()
                        fg_inds = fg_inds[fg_score_ascend]
                        fg_inds = fg_inds[0:fg_num]

                    if bg_num > 0 and bg_num != bg_inds.shape[0]:
                        scores = prob_detach[bind, bg_inds, labels[
                            bind, bg_inds].astype(int)]
                        bg_score_ascend = (scores).argsort()
                        bg_inds = bg_inds[bg_score_ascend]
                        bg_inds = bg_inds[0:bg_num]

                else:

                    if fg_num > 0 and fg_num != fg_inds.shape[0]:
                        fg_inds = np.random.choice(
                            fg_inds, fg_num, replace=False)

                    if bg_num > 0 and bg_num != bg_inds.shape[0]:
                        bg_inds = np.random.choice(
                            bg_inds, bg_num, replace=False)

                labels_weight[bind, bg_inds] = BG_ENC
                labels_weight[bind, fg_inds] = FG_ENC
                bbox_weights[bind, fg_inds] = 1

                # ----------------------------------------
                # compute IoU stats
                # ----------------------------------------

                if fg_num > 0:

                    # compile deltas pred (Variable)
                    bbox_x_bind = bbox_x[bind, :]
                    bbox_x_bind_unsqueeze = fluid.layers.unsqueeze(
                        bbox_x_bind, axes=1)
                    bbox_y_bind = bbox_y[bind, :]
                    bbox_y_bind_unsqueeze = fluid.layers.unsqueeze(
                        bbox_y_bind, axes=1)
                    bbox_w_bind = bbox_w[bind, :]
                    bbox_w_bind_unsqueeze = fluid.layers.unsqueeze(
                        bbox_w_bind, axes=1)
                    bbox_h_bind = bbox_h[bind, :]
                    bbox_h_bind_unsqueeze = fluid.layers.unsqueeze(
                        bbox_h_bind, axes=1)
                    deltas_2d = fluid.layers.concat(
                        (bbox_x_bind_unsqueeze, bbox_y_bind_unsqueeze,
                         bbox_w_bind_unsqueeze, bbox_h_bind_unsqueeze),
                        axis=1)

                    # compile deltas targets (nparray)
                    deltas_2d_tar = np.concatenate(
                        (bbox_x_tar[bind, :, np.newaxis],
                         bbox_y_tar[bind, :, np.newaxis],
                         bbox_w_tar[bind, :, np.newaxis],
                         bbox_h_tar[bind, :, np.newaxis]),
                        axis=1).astype('float32')

                    # move to gpu
                    deltas_2d_tar = to_variable(deltas_2d_tar)
                    deltas_2d_tar.stop_gradient = True

                    means = self.bbox_means[0, :]
                    stds = self.bbox_stds[0, :]

                    #variable
                    coords_2d = bbox_transform_inv(
                        rois, deltas_2d, means=means, stds=stds)
                    coords_2d_tar = bbox_transform_inv(
                        rois, deltas_2d_tar, means=means, stds=stds)

                    #vaiable
                    ious_2d_var = iou(coords_2d, coords_2d_tar, mode='list')
                    ious_2d_var_shape = ious_2d_var.shape
                    ious_2d_fg_mask = np.zeros(ious_2d_var_shape).astype(
                        'float32')
                    ious_2d_fg_mask[fg_inds] = 1
                    ious_2d_var = ious_2d_var * to_variable(ious_2d_fg_mask)
                    ious_2d_var_list.append(ious_2d_var)

                    bbox_x3d_dn_fg = bbox_x3d_dn.numpy()[bind, fg_inds]
                    bbox_y3d_dn_fg = bbox_y3d_dn.numpy()[bind, fg_inds]

                    src_anchors = self.anchors[rois[fg_inds, 4].astype('int64')]
                    src_anchors = to_variable(src_anchors).astype('float32')
                    src_anchors.stop_gradient = True
                    if len(src_anchors.shape) == 1:
                        src_anchors = fluid.layers.unsqueeze(
                            input=src_anchors, axes=0)

                    #nparray
                    bbox_x3d_dn_fg = bbox_x3d_dn.numpy()[bind, fg_inds]
                    bbox_y3d_dn_fg = bbox_y3d_dn.numpy()[bind, fg_inds]
                    bbox_z3d_dn_fg = bbox_z3d_dn.numpy()[bind, fg_inds]
                    bbox_w3d_dn_fg = bbox_w3d_dn.numpy()[bind, fg_inds]
                    bbox_h3d_dn_fg = bbox_h3d_dn.numpy()[bind, fg_inds]
                    bbox_l3d_dn_fg = bbox_l3d_dn.numpy()[bind, fg_inds]
                    bbox_ry3d_dn_fg = bbox_ry3d_dn.numpy()[bind, fg_inds]

                    # re-scale all 2D back to original
                    bbox_x3d_dn_fg /= imobj['scale_factor']
                    bbox_y3d_dn_fg /= imobj['scale_factor']

                    coords_2d = fluid.layers.concat(
                        (to_variable(bbox_x3d_dn_fg[np.newaxis, :] *
                                     bbox_z3d_dn_fg[np.newaxis, :]),
                         to_variable(bbox_y3d_dn_fg[np.newaxis, :] *
                                     bbox_z3d_dn_fg[np.newaxis, :]),
                         to_variable(bbox_z3d_dn_fg[np.newaxis, :])),
                        axis=0)
                    coords_2d = fluid.layers.concat(
                        (coords_2d,
                         to_variable(np.ones([1, coords_2d.shape[1]])).astype(
                             'float32')),
                        axis=0)

                    coords_3d = fluid.layers.matmul(p2_inv, coords_2d)

                    bbox_x3d_proj[bind, fg_inds] = coords_3d[0, :].numpy()
                    bbox_y3d_proj[bind, fg_inds] = coords_3d[1, :].numpy()
                    bbox_z3d_proj[bind, fg_inds] = coords_3d[2, :].numpy()

                    # absolute targets
                    bbox_z3d_dn_tar = bbox_z3d_tar[
                        bind, fg_inds] * self.bbox_stds[:, 6][
                            0] + self.bbox_means[:, 6][0]
                    bbox_z3d_dn_tar = to_variable(bbox_z3d_dn_tar).astype(
                        'float32')
                    bbox_z3d_dn_tar.stop_gradient = True
                    bbox_z3d_dn_tar = src_anchors[:, 4] + bbox_z3d_dn_tar

                    bbox_ry3d_dn_tar = bbox_ry3d_tar[
                        bind, fg_inds] * self.bbox_stds[:, 10][
                            0] + self.bbox_means[:, 10][0]
                    bbox_ry3d_dn_tar = to_variable(bbox_ry3d_dn_tar).astype(
                        'float32')

                    bbox_ry3d_dn_tar.stop_gradient = True
                    bbox_ry3d_dn_tar = src_anchors[:, 8] + bbox_ry3d_dn_tar

                    bbox_z3d_dn_fg = to_variable(bbox_z3d_dn_fg)
                    bbox_ry3d_dn_fg = to_variable(bbox_ry3d_dn_fg)
                    bbox_abs_z3d_var = fluid.layers.abs(bbox_z3d_dn_tar -
                                                        bbox_z3d_dn_fg)
                    coords_abs_z[bind, fg_inds] = bbox_abs_z3d_var.numpy()
                    bbox_abs_ry3d_var = fluid.layers.abs(bbox_ry3d_dn_tar -
                                                         bbox_ry3d_dn_fg)
                    coords_abs_ry[bind, fg_inds] = bbox_abs_ry3d_var.numpy()

            else:

                bg_inds = np.arange(0, rois.shape[0])

                if self.box_samples == np.inf: bg_num = len(bg_inds)
                else:
                    bg_num = min(
                        round(self.box_samples * (1 - self.fg_fraction)),
                        len(bg_inds))

                if self.hard_negatives:

                    if bg_num > 0 and bg_num != bg_inds.shape[0]:
                        scores = prob_detach[bind, bg_inds, labels[
                            bind, bg_inds].astype(int)]
                        bg_score_ascend = (scores).argsort()
                        bg_inds = bg_inds[bg_score_ascend]
                        bg_inds = bg_inds[0:bg_num]

                else:

                    if bg_num > 0 and bg_num != bg_inds.shape[0]:
                        bg_inds = np.random.choice(
                            bg_inds, bg_num, replace=False)

                labels[bind, :] = 0
                labels_weight[bind, bg_inds] = BG_ENC

            # grab label predictions (for weighing purposes) dtype: nparray
            active = labels[bind, :] != IGN_FLAG
            labels_scores[bind, active] = prob_detach[bind, active, labels[
                bind, active].astype(int)]

        # ----------------------------------------
        # useful statistics
        # ----------------------------------------

        fg_inds_all = np.flatnonzero((labels > 0) & (labels != IGN_FLAG))
        bg_inds_all = np.flatnonzero((labels == 0) & (labels != IGN_FLAG))

        fg_inds_unravel = np.unravel_index(fg_inds_all, prob_detach.shape[0:2])
        bg_inds_unravel = np.unravel_index(bg_inds_all, prob_detach.shape[0:2])

        cls_pred = np.argmax(cls.detach().numpy(), axis=2)

        if self.cls_2d_lambda and len(fg_inds_all) > 0:
            acc_fg = np.mean(
                cls_pred[fg_inds_unravel] == labels[fg_inds_unravel])
            stats.append({
                'name': 'fg',
                'val': acc_fg,
                'format': '{:0.2f}',
                'group': 'acc'
            })

        if self.cls_2d_lambda and len(bg_inds_all) > 0:
            acc_bg = np.mean(
                cls_pred[bg_inds_unravel] == labels[bg_inds_unravel])
            stats.append({
                'name': 'bg',
                'val': acc_bg,
                'format': '{:0.2f}',
                'group': 'acc'
            })

        # ----------------------------------------
        # box weighting
        # ----------------------------------------

        fg_inds = np.flatnonzero(labels_weight == FG_ENC)
        bg_inds = np.flatnonzero(labels_weight == BG_ENC)
        active_inds = np.concatenate((fg_inds, bg_inds), axis=0)

        fg_num = len(fg_inds)
        bg_num = len(bg_inds)

        labels_weight[...] = 0.0
        box_samples = fg_num + bg_num

        fg_inds_unravel = np.unravel_index(fg_inds, labels_weight.shape)
        bg_inds_unravel = np.unravel_index(bg_inds, labels_weight.shape)
        active_inds_unravel = np.unravel_index(active_inds, labels_weight.shape)

        labels_weight[active_inds_unravel] = 1.0

        if self.fg_fraction is not None:

            if fg_num > 0:

                fg_weight = (self.fg_fraction /
                             (1 - self.fg_fraction)) * (bg_num / fg_num)
                labels_weight[fg_inds_unravel] = fg_weight
                labels_weight[bg_inds_unravel] = 1.0

            else:
                labels_weight[bg_inds_unravel] = 1.0

        # different method of doing hard negative mining
        # use the scores to normalize the importance of each sample
        # hence, encourages the network to get all "correct" rather than
        # becoming more correct at a decision it is already good at
        # this method is equivelent to the focal loss with additional mean scaling
        if self.focal_loss:

            weights_sum = 0

            # re-weight bg
            if bg_num > 0:
                bg_scores = labels_scores[bg_inds_unravel]
                bg_weights = (1 - bg_scores)**self.focal_loss
                weights_sum += np.sum(bg_weights)
                labels_weight[bg_inds_unravel] *= bg_weights

            # re-weight fg
            if fg_num > 0:
                fg_scores = labels_scores[fg_inds_unravel]
                fg_weights = (1 - fg_scores)**self.focal_loss
                weights_sum += np.sum(fg_weights)
                labels_weight[fg_inds_unravel] *= fg_weights

        # ----------------------------------------
        # classification loss
        # ----------------------------------------
        labels_weight = labels_weight.view()
        labels_weight.shape = np.product(labels_weight.shape)

        active = labels_weight > 0

        labels_weight_active = labels_weight[active]

        labels_weight_active = to_variable(labels_weight_active)
        labels_weight_active = labels_weight_active.astype('float32')
        labels_weight_active.stop_gradient = True

        labels = labels.view().astype('int64')
        labels.shape = np.product(labels.shape)
        labels_active = labels[active]
        labels_active = to_variable(labels_active)
        labels_active.stop_gradient = True

        active_index = np.flatnonzero(active)
        cls_reshape = fluid.layers.reshape(cls, shape=[-1, cls.shape[2]])
        active_index_var = to_variable(active_index)
        active_index_var.stop_gradient = True
        cls_active = fluid.layers.gather(cls_reshape, index=active_index_var)

        if self.cls_2d_lambda:

            # cls loss
            if np.any(active):
                labels_active = fluid.layers.reshape(
                    labels_active, shape=[-1, 1])
                loss_cls = fluid.layers.softmax_with_cross_entropy(
                    cls_active, labels_active, ignore_index=IGN_FLAG)
                labels_weight_active = fluid.layers.unsqueeze(
                    labels_weight_active, axes=1)
                loss_cls = fluid.layers.elementwise_mul(loss_cls,
                                                        labels_weight_active)

                # simple gradient clipping
                loss_cls = fluid.layers.clip(loss_cls, min=0.0, max=2000.0)

                # take mean and scale lambda
                loss_cls = fluid.layers.mean(loss_cls)
                loss_cls *= self.cls_2d_lambda

                loss += loss_cls

                stats.append({
                    'name': 'cls',
                    'val': loss_cls.numpy(),
                    'format': '{:0.4f}',
                    'group': 'loss'
                })

        # ----------------------------------------
        # bbox regression loss
        # ----------------------------------------

        if np.sum(bbox_weights) > 0:

            bbox_total_nums = np.product(bbox_weights.shape)
            bbox_weights = bbox_weights.view().astype('float32')
            bbox_weights.shape = bbox_total_nums

            active = bbox_weights > 0
            active_index = np.flatnonzero(active)
            active_len = active_index.size
            active_index_var = to_variable(active_index)
            active_index_var.stop_gradient = True
            bbox_weights.shape = 1, bbox_total_nums
            bbox_weights_active = bbox_weights[:, active]
            bbox_weights_active = to_variable(bbox_weights_active)
            bbox_weights_active.stop_gradient = True

            if self.bbox_2d_lambda:

                # bbox loss 2d
                bbox_x_tar = bbox_x_tar.view().astype('float32')
                bbox_x_tar.shape = 1, bbox_total_nums
                bbox_x_tar_active = bbox_x_tar[:, active]
                bbox_x_tar_active = to_variable(bbox_x_tar_active)
                bbox_x_tar_active.stop_gradient = True

                bbox_y_tar = bbox_y_tar.view().astype('float32')
                bbox_y_tar.shape = 1, bbox_total_nums
                bbox_y_tar_active = bbox_y_tar[:, active]
                bbox_y_tar_active = to_variable(bbox_y_tar_active)
                bbox_y_tar_active.stop_gradient = True

                bbox_w_tar = bbox_w_tar.view().astype('float32')
                bbox_w_tar.shape = 1, bbox_total_nums
                bbox_w_tar_active = bbox_w_tar[:, active]
                bbox_w_tar_active = to_variable(bbox_w_tar_active)
                bbox_w_tar_active.stop_gradient = True

                bbox_h_tar = bbox_h_tar.view().astype('float32')
                bbox_h_tar.shape = 1, bbox_total_nums
                bbox_h_tar_active = bbox_h_tar[:, active]
                bbox_h_tar_active = to_variable(bbox_h_tar_active)
                bbox_h_tar_active.stop_gradient = True

                bbox_x = fluid.layers.reshape(bbox_x, shape=[-1])
                bbox_x_active = fluid.layers.gather(bbox_x, active_index_var)
                bbox_x_active = fluid.layers.unsqueeze(bbox_x_active, axes=0)

                bbox_y = fluid.layers.reshape(bbox_y, shape=[-1])
                bbox_y_active = fluid.layers.gather(bbox_y, active_index_var)
                bbox_y_active = fluid.layers.unsqueeze(bbox_y_active, axes=0)

                bbox_w = fluid.layers.reshape(bbox_w, shape=[-1])
                bbox_w_active = fluid.layers.gather(bbox_w, active_index_var)
                bbox_w_active = fluid.layers.unsqueeze(bbox_w_active, axes=0)

                bbox_h = fluid.layers.reshape(bbox_h, shape=[-1])
                bbox_h_active = fluid.layers.gather(bbox_h, active_index_var)
                bbox_h_active = fluid.layers.unsqueeze(bbox_h_active, axes=0)

                loss_bbox_x = fluid.layers.smooth_l1(
                    bbox_x_active,
                    bbox_x_tar_active,
                    outside_weight=bbox_weights_active)
                loss_bbox_y = fluid.layers.smooth_l1(
                    bbox_y_active,
                    bbox_y_tar_active,
                    outside_weight=bbox_weights_active)
                loss_bbox_w = fluid.layers.smooth_l1(
                    bbox_w_active,
                    bbox_w_tar_active,
                    outside_weight=bbox_weights_active)
                loss_bbox_h = fluid.layers.smooth_l1(
                    bbox_h_active,
                    bbox_h_tar_active,
                    outside_weight=bbox_weights_active)

                bbox_2d_loss = (
                    loss_bbox_x + loss_bbox_y + loss_bbox_w + loss_bbox_h
                ) / active_len
                bbox_2d_loss *= self.bbox_2d_lambda

                loss += bbox_2d_loss
                stats.append({
                    'name': 'bbox_2d',
                    'val': bbox_2d_loss.numpy(),
                    'format': '{:0.4f}',
                    'group': 'loss'
                })

            if self.bbox_3d_lambda:

                # bbox loss 3d
                bbox_x3d_tar = bbox_x3d_tar.view().astype('float32')
                bbox_x3d_tar.shape = 1, bbox_total_nums
                bbox_x3d_tar_active = bbox_x3d_tar[:, active]
                bbox_x3d_tar_active = to_variable(bbox_x3d_tar_active)
                bbox_x3d_tar_active.stop_gradient = True

                bbox_y3d_tar = bbox_y3d_tar.view().astype('float32')
                bbox_y3d_tar.shape = 1, bbox_total_nums
                bbox_y3d_tar_active = bbox_y3d_tar[:, active]
                bbox_y3d_tar_active = to_variable(bbox_y3d_tar_active)
                bbox_y3d_tar_active.stop_gradient = True

                bbox_z3d_tar = bbox_z3d_tar.view().astype('float32')
                bbox_z3d_tar.shape = 1, bbox_total_nums
                bbox_z3d_tar_active = bbox_z3d_tar[:, active]
                bbox_z3d_tar_active = to_variable(bbox_z3d_tar_active)
                bbox_z3d_tar_active.stop_gradient = True

                bbox_w3d_tar = bbox_w3d_tar.view().astype('float32')
                bbox_w3d_tar.shape = 1, bbox_total_nums
                bbox_w3d_tar_active = bbox_w3d_tar[:, active]
                bbox_w3d_tar_active = to_variable(bbox_w3d_tar_active)
                bbox_w3d_tar_active.stop_gradient = True

                bbox_h3d_tar = bbox_h3d_tar.view().astype('float32')
                bbox_h3d_tar.shape = 1, bbox_total_nums
                bbox_h3d_tar_active = bbox_h3d_tar[:, active]
                bbox_h3d_tar_active = to_variable(bbox_h3d_tar_active)
                bbox_h3d_tar_active.stop_gradient = True

                bbox_l3d_tar = bbox_l3d_tar.view().astype('float32')
                bbox_l3d_tar.shape = 1, bbox_total_nums
                bbox_l3d_tar_active = bbox_l3d_tar[:, active]
                bbox_l3d_tar_active = to_variable(bbox_l3d_tar_active)
                bbox_l3d_tar_active.stop_gradient = True

                bbox_ry3d_tar = bbox_ry3d_tar.view().astype('float32')
                bbox_ry3d_tar.shape = 1, bbox_total_nums
                bbox_ry3d_tar_active = bbox_ry3d_tar[:, active]
                bbox_ry3d_tar_active = to_variable(bbox_ry3d_tar_active)
                bbox_ry3d_tar_active.stop_gradient = True

                bbox_x3d = fluid.layers.reshape(bbox_x3d, shape=[-1])
                bbox_x3d_active = fluid.layers.gather(bbox_x3d,
                                                      active_index_var)
                bbox_x3d_active = fluid.layers.unsqueeze(
                    bbox_x3d_active, axes=0)

                bbox_y3d = fluid.layers.reshape(bbox_y3d, shape=[-1])
                bbox_y3d_active = fluid.layers.gather(bbox_y3d,
                                                      active_index_var)
                bbox_y3d_active = fluid.layers.unsqueeze(
                    bbox_y3d_active, axes=0)

                bbox_z3d = fluid.layers.reshape(bbox_z3d, shape=[-1])
                bbox_z3d_active = fluid.layers.gather(bbox_z3d,
                                                      active_index_var)
                bbox_z3d_active = fluid.layers.unsqueeze(
                    bbox_z3d_active, axes=0)

                bbox_w3d = fluid.layers.reshape(bbox_w3d, shape=[-1])
                bbox_w3d_active = fluid.layers.gather(bbox_w3d,
                                                      active_index_var)
                bbox_w3d_active = fluid.layers.unsqueeze(
                    bbox_w3d_active, axes=0)

                bbox_h3d = fluid.layers.reshape(bbox_h3d, shape=[-1])
                bbox_h3d_active = fluid.layers.gather(bbox_h3d,
                                                      active_index_var)
                bbox_h3d_active = fluid.layers.unsqueeze(
                    bbox_h3d_active, axes=0)

                bbox_l3d = fluid.layers.reshape(bbox_l3d, shape=[-1])
                bbox_l3d_active = fluid.layers.gather(bbox_l3d,
                                                      active_index_var)
                bbox_l3d_active = fluid.layers.unsqueeze(
                    bbox_l3d_active, axes=0)

                bbox_ry3d = fluid.layers.reshape(bbox_ry3d, shape=[-1])
                bbox_ry3d_active = fluid.layers.gather(bbox_ry3d,
                                                       active_index_var)
                bbox_ry3d_active = fluid.layers.unsqueeze(
                    bbox_ry3d_active, axes=0)

                loss_bbox_x3d = fluid.layers.smooth_l1(
                    bbox_x3d_active.astype('float32'),
                    bbox_x3d_tar_active.astype('float32'),
                    outside_weight=bbox_weights_active.astype('float32'))
                loss_bbox_y3d = fluid.layers.smooth_l1(
                    bbox_y3d_active.astype('float32'),
                    bbox_y3d_tar_active.astype('float32'),
                    outside_weight=bbox_weights_active.astype('float32'))
                loss_bbox_z3d = fluid.layers.smooth_l1(
                    bbox_z3d_active.astype('float32'),
                    bbox_z3d_tar_active.astype('float32'),
                    outside_weight=bbox_weights_active.astype('float32'))
                loss_bbox_w3d = fluid.layers.smooth_l1(
                    bbox_w3d_active.astype('float32'),
                    bbox_w3d_tar_active.astype('float32'),
                    outside_weight=bbox_weights_active.astype('float32'))
                loss_bbox_h3d = fluid.layers.smooth_l1(
                    bbox_h3d_active.astype('float32'),
                    bbox_h3d_tar_active.astype('float32'),
                    outside_weight=bbox_weights_active.astype('float32'))
                loss_bbox_l3d = fluid.layers.smooth_l1(
                    bbox_l3d_active.astype('float32'),
                    bbox_l3d_tar_active.astype('float32'),
                    outside_weight=bbox_weights_active.astype('float32'))
                loss_bbox_ry3d = fluid.layers.smooth_l1(
                    bbox_ry3d_active.astype('float32'),
                    bbox_ry3d_tar_active.astype('float32'),
                    outside_weight=bbox_weights_active.astype('float32'))

                bbox_3d_loss = (loss_bbox_x3d + loss_bbox_y3d + loss_bbox_z3d)
                bbox_3d_loss += (loss_bbox_w3d + loss_bbox_h3d + loss_bbox_l3d +
                                 loss_bbox_ry3d)
                bbox_3d_loss = bbox_3d_loss / active_len

                bbox_3d_loss *= self.bbox_3d_lambda
                bbox_3d_loss = bbox_3d_loss
                loss += bbox_3d_loss
                stats.append({
                    'name': 'bbox_3d',
                    'val': bbox_3d_loss.numpy(),
                    'format': '{:0.4f}',
                    'group': 'loss'
                })

            if self.bbox_3d_proj_lambda:

                # bbox loss 3d
                bbox_x3d_proj_tar = bbox_x3d_proj_tar.view().astype('float32')
                bbox_x3d_proj_tar.shape = 1, bbox_total_nums
                bbox_x3d_proj_tar_active = bbox_x3d_proj_tar[:, active]
                bbox_x3d_proj_tar_active = to_variable(bbox_x3d_proj_tar_active)
                bbox_x3d_proj_tar_active.stop_gradient = True

                bbox_y3d_proj_tar = bbox_y3d_proj_tar.view().astype('float32')
                bbox_y3d_proj_tar.shape = 1, bbox_total_nums
                bbox_y3d_proj_tar_active = bbox_y3d_proj_tar[:, active]
                bbox_y3d_proj_tar_active = to_variable(bbox_y3d_proj_tar_active)
                bbox_y3d_proj_tar_active.stop_gradient = True

                bbox_z3d_proj_tar = bbox_z3d_proj_tar.view().astype('float32')
                bbox_z3d_proj_tar.shape = 1, bbox_total_nums
                bbox_z3d_proj_tar_active = bbox_z3d_proj_tar[:, active]
                bbox_z3d_proj_tar_active = to_variable(bbox_z3d_proj_tar_active)
                bbox_z3d_proj_tar_active.stop_gradient = True

                bbox_x3d_proj = bbox_x3d_proj.view()
                bbox_x3d_proj.shape = 1, bbox_total_nums
                bbox_x3d_proj_active = bbox_x3d_proj[:, active]
                bbox_x3d_proj_active = to_variable(bbox_x3d_proj_active)

                bbox_y3d_proj = bbox_y3d_proj.view()
                bbox_y3d_proj.shape = 1, bbox_total_nums
                bbox_y3d_proj_active = bbox_y3d_proj[:, active]
                bbox_y3d_proj_active = to_variable(bbox_y3d_proj_active)
                bbox_y3d_proj_active.stop_gradient = True

                bbox_z3d_proj = bbox_z3d_proj.view()
                bbox_z3d_proj.shape = 1, bbox_total_nums
                bbox_z3d_proj_active = bbox_z3d_proj[:, active]
                bbox_z3d_proj_active = to_variable(bbox_z3d_proj_active)
                bbox_z3d_proj_active.stop_gradient = True

                loss_bbox_x3d_proj = fluid.layers.smooth_l1(
                    bbox_x3d_proj_active.astype('float32'),
                    bbox_x3d_proj_tar_active.astype('float32'),
                    outside_weight=bbox_weights_active.astype('float32'))
                loss_bbox_y3d_proj = fluid.layers.smooth_l1(
                    bbox_y3d_proj_active.astype('float32'),
                    bbox_y3d_proj_tar_active.astype('float32'),
                    outside_weight=bbox_weights_active.astype('float32'))
                loss_bbox_z3d_proj = fluid.layers.smooth_l1(
                    bbox_z3d_proj_active.astype('float32'),
                    bbox_z3d_proj_tar_active.astype('float32'),
                    outside_weight=bbox_weights_active.astype('float32'))

                bbox_3d_proj_loss = (
                    loss_bbox_x3d_proj + loss_bbox_y3d_proj + loss_bbox_z3d_proj
                )
                bbox_3d_proj_loss = bbox_3d_proj_loss / active_len

                bbox_3d_proj_loss *= self.bbox_3d_proj_lambda

                bbox_3d_proj_loss = bbox_3d_proj_loss
                loss += bbox_3d_proj_loss
                stats.append({
                    'name': 'bbox_3d_proj',
                    'val': bbox_3d_proj_loss.numpy(),
                    'format': '{:0.4f}',
                    'group': 'loss'
                })

            coords_abs_z = fluid.layers.reshape(
                to_variable(coords_abs_z), shape=[-1])
            coords_abs_z_np = coords_abs_z.numpy()
            coords_abs_z_active = coords_abs_z_np[active]
            coords_abs_z = to_variable(coords_abs_z_active)
            coords_abs_z_mean = fluid.layers.mean(coords_abs_z)
            stats.append({
                'name': 'z',
                'val': coords_abs_z_mean.numpy(),
                'format': '{:0.2f}',
                'group': 'misc'
            })

            coords_abs_ry = fluid.layers.reshape(
                to_variable(coords_abs_ry), shape=[-1])
            coords_abs_ry_np = coords_abs_ry.numpy()
            coords_abs_ry_active = coords_abs_ry_np[active]
            coords_abs_ry = to_variable(coords_abs_ry_active)
            coords_abs_ry_mean = fluid.layers.mean(coords_abs_ry)
            stats.append({
                'name': 'ry',
                'val': coords_abs_ry_mean.numpy(),
                'format': '{:0.2f}',
                'group': 'misc'
            })

            ious_2d = fluid.layers.concat(ious_2d_var_list, axis=0)
            ious_2d = fluid.layers.reshape(ious_2d, shape=[-1])

            ious_2d_active = fluid.layers.gather(ious_2d, active_index_var)
            ious_2d_mean = fluid.layers.mean(ious_2d_active)
            stats.append({
                'name': 'iou',
                'val': ious_2d_mean.numpy(),
                'format': '{:0.2f}',
                'group': 'acc'
            })

            # use a 2d IoU based log loss
            if self.iou_2d_lambda:
                iou_2d_loss = -fluid.layers.log(ious_2d_active)
                iou_2d_loss = (iou_2d_loss * bbox_weights_active)
                iou_2d_loss = fluid.layers.mean(iou_2d_loss)

                iou_2d_loss *= self.iou_2d_lambda
                loss += iou_2d_loss

                stats.append({
                    'name': 'iou',
                    'val': iou_2d_loss.numpy(),
                    'format': '{:0.4f}',
                    'group': 'loss'
                })

        return loss, stats
