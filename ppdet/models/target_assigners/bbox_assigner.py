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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import paddle.fluid as fluid

__all__ = ['BBoxAssigner']


class BBoxAssigner(object):
    """
    Get the sampled proposal RoIs and the target bounding-boxes (target
    bounding-box regression deltas given proposal RoIs and ground-truth
    boxes). And for sampled target bbox, assign the classification
    (class label) and the inside weights and outside weights for regression
    loss.

    Args:
        cfg (AttrDict): All configuration.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.use_random = getattr(cfg.TRAIN, 'RANDOM', True)
        self.class_num = cfg.DATA.CLASS_NUM

        local_cfg = cfg.RPN_HEAD.PROPOSAL

        self.batch_size_per_im = local_cfg.BATCH_SIZE_PER_IM
        self.fg_fractrion = local_cfg.FG_FRACTION
        self.fg_thresh = local_cfg.FG_THRESH
        self.bg_thresh_hi = local_cfg.BG_THRESH_HI
        self.bg_thresh_lo = local_cfg.BG_THRESH_LO
        self.bbox_reg_weights = local_cfg.BBOX_REG_WEIGHTS

    def get_sampled_rois_and_targets(self, input_rois, feed_vars):
        """
        Get the sampled proposal RoIs and the target bounding-boxes (target
        bounding-box regression deltas given proposal RoIs and ground-truth
        boxes). And for sampled target bbox, assign the classification
        (class label) and the inside weights and outside weights for regression
        loss.

        Args:
            input_rois (Variable): Input RoI bboxes with shape [P, 4]. P is
                the number of RoIs.
            feed_vars (dict): The input dictionary consist of ground truth.

        Returns:
            rois(Variable): RoI with shape [P, 4]. P is usually equal to
                batch_size_per_im * batch_size, each element
                is a bounding box with [xmin, ymin, xmax, ymax] format.
            labels_int32(Variable): Class label of a RoI with shape [P, 1].
            bbox_targets(Variable): Box label of a RoI with shape
                [P, 4 * class_nums].
            bbox_inside_weights(Variable): Indicates whether a box should
                contribute to loss. Same shape as bbox_targets.
            bbox_outside_weights(Variable): Indicates whether a box should
                contribute to loss. Same shape as bbox_targets.
        """
        if not feed_vars['gt_label']:
            raise ValueError("{} has no gt_label".format(feed_vars))
        if not feed_vars['is_crowd']:
            raise ValueError("{} has no gt_label".format(feed_vars))
        if not feed_vars['gt_box']:
            raise ValueError("{} has no gt_box".format(feed_vars))
        if not feed_vars['im_info']:
            raise ValueError("{} has no im_info".format(feed_vars))

        # outs = (rois, labels_int32, bbox_targets,
        #         bbox_inside_weights, bbox_outside_weights)
        outs = fluid.layers.generate_proposal_labels(
            rpn_rois=input_rois,
            gt_classes=feed_vars['gt_label'],
            is_crowd=feed_vars['is_crowd'],
            gt_boxes=feed_vars['gt_box'],
            im_info=feed_vars['im_info'],
            batch_size_per_im=self.batch_size_per_im,
            fg_fraction=self.fg_fractrion,
            fg_thresh=self.fg_thresh,
            bg_thresh_hi=self.bg_thresh_hi,
            bg_thresh_lo=self.bg_thresh_lo,
            bbox_reg_weights=self.bbox_reg_weights,
            class_nums=self.class_num,
            use_random=self.use_random)
        return outs


class CascadeBBoxAssigner(BBoxAssigner):
    """
    Get the sampled proposal RoIs and the target bounding-boxes (target
    bounding-box regression deltas given proposal RoIs and ground-truth
    boxes). And for sampled target bbox, assign the classification
    (class label) and the inside weights and outside weights for regression
    loss.

    Args:
        cfg (AttrDict): All configuration.
    """

    def __init__(self, cfg):
        super(CascadeBBoxAssigner, self).__init__(cfg)

    def get_sampled_rois_and_targets(self,
                                     input_rois,
                                     feed_vars,
                                     is_cls_agnostic=False,
                                     is_cascade_rcnn=False,
                                     cascade_curr_stage=0):
        """
        Get the sampled proposal RoIs and the target bounding-boxes (target
        bounding-box regression deltas given proposal RoIs and ground-truth
        boxes). And for sampled target bbox, assign the classification
        (class label) and the inside weights and outside weights for regression
        loss.

        Args:
            input_rois (Variable): input RoI bboxes.
            feed_vars (dict): the

        Returns:
            The last variable in endpoint-th stage.
        """
        if not feed_vars['gt_label']:
            raise ValueError("{} has no gt_label".format(feed_vars))
        if not feed_vars['is_crowd']:
            raise ValueError("{} has no gt_label".format(feed_vars))
        if not feed_vars['gt_box']:
            raise ValueError("{} has no gt_box".format(feed_vars))
        if not feed_vars['im_info']:
            raise ValueError("{} has no im_info".format(feed_vars))
        curr_bbox_reg_w = [
            1. / self.bbox_reg_weights[cascade_curr_stage],
            1. / self.bbox_reg_weights[cascade_curr_stage],
            2. / self.bbox_reg_weights[cascade_curr_stage],
            2. / self.bbox_reg_weights[cascade_curr_stage],
        ]
        outs = fluid.layers.generate_proposal_labels(
            rpn_rois=input_rois,
            gt_classes=feed_vars['gt_label'],
            is_crowd=feed_vars['is_crowd'],
            gt_boxes=feed_vars['gt_box'],
            im_info=feed_vars['im_info'],
            batch_size_per_im=self.batch_size_per_im,
            fg_thresh=self.fg_thresh[cascade_curr_stage],
            bg_thresh_hi=self.bg_thresh_hi[cascade_curr_stage],
            bg_thresh_lo=self.bg_thresh_lo[cascade_curr_stage],
            bbox_reg_weights=curr_bbox_reg_w,
            use_random=False,
            class_nums=self.class_num if not is_cls_agnostic else 2,
            is_cls_agnostic=is_cls_agnostic,
            is_cascade_rcnn=is_cascade_rcnn, )

        return outs
