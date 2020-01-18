# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import numpy as np
import paddle.fluid as fluid
from paddle.fluid.dygraph.base import to_variable
#from cyops.target import rpn_target_assign, generate_proposal_labels  
from pyops.target import rpn_target_assign, generate_proposal_labels, generate_mask_labels


def rpn_target_assign_pyfn(bbox_pred,
                           cls_logits,
                           anchor_box,
                           anchor_var,
                           gt_boxes,
                           is_crowd,
                           im_info,
                           rpn_straddle_thresh,
                           rpn_batch_size_per_im,
                           rpn_positive_overlap,
                           rpn_negative_overlap,
                           rpn_fg_fraction,
                           use_random=True):

    anchor_box = anchor_box.numpy()
    gt_boxes = gt_boxes.numpy()
    is_crowd = is_crowd.numpy()
    im_info = im_info.numpy()

    loc_indexes, score_indexes, tgt_labels, tgt_bboxes, bbox_inside_weights = rpn_target_assign(
        anchor_box,
        gt_boxes,
        is_crowd,
        im_info,
        rpn_straddle_thresh,
        rpn_batch_size_per_im,
        rpn_positive_overlap,
        rpn_negative_overlap,
        rpn_fg_fraction,
        use_random, )

    loc_indexes = to_variable(loc_indexes)
    score_indexes = to_variable(score_indexes)
    tgt_labels = to_variable(tgt_labels)
    tgt_bboxes = to_variable(tgt_bboxes)
    bbox_inside_weights = to_variable(bbox_inside_weights)

    loc_indexes.stop_gradient = True
    score_indexes.stop_gradient = True
    tgt_labels.stop_gradient = True

    cls_logits = fluid.layers.reshape(x=cls_logits, shape=(-1, ))
    bbox_pred = fluid.layers.reshape(x=bbox_pred, shape=(-1, 4))
    pred_cls_logits = fluid.layers.gather(cls_logits, score_indexes)
    pred_bbox_pred = fluid.layers.gather(bbox_pred, loc_indexes)

    return pred_cls_logits, pred_bbox_pred, tgt_labels, tgt_bboxes, bbox_inside_weights


def generate_proposal_labels_pyfn(rpn_rois,
                                  rpn_rois_lod,
                                  gt_classes,
                                  is_crowd,
                                  gt_boxes,
                                  im_info,
                                  batch_size_per_im,
                                  fg_fraction,
                                  fg_thresh,
                                  bg_thresh_hi,
                                  bg_thresh_lo,
                                  bbox_reg_weights,
                                  class_nums,
                                  use_random=True,
                                  is_cls_agnostic=False,
                                  is_cascade_rcnn=False):
    rpn_rois = rpn_rois.numpy()
    rpn_rois_lod = rpn_rois_lod.numpy()
    gt_classes = gt_classes.numpy()
    gt_boxes = gt_boxes.numpy()
    is_crowd = is_crowd.numpy()
    im_info = im_info.numpy()

    outs = generate_proposal_labels(
        rpn_rois, rpn_rois_lod, gt_classes, is_crowd, gt_boxes, im_info,
        batch_size_per_im, fg_fraction, fg_thresh, bg_thresh_hi, bg_thresh_lo,
        np.asarray(bbox_reg_weights), class_nums, use_random, is_cls_agnostic,
        is_cascade_rcnn)

    outs = [to_variable(v) for v in outs]
    for v in outs:
        v.stop_gradient = True
    return outs


def generate_mask_labels_pyfn(im_info, gt_classes, is_crowd, gt_segms, rois,
                              rois_lod, labels_int32, num_classes, resolution):
    #print(type(im_info), type(gt_classes), type(is_crowd), type(gt_segms), type(rois), type(rois_lod), type(labels_int32), type(num_classes), type(resolution))
    im_info = im_info.numpy()
    gt_classes = gt_classes.numpy()
    is_crowd = is_crowd.numpy()
    gt_segms = gt_segms.numpy()
    rois = rois.numpy()
    rois_lod = rois_lod.numpy()
    labels_int32 = labels_int32.numpy()
    #print("gt_segms: ", gt_segms.shape) 
    outs = generate_mask_labels(im_info, gt_classes, is_crowd, gt_segms, rois,
                                rois_lod, labels_int32, num_classes, resolution)

    outs = [to_variable(v) for v in outs]
    for v in outs:
        v.stop_gradient = True
    return outs
