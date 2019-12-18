#  Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import numpy as np
from utils.config import cfg
from utils import calibration as calib
import utils.cyops.kitti_utils as kitti_utils 

__all__ = ['save_rpn_feature', 'save_kitti_result', 'save_kitti_format']


def save_rpn_feature(rets, kitti_features_dir):
    """
    save rpn features for RCNN offline training
    """

    sample_id = rets['sample_id'][0]
    backbone_xyz = rets['backbone_xyz'][0]
    backbone_feature = rets['backbone_feature'][0]
    pts_features = rets['pts_features'][0]
    seg_mask = rets['seg_mask'][0]
    rpn_cls = rets['rpn_cls'][0]

    for i in range(len(sample_id)):
        pts_intensity = pts_features[i, :, 0]
        s_id = sample_id[i, 0]

        output_file = os.path.join(kitti_features_dir, '%06d.npy' % s_id)
        xyz_file = os.path.join(kitti_features_dir, '%06d_xyz.npy' % s_id)
        seg_file = os.path.join(kitti_features_dir, '%06d_seg.npy' % s_id)
        intensity_file = os.path.join(
            kitti_features_dir, '%06d_intensity.npy' % s_id)
        np.save(output_file, backbone_feature[i])
        np.save(xyz_file, backbone_xyz[i])
        np.save(seg_file, seg_mask[i])
        np.save(intensity_file, pts_intensity)
        rpn_scores_raw_file = os.path.join(
            kitti_features_dir, '%06d_rawscore.npy' % s_id)
        np.save(rpn_scores_raw_file, rpn_cls[i])


def save_kitti_result(rets, seg_output_dir, kitti_output_dir, reader, classes):
    sample_id = rets['sample_id'][0]
    roi_scores_row = rets['roi_scores_row'][0]
    bboxes3d = rets['rois'][0]
    pts_rect = rets['pts_rect'][0]
    seg_mask = rets['seg_mask'][0]
    rpn_cls_label = rets['rpn_cls_label'][0]
    gt_boxes3d = rets['gt_boxes3d'][0]
    gt_boxes3d_num = rets['gt_boxes3d'][1]

    for i in range(len(sample_id)):
        s_id = sample_id[i, 0]

        seg_result_data = np.concatenate((pts_rect[i].reshape(-1, 3),
                                          rpn_cls_label[i].reshape(-1, 1),
                                          seg_mask[i].reshape(-1, 1)),
                                         axis=1).astype('float16')
        seg_output_file = os.path.join(seg_output_dir, '%06d.npy' % s_id)
        np.save(seg_output_file, seg_result_data)

        scores = roi_scores_row[i, :]
        bbox3d = bboxes3d[i, :]
        img_shape = reader.get_image_shape(s_id)
        calib = reader.get_calib(s_id)

        corners3d = kitti_utils.boxes3d_to_corners3d(bbox3d)
        img_boxes, _ = calib.corners3d_to_img_boxes(corners3d)

        img_boxes[:, 0] = np.clip(img_boxes[:, 0], 0, img_shape[1] - 1)
        img_boxes[:, 1] = np.clip(img_boxes[:, 1], 0, img_shape[0] - 1)
        img_boxes[:, 2] = np.clip(img_boxes[:, 2], 0, img_shape[1] - 1)
        img_boxes[:, 3] = np.clip(img_boxes[:, 3], 0, img_shape[0] - 1)

        img_boxes_w = img_boxes[:, 2] - img_boxes[:, 0]
        img_boxes_h = img_boxes[:, 3] - img_boxes[:, 1]
        box_valid_mask = np.logical_and(
            img_boxes_w < img_shape[1] * 0.8, img_boxes_h < img_shape[0] * 0.8)

        kitti_output_file = os.path.join(kitti_output_dir, '%06d.txt' % s_id)
        with open(kitti_output_file, 'w') as f:
            for k in range(bbox3d.shape[0]):
                if box_valid_mask[k] == 0:
                    continue
                x, z, ry = bbox3d[k, 0], bbox3d[k, 2], bbox3d[k, 6]
                beta = np.arctan2(z, x)
                alpha = -np.sign(beta) * np.pi / 2 + beta + ry

                f.write('{} -1 -1 {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}\n'.format(
                    classes, alpha, img_boxes[k, 0], img_boxes[k, 1], img_boxes[k, 2], img_boxes[k, 3],
                    bbox3d[k, 3], bbox3d[k, 4], bbox3d[k, 5], bbox3d[k, 0], bbox3d[k, 1], bbox3d[k, 2],
                    bbox3d[k, 6], scores[k]))


def save_kitti_format(sample_id, calib, bbox3d, kitti_output_dir, scores, img_shape):
    corners3d = kitti_utils.boxes3d_to_corners3d(bbox3d)
    img_boxes, _ = calib.corners3d_to_img_boxes(corners3d)
    img_boxes[:, 0] = np.clip(img_boxes[:, 0], 0, img_shape[1] - 1)
    img_boxes[:, 1] = np.clip(img_boxes[:, 1], 0, img_shape[0] - 1)
    img_boxes[:, 2] = np.clip(img_boxes[:, 2], 0, img_shape[1] - 1)
    img_boxes[:, 3] = np.clip(img_boxes[:, 3], 0, img_shape[0] - 1)

    img_boxes_w = img_boxes[:, 2] - img_boxes[:, 0]
    img_boxes_h = img_boxes[:, 3] - img_boxes[:, 1]
    box_valid_mask = np.logical_and(img_boxes_w < img_shape[1] * 0.8, img_boxes_h < img_shape[0] * 0.8)

    kitti_output_file = os.path.join(kitti_output_dir, '%06d.txt' % sample_id)
    with open(kitti_output_file, 'w') as f:
        for k in range(bbox3d.shape[0]):
            if box_valid_mask[k] == 0:
                continue
            x, z, ry = bbox3d[k, 0], bbox3d[k, 2], bbox3d[k, 6]
            beta = np.arctan2(z, x)
            alpha = -np.sign(beta) * np.pi / 2 + beta + ry

            f.write('%s -1 -1 %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n' %
                  (cfg.CLASSES, alpha, img_boxes[k, 0], img_boxes[k, 1], img_boxes[k, 2], img_boxes[k, 3],
                   bbox3d[k, 3], bbox3d[k, 4], bbox3d[k, 5], bbox3d[k, 0], bbox3d[k, 1], bbox3d[k, 2],
                   bbox3d[k, 6], scores[k]))

