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
"""
This code is based on https://github.com/sshaoshuai/PointRCNN/blob/master/lib/datasets/kitti_rcnn_dataset.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import signal
import logging
import multiprocessing
import numpy as np
import scipy
from scipy.spatial import Delaunay
try:
    import cPickle as pickle
except:
    import pickle

import pts_utils
import utils.cyops.kitti_utils as kitti_utils
import utils.cyops.roipool3d_utils as roipool3d_utils
from data.kitti_dataset import KittiDataset
from utils.config import cfg
from collections import OrderedDict

__all__ = ["KittiRCNNReader"]

logger = logging.getLogger(__name__)


def has_empty(data):
    for d in data:
        if isinstance(d, np.ndarray) and len(d) == 0:
            return True
    return False


def in_hull(p, hull):
    """
    :param p: (N, K) test points
    :param hull: (M, K) M corners of a box
    :return (N) bool
    """
    try:
        if not isinstance(hull, Delaunay):
            hull = Delaunay(hull)
        flag = hull.find_simplex(p) >= 0
    except scipy.spatial.qhull.QhullError:
        logger.debug('Warning: not a hull.')
        flag = np.zeros(p.shape[0], dtype=np.bool)

    return flag


class KittiRCNNReader(KittiDataset):
    def __init__(self, data_dir, npoints=16384, split='train', classes='Car', mode='TRAIN',
                 random_select=True, rcnn_training_roi_dir=None, rcnn_training_feature_dir=None,
                 rcnn_eval_roi_dir=None, rcnn_eval_feature_dir=None, gt_database_dir=None):
        super(KittiRCNNReader, self).__init__(data_dir=data_dir, split=split)
        if classes == 'Car':
            self.classes = ('Background', 'Car')
            aug_scene_data_dir = os.path.join(data_dir, 'KITTI', 'aug_scene')
        elif classes == 'People':
            self.classes = ('Background', 'Pedestrian', 'Cyclist')
        elif classes == 'Pedestrian':
            self.classes = ('Background', 'Pedestrian')
            aug_scene_data_dir = os.path.join(data_dir, 'KITTI', 'aug_scene_ped')
        elif classes == 'Cyclist':
            self.classes = ('Background', 'Cyclist')
            aug_scene_data_dir = os.path.join(data_dir, 'KITTI', 'aug_scene_cyclist')
        else:
            assert False, "Invalid classes: %s" % classes

        self.num_classes = len(self.classes)

        self.npoints = npoints
        self.sample_id_list = []
        self.random_select = random_select

        if split == 'train_aug':
            self.aug_label_dir = os.path.join(aug_scene_data_dir, 'training', 'aug_label')
            self.aug_pts_dir = os.path.join(aug_scene_data_dir, 'training', 'rectified_data')
        else:
            self.aug_label_dir = os.path.join(aug_scene_data_dir, 'training', 'aug_label')
            self.aug_pts_dir = os.path.join(aug_scene_data_dir, 'training', 'rectified_data')

        # for rcnn training
        self.rcnn_training_bbox_list = []
        self.rpn_feature_list = {}
        self.pos_bbox_list = []
        self.neg_bbox_list = []
        self.far_neg_bbox_list = []
        self.rcnn_eval_roi_dir = rcnn_eval_roi_dir
        self.rcnn_eval_feature_dir = rcnn_eval_feature_dir
        self.rcnn_training_roi_dir = rcnn_training_roi_dir
        self.rcnn_training_feature_dir = rcnn_training_feature_dir

        self.gt_database = None

        if not self.random_select:
            logger.warning('random select is False')

        assert mode in ['TRAIN', 'EVAL', 'TEST'], 'Invalid mode: %s' % mode
        self.mode = mode

        if cfg.RPN.ENABLED:
            if gt_database_dir is not None:
                self.gt_database = pickle.load(open(gt_database_dir, 'rb'))

                if cfg.GT_AUG_HARD_RATIO > 0:
                    easy_list, hard_list = [], []
                    for k in range(self.gt_database.__len__()):
                        obj = self.gt_database[k]
                        if obj['points'].shape[0] > 100:
                            easy_list.append(obj)
                        else:
                            hard_list.append(obj)
                    self.gt_database = [easy_list, hard_list]
                    logger.info('Loading gt_database(easy(pt_num>100): %d, hard(pt_num<=100): %d) from %s'
                                % (len(easy_list), len(hard_list), gt_database_dir))
                else:
                    logger.info('Loading gt_database(%d) from %s' % (len(self.gt_database), gt_database_dir))

            if mode == 'TRAIN':
                self.preprocess_rpn_training_data()
            else:
                self.sample_id_list = [int(sample_id) for sample_id in self.image_idx_list]
                logger.info('Load testing samples from %s' % self.imageset_dir)
                logger.info('Done: total test samples %d' % len(self.sample_id_list))
        elif cfg.RCNN.ENABLED:
            for idx in range(0, self.num_sample):
                sample_id = int(self.image_idx_list[idx])
                obj_list = self.filtrate_objects(self.get_label(sample_id))
                if len(obj_list) == 0:
                    # logger.info('No gt classes: %06d' % sample_id)
                    continue
                self.sample_id_list.append(sample_id)

            logger.info('Done: filter %s results for rcnn training: %d / %d\n' %
                  (self.mode, len(self.sample_id_list), len(self.image_idx_list)))

    def preprocess_rpn_training_data(self):
        """
        Discard samples which don't have current classes, which will not be used for training.
        Valid sample_id is stored in self.sample_id_list
        """
        logger.info('Loading %s samples from %s ...' % (self.mode, self.label_dir))
        for idx in range(0, self.num_sample):
            sample_id = int(self.image_idx_list[idx])
            obj_list = self.filtrate_objects(self.get_label(sample_id))
            if len(obj_list) == 0:
                logger.debug('No gt classes: %06d' % sample_id)
                continue
            self.sample_id_list.append(sample_id)

        logger.info('Done: filter %s results: %d / %d\n' % (self.mode, len(self.sample_id_list),
                                                                 len(self.image_idx_list)))

    def get_label(self, idx):
        if idx < 10000:
            label_file = os.path.join(self.label_dir, '%06d.txt' % idx)
        else:
            label_file = os.path.join(self.aug_label_dir, '%06d.txt' % idx)

        assert os.path.exists(label_file)
        return kitti_utils.get_objects_from_label(label_file)

    def get_image(self, idx):
        return super(KittiRCNNReader, self).get_image(idx % 10000)

    def get_image_shape(self, idx):
        return super(KittiRCNNReader, self).get_image_shape(idx % 10000)

    def get_calib(self, idx):
        return super(KittiRCNNReader, self).get_calib(idx % 10000)

    def get_road_plane(self, idx):
        return super(KittiRCNNReader, self).get_road_plane(idx % 10000)

    @staticmethod
    def get_rpn_features(rpn_feature_dir, idx):
        rpn_feature_file = os.path.join(rpn_feature_dir, '%06d.npy' % idx)
        rpn_xyz_file = os.path.join(rpn_feature_dir, '%06d_xyz.npy' % idx)
        rpn_intensity_file = os.path.join(rpn_feature_dir, '%06d_intensity.npy' % idx)
        if cfg.RCNN.USE_SEG_SCORE:
            rpn_seg_file = os.path.join(rpn_feature_dir, '%06d_rawscore.npy' % idx)
            rpn_seg_score = np.load(rpn_seg_file).reshape(-1)
            rpn_seg_score = torch.sigmoid(torch.from_numpy(rpn_seg_score)).numpy()
        else:
            rpn_seg_file = os.path.join(rpn_feature_dir, '%06d_seg.npy' % idx)
            rpn_seg_score = np.load(rpn_seg_file).reshape(-1)
        return np.load(rpn_xyz_file), np.load(rpn_feature_file), np.load(rpn_intensity_file).reshape(-1), rpn_seg_score

    def filtrate_objects(self, obj_list):
        """
        Discard objects which are not in self.classes (or its similar classes)
        :param obj_list: list
        :return: list
        """
        type_whitelist = self.classes
        if self.mode == 'TRAIN' and cfg.INCLUDE_SIMILAR_TYPE:
            type_whitelist = list(self.classes)
            if 'Car' in self.classes:
                type_whitelist.append('Van')
            if 'Pedestrian' in self.classes:  # or 'Cyclist' in self.classes:
                type_whitelist.append('Person_sitting')

        valid_obj_list = []
        for obj in obj_list:
            if obj.cls_type not in type_whitelist:  # rm Van, 20180928
                continue
            if self.mode == 'TRAIN' and cfg.PC_REDUCE_BY_RANGE and (self.check_pc_range(obj.pos) is False):
                continue
            valid_obj_list.append(obj)
        return valid_obj_list

    @staticmethod
    def filtrate_dc_objects(obj_list):
        valid_obj_list = []
        for obj in obj_list:
            if obj.cls_type in ['DontCare']:
                continue
            valid_obj_list.append(obj)

        return valid_obj_list

    @staticmethod
    def check_pc_range(xyz):
        """
        :param xyz: [x, y, z]
        :return:
        """
        x_range, y_range, z_range = cfg.PC_AREA_SCOPE
        if (x_range[0] <= xyz[0] <= x_range[1]) and (y_range[0] <= xyz[1] <= y_range[1]) and \
                (z_range[0] <= xyz[2] <= z_range[1]):
            return True
        return False

    @staticmethod
    def get_valid_flag(pts_rect, pts_img, pts_rect_depth, img_shape):
        """
        Valid point should be in the image (and in the PC_AREA_SCOPE)
        :param pts_rect:
        :param pts_img:
        :param pts_rect_depth:
        :param img_shape:
        :return:
        """
        val_flag_1 = np.logical_and(pts_img[:, 0] >= 0, pts_img[:, 0] < img_shape[1])
        val_flag_2 = np.logical_and(pts_img[:, 1] >= 0, pts_img[:, 1] < img_shape[0])
        val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
        pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)

        if cfg.PC_REDUCE_BY_RANGE:
            x_range, y_range, z_range = cfg.PC_AREA_SCOPE
            pts_x, pts_y, pts_z = pts_rect[:, 0], pts_rect[:, 1], pts_rect[:, 2]
            range_flag = (pts_x >= x_range[0]) & (pts_x <= x_range[1]) \
                         & (pts_y >= y_range[0]) & (pts_y <= y_range[1]) \
                         & (pts_z >= z_range[0]) & (pts_z <= z_range[1])
            pts_valid_flag = pts_valid_flag & range_flag
        return pts_valid_flag

    def get_rpn_sample(self, index):
        sample_id = int(self.sample_id_list[index])
        if sample_id < 10000:
            calib = self.get_calib(sample_id)
            # img = self.get_image(sample_id)
            img_shape = self.get_image_shape(sample_id)
            pts_lidar = self.get_lidar(sample_id)

            # get valid point (projected points should be in image)
            pts_rect = calib.lidar_to_rect(pts_lidar[:, 0:3])
            pts_intensity = pts_lidar[:, 3]
        else:
            calib = self.get_calib(sample_id % 10000)
            # img = self.get_image(sample_id % 10000)
            img_shape = self.get_image_shape(sample_id % 10000)

            pts_file = os.path.join(self.aug_pts_dir, '%06d.bin' % sample_id)
            assert os.path.exists(pts_file), '%s' % pts_file
            aug_pts = np.fromfile(pts_file, dtype=np.float32).reshape(-1, 4)
            pts_rect, pts_intensity = aug_pts[:, 0:3], aug_pts[:, 3]

        pts_img, pts_rect_depth = calib.rect_to_img(pts_rect)
        pts_valid_flag = self.get_valid_flag(pts_rect, pts_img, pts_rect_depth, img_shape)

        pts_rect = pts_rect[pts_valid_flag][:, 0:3]
        pts_intensity = pts_intensity[pts_valid_flag]

        if cfg.GT_AUG_ENABLED and self.mode == 'TRAIN':
            # all labels for checking overlapping
            all_gt_obj_list = self.filtrate_dc_objects(self.get_label(sample_id))
            all_gt_boxes3d = kitti_utils.objs_to_boxes3d(all_gt_obj_list)

            gt_aug_flag = False
            if np.random.rand() < cfg.GT_AUG_APPLY_PROB:
                # augment one scene
                gt_aug_flag, pts_rect, pts_intensity, extra_gt_boxes3d, extra_gt_obj_list = \
                    self.apply_gt_aug_to_one_scene(sample_id, pts_rect, pts_intensity, all_gt_boxes3d)

        # generate inputs
        if self.mode == 'TRAIN' or self.random_select:
            if self.npoints < len(pts_rect):
                pts_depth = pts_rect[:, 2]
                pts_near_flag = pts_depth < 40.0
                far_idxs_choice = np.where(pts_near_flag == 0)[0]
                near_idxs = np.where(pts_near_flag == 1)[0]
                near_idxs_choice = np.random.choice(near_idxs, self.npoints - len(far_idxs_choice), replace=False)

                choice = np.concatenate((near_idxs_choice, far_idxs_choice), axis=0) \
                    if len(far_idxs_choice) > 0 else near_idxs_choice
                np.random.shuffle(choice)
            else:
                choice = np.arange(0, len(pts_rect), dtype=np.int32)
                if self.npoints > len(pts_rect):
                    extra_choice = np.random.choice(choice, self.npoints - len(pts_rect), replace=False)
                    choice = np.concatenate((choice, extra_choice), axis=0)
                np.random.shuffle(choice)

            ret_pts_rect = pts_rect[choice, :]
            ret_pts_intensity = pts_intensity[choice] - 0.5  # translate intensity to [-0.5, 0.5]
        else:
            ret_pts_rect = np.zeros((self.npoints, pts_rect.shape[1])).astype(pts_rect.dtype)
            num_ = min(self.npoints, pts_rect.shape[0])
            ret_pts_rect[:num_] = pts_rect[:num_]

            ret_pts_intensity = pts_intensity - 0.5

        pts_features = [ret_pts_intensity.reshape(-1, 1)]
        ret_pts_features = np.concatenate(pts_features, axis=1) if pts_features.__len__() > 1 else pts_features[0]

        sample_info = {'sample_id': sample_id, 'random_select': self.random_select}

        if self.mode == 'TEST':
            if cfg.RPN.USE_INTENSITY:
                pts_input = np.concatenate((ret_pts_rect, ret_pts_features), axis=1)  # (N, C)
            else:
                pts_input = ret_pts_rect
            sample_info['pts_input'] = pts_input
            sample_info['pts_rect'] = ret_pts_rect
            sample_info['pts_features'] = ret_pts_features
            return sample_info

        gt_obj_list = self.filtrate_objects(self.get_label(sample_id))
        if cfg.GT_AUG_ENABLED and self.mode == 'TRAIN' and gt_aug_flag:
            gt_obj_list.extend(extra_gt_obj_list)
        gt_boxes3d = kitti_utils.objs_to_boxes3d(gt_obj_list)

        gt_alpha = np.zeros((gt_obj_list.__len__()), dtype=np.float32)
        for k, obj in enumerate(gt_obj_list):
            gt_alpha[k] = obj.alpha

        # data augmentation
        aug_pts_rect = ret_pts_rect.copy()
        aug_gt_boxes3d = gt_boxes3d.copy()
        if cfg.AUG_DATA and self.mode == 'TRAIN':
            aug_pts_rect, aug_gt_boxes3d, aug_method = self.data_augmentation(aug_pts_rect, aug_gt_boxes3d, gt_alpha,
                                                                              sample_id)
            sample_info['aug_method'] = aug_method

        # prepare input
        if cfg.RPN.USE_INTENSITY:
            pts_input = np.concatenate((aug_pts_rect, ret_pts_features), axis=1)  # (N, C)
        else:
            pts_input = aug_pts_rect

        if cfg.RPN.FIXED:
            sample_info['pts_input'] = pts_input
            sample_info['pts_rect'] = aug_pts_rect
            sample_info['pts_features'] = ret_pts_features
            sample_info['gt_boxes3d'] = aug_gt_boxes3d
            return sample_info

        if self.mode == 'EVAL' and aug_gt_boxes3d.shape[0] == 0:
            aug_gt_boxes3d = np.zeros((1, aug_gt_boxes3d.shape[1]))

        # generate training labels
        rpn_cls_label, rpn_reg_label = self.generate_rpn_training_labels(aug_pts_rect, aug_gt_boxes3d)
        sample_info['pts_input'] = pts_input
        sample_info['pts_rect'] = aug_pts_rect
        sample_info['pts_features'] = ret_pts_features
        sample_info['rpn_cls_label'] = rpn_cls_label
        sample_info['rpn_reg_label'] = rpn_reg_label
        sample_info['gt_boxes3d'] = aug_gt_boxes3d
        return sample_info

    def apply_gt_aug_to_one_scene(self, sample_id, pts_rect, pts_intensity, all_gt_boxes3d):
        """
        :param pts_rect: (N, 3)
        :param all_gt_boxex3d: (M2, 7)
        :return:
        """
        assert self.gt_database is not None
        # extra_gt_num = np.random.randint(10, 15)
        # try_times = 50
        if cfg.GT_AUG_RAND_NUM:
            extra_gt_num = np.random.randint(10, cfg.GT_EXTRA_NUM)
        else:
            extra_gt_num = cfg.GT_EXTRA_NUM
        try_times = 100
        cnt = 0
        cur_gt_boxes3d = all_gt_boxes3d.copy()
        cur_gt_boxes3d[:, 4] += 0.5  # TODO: consider different objects
        cur_gt_boxes3d[:, 5] += 0.5  # enlarge new added box to avoid too nearby boxes
        cur_gt_corners = kitti_utils.boxes3d_to_corners3d(cur_gt_boxes3d)

        extra_gt_obj_list = []
        extra_gt_boxes3d_list = []
        new_pts_list, new_pts_intensity_list = [], []
        src_pts_flag = np.ones(pts_rect.shape[0], dtype=np.int32)

        road_plane = self.get_road_plane(sample_id)
        a, b, c, d = road_plane

        while try_times > 0:
            if cnt > extra_gt_num:
                break

            try_times -= 1
            if cfg.GT_AUG_HARD_RATIO > 0:
                p = np.random.rand()
                if p > cfg.GT_AUG_HARD_RATIO:
                    # use easy sample
                    rand_idx = np.random.randint(0, len(self.gt_database[0]))
                    new_gt_dict = self.gt_database[0][rand_idx]
                else:
                    # use hard sample
                    rand_idx = np.random.randint(0, len(self.gt_database[1]))
                    new_gt_dict = self.gt_database[1][rand_idx]
            else:
                rand_idx = np.random.randint(0, self.gt_database.__len__())
                new_gt_dict = self.gt_database[rand_idx]

            new_gt_box3d = new_gt_dict['gt_box3d'].copy()
            new_gt_points = new_gt_dict['points'].copy()
            new_gt_intensity = new_gt_dict['intensity'].copy()
            new_gt_obj = new_gt_dict['obj']
            center = new_gt_box3d[0:3]
            if cfg.PC_REDUCE_BY_RANGE and (self.check_pc_range(center) is False):
                continue

            if new_gt_points.__len__() < 5:  # too few points
                continue

            # put it on the road plane
            cur_height = (-d - a * center[0] - c * center[2]) / b
            move_height = new_gt_box3d[1] - cur_height
            new_gt_box3d[1] -= move_height
            new_gt_points[:, 1] -= move_height
            new_gt_obj.pos[1] -= move_height

            new_enlarged_box3d = new_gt_box3d.copy()
            new_enlarged_box3d[4] += 0.5
            new_enlarged_box3d[5] += 0.5  # enlarge new added box to avoid too nearby boxes

            cnt += 1
            new_corners = kitti_utils.boxes3d_to_corners3d(new_enlarged_box3d.reshape(1, 7))
            iou3d = kitti_utils.get_iou3d(new_corners, cur_gt_corners)
            valid_flag = iou3d.max() < 1e-8
            if not valid_flag:
                continue

            enlarged_box3d = new_gt_box3d.copy()
            enlarged_box3d[3] += 2  # remove the points above and below the object

            boxes_pts_mask_list = pts_utils.pts_in_boxes3d(pts_rect,
                    enlarged_box3d.reshape(1, 7))
            pt_mask_flag = (boxes_pts_mask_list[0] == 1)
            src_pts_flag[pt_mask_flag] = 0  # remove the original points which are inside the new box

            new_pts_list.append(new_gt_points)
            new_pts_intensity_list.append(new_gt_intensity)
            cur_gt_boxes3d = np.concatenate((cur_gt_boxes3d, new_enlarged_box3d.reshape(1, 7)), axis=0)
            cur_gt_corners = np.concatenate((cur_gt_corners, new_corners), axis=0)
            extra_gt_boxes3d_list.append(new_gt_box3d.reshape(1, 7))
            extra_gt_obj_list.append(new_gt_obj)

        if new_pts_list.__len__() == 0:
            return False, pts_rect, pts_intensity, None, None

        extra_gt_boxes3d = np.concatenate(extra_gt_boxes3d_list, axis=0)
        # remove original points and add new points
        pts_rect = pts_rect[src_pts_flag == 1]
        pts_intensity = pts_intensity[src_pts_flag == 1]
        new_pts_rect = np.concatenate(new_pts_list, axis=0)
        new_pts_intensity = np.concatenate(new_pts_intensity_list, axis=0)
        pts_rect = np.concatenate((pts_rect, new_pts_rect), axis=0)
        pts_intensity = np.concatenate((pts_intensity, new_pts_intensity), axis=0)

        return True, pts_rect, pts_intensity, extra_gt_boxes3d, extra_gt_obj_list

    def rotate_box3d_along_y(self, box3d, rot_angle):
        old_x, old_z, ry = box3d[0], box3d[2], box3d[6]
        old_beta = np.arctan2(old_z, old_x)
        alpha = -np.sign(old_beta) * np.pi / 2 + old_beta + ry
        box3d = kitti_utils.rotate_pc_along_y(box3d.reshape(1, 7), rot_angle=rot_angle)[0]
        new_x, new_z = box3d[0], box3d[2]
        new_beta = np.arctan2(new_z, new_x)
        box3d[6] = np.sign(new_beta) * np.pi / 2 + alpha - new_beta
        return box3d

    def data_augmentation(self, aug_pts_rect, aug_gt_boxes3d, gt_alpha, sample_id=None, mustaug=False, stage=1):
        """
        :param aug_pts_rect: (N, 3)
        :param aug_gt_boxes3d: (N, 7)
        :param gt_alpha: (N)
        :return:
        """
        aug_list = cfg.AUG_METHOD_LIST
        aug_enable = 1 - np.random.rand(3)
        if mustaug is True:
            aug_enable[0] = -1
            aug_enable[1] = -1
        aug_method = []
        if 'rotation' in aug_list and aug_enable[0] < cfg.AUG_METHOD_PROB[0]:
            angle = np.random.uniform(-np.pi / cfg.AUG_ROT_RANGE, np.pi / cfg.AUG_ROT_RANGE)
            aug_pts_rect = kitti_utils.rotate_pc_along_y(aug_pts_rect, rot_angle=angle)
            if stage == 1:
                # xyz change, hwl unchange
                aug_gt_boxes3d = kitti_utils.rotate_pc_along_y(aug_gt_boxes3d, rot_angle=angle)

                # calculate the ry after rotation
                x, z = aug_gt_boxes3d[:, 0], aug_gt_boxes3d[:, 2]
                beta = np.arctan2(z, x)
                new_ry = np.sign(beta) * np.pi / 2 + gt_alpha - beta
                aug_gt_boxes3d[:, 6] = new_ry  # TODO: not in [-np.pi / 2, np.pi / 2]
            elif stage == 2:
                # for debug stage-2, this implementation has little float precision difference with the above one
                assert aug_gt_boxes3d.shape[0] == 2
                aug_gt_boxes3d[0] = self.rotate_box3d_along_y(aug_gt_boxes3d[0], angle)
                aug_gt_boxes3d[1] = self.rotate_box3d_along_y(aug_gt_boxes3d[1], angle)
            else:
                raise NotImplementedError

            aug_method.append(['rotation', angle])

        if 'scaling' in aug_list and aug_enable[1] < cfg.AUG_METHOD_PROB[1]:
            scale = np.random.uniform(0.95, 1.05)
            aug_pts_rect = aug_pts_rect * scale
            aug_gt_boxes3d[:, 0:6] = aug_gt_boxes3d[:, 0:6] * scale
            aug_method.append(['scaling', scale])

        if 'flip' in aug_list and aug_enable[2] < cfg.AUG_METHOD_PROB[2]:
            # flip horizontal
            aug_pts_rect[:, 0] = -aug_pts_rect[:, 0]
            aug_gt_boxes3d[:, 0] = -aug_gt_boxes3d[:, 0]
            # flip orientation: ry > 0: pi - ry, ry < 0: -pi - ry
            if stage == 1:
                aug_gt_boxes3d[:, 6] = np.sign(aug_gt_boxes3d[:, 6]) * np.pi - aug_gt_boxes3d[:, 6]
            elif stage == 2:
                assert aug_gt_boxes3d.shape[0] == 2
                aug_gt_boxes3d[0, 6] = np.sign(aug_gt_boxes3d[0, 6]) * np.pi - aug_gt_boxes3d[0, 6]
                aug_gt_boxes3d[1, 6] = np.sign(aug_gt_boxes3d[1, 6]) * np.pi - aug_gt_boxes3d[1, 6]
            else:
                raise NotImplementedError

            aug_method.append('flip')

        return aug_pts_rect, aug_gt_boxes3d, aug_method

    @staticmethod
    def generate_rpn_training_labels(pts_rect, gt_boxes3d):
        cls_label = np.zeros((pts_rect.shape[0]), dtype=np.int32)
        reg_label = np.zeros((pts_rect.shape[0], 7), dtype=np.float32)  # dx, dy, dz, ry, h, w, l
        gt_corners = kitti_utils.boxes3d_to_corners3d(gt_boxes3d, rotate=True)
        extend_gt_boxes3d = kitti_utils.enlarge_box3d(gt_boxes3d, extra_width=0.2)
        extend_gt_corners = kitti_utils.boxes3d_to_corners3d(extend_gt_boxes3d, rotate=True)
        for k in range(gt_boxes3d.shape[0]):
            box_corners = gt_corners[k]
            fg_pt_flag = in_hull(pts_rect, box_corners)
            fg_pts_rect = pts_rect[fg_pt_flag]
            cls_label[fg_pt_flag] = 1

            # enlarge the bbox3d, ignore nearby points
            extend_box_corners = extend_gt_corners[k]
            fg_enlarge_flag = in_hull(pts_rect, extend_box_corners)
            ignore_flag = np.logical_xor(fg_pt_flag, fg_enlarge_flag)
            cls_label[ignore_flag] = -1

            # pixel offset of object center
            center3d = gt_boxes3d[k][0:3].copy()  # (x, y, z)
            center3d[1] -= gt_boxes3d[k][3] / 2
            reg_label[fg_pt_flag, 0:3] = center3d - fg_pts_rect  # Now y is the true center of 3d box 20180928

            # size and angle encoding
            reg_label[fg_pt_flag, 3] = gt_boxes3d[k][3]  # h
            reg_label[fg_pt_flag, 4] = gt_boxes3d[k][4]  # w
            reg_label[fg_pt_flag, 5] = gt_boxes3d[k][5]  # l
            reg_label[fg_pt_flag, 6] = gt_boxes3d[k][6]  # ry

        return cls_label, reg_label

    def get_rcnn_sample_jit(self, index):
        sample_id = int(self.sample_id_list[index])
        rpn_xyz, rpn_features, rpn_intensity, seg_mask = \
            self.get_rpn_features(self.rcnn_training_feature_dir, sample_id)

        # load rois and gt_boxes3d for this sample
        roi_file = os.path.join(self.rcnn_training_roi_dir, '%06d.txt' % sample_id)
        roi_obj_list = kitti_utils.get_objects_from_label(roi_file)
        roi_boxes3d = kitti_utils.objs_to_boxes3d(roi_obj_list)
        # roi_scores is not used currently
        # roi_scores = kitti_utils.objs_to_scores(roi_obj_list)

        gt_obj_list = self.filtrate_objects(self.get_label(sample_id))
        gt_boxes3d = kitti_utils.objs_to_boxes3d(gt_obj_list)
        sample_info = OrderedDict()
        sample_info["sample_id"] = sample_id
        sample_info['rpn_xyz'] = rpn_xyz
        sample_info['rpn_features'] = rpn_features
        sample_info['rpn_intensity'] = rpn_intensity
        sample_info['seg_mask'] = seg_mask
        sample_info['roi_boxes3d'] = roi_boxes3d
        sample_info['pts_depth'] = np.linalg.norm(rpn_xyz, ord=2, axis=1)
        sample_info['gt_boxes3d'] = gt_boxes3d

        return sample_info

    def sample_bg_inds(self, hard_bg_inds, easy_bg_inds, bg_rois_per_this_image):
        if hard_bg_inds.size > 0 and easy_bg_inds.size > 0:
            hard_bg_rois_num = int(bg_rois_per_this_image * cfg.RCNN.HARD_BG_RATIO)
            easy_bg_rois_num = bg_rois_per_this_image - hard_bg_rois_num

            # sampling hard bg
            rand_num = np.floor(np.random.rand(hard_bg_rois_num) * hard_bg_inds.size).astype(np.int32)
            hard_bg_inds = hard_bg_inds[rand_num]
            # sampling easy bg
            rand_num = np.floor(np.random.rand(easy_bg_rois_num) * easy_bg_inds.size).astype(np.int32)
            easy_bg_inds = easy_bg_inds[rand_num]

            bg_inds = np.concatenate([hard_bg_inds, easy_bg_inds], axis=0)
        elif hard_bg_inds.size > 0 and easy_bg_inds.size == 0:
            hard_bg_rois_num = bg_rois_per_this_image
            # sampling hard bg
            rand_num = np.floor(np.random.rand(hard_bg_rois_num) * hard_bg_inds.size).astype(np.int32)
            bg_inds = hard_bg_inds[rand_num]
        elif hard_bg_inds.size == 0 and easy_bg_inds.size > 0:
            easy_bg_rois_num = bg_rois_per_this_image
            # sampling easy bg
            rand_num = np.floor(np.random.rand(easy_bg_rois_num) * easy_bg_inds.size).astype(np.int32)
            bg_inds = easy_bg_inds[rand_num]
        else:
            raise NotImplementedError

        return bg_inds

    def aug_roi_by_noise_batch(self, roi_boxes3d, gt_boxes3d, aug_times=10):
        """
        :param roi_boxes3d: (N, 7)
        :param gt_boxes3d: (N, 7)
        :return:
        """
        iou_of_rois = np.zeros(roi_boxes3d.shape[0], dtype=np.float32)
        for k in range(roi_boxes3d.__len__()):
            temp_iou = cnt = 0
            roi_box3d = roi_boxes3d[k]
            gt_box3d = gt_boxes3d[k]
            pos_thresh = min(cfg.RCNN.REG_FG_THRESH, cfg.RCNN.CLS_FG_THRESH)
            gt_corners = kitti_utils.boxes3d_to_corners3d(gt_box3d.reshape(1, 7), True)
            aug_box3d = roi_box3d
            while temp_iou < pos_thresh and cnt < aug_times:
                if np.random.rand() < 0.2:
                    aug_box3d = roi_box3d  # p=0.2 to keep the original roi box
                else:
                    aug_box3d = self.random_aug_box3d(roi_box3d)
                aug_corners = kitti_utils.boxes3d_to_corners3d(aug_box3d.reshape(1, 7), True)
                iou3d = kitti_utils.get_iou3d(aug_corners, gt_corners)
                temp_iou = iou3d[0][0]
                cnt += 1
            roi_boxes3d[k] = aug_box3d
            iou_of_rois[k] = temp_iou
        return roi_boxes3d, iou_of_rois

    @staticmethod
    def canonical_transform_batch(pts_input, roi_boxes3d, gt_boxes3d):
        """
        :param pts_input: (N, npoints, 3 + C)
        :param roi_boxes3d: (N, 7)
        :param gt_boxes3d: (N, 7)
        :return:
        """
        roi_ry = roi_boxes3d[:, 6] % (2 * np.pi)  # 0 ~ 2pi
        roi_center = roi_boxes3d[:, 0:3]
        # shift to center
        pts_input[:, :, [0, 1, 2]] = pts_input[:, :, [0, 1, 2]] - roi_center.reshape(-1, 1, 3)
        gt_boxes3d_ct = np.copy(gt_boxes3d)
        gt_boxes3d_ct[:, 0:3] = gt_boxes3d_ct[:, 0:3] - roi_center
        # rotate to the direction of head
        gt_boxes3d_ct = kitti_utils.rotate_pc_along_y_np(
            gt_boxes3d_ct.reshape(-1, 1, 7),
            roi_ry,
        )
        # TODO: check here
        gt_boxes3d_ct = gt_boxes3d_ct.reshape(-1,7)
        gt_boxes3d_ct[:, 6] = gt_boxes3d_ct[:, 6] - roi_ry
        pts_input = kitti_utils.rotate_pc_along_y_np(
            pts_input, 
            roi_ry
        )
        return pts_input, gt_boxes3d_ct

    def get_rcnn_training_sample_batch(self, index):
        sample_id = int(self.sample_id_list[index])
        rpn_xyz, rpn_features, rpn_intensity, seg_mask = \
            self.get_rpn_features(self.rcnn_training_feature_dir, sample_id)

        # load rois and gt_boxes3d for this sample
        roi_file = os.path.join(self.rcnn_training_roi_dir, '%06d.txt' % sample_id)
        roi_obj_list = kitti_utils.get_objects_from_label(roi_file)
        roi_boxes3d = kitti_utils.objs_to_boxes3d(roi_obj_list)
        # roi_scores = kitti_utils.objs_to_scores(roi_obj_list)

        gt_obj_list = self.filtrate_objects(self.get_label(sample_id))
        gt_boxes3d = kitti_utils.objs_to_boxes3d(gt_obj_list)

        # calculate original iou
        iou3d = kitti_utils.get_iou3d(kitti_utils.boxes3d_to_corners3d(roi_boxes3d, True),
                                      kitti_utils.boxes3d_to_corners3d(gt_boxes3d, True))
        max_overlaps, gt_assignment = iou3d.max(axis=1), iou3d.argmax(axis=1)
        max_iou_of_gt, roi_assignment = iou3d.max(axis=0), iou3d.argmax(axis=0)
        roi_assignment = roi_assignment[max_iou_of_gt > 0].reshape(-1)

        # sample fg, easy_bg, hard_bg
        fg_rois_per_image = int(np.round(cfg.RCNN.FG_RATIO * cfg.RCNN.ROI_PER_IMAGE))
        fg_thresh = min(cfg.RCNN.REG_FG_THRESH, cfg.RCNN.CLS_FG_THRESH)
        fg_inds = np.nonzero(max_overlaps >= fg_thresh)[0]
        fg_inds = np.concatenate((fg_inds, roi_assignment), axis=0)  # consider the roi which has max_overlaps with gt as fg

        easy_bg_inds = np.nonzero((max_overlaps < cfg.RCNN.CLS_BG_THRESH_LO))[0]
        hard_bg_inds = np.nonzero((max_overlaps < cfg.RCNN.CLS_BG_THRESH) &
                                  (max_overlaps >= cfg.RCNN.CLS_BG_THRESH_LO))[0]

        fg_num_rois = fg_inds.size
        bg_num_rois = hard_bg_inds.size + easy_bg_inds.size

        if fg_num_rois > 0 and bg_num_rois > 0:
            # sampling fg
            fg_rois_per_this_image = min(fg_rois_per_image, fg_num_rois)
            rand_num = np.random.permutation(fg_num_rois)
            fg_inds = fg_inds[rand_num[:fg_rois_per_this_image]]

            # sampling bg
            bg_rois_per_this_image = cfg.RCNN.ROI_PER_IMAGE  - fg_rois_per_this_image
            bg_inds = self.sample_bg_inds(hard_bg_inds, easy_bg_inds, bg_rois_per_this_image)

        elif fg_num_rois > 0 and bg_num_rois == 0:
            # sampling fg
            rand_num = np.floor(np.random.rand(cfg.RCNN.ROI_PER_IMAGE ) * fg_num_rois)
            # rand_num = torch.from_numpy(rand_num).type_as(gt_boxes3d).long()
            fg_inds = fg_inds[rand_num]
            fg_rois_per_this_image = cfg.RCNN.ROI_PER_IMAGE
            bg_rois_per_this_image = 0
        elif bg_num_rois > 0 and fg_num_rois == 0:
            # sampling bg
            bg_rois_per_this_image = cfg.RCNN.ROI_PER_IMAGE
            bg_inds = self.sample_bg_inds(hard_bg_inds, easy_bg_inds, bg_rois_per_this_image)
            fg_rois_per_this_image = 0
        else:
            import pdb
            pdb.set_trace()
            raise NotImplementedError

        # augment the rois by noise
        roi_list, roi_iou_list, roi_gt_list = [], [], []
        if fg_rois_per_this_image > 0:
            fg_rois_src = roi_boxes3d[fg_inds].copy()
            gt_of_fg_rois = gt_boxes3d[gt_assignment[fg_inds]]
            fg_rois, fg_iou3d = self.aug_roi_by_noise_batch(fg_rois_src, gt_of_fg_rois, aug_times=10)
            roi_list.append(fg_rois)
            roi_iou_list.append(fg_iou3d)
            roi_gt_list.append(gt_of_fg_rois)

        if bg_rois_per_this_image > 0:
            bg_rois_src = roi_boxes3d[bg_inds].copy()
            gt_of_bg_rois = gt_boxes3d[gt_assignment[bg_inds]]
            bg_rois, bg_iou3d = self.aug_roi_by_noise_batch(bg_rois_src, gt_of_bg_rois, aug_times=1)
            roi_list.append(bg_rois)
            roi_iou_list.append(bg_iou3d)
            roi_gt_list.append(gt_of_bg_rois)

        rois = np.concatenate(roi_list, axis=0)
        iou_of_rois = np.concatenate(roi_iou_list, axis=0)
        gt_of_rois = np.concatenate(roi_gt_list, axis=0)

        # collect extra features for point cloud pooling
        if cfg.RCNN.USE_INTENSITY:
            pts_extra_input_list = [rpn_intensity.reshape(-1, 1), seg_mask.reshape(-1, 1)]
        else:
            pts_extra_input_list = [seg_mask.reshape(-1, 1)]

        if cfg.RCNN.USE_DEPTH:
            pts_depth = (np.linalg.norm(rpn_xyz, ord=2, axis=1) / 70.0) - 0.5
            pts_extra_input_list.append(pts_depth.reshape(-1, 1))
        pts_extra_input = np.concatenate(pts_extra_input_list, axis=1)

        # pts, pts_feature, boxes3d, pool_extra_width, sampled_pt_num
        pts_input, pts_features, pts_empty_flag = roipool3d_utils.roipool3d_cpu(
            rpn_xyz, rpn_features, rois, pts_extra_input,
            cfg.RCNN.POOL_EXTRA_WIDTH,
            sampled_pt_num=cfg.RCNN.NUM_POINTS,
            #canonical_transform=False
        )

        # data augmentation
        if cfg.AUG_DATA and self.mode == 'TRAIN':
            for k in range(rois.__len__()):
                aug_pts = pts_input[k, :, 0:3].copy()
                aug_gt_box3d = gt_of_rois[k].copy()
                aug_roi_box3d = rois[k].copy()

                # calculate alpha by ry
                temp_boxes3d = np.concatenate([aug_roi_box3d.reshape(1, 7), aug_gt_box3d.reshape(1, 7)], axis=0)
                temp_x, temp_z, temp_ry = temp_boxes3d[:, 0], temp_boxes3d[:, 2], temp_boxes3d[:, 6]
                temp_beta = np.arctan2(temp_z, temp_x).astype(np.float64)
                temp_alpha = -np.sign(temp_beta) * np.pi / 2 + temp_beta + temp_ry

                # data augmentation
                aug_pts, aug_boxes3d, aug_method = self.data_augmentation(aug_pts, temp_boxes3d, temp_alpha,
                                                                          mustaug=True, stage=2)

                # assign to original data
                pts_input[k, :, 0:3] = aug_pts
                rois[k] = aug_boxes3d[0]
                gt_of_rois[k] = aug_boxes3d[1]

        valid_mask = (pts_empty_flag == 0).astype(np.int32)
        # regression valid mask
        reg_valid_mask = (iou_of_rois > cfg.RCNN.REG_FG_THRESH).astype(np.int32) & valid_mask

        # classification label
        cls_label = (iou_of_rois > cfg.RCNN.CLS_FG_THRESH).astype(np.int32)
        invalid_mask = (iou_of_rois > cfg.RCNN.CLS_BG_THRESH) & (iou_of_rois < cfg.RCNN.CLS_FG_THRESH)
        cls_label[invalid_mask] = -1
        cls_label[valid_mask == 0] = -1

        # canonical transform and sampling
        pts_input_ct, gt_boxes3d_ct = self.canonical_transform_batch(pts_input, rois, gt_of_rois)

        pts_input_ = np.concatenate((pts_input_ct, pts_features), axis=-1)
        sample_info = OrderedDict()

        sample_info['sample_id'] = sample_id
        sample_info['pts_input'] = pts_input_
        sample_info['pts_feature'] = pts_features
        sample_info['roi_boxes3d'] = rois
        sample_info['cls_label'] = cls_label
        sample_info['reg_valid_mask'] = reg_valid_mask
        sample_info['gt_boxes3d_ct'] = gt_boxes3d_ct
        sample_info['gt_of_rois'] = gt_of_rois
        return sample_info

    @staticmethod
    def random_aug_box3d(box3d):
        """
        :param box3d: (7) [x, y, z, h, w, l, ry]
        random shift, scale, orientation
        """
        if cfg.RCNN.REG_AUG_METHOD == 'single':
            pos_shift = (np.random.rand(3) - 0.5)  # [-0.5 ~ 0.5]
            hwl_scale = (np.random.rand(3) - 0.5) / (0.5 / 0.15) + 1.0  #
            angle_rot = (np.random.rand(1) - 0.5) / (0.5 / (np.pi / 12))  # [-pi/12 ~ pi/12]

            aug_box3d = np.concatenate([box3d[0:3] + pos_shift, box3d[3:6] * hwl_scale,
                                        box3d[6:7] + angle_rot])
            return aug_box3d
        elif cfg.RCNN.REG_AUG_METHOD == 'multiple':
            # pos_range, hwl_range, angle_range, mean_iou
            range_config = [[0.2, 0.1, np.pi / 12, 0.7],
                            [0.3, 0.15, np.pi / 12, 0.6],
                            [0.5, 0.15, np.pi / 9, 0.5],
                            [0.8, 0.15, np.pi / 6, 0.3],
                            [1.0, 0.15, np.pi / 3, 0.2]]
            idx = np.random.randint(len(range_config))

            pos_shift = ((np.random.rand(3) - 0.5) / 0.5) * range_config[idx][0]
            hwl_scale = ((np.random.rand(3) - 0.5) / 0.5) * range_config[idx][1] + 1.0
            angle_rot = ((np.random.rand(1) - 0.5) / 0.5) * range_config[idx][2]

            aug_box3d = np.concatenate([box3d[0:3] + pos_shift, box3d[3:6] * hwl_scale, box3d[6:7] + angle_rot])
            return aug_box3d
        elif cfg.RCNN.REG_AUG_METHOD == 'normal':
            x_shift = np.random.normal(loc=0, scale=0.3)
            y_shift = np.random.normal(loc=0, scale=0.2)
            z_shift = np.random.normal(loc=0, scale=0.3)
            h_shift = np.random.normal(loc=0, scale=0.25)
            w_shift = np.random.normal(loc=0, scale=0.15)
            l_shift = np.random.normal(loc=0, scale=0.5)
            ry_shift = ((np.random.rand() - 0.5) / 0.5) * np.pi / 12

            aug_box3d = np.array([box3d[0] + x_shift, box3d[1] + y_shift, box3d[2] + z_shift, box3d[3] + h_shift,
                                  box3d[4] + w_shift, box3d[5] + l_shift, box3d[6] + ry_shift])
            return aug_box3d
        else:
            raise NotImplementedError

    def get_proposal_from_file(self, index):
        sample_id = int(self.image_idx_list[index])
        proposal_file = os.path.join(self.rcnn_eval_roi_dir, '%06d.txt' % sample_id)
        roi_obj_list = kitti_utils.get_objects_from_label(proposal_file)

        rpn_xyz, rpn_features, rpn_intensity, seg_mask = self.get_rpn_features(self.rcnn_eval_feature_dir, sample_id)
        pts_rect, pts_rpn_features, pts_intensity = rpn_xyz, rpn_features, rpn_intensity

        roi_box3d_list, roi_scores = [], []
        for obj in roi_obj_list:
            box3d = np.array([obj.pos[0], obj.pos[1], obj.pos[2], obj.h, obj.w, obj.l, obj.ry], dtype=np.float32)
            roi_box3d_list.append(box3d.reshape(1, 7))
            roi_scores.append(obj.score)

        roi_boxes3d = np.concatenate(roi_box3d_list, axis=0)  # (N, 7)
        roi_scores = np.array(roi_scores, dtype=np.float32)  # (N)

        if cfg.RCNN.ROI_SAMPLE_JIT:
            sample_dict = {'sample_id': sample_id,
                           'rpn_xyz': rpn_xyz,
                           'rpn_features': rpn_features,
                           'seg_mask': seg_mask,
                           'roi_boxes3d': roi_boxes3d,
                           'roi_scores': roi_scores,
                           'pts_depth': np.linalg.norm(rpn_xyz, ord=2, axis=1)}

            if self.mode != 'TEST':
                gt_obj_list = self.filtrate_objects(self.get_label(sample_id))
                gt_boxes3d = kitti_utils.objs_to_boxes3d(gt_obj_list)

                roi_corners = kitti_utils.boxes3d_to_corners3d(roi_boxes3d,True)
                gt_corners = kitti_utils.boxes3d_to_corners3d(gt_boxes3d,True)
                iou3d = kitti_utils.get_iou3d(roi_corners, gt_corners)
                if gt_boxes3d.shape[0] > 0:
                    gt_iou = iou3d.max(axis=1)
                else:
                    gt_iou = np.zeros(roi_boxes3d.shape[0]).astype(np.float32)

                sample_dict['gt_boxes3d'] = gt_boxes3d
                sample_dict['gt_iou'] = gt_iou
            return sample_dict

        if cfg.RCNN.USE_INTENSITY:
            pts_extra_input_list = [pts_intensity.reshape(-1, 1), seg_mask.reshape(-1, 1)]
        else:
            pts_extra_input_list = [seg_mask.reshape(-1, 1)]

        if cfg.RCNN.USE_DEPTH:
            cur_depth = np.linalg.norm(pts_rect, axis=1, ord=2)
            cur_depth_norm = (cur_depth / 70.0) - 0.5
            pts_extra_input_list.append(cur_depth_norm.reshape(-1, 1))

        pts_extra_input = np.concatenate(pts_extra_input_list, axis=1)
        pts_input, pts_features, _ = roipool3d_utils.roipool3d_cpu(
            pts_rect, pts_rpn_features, roi_boxes3d, pts_extra_input, 
            cfg.RCNN.POOL_EXTRA_WIDTH, sampled_pt_num=cfg.RCNN.NUM_POINTS,
            canonical_transform=True
        )
        pts_input = np.concatenate((pts_input, pts_features), axis=-1)
        
        sample_dict = OrderedDict()
        sample_dict['sample_id'] = sample_id 
        sample_dict['pts_input'] = pts_input 
        sample_dict['pts_feature'] = pts_features 
        sample_dict['roi_boxes3d'] = roi_boxes3d 
        sample_dict['roi_scores'] = roi_scores 
        #sample_dict['roi_size'] = roi_boxes3d[:, 3:6]

        if self.mode == 'TEST':
            return sample_dict

        gt_obj_list = self.filtrate_objects(self.get_label(sample_id))
        gt_boxes3d = np.zeros((gt_obj_list.__len__(), 7), dtype=np.float32)

        for k, obj in enumerate(gt_obj_list):
            gt_boxes3d[k, 0:3], gt_boxes3d[k, 3], gt_boxes3d[k, 4], gt_boxes3d[k, 5], gt_boxes3d[k, 6] \
                = obj.pos, obj.h, obj.w, obj.l, obj.ry

        if gt_boxes3d.__len__() == 0:
            gt_iou = np.zeros((roi_boxes3d.shape[0]), dtype=np.float32)
        else:
            roi_corners = kitti_utils.boxes3d_to_corners3d(roi_boxes3d,True)
            gt_corners = kitti_utils.boxes3d_to_corners3d(gt_boxes3d,True)
            iou3d = kitti_utils.get_iou3d(roi_corners, gt_corners)
            gt_iou = iou3d.max(axis=1)

        sample_dict['gt_iou'] = gt_iou
        sample_dict['gt_boxes3d'] = gt_boxes3d
        
        return sample_dict

    def __len__(self):
        if cfg.RPN.ENABLED:
            return len(self.sample_id_list)
        elif cfg.RCNN.ENABLED:
            if self.mode == 'TRAIN':
                return len(self.sample_id_list)
            else:
                return len(self.image_idx_list)
        else:
            raise NotImplementedError

    def __getitem__(self, index):
        if cfg.RPN.ENABLED:
            return self.get_rpn_sample(index)
        elif cfg.RCNN.ENABLED:
            if self.mode == 'TRAIN':
                if cfg.RCNN.ROI_SAMPLE_JIT:
                    return self.get_rcnn_sample_jit(index)
                else:
                    return self.get_rcnn_training_sample_batch(index)
            else:
                return self.get_proposal_from_file(index)
        else:
            raise NotImplementedError

    def padding_batch(self, batch_data, batch_size):
        max_roi = 0
        max_gt = 0 
        
        for k in range(batch_size):
            # roi_boxes3d
            max_roi = max(max_roi, batch_data[k][3].shape[0])
            # gt_boxes3d
            max_gt = max(max_gt, batch_data[k][-1].shape[0])
        batch_roi_boxes3d = np.zeros((batch_size, max_roi, 7))
        batch_gt_boxes3d = np.zeros((batch_size, max_gt, 7), dtype=np.float32)
        
        for i, data in enumerate(batch_data):
            roi_num = data[3].shape[0]
            gt_num = data[-1].shape[0]
            batch_roi_boxes3d[i,:roi_num,:] = data[3]
            batch_gt_boxes3d[i,:gt_num,:] = data[-1]

        new_batch = []
        for i, data in enumerate(batch_data):
            new_batch.append(data[:3])
            # roi_boxes3d
            new_batch[i].append(batch_roi_boxes3d[i])
            # ... 
            new_batch[i].extend(data[4:7])
            # gt_boxes3d
            new_batch[i].append(batch_gt_boxes3d[i])
        return new_batch

    def padding_batch_eval(self, batch_data, batch_size):
        max_pts = 0 
        max_feats = 0
        max_roi = 0
        max_score = 0
        max_iou = 0
        max_gt = 0
        
        for k in range(batch_size):
            # pts_input
            max_pts = max(max_pts, batch_data[k][1].shape[0])
            # pts_feature
            max_feats = max(max_feats, batch_data[k][2].shape[0])
            # roi_boxes3d
            max_roi = max(max_roi, batch_data[k][3].shape[0])
            # gt_iou 
            max_iou = max(max_iou, batch_data[k][-2].shape[0])
            # gt_boxes3d
            max_gt = max(max_gt, batch_data[k][-1].shape[0])
        batch_pts_input = np.zeros((batch_size, max_pts, 512, 133), dtype=np.float32)
        batch_pts_feat = np.zeros((batch_size, max_feats, 512, 128), dtype=np.float32)
        batch_roi_boxes3d = np.zeros((batch_size, max_roi, 7), dtype=np.float32)
        batch_gt_iou = np.zeros((batch_size, max_iou), dtype=np.float32)
        batch_gt_boxes3d = np.zeros((batch_size, max_gt, 7), dtype=np.float32)
        
        for i, data in enumerate(batch_data):
            # num
            pts_num = data[1].shape[0]
            pts_feat_num = data[2].shape[0]
            roi_num = data[3].shape[0]
            iou_num = data[-2].shape[0]
            gt_num = data[-1].shape[0]
            # data
            batch_pts_input[i, :pts_num, :, :] = data[1]
            batch_pts_feat[i, :pts_feat_num, :, :] = data[2]
            batch_roi_boxes3d[i,:roi_num,:] = data[3]
            batch_gt_iou[i,:iou_num] = data[-2] 
            batch_gt_boxes3d[i,:gt_num,:] = data[-1]
            
        new_batch = []
        for i, data in enumerate(batch_data):
            new_batch.append(data[:1])
            new_batch[i].append(batch_pts_input[i])
            new_batch[i].append(batch_pts_feat[i])
            new_batch[i].append(batch_roi_boxes3d[i])
            new_batch[i].append(data[4])
            new_batch[i].append(batch_gt_iou[i])
            new_batch[i].append(batch_gt_boxes3d[i])
        return new_batch

    def get_reader(self, batch_size, fields, drop_last=False):
        def reader():
            batch_out = []
            idxs = np.arange(self.__len__())
            if self.mode == 'TRAIN':
                np.random.shuffle(idxs)
            for idx in idxs:
                sample_all = self.__getitem__(idx)
                sample = [sample_all[f] for f in fields]
                if has_empty(sample):
                    logger.info("sample field: %d has empty field"%len(sample))
                    continue
                batch_out.append(sample)
                if len(batch_out) >= batch_size:
                    if cfg.RPN.ENABLED:
                        yield batch_out
                    else:
                        if self.mode == 'TRAIN':
                            yield self.padding_batch(batch_out, batch_size)
                        elif self.mode == 'EVAL':
                            # batch_size can should be 1 in rcnn_offline eval currently
                            # if batch_size > 1, batch should be padded as follow
                            # yield self.padding_batch_eval(batch_out, batch_size)
                            yield batch_out
                        else:
                            logger.error("not only support train/eval padding")
                    batch_out = []
            if not drop_last:
                if len(batch_out) > 0:
                    yield batch_out
        return reader

    def get_multiprocess_reader(self, batch_size, fields, proc_num=8, max_queue_len=128, drop_last=False):
        def read_to_queue(idxs, queue):
            for idx in idxs:
                sample_all = self.__getitem__(idx)
                sample = [sample_all[f] for f in fields]
                queue.put(sample)
            queue.put(None)

        def reader():
            sample_num = self.__len__()
            idxs = np.arange(self.__len__())
            if self.mode == 'TRAIN':
                np.random.shuffle(idxs)

            proc_idxs = []
            proc_sample_num = int(sample_num / proc_num)
            start_idx = 0
            for i in range(proc_num - 1):
                proc_idxs.append(idxs[start_idx:start_idx + proc_sample_num])
                start_idx += proc_sample_num
            proc_idxs.append(idxs[start_idx:])

            queue = multiprocessing.Queue(max_queue_len)
            p_list = []
            for i in range(proc_num):
                p_list.append(multiprocessing.Process(
                    target=read_to_queue, args=(proc_idxs[i], queue,)))
                p_list[-1].start()

            finish_num = 0
            batch_out = []
            while finish_num < len(p_list):
                sample = queue.get()
                if sample is None:
                    finish_num += 1
                else:
                    batch_out.append(sample)
                    if len(batch_out) == batch_size:
                        yield batch_out
                        batch_out = []

            # join process
            for p in p_list:
                if p.is_alive():
                    p.join()

        return reader


def _term_reader(signum, frame):
    logger.info('pid {} terminated, terminate reader process '
                'group {}...'.format(os.getpid(), os.getpgrp()))
    os.killpg(os.getpgid(os.getpid()), signal.SIGKILL)

signal.signal(signal.SIGINT, _term_reader)

