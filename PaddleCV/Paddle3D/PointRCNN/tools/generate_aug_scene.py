"""
Generate GT database
This code is based on https://github.com/sshaoshuai/PointRCNN/blob/master/tools/generate_aug_scene.py
"""

import os
import numpy as np
import pickle

import pts_utils
import utils.cyops.kitti_utils as kitti_utils 
from utils.box_utils import boxes_iou3d
from utils import calibration as calib
from data.kitti_dataset import KittiDataset
import argparse

np.random.seed(1024)

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='generator')
parser.add_argument('--class_name', type=str, default='Car')
parser.add_argument('--data_dir', type=str, default='./data')
parser.add_argument('--save_dir', type=str, default='./data/KITTI/aug_scene/training')
parser.add_argument('--split', type=str, default='train')
parser.add_argument('--gt_database_dir', type=str, default='./data/gt_database/train_gt_database_3level_Car.pkl')
parser.add_argument('--include_similar', action='store_true', default=False)
parser.add_argument('--aug_times', type=int, default=4)
args = parser.parse_args()

PC_REDUCE_BY_RANGE = True
if args.class_name == 'Car':
    PC_AREA_SCOPE = np.array([[-40, 40], [-1, 3], [0, 70.4]])  # x, y, z scope in rect camera coords
else:
    PC_AREA_SCOPE = np.array([[-30, 30], [-1, 3], [0, 50]])


def log_print(info, fp=None):
    print(info)
    if fp is not None:
        # print(info, file=fp)
        fp.write(info+"\n")


def save_kitti_format(calib, bbox3d, obj_list, img_shape, save_fp):
    corners3d = kitti_utils.boxes3d_to_corners3d(bbox3d)
    img_boxes, _ = calib.corners3d_to_img_boxes(corners3d)

    img_boxes[:, 0] = np.clip(img_boxes[:, 0], 0, img_shape[1] - 1)
    img_boxes[:, 1] = np.clip(img_boxes[:, 1], 0, img_shape[0] - 1)
    img_boxes[:, 2] = np.clip(img_boxes[:, 2], 0, img_shape[1] - 1)
    img_boxes[:, 3] = np.clip(img_boxes[:, 3], 0, img_shape[0] - 1)

    # Discard boxes that are larger than 80% of the image width OR height
    img_boxes_w = img_boxes[:, 2] - img_boxes[:, 0]
    img_boxes_h = img_boxes[:, 3] - img_boxes[:, 1]
    box_valid_mask = np.logical_and(img_boxes_w < img_shape[1] * 0.8, img_boxes_h < img_shape[0] * 0.8)

    for k in range(bbox3d.shape[0]):
        if box_valid_mask[k] == 0:
            continue
        x, z, ry = bbox3d[k, 0], bbox3d[k, 2], bbox3d[k, 6]
        beta = np.arctan2(z, x)
        alpha = -np.sign(beta) * np.pi / 2 + beta + ry

        save_fp.write('%s %.2f %d %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n' %
              (args.class_name, obj_list[k].trucation, int(obj_list[k].occlusion), alpha, img_boxes[k, 0], img_boxes[k, 1],
               img_boxes[k, 2], img_boxes[k, 3],
               bbox3d[k, 3], bbox3d[k, 4], bbox3d[k, 5], bbox3d[k, 0], bbox3d[k, 1], bbox3d[k, 2],
               bbox3d[k, 6]))


class AugSceneGenerator(KittiDataset):
    def __init__(self, root_dir, gt_database=None, split='train', classes=args.class_name):
        super(AugSceneGenerator, self).__init__(root_dir, split=split)
        self.gt_database = None
        if classes == 'Car':
            self.classes = ('Background', 'Car')
        elif classes == 'People':
            self.classes = ('Background', 'Pedestrian', 'Cyclist')
        elif classes == 'Pedestrian':
            self.classes = ('Background', 'Pedestrian')
        elif classes == 'Cyclist':
            self.classes = ('Background', 'Cyclist')
        else:
            assert False, "Invalid classes: %s" % classes

        self.gt_database = gt_database

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, item):
        raise NotImplementedError

    def filtrate_dc_objects(self, obj_list):
        valid_obj_list = []
        for obj in obj_list:
            if obj.cls_type in ['DontCare']:
                continue
            valid_obj_list.append(obj)

        return valid_obj_list

    def filtrate_objects(self, obj_list):
        valid_obj_list = []
        type_whitelist = self.classes
        if args.include_similar:
            type_whitelist = list(self.classes)
            if 'Car' in self.classes:
                type_whitelist.append('Van')
            if 'Pedestrian' in self.classes or 'Cyclist' in self.classes:
                type_whitelist.append('Person_sitting')

        for obj in obj_list:
            if obj.cls_type in type_whitelist:
                valid_obj_list.append(obj)
        return valid_obj_list

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

        if PC_REDUCE_BY_RANGE:
            x_range, y_range, z_range = PC_AREA_SCOPE
            pts_x, pts_y, pts_z = pts_rect[:, 0], pts_rect[:, 1], pts_rect[:, 2]
            range_flag = (pts_x >= x_range[0]) & (pts_x <= x_range[1]) \
                         & (pts_y >= y_range[0]) & (pts_y <= y_range[1]) \
                         & (pts_z >= z_range[0]) & (pts_z <= z_range[1])
            pts_valid_flag = pts_valid_flag & range_flag
        return pts_valid_flag

    @staticmethod
    def check_pc_range(xyz):
        """
        :param xyz: [x, y, z]
        :return:
        """
        x_range, y_range, z_range = PC_AREA_SCOPE
        if (x_range[0] <= xyz[0] <= x_range[1]) and (y_range[0] <= xyz[1] <= y_range[1]) and \
                (z_range[0] <= xyz[2] <= z_range[1]):
            return True
        return False

    def aug_one_scene(self, sample_id, pts_rect, pts_intensity, all_gt_boxes3d):
        """
        :param pts_rect: (N, 3)
        :param gt_boxes3d: (M1, 7)
        :param all_gt_boxex3d: (M2, 7)
        :return:
        """
        assert self.gt_database is not None
        extra_gt_num = np.random.randint(10, 15)
        try_times = 50
        cnt = 0
        cur_gt_boxes3d = all_gt_boxes3d.copy()
        cur_gt_boxes3d[:, 4] += 0.5
        cur_gt_boxes3d[:, 5] += 0.5  # enlarge new added box to avoid too nearby boxes

        extra_gt_obj_list = []
        extra_gt_boxes3d_list = []
        new_pts_list, new_pts_intensity_list = [], []
        src_pts_flag = np.ones(pts_rect.shape[0], dtype=np.int32)

        road_plane = self.get_road_plane(sample_id)
        a, b, c, d = road_plane

        while try_times > 0:
            try_times -= 1

            rand_idx = np.random.randint(0, self.gt_database.__len__() - 1)

            new_gt_dict = self.gt_database[rand_idx]
            new_gt_box3d = new_gt_dict['gt_box3d'].copy()
            new_gt_points = new_gt_dict['points'].copy()
            new_gt_intensity = new_gt_dict['intensity'].copy()
            new_gt_obj = new_gt_dict['obj']
            center = new_gt_box3d[0:3]
            if PC_REDUCE_BY_RANGE and (self.check_pc_range(center) is False):
                continue
            if cnt > extra_gt_num:
                break
            if new_gt_points.__len__() < 5:  # too few points
                continue

            # put it on the road plane
            cur_height = (-d - a * center[0] - c * center[2]) / b
            move_height = new_gt_box3d[1] - cur_height
            new_gt_box3d[1] -= move_height
            new_gt_points[:, 1] -= move_height

            cnt += 1

            iou3d = boxes_iou3d(new_gt_box3d.reshape(1, 7), cur_gt_boxes3d)

            valid_flag = iou3d.max() < 1e-8
            if not valid_flag:
                continue

            enlarged_box3d = new_gt_box3d.copy()
            enlarged_box3d[3] += 2  # remove the points above and below the object
            boxes_pts_mask_list = pts_utils.pts_in_boxes3d(pts_rect, enlarged_box3d.reshape(1, 7))
            pt_mask_flag = (boxes_pts_mask_list[0] == 1)
            src_pts_flag[pt_mask_flag] = 0  # remove the original points which are inside the new box

            new_pts_list.append(new_gt_points)
            new_pts_intensity_list.append(new_gt_intensity)
            enlarged_box3d = new_gt_box3d.copy()
            enlarged_box3d[4] += 0.5
            enlarged_box3d[5] += 0.5  # enlarge new added box to avoid too nearby boxes
            cur_gt_boxes3d = np.concatenate((cur_gt_boxes3d, enlarged_box3d.reshape(1, 7)), axis=0)
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

    def aug_one_epoch_scene(self, base_id, data_save_dir, label_save_dir, split_list, log_fp=None):
        for idx, sample_id in enumerate(self.image_idx_list):
            sample_id = int(sample_id)
            print('process gt sample (%s, id=%06d)' % (args.split, sample_id))

            pts_lidar = self.get_lidar(sample_id)
            calib = self.get_calib(sample_id)
            pts_rect = calib.lidar_to_rect(pts_lidar[:, 0:3])
            pts_img, pts_rect_depth = calib.rect_to_img(pts_rect)
            img_shape = self.get_image_shape(sample_id)

            pts_valid_flag = self.get_valid_flag(pts_rect, pts_img, pts_rect_depth, img_shape)
            pts_rect = pts_rect[pts_valid_flag][:, 0:3]
            pts_intensity = pts_lidar[pts_valid_flag][:, 3]

            # all labels for checking overlapping
            all_obj_list = self.filtrate_dc_objects(self.get_label(sample_id))
            all_gt_boxes3d = np.zeros((all_obj_list.__len__(), 7), dtype=np.float32)
            for k, obj in enumerate(all_obj_list):
                all_gt_boxes3d[k, 0:3], all_gt_boxes3d[k, 3], all_gt_boxes3d[k, 4], all_gt_boxes3d[k, 5], \
                all_gt_boxes3d[k, 6] = obj.pos, obj.h, obj.w, obj.l, obj.ry

            # gt_boxes3d of current label
            obj_list = self.filtrate_objects(self.get_label(sample_id))
            if args.class_name != 'Car' and obj_list.__len__() == 0:
                continue

            # augment one scene
            aug_flag, pts_rect, pts_intensity, extra_gt_boxes3d, extra_gt_obj_list = \
                self.aug_one_scene(sample_id, pts_rect, pts_intensity, all_gt_boxes3d)

            # save augment result to file
            pts_info = np.concatenate((pts_rect, pts_intensity.reshape(-1, 1)), axis=1)
            bin_file = os.path.join(data_save_dir, '%06d.bin' % (base_id + sample_id))
            pts_info.astype(np.float32).tofile(bin_file)

            # save filtered original gt_boxes3d
            label_save_file = os.path.join(label_save_dir, '%06d.txt' % (base_id + sample_id))
            with open(label_save_file, 'w') as f:
                for obj in obj_list:
                    f.write(obj.to_kitti_format() + '\n')

                if aug_flag:
                    # augment successfully
                    save_kitti_format(calib, extra_gt_boxes3d, extra_gt_obj_list, img_shape=img_shape, save_fp=f)
                else:
                    extra_gt_boxes3d = np.zeros((0, 7), dtype=np.float32)
            log_print('Save to file (new_obj: %s): %s' % (extra_gt_boxes3d.__len__(), label_save_file), fp=log_fp)
            split_list.append('%06d' % (base_id + sample_id))

    def generate_aug_scene(self, aug_times, log_fp=None):
        data_save_dir = os.path.join(args.save_dir, 'rectified_data')
        label_save_dir = os.path.join(args.save_dir, 'aug_label')
        if not os.path.isdir(data_save_dir):
            os.makedirs(data_save_dir)
        if not os.path.isdir(label_save_dir):
            os.makedirs(label_save_dir)

        split_file = os.path.join(args.save_dir, '%s_aug.txt' % args.split)
        split_list = self.image_idx_list[:]
        for epoch in range(aug_times):
            base_id = (epoch + 1) * 10000
            self.aug_one_epoch_scene(base_id, data_save_dir, label_save_dir, split_list, log_fp=log_fp)

        with open(split_file, 'w') as f:
            for idx, sample_id in enumerate(split_list):
                f.write(str(sample_id) + '\n')
        log_print('Save split file to %s' % split_file, fp=log_fp)
        target_dir = os.path.join(args.data_dir, 'KITTI/ImageSets/')
        os.system('cp %s %s' % (split_file, target_dir))
        log_print('Copy split file from %s to %s' % (split_file, target_dir), fp=log_fp)


if __name__ == '__main__':
    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)
    info_file = os.path.join(args.save_dir, 'log_info.txt')

    if args.mode == 'generator':
        log_fp = open(info_file, 'w')

        gt_database = pickle.load(open(args.gt_database_dir, 'rb'))
        log_print('Loading gt_database(%d) from %s' % (gt_database.__len__(), args.gt_database_dir), fp=log_fp)

        dataset = AugSceneGenerator(root_dir=args.data_dir, gt_database=gt_database, split=args.split)
        dataset.generate_aug_scene(aug_times=args.aug_times, log_fp=log_fp)

        log_fp.close()

    else:
        pass

