"""
Generate GT database
This code is based on https://github.com/sshaoshuai/PointRCNN/blob/master/tools/generate_gt_database.py
"""

import os
import numpy as np
import pickle

from data.kitti_dataset import KittiDataset
import pts_utils 
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='./data')
parser.add_argument('--save_dir', type=str, default='./data/gt_database')
parser.add_argument('--class_name', type=str, default='Car')
parser.add_argument('--split', type=str, default='train')
args = parser.parse_args()


class GTDatabaseGenerator(KittiDataset):
    def __init__(self, root_dir, split='train', classes=args.class_name):
        super(GTDatabaseGenerator, self).__init__(root_dir, split=split)
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

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, item):
        raise NotImplementedError

    def filtrate_objects(self, obj_list):
        valid_obj_list = []
        for obj in obj_list:
            if obj.cls_type not in self.classes:
                continue
            if obj.level_str not in ['Easy', 'Moderate', 'Hard']:
                continue
            valid_obj_list.append(obj)

        return valid_obj_list

    def generate_gt_database(self):
        gt_database = []
        for idx, sample_id in enumerate(self.image_idx_list):
            sample_id = int(sample_id)
            print('process gt sample (id=%06d)' % sample_id)

            pts_lidar = self.get_lidar(sample_id)
            calib = self.get_calib(sample_id)
            pts_rect = calib.lidar_to_rect(pts_lidar[:, 0:3])
            pts_intensity = pts_lidar[:, 3]

            obj_list = self.filtrate_objects(self.get_label(sample_id))

            gt_boxes3d = np.zeros((obj_list.__len__(), 7), dtype=np.float32)
            for k, obj in enumerate(obj_list):
                gt_boxes3d[k, 0:3], gt_boxes3d[k, 3], gt_boxes3d[k, 4], gt_boxes3d[k, 5], gt_boxes3d[k, 6] \
                    = obj.pos, obj.h, obj.w, obj.l, obj.ry

            if gt_boxes3d.__len__() == 0:
                print('No gt object')
                continue

            boxes_pts_mask_list = pts_utils.pts_in_boxes3d(pts_rect, gt_boxes3d)

            for k in range(boxes_pts_mask_list.shape[0]):
                pt_mask_flag = (boxes_pts_mask_list[k] == 1)
                cur_pts = pts_rect[pt_mask_flag].astype(np.float32)
                cur_pts_intensity = pts_intensity[pt_mask_flag].astype(np.float32)
                sample_dict = {'sample_id': sample_id,
                               'cls_type': obj_list[k].cls_type,
                               'gt_box3d': gt_boxes3d[k],
                               'points': cur_pts,
                               'intensity': cur_pts_intensity,
                               'obj': obj_list[k]}
                gt_database.append(sample_dict)

        save_file_name = os.path.join(args.save_dir, '%s_gt_database_3level_%s.pkl' % (args.split, self.classes[-1]))
        with open(save_file_name, 'wb') as f:
            pickle.dump(gt_database, f)

        self.gt_database = gt_database
        print('Save refine training sample info file to %s' % save_file_name)


if __name__ == '__main__':
    dataset = GTDatabaseGenerator(root_dir=args.data_dir, split=args.split)
    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)

    dataset.generate_gt_database()

