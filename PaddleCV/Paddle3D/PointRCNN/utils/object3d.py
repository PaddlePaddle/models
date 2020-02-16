"""
This code is borrow from https://github.com/sshaoshuai/PointRCNN/blob/master/lib/utils/object3d.py
"""
import numpy as np


def cls_type_to_id(cls_type):
    type_to_id = {'Car': 1, 'Pedestrian': 2, 'Cyclist': 3, 'Van': 4}
    if cls_type not in type_to_id.keys():
        return -1
    return type_to_id[cls_type]


def get_objects_from_label(label_file):
    with open(label_file, 'r') as f:
        lines = f.readlines()
    objects = [Object3d(line) for line in lines]
    return objects


class Object3d(object):
    def __init__(self, line):
        label = line.strip().split(' ')
        self.src = line
        self.cls_type = label[0]
        self.cls_id = cls_type_to_id(self.cls_type)
        self.trucation = float(label[1])
        self.occlusion = float(label[2])  # 0:fully visible 1:partly occluded 2:largely occluded 3:unknown
        self.alpha = float(label[3])
        self.box2d = np.array((float(label[4]), float(label[5]), float(label[6]), float(label[7])), dtype=np.float32)
        self.h = float(label[8])
        self.w = float(label[9])
        self.l = float(label[10])
        self.pos = np.array((float(label[11]), float(label[12]), float(label[13])), dtype=np.float32)
        self.dis_to_cam = np.linalg.norm(self.pos)
        self.ry = float(label[14])
        self.score = float(label[15]) if label.__len__() == 16 else -1.0
        self.level_str = None
        self.level = self.get_obj_level()

    def get_obj_level(self):
        height = float(self.box2d[3]) - float(self.box2d[1]) + 1

        if height >= 40 and self.trucation <= 0.15 and self.occlusion <= 0:
            self.level_str = 'Easy'
            return 1  # Easy
        elif height >= 25 and self.trucation <= 0.3 and self.occlusion <= 1:
            self.level_str = 'Moderate'
            return 2  # Moderate
        elif height >= 25 and self.trucation <= 0.5 and self.occlusion <= 2:
            self.level_str = 'Hard'
            return 3  # Hard
        else:
            self.level_str = 'UnKnown'
            return 4

    def generate_corners3d(self):
        """
        generate corners3d representation for this object
        :return corners_3d: (8, 3) corners of box3d in camera coord
        """
        l, h, w = self.l, self.h, self.w
        x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
        y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
        z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

        R = np.array([[np.cos(self.ry), 0, np.sin(self.ry)],
                      [0, 1, 0],
                      [-np.sin(self.ry), 0, np.cos(self.ry)]])
        corners3d = np.vstack([x_corners, y_corners, z_corners])  # (3, 8)
        corners3d = np.dot(R, corners3d).T
        corners3d = corners3d + self.pos
        return corners3d

    def to_bev_box2d(self, oblique=True, voxel_size=0.1):
        """
        :param bev_shape: (2) for bev shape (h, w), => (y_max, x_max) in image
        :param voxel_size: float, 0.1m
        :param oblique:
        :return: box2d (4, 2)/ (4) in image coordinate
        """
        if oblique:
            corners3d = self.generate_corners3d()
            xz_corners = corners3d[0:4, [0, 2]]
            box2d = np.zeros((4, 2), dtype=np.int32)
            box2d[:, 0] = ((xz_corners[:, 0] - Object3d.MIN_XZ[0]) / voxel_size).astype(np.int32)
            box2d[:, 1] = Object3d.BEV_SHAPE[0] - 1 - ((xz_corners[:, 1] - Object3d.MIN_XZ[1]) / voxel_size).astype(np.int32)
            box2d[:, 0] = np.clip(box2d[:, 0], 0, Object3d.BEV_SHAPE[1])
            box2d[:, 1] = np.clip(box2d[:, 1], 0, Object3d.BEV_SHAPE[0])
        else:
            box2d = np.zeros(4, dtype=np.int32)
            # discrete_center = np.floor((self.pos / voxel_size)).astype(np.int32)
            cu = np.floor((self.pos[0] - Object3d.MIN_XZ[0]) / voxel_size).astype(np.int32)
            cv = Object3d.BEV_SHAPE[0] - 1 - ((self.pos[2] - Object3d.MIN_XZ[1]) / voxel_size).astype(np.int32)
            half_l, half_w = int(self.l / voxel_size / 2), int(self.w / voxel_size / 2)
            box2d[0], box2d[1] = cu - half_l, cv - half_w
            box2d[2], box2d[3] = cu + half_l, cv + half_w

        return box2d

    def to_str(self):
        print_str = '%s %.3f %.3f %.3f box2d: %s hwl: [%.3f %.3f %.3f] pos: %s ry: %.3f' \
                     % (self.cls_type, self.trucation, self.occlusion, self.alpha, self.box2d, self.h, self.w, self.l,
                        self.pos, self.ry)
        return print_str

    def to_kitti_format(self):
        kitti_str = '%s %.2f %d %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f' \
                    % (self.cls_type, self.trucation, int(self.occlusion), self.alpha, self.box2d[0], self.box2d[1],
                       self.box2d[2], self.box2d[3], self.h, self.w, self.l, self.pos[0], self.pos[1], self.pos[2],
                       self.ry)
        return kitti_str

