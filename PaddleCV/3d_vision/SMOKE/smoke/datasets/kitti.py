# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import csv
import logging
import random

import paddle
import numpy as np
from PIL import Image

from smoke.cvlibs import manager
from smoke.transforms import Compose

from smoke.utils.heatmap_coder import (
    get_transfrom_matrix,
    affine_transform,
    gaussian_radius,
    draw_umich_gaussian,
    encode_label
)


@manager.DATASETS.add_component
class KITTI(paddle.io.Dataset):
    """Parsing KITTI format dataset

    Args:
        Dataset (class):
    """
    def __init__(self, dataset_root, mode="train", transforms=None, flip_prob=0.5, aug_prob=0.3):
        super().__init__()

        self.TYPE_ID_CONVERSION = {
            'Car': 0,
            'Cyclist': 1,
            'Pedestrian': 2,
        }
        
        mode = mode.lower()

        self.image_dir = os.path.join(dataset_root, "image_2")
        self.label_dir = os.path.join(dataset_root, "label_2")
        self.calib_dir = os.path.join(dataset_root, "calib")
        
        if mode.lower() not in ['train', 'val', 'trainval', 'test']:
            raise ValueError(
                "mode should be 'train', 'val', 'trainval' or 'test', but got {}.".format(
                    mode))
        imageset_txt = os.path.join(dataset_root, "ImageSets", "{}.txt".format(mode))

        self.is_train = True if mode in ["train", "trainval"] else False
        self.transforms = Compose(transforms)

        image_files = []
        for line in open(imageset_txt, "r"):
            base_name = line.replace("\n", "")
            image_name = base_name + ".png"
            image_files.append(image_name)
        self.image_files = image_files
        self.label_files = [i.replace(".png", ".txt") for i in self.image_files]
        self.num_samples = len(self.image_files)
        self.classes = ("Car", "Cyclist", "Pedestrian")


        self.flip_prob = flip_prob if self.is_train else 0.0
        self.aug_prob = aug_prob if self.is_train else 0.0
        self.shift_scale = (0.2, 0.4)
        self.num_classes = len(self.classes)

        self.input_width = 1280
        self.input_height = 384
        self.output_width = self.input_width // 4
        self.output_height = self.input_height // 4
        self.max_objs = 50

        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing KITTI {} set with {} files loaded".format(mode, self.num_samples))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # load default parameter here
        original_idx = self.label_files[idx].replace(".txt", "")
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        img = Image.open(img_path)
        anns, K = self.load_annotations(idx)
        
        center = np.array([i / 2 for i in img.size], dtype=np.float32)
        size = np.array([i for i in img.size], dtype=np.float32)

        """
        resize, horizontal flip, and affine augmentation are performed here.
        since it is complicated to compute heatmap w.r.t transform.
        """
        flipped = False
        if (self.is_train) and (random.random() < self.flip_prob):
            flipped = True
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            center[0] = size[0] - center[0] - 1
            K[0, 2] = size[0] - K[0, 2] - 1

        affine = False
        if (self.is_train) and (random.random() < self.aug_prob):
            affine = True
            shift, scale = self.shift_scale[0], self.shift_scale[1]
            shift_ranges = np.arange(-shift, shift + 0.1, 0.1)
            center[0] += size[0] * random.choice(shift_ranges)
            center[1] += size[1] * random.choice(shift_ranges)

            scale_ranges = np.arange(1 - scale, 1 + scale + 0.1, 0.1)
            size *= random.choice(scale_ranges)

        center_size = [center, size]
        trans_affine = get_transfrom_matrix(
            center_size,
            [self.input_width, self.input_height]
        )
        trans_affine_inv = np.linalg.inv(trans_affine)
        img = img.transform(
            (self.input_width, self.input_height),
            method=Image.AFFINE,
            data=trans_affine_inv.flatten()[:6],
            resample=Image.BILINEAR,
        )


        trans_mat = get_transfrom_matrix(
            center_size,
            [self.output_width, self.output_height]
        )

        if not self.is_train:
            # for inference we parametrize with original size
            target = {}
            target["image_size"] = size
            target["is_train"] = self.is_train
            target["trans_mat"] = trans_mat
            target["K"] = K
            if self.transforms is not None:
                img, target = self.transforms(img, target)

            return np.array(img), target, original_idx

        heat_map = np.zeros([self.num_classes, self.output_height, self.output_width], dtype=np.float32)
        regression = np.zeros([self.max_objs, 3, 8], dtype=np.float32)
        cls_ids = np.zeros([self.max_objs], dtype=np.int32)
        proj_points = np.zeros([self.max_objs, 2], dtype=np.int32)
        p_offsets = np.zeros([self.max_objs, 2], dtype=np.float32)
        c_offsets = np.zeros([self.max_objs, 2], dtype=np.float32)
        dimensions = np.zeros([self.max_objs, 3], dtype=np.float32)
        locations = np.zeros([self.max_objs, 3], dtype=np.float32)
        rotys = np.zeros([self.max_objs], dtype=np.float32)
        reg_mask = np.zeros([self.max_objs], dtype=np.uint8)
        flip_mask = np.zeros([self.max_objs], dtype=np.uint8)
        bbox2d_size = np.zeros([self.max_objs, 2], dtype=np.float32)

        for i, a in enumerate(anns):
            if i == self.max_objs:
                break
            a = a.copy()
            cls = a["label"]

            locs = np.array(a["locations"])
            rot_y = np.array(a["rot_y"])
            if flipped:
                locs[0] *= -1
                rot_y *= -1

            point, box2d, box3d = encode_label(
                K, rot_y, a["dimensions"], locs
            )
            if np.all(box2d == 0):
                continue
            point = affine_transform(point, trans_mat)
            box2d[:2] = affine_transform(box2d[:2], trans_mat)
            box2d[2:] = affine_transform(box2d[2:], trans_mat)
            box2d[[0, 2]] = box2d[[0, 2]].clip(0, self.output_width - 1)
            box2d[[1, 3]] = box2d[[1, 3]].clip(0, self.output_height - 1)
            h, w = box2d[3] - box2d[1], box2d[2] - box2d[0]
            center = np.array([(box2d[0] + box2d[2]) / 2, (box2d[1] + box2d[3]) /2], dtype=np.float32)

            if (0 < center[0] < self.output_width) and (0 < center[1] < self.output_height):
                point_int = center.astype(np.int32)
                p_offset = point - point_int
                c_offset = center - point_int
                radius = gaussian_radius(h, w)
                radius = max(0, int(radius))
                heat_map[cls] = draw_umich_gaussian(heat_map[cls], point_int, radius)

                cls_ids[i] = cls
                regression[i] = box3d
                proj_points[i] = point_int
                p_offsets[i] = p_offset
                c_offsets[i] = c_offset
                dimensions[i] = np.array(a["dimensions"])
                locations[i] = locs
                rotys[i] = rot_y
                reg_mask[i] = 1 if not affine else 0
                flip_mask[i] = 1 if not affine and flipped else 0

                # targets for 2d bbox
                bbox2d_size[i, 0] = w
                bbox2d_size[i, 1] = h
        
        target = {}
        target["image_size"] = np.array(img.size)
        target["is_train"] = self.is_train
        target["trans_mat"] = trans_mat
        target["K"] = K
        target["hm"] = heat_map
        target["reg"] = regression
        target["cls_ids"] = cls_ids
        target["proj_p"] = proj_points
        target["dimensions"] = dimensions
        target["locations"] = locations
        target["rotys"] = rotys
        target["reg_mask"] = reg_mask
        target["flip_mask"] = flip_mask
        target["bbox_size"] = bbox2d_size
        target["c_offsets"] = c_offsets

        if self.transforms is not None:
            img, target = self.transforms(img, target)


        return np.array(img), target, original_idx


    def load_annotations(self, idx):
        """load kitti label by given index

        Args:
            idx (int): which label to load

        Returns:
            (list[dict], np.ndarray(float32, 3x3)): labels and camera intrinsic matrix
        """
        annotations = []
        file_name = self.label_files[idx]
        fieldnames = ['type', 'truncated', 'occluded', 'alpha', 'xmin', 'ymin', 'xmax', 'ymax', 'dh', 'dw',
                      'dl', 'lx', 'ly', 'lz', 'ry']

        if self.is_train:
            if os.path.exists(os.path.join(self.label_dir, file_name)):
                with open(os.path.join(self.label_dir, file_name), 'r') as csv_file:
                    reader = csv.DictReader(csv_file, delimiter=' ', fieldnames=fieldnames)

                    for line, row in enumerate(reader):
                        if (float(row["xmax"]) == 0.) | (float(row["ymax"]) == 0.):
                            continue
                        if row["type"] in self.classes:
                            annotations.append({
                                "class": row["type"],
                                "label": self.TYPE_ID_CONVERSION[row["type"]],
                                "truncation": float(row["truncated"]),
                                "occlusion": float(row["occluded"]),
                                "alpha": float(row["alpha"]),
                                "dimensions": [float(row['dl']), float(row['dh']), float(row['dw'])],
                                "locations": [float(row['lx']), float(row['ly']), float(row['lz'])],
                                "rot_y": float(row["ry"])
                            })

        # get camera intrinsic matrix K
        with open(os.path.join(self.calib_dir, file_name), 'r') as csv_file:
            reader = csv.reader(csv_file, delimiter=' ')
            for line, row in enumerate(reader):
                if row[0] == 'P2:':
                    K = row[1:]
                    K = [float(i) for i in K]
                    K = np.array(K, dtype=np.float32).reshape(3, 4)
                    K = K[:3, :3]
                    break

        return annotations, K
