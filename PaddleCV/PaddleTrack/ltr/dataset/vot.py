import os
from .base_dataset import BaseDataset
from ltr.data.image_loader import default_image_loader
import numpy as np
import cv2 as cv
from collections import OrderedDict
from ltr.admin.environment import env_settings


def get_axis_aligned_bbox(region):
    region = np.array(region)
    if len(region.shape) == 3:
        # region (1,4,2)
        region = np.array([region[0][0][0], region[0][0][1], region[0][1][0], region[0][1][1],
                           region[0][2][0], region[0][2][1], region[0][3][0], region[0][3][1]])

    cx = np.mean(region[0::2])
    cy = np.mean(region[1::2])
    x1 = min(region[0::2])

    x2 = max(region[0::2])
    y1 = min(region[1::2])
    y2 = max(region[1::2])

    A1 = np.linalg.norm(region[0:2] - region[2:4]) * np.linalg.norm(region[2:4] - region[4:6])
    A2 = (x2 - x1) * (y2 - y1)
    s = np.sqrt(A1 / A2)
    w = s * (x2 - x1) + 1
    h = s * (y2 - y1) + 1

    x11 = cx - w // 2
    y11 = cy - h // 2

    return x11, y11, w, h


class VOT(BaseDataset):
    def __init__(self, root=None, image_loader=default_image_loader):
        # root = env_settings().vot_dir if root is None else root
        assert root is not None
        super().__init__(root, image_loader)

        self.sequence_list = self._get_sequence_list()
        self.ann = self._get_annotations()

    def _get_sequence_list(self):
        seq_list = []
        for d in os.listdir(self.root):
            if os.path.isdir(os.path.join(self.root, d)):
                seq_list.append(d)
        return sorted(seq_list)

    def _get_annotations(self):
        ann = {}
        for seq in self.sequence_list:
            ann[seq] = {'bbox': [],
                        'rbb': []}
            with open(os.path.join(self.root, seq, 'groundtruth.txt')) as f:
                lines = [l.strip().split(',') for l in f.readlines()]
                for l in lines:
                    vs = [float(v) for v in l]
                    if len(vs) == 4:
                        polys = [vs[0], vs[1] + vs[3] - 1,
                                 vs[0], vs[1],
                                 vs[0] + vs[2] - 1, vs[1],
                                 vs[0] + vs[2] - 1, vs[1] + vs[3] - 1]
                    else:
                        polys = vs

                    box = get_axis_aligned_bbox(polys)
                    rbb = cv.minAreaRect(np.int0(np.array(polys).reshape((-1, 2))))
                    # assume small rotation angle, switch height, width
                    if rbb[2] < -45:
                        angle = rbb[2] + 90
                        height = rbb[1][0]
                        width = rbb[1][1]
                    else:
                        angle = rbb[2]
                        height = rbb[1][1]
                        width = rbb[1][0]
                    rbb = [rbb[0][0], rbb[0][1], width, height, angle]
                    ann[seq]['bbox'].append(box)
                    ann[seq]['rbb'].append(rbb)
        return ann

    def is_video_sequence(self):
        return True

    def get_name(self):
        return 'vot'

    def get_num_sequences(self):
        return len(self.sequence_list)

    def get_sequence_info(self, seq_id):
        anno = self._get_anno(seq_id)
        target_visible = (anno[:, 2] > 0) & (anno[:, 3] > 0)
        return anno, target_visible

    def _get_anno(self, seq_id):
        anno = self.ann[self.sequence_list[seq_id]]['bbox']
        return np.reshape(np.array(anno), (-1, 4))

    def get_meta_info(self, seq_id):
        object_meta = OrderedDict({'object_class': None,
                                   'motion_class': None,
                                   'major_class': None,
                                   'root_class': None,
                                   'motion_adverb': None})
        return object_meta

    def _get_sequence_path(self, seq_id):
        return os.path.join(self.root, self.sequence_list[seq_id])

    def _get_frame_path(self, seq_path, frame_id):
        return os.path.join(seq_path, 'color', '{:08}.jpg'.format(frame_id + 1))  # frames start from 1

    def _get_frame(self, seq_path, frame_id):
        return self.image_loader(self._get_frame_path(seq_path, frame_id))

    def get_frames(self, seq_id=None, frame_ids=None, anno=None):
        seq_path = self._get_sequence_path(seq_id)
        frame_list = [self._get_frame(seq_path, f_id) for f_id in frame_ids]

        if anno is None:
            anno = self._get_anno(seq_id)

        anno_frames = [anno[f_id, :] for f_id in frame_ids]

        object_meta = self.get_meta_info(seq_id)

        return frame_list, anno_frames, object_meta
