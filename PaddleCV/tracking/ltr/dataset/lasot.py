import os
import os.path
import numpy as np
import pandas
import csv
from collections import OrderedDict
from .base_dataset import BaseDataset
from ltr.data.image_loader import default_image_loader
from ltr.admin.environment import env_settings


class Lasot(BaseDataset):
    """ LaSOT dataset.

    Publication:
        LaSOT: A High-quality Benchmark for Large-scale Single Object Tracking
        Heng Fan, Liting Lin, Fan Yang, Peng Chu, Ge Deng, Sijia Yu, Hexin Bai, Yong Xu, Chunyuan Liao and Haibin Ling
        CVPR, 2019
        https://arxiv.org/pdf/1809.07845.pdf

    Download the dataset from https://cis.temple.edu/lasot/download.html
    """

    def __init__(self,
                 root=None,
                 filter=None,
                 image_loader=default_image_loader,
                 vid_ids=None,
                 split=None):
        """
        args:
            root - path to the lasot dataset.
            image_loader (jpeg4py_loader) -  The function to read the images. jpeg4py (https://github.com/ajkxyz/jpeg4py)
                                            is used by default.
            vid_ids - List containing the ids of the videos (1 - 20) used for training. If vid_ids = [1, 3, 5], then the
                    videos with subscripts -1, -3, and -5 from each class will be used for training.
            split - If split='train', the official train split (protocol-II) is used for training. Note: Only one of
                    vid_ids or split option can be used at a time.
        """
        root = env_settings().lasot_dir if root is None else root
        super().__init__(root, image_loader)

        self.sequence_list = self._build_sequence_list(vid_ids, split)
        self.filter = filter

    def _build_sequence_list(self, vid_ids=None, split=None):
        if split is not None:
            if vid_ids is not None:
                raise ValueError('Cannot set both split_name and vid_ids.')
            ltr_path = os.path.join(
                os.path.dirname(os.path.realpath(__file__)), '..')
            if split == 'train':
                file_path = os.path.join(ltr_path, 'data_specs',
                                         'lasot_train_split.txt')
            else:
                raise ValueError('Unknown split name.')
            sequence_list = pandas.read_csv(
                file_path, header=None, squeeze=True).values.tolist()
        elif vid_ids is not None:
            sequence_list = [
                c + '-' + str(v) for c in self.class_list for v in vid_ids
            ]
        else:
            raise ValueError('Set either split_name or vid_ids.')

        return sequence_list

    def get_name(self):
        return 'lasot'

    def get_num_sequences(self):
        return len(self.sequence_list)

    def _read_anno(self, seq_path):
        anno_file = os.path.join(seq_path, "groundtruth.txt")
        gt = pandas.read_csv(
            anno_file,
            delimiter=',',
            header=None,
            dtype=np.float32,
            na_filter=False,
            low_memory=False).values
        return np.array(gt)

    def _read_target_visible(self, seq_path, anno):
        # Read full occlusion and out_of_view
        occlusion_file = os.path.join(seq_path, "full_occlusion.txt")
        out_of_view_file = os.path.join(seq_path, "out_of_view.txt")

        with open(occlusion_file, 'r', newline='') as f:
            occlusion = np.array([int(v) for v in list(csv.reader(f))[0]],
                                 'byte')
        with open(out_of_view_file, 'r') as f:
            out_of_view = np.array([int(v) for v in list(csv.reader(f))[0]],
                                   'byte')

        target_visible = ~occlusion & ~out_of_view & (anno[:, 2] > 0) & (
            anno[:, 3] > 0)

        return target_visible

    def _get_sequence_path(self, seq_id):
        seq_name = self.sequence_list[seq_id]
        class_name = seq_name.split('-')[0]
        vid_id = seq_name.split('-')[1]

        return os.path.join(self.root, class_name, class_name + '-' + vid_id)

    def get_sequence_info(self, seq_id):
        seq_path = self._get_sequence_path(seq_id)
        anno = self._read_anno(seq_path)
        target_visible = self._read_target_visible(seq_path, anno)
        if self.filter is not None:
            target_large = (anno[:, 2] * anno[:, 3] > 30 * 30)
            ratio = anno[:, 2] / anno[:, 3]
            target_reasonable_ratio = (10 > ratio) & (ratio > 0.1)
            target_visible = target_visible & target_reasonable_ratio & target_large
        return anno, target_visible

    def _get_frame_path(self, seq_path, frame_id):
        return os.path.join(
            seq_path, 'img',
            '{:08}.jpg'.format(frame_id + 1))  # frames start from 1

    def _get_frame(self, seq_path, frame_id):
        return self.image_loader(self._get_frame_path(seq_path, frame_id))

    def _get_class(self, seq_path):
        obj_class = seq_path.split('/')[-2]
        return obj_class

    def get_frames(self, seq_id, frame_ids, anno=None):
        seq_path = self._get_sequence_path(seq_id)

        obj_class = self._get_class(seq_path)
        frame_list = [self._get_frame(seq_path, f_id) for f_id in frame_ids]

        if anno is None:
            anno = self._read_anno(seq_path)

        # Return as list of tensors
        anno_frames = [anno[f_id, :] for f_id in frame_ids]

        object_meta = OrderedDict({
            'object_class': obj_class,
            'motion_class': None,
            'major_class': None,
            'root_class': None,
            'motion_adverb': None
        })

        return frame_list, anno_frames, object_meta
