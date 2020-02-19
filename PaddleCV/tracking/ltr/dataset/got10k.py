import os
import os.path
import numpy as np
import csv
import pandas
from collections import OrderedDict
from .base_dataset import BaseDataset
from ltr.data.image_loader import default_image_loader
from ltr.admin.environment import env_settings


class Got10k(BaseDataset):
    """ GOT-10k dataset.

    Publication:
        GOT-10k: A Large High-Diversity Benchmark for Generic Object Tracking in the Wild
        Lianghua Huang, Xin Zhao, and Kaiqi Huang
        arXiv:1810.11981, 2018
        https://arxiv.org/pdf/1810.11981.pdf

    Download dataset from http://got-10k.aitestunion.com/downloads
    """

    def __init__(self,
                 root=None,
                 filter=None,
                 image_loader=default_image_loader,
                 split=None,
                 seq_ids=None):
        """
        args:
            root - path to the got-10k training data. Note: This should point to the 'train' folder inside GOT-10k
            image_loader (jpeg4py_loader) -  The function to read the images. jpeg4py (https://github.com/ajkxyz/jpeg4py)
                                            is used by default.
            split - 'train' or 'val'. Note: The validation split here is a subset of the official got-10k train split,
                    not NOT the official got-10k validation split. To use the official validation split, provide that as
                    the root folder instead.
            seq_ids - List containing the ids of the videos to be used for training. Note: Only one of 'split' or 'seq_ids'
                        options can be used at the same time.
        """
        root = env_settings().got10k_dir if root is None else root
        super().__init__(root, image_loader)

        # all folders inside the root
        self.sequence_list = self._get_sequence_list()

        if split == 'vot-train':
            ltr_path = os.path.join(
                os.path.dirname(os.path.realpath(__file__)), '..')
            with open(
                    os.path.join(ltr_path, 'data_specs',
                                 'got10k_prohibited_for_VOT.txt')) as f:
                prohibited = [l.strip() for l in f.readlines()]
            print('GOT10K before: {}'.format(len(self.sequence_list)))
            self.sequence_list = [
                x for x in self.sequence_list if x not in prohibited
            ]
            print('GOT10K after: {}'.format(len(self.sequence_list)))
        else:
            # seq_id is the index of the folder inside the got10k root path
            if split is not None:
                if seq_ids is not None:
                    raise ValueError('Cannot set both split_name and seq_ids.')
                ltr_path = os.path.join(
                    os.path.dirname(os.path.realpath(__file__)), '..')
                if split == 'train':
                    file_path = os.path.join(ltr_path, 'data_specs',
                                             'got10k_train_split.txt')
                elif split == 'val':
                    file_path = os.path.join(ltr_path, 'data_specs',
                                             'got10k_val_split.txt')
                else:
                    raise ValueError('Unknown split name.')
                seq_ids = pandas.read_csv(
                    file_path, header=None, squeeze=True,
                    dtype=np.int64).values.tolist()
            elif seq_ids is None:
                seq_ids = list(range(0, len(self.sequence_list)))
            # self.seq_ids = seq_ids

            self.sequence_list = [self.sequence_list[i] for i in seq_ids]

        self.sequence_meta_info = self._load_meta_info()
        self.filter = filter

    def get_name(self):
        return 'got10k'

    def _load_meta_info(self):
        sequence_meta_info = {
            s: self._read_meta(os.path.join(self.root, s))
            for s in self.sequence_list
        }
        return sequence_meta_info

    def _read_meta(self, seq_path):
        try:
            with open(os.path.join(seq_path, 'meta_info.ini')) as f:
                meta_info = f.readlines()
            object_meta = OrderedDict({
                'object_class': meta_info[5].split(': ')[-1][:-1],
                'motion_class': meta_info[6].split(': ')[-1][:-1],
                'major_class': meta_info[7].split(': ')[-1][:-1],
                'root_class': meta_info[8].split(': ')[-1][:-1],
                'motion_adverb': meta_info[9].split(': ')[-1][:-1]
            })
        except:
            object_meta = OrderedDict({
                'object_class': None,
                'motion_class': None,
                'major_class': None,
                'root_class': None,
                'motion_adverb': None
            })
        return object_meta

    def _get_sequence_list(self):
        with open(os.path.join(self.root, 'list.txt')) as f:
            # dir_names = f.readlines()
            dir_list = list(csv.reader(f))
        dir_list = [dir_name[0] for dir_name in dir_list]
        return dir_list

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
        occlusion_file = os.path.join(seq_path, "absence.label")
        cover_file = os.path.join(seq_path, "cover.label")

        with open(occlusion_file, 'r', newline='') as f:
            occlusion = np.array([int(v[0]) for v in csv.reader(f)], 'byte')
        with open(cover_file, 'r', newline='') as f:
            cover = np.array([int(v[0]) for v in csv.reader(f)], 'byte')

        target_visible = ~occlusion & (cover > 0) & (anno[:, 2] > 0) & (
            anno[:, 3] > 0)

        return target_visible

    def _get_sequence_path(self, seq_id):
        return os.path.join(self.root, self.sequence_list[seq_id])

    def get_sequence_info(self, seq_id):
        seq_path = self._get_sequence_path(seq_id)
        anno = self._read_anno(seq_path)
        target_visible = self._read_target_visible(seq_path, anno)
        if self.filter:
            target_large = (anno[:, 2] * anno[:, 3] > 30 * 30)
            ratio = anno[:, 2] / anno[:, 3]
            target_reasonable_ratio = (10 > ratio) & (ratio > 0.1)
            target_visible = target_visible & target_large & target_reasonable_ratio
        return anno, target_visible

    def _get_frame_path(self, seq_path, frame_id):
        return os.path.join(
            seq_path, '{:08}.jpg'.format(frame_id + 1))  # frames start from 1

    def _get_frame(self, seq_path, frame_id):
        return self.image_loader(self._get_frame_path(seq_path, frame_id))

    def get_frames(self, seq_id, frame_ids, anno=None):
        seq_path = self._get_sequence_path(seq_id)
        obj_meta = self.sequence_meta_info[self.sequence_list[seq_id]]

        frame_list = [self._get_frame(seq_path, f_id) for f_id in frame_ids]

        if anno is None:
            anno = self._read_anno(seq_path)

        # Return as list of tensors
        anno_frames = [anno[f_id, :] for f_id in frame_ids]

        return frame_list, anno_frames, obj_meta
