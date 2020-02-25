import os
from .base_dataset import BaseDataset
from ltr.data.image_loader import default_image_loader
import xml.etree.ElementTree as ET
import json
import pickle
from collections import OrderedDict
import numpy as np
import nltk
from nltk.corpus import wordnet
from ltr.admin.environment import env_settings


def get_target_to_image_ratio(seq):
    anno = np.array(seq['anno'])
    img_sz = np.array(seq['image_size'])
    return np.sqrt(anno[0, 2:4].prod() / (img_sz.prod()))


class YoutubeBB(BaseDataset):
    """ YoutubeBB dataset.

    Publication:
        ImageNet Large Scale Visual Recognition Challenge
        Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy,
        Aditya Khosla, Michael Bernstein, Alexander C. Berg and Li Fei-Fei
        IJCV, 2015
        https://arxiv.org/pdf/1409.0575.pdf

    Download the dataset from http://image-net.org/
    """

    def __init__(self,
                 root=None,
                 filter=None,
                 image_loader=default_image_loader,
                 min_length=0,
                 max_target_area=1):
        """
        args:
            root - path to the imagenet vid dataset.
            image_loader (jpeg4py_loader) -  The function to read the images. jpeg4py (https://github.com/ajkxyz/jpeg4py)
                                            is used by default.
            min_length - Minimum allowed sequence length.
            max_target_area - max allowed ratio between target area and image area. Can be used to filter out targets
                                which cover complete image.
        """
        super().__init__(root, image_loader)

        meta_file = os.path.join(root, 'ytb_meta.pickle')
        with open(meta_file, 'rb') as f:
            meta = pickle.load(f)

        sequence_list = []
        for video_name, video_info in meta:
            if 'ILSVRC' not in video_name:
                seq_info = {}
                for trkid in video_info:
                    if len(video_info[trkid]['img']) > 2:
                        seq_info['video_name'] = video_name
                        seq_info['anno'] = video_info[trkid]['box']
                        seq_info['img_paths'] = video_info[trkid]['img']
                        sequence_list.append(seq_info)

        print('num_sequences: {}'.format(len(sequence_list)))
        self.sequence_list = sequence_list

        # Filter the sequences based on min_length and max_target_area in the first frame
        # self.sequence_list = [x for x in self.sequence_list if len(x['anno']) >= min_length and
        #                       get_target_to_image_ratio(x) < max_target_area]
        self.filter = filter

    def get_name(self):
        return 'youtubebb'

    def get_num_sequences(self):
        return len(self.sequence_list)

    def get_sequence_info(self, seq_id):
        anno = np.array(self.sequence_list[seq_id]['anno'])
        target_visible = (anno[:, 2] > 0) & (anno[:, 3] > 0)
        if self.filter is not None:
            target_large = (anno[:, 2] * anno[:, 3] > 30 * 30)
            target_resonable = (anno[:, 2] * anno[:, 3] < 500 * 500)
            ratio = anno[:, 2] / anno[:, 3]
            target_reasonable_ratio = (10 > ratio) & (ratio > 0.1)
            target_visible = target_visible & target_reasonable_ratio & target_large & target_resonable
        return anno, target_visible

    def _get_frame(self, sequence, frame_id):
        frame_path = os.path.join(self.root, sequence['video_name'],
                                  sequence['img_paths'][frame_id] + '.jpg')
        return self.image_loader(frame_path)

    def get_frames(self, seq_id, frame_ids, anno=None):
        sequence = self.sequence_list[seq_id]
        frame_list = [self._get_frame(sequence, f) for f in frame_ids]

        if anno is None:
            anno = sequence['anno']

        # Return as list of tensors
        anno_frames = [anno[f_id, :] for f_id in frame_ids]

        # added the class info to the meta info
        object_meta = OrderedDict({
            'object_class': None,
            'motion_class': None,
            'major_class': None,
            'root_class': None,
            'motion_adverb': None
        })

        return frame_list, anno_frames, object_meta
