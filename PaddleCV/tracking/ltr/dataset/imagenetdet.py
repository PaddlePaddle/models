import os
import numpy as np
from .base_dataset import BaseDataset
from ltr.data.image_loader import default_image_loader
import xml.etree.ElementTree as ET
import glob
import json
from collections import OrderedDict
import nltk
from nltk.corpus import wordnet
from ltr.admin.environment import env_settings


class ImagenetDET(BaseDataset):
    """ Imagenet DET dataset.

    Publication:
        ImageNet Large Scale Visual Recognition Challenge
        Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy,
        Aditya Khosla, Michael Bernstein, Alexander C. Berg and Li Fei-Fei
        IJCV, 2015
        https://arxiv.org/pdf/1409.0575.pdf

    Download the dataset from http://image-net.org/
    """

    def __init__(self, root=None, filter=None, image_loader=default_image_loader):
        """
        args:
            root - path to the imagenet det dataset.
            image_loader (jpeg4py_loader) -  The function to read the images. jpeg4py (https://github.com/ajkxyz/jpeg4py)
                                            is used by default.
        """
        root = env_settings().imagenetdet_dir if root is None else root
        super().__init__(root, image_loader)
        self.filter = filter

        self.set_list = ['ILSVRC2013_train', 'ILSVRC2014_train_0000',
                         'ILSVRC2014_train_0001', 'ILSVRC2014_train_0002',
                         'ILSVRC2014_train_0003', 'ILSVRC2014_train_0004',
                         'ILSVRC2014_train_0005', 'ILSVRC2014_train_0006']

        cache_file = os.path.join(root, 'cache.json')
        if os.path.isfile(cache_file):
            # If available, load the pre-processed cache file containing meta-info for each sequence
            with open(cache_file, 'r') as f:
                sequence_list_dict = json.load(f)

            self.sequence_list = sequence_list_dict
        else:
            # Else process the imagenet annotations and generate the cache file
            self.sequence_list = self._process_anno(root)

            with open(cache_file, 'w') as f:
                json.dump(self.sequence_list, f)

    def is_video_sequence(self):
        return False

    def get_name(self):
        return 'imagenetdet'

    def get_num_sequences(self):
        return len(self.sequence_list)

    def get_sequence_info(self, seq_id):
        anno = self._get_anno(seq_id)
        target_visible = (anno[:, 2] > 0) & (anno[:, 3] > 0)
        if self.filter:
            target_large = (anno[:, 2] * anno[:, 3] > 30 * 30)
            ratio = anno[:, 2] / anno[:, 3]
            target_reasonable_ratio = (10 > ratio) & (ratio > 0.1)
            target_visible = target_visible & target_reasonable_ratio & target_large
        return anno, target_visible

    def _get_anno(self, seq_id):
        anno = self.sequence_list[seq_id]['anno']
        return np.reshape(np.array(anno), (1, 4))

    def _get_frames(self, seq_id):
        set_name = self.set_list[self.sequence_list[seq_id]['set_id']]
        folder = self.sequence_list[seq_id]['folder']
        if folder == set_name:
            folder = ''
        filename = self.sequence_list[seq_id]['filename']

        frame_path = os.path.join(self.root, 'Data', 'DET', 'train', set_name, folder,
                                  '{:s}.JPEG'.format(filename))
        return self.image_loader(frame_path)

    def get_frames(self, seq_id, frame_ids, anno=None):
        # ImageNet DET is an image dataset. Thus we replicate the image denoted by seq_id len(frame_ids) times, and return a
        # list containing these replicated images.
        frame = self._get_frames(seq_id)

        frame_list = [frame.copy() for _ in frame_ids]

        if anno is None:
            anno = self._get_anno(seq_id)

        anno_frames = [anno.copy()[0, :] for _ in frame_ids]

        object_meta = OrderedDict({'object_class': self.sequence_list[seq_id]['class_name'],
                                   'motion_class': None,
                                   'major_class': None,
                                   'root_class': None,
                                   'motion_adverb': None})

        return frame_list, anno_frames, object_meta

    def _process_anno(self, root):
        # Builds individual tracklets
        base_det_anno_path = os.path.join(root, 'Annotations', 'DET', 'train')

        all_sequences = []
        for set_id, set in enumerate(self.set_list):
            if set_id == 0:
                xmls = sorted(glob.glob(os.path.join(base_det_anno_path, set, '*', '*.xml')))
            else:
                xmls = sorted(glob.glob(os.path.join(base_det_anno_path, set, '*.xml')))
            for xml in xmls:
                xmltree = ET.parse(xml)
                folder = xmltree.find('folder').text
                filename = xmltree.find('filename').text
                image_size = [int(xmltree.find('size/width').text), int(xmltree.find('size/height').text)]
                objects = xmltree.findall('object')
                # Find all objects
                for id, object_iter in enumerate(objects):
                    bndbox = object_iter.find('bndbox')
                    x1 = int(bndbox.find('xmin').text)
                    y1 = int(bndbox.find('ymin').text)
                    x2 = int(bndbox.find('xmax').text)
                    y2 = int(bndbox.find('ymax').text)
                    object_anno = [x1, y1, x2 - x1, y2 - y1]
                    class_name = None
                    if x2 <= x1 or y2 <= y1:
                        continue

                    new_sequence = {'set_id': set_id, 'folder': folder, 'filename': filename,
                                    'class_name': class_name, 'anno': object_anno, 'image_size': image_size}
                    all_sequences.append(new_sequence)

        return all_sequences
