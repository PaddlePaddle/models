import os
import numpy as np
from .base_dataset import BaseDataset
from ltr.data.image_loader import default_image_loader
import xml.etree.ElementTree as ET
import json
from collections import OrderedDict
import nltk
from nltk.corpus import wordnet
from ltr.admin.environment import env_settings


def get_target_to_image_ratio(seq):
    anno = np.array(seq['anno'])
    img_sz = np.array(seq['image_size'])
    return np.sqrt(anno[0, 2:4].prod() / (img_sz.prod()))


class ImagenetVID(BaseDataset):
    """ Imagenet VID dataset.

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
        root = env_settings().imagenet_dir if root is None else root
        super().__init__(root, image_loader)

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

        # Filter the sequences based on min_length and max_target_area in the first frame
        self.sequence_list = [
            x for x in self.sequence_list
            if len(x['anno']) >= min_length and get_target_to_image_ratio(x) <
            max_target_area
        ]
        self.filter = filter

    def get_name(self):
        return 'imagenetvid'

    def get_num_sequences(self):
        return len(self.sequence_list)

    def get_sequence_info(self, seq_id):
        anno = np.array(self.sequence_list[seq_id]['anno'])
        target_visible = np.array(self.sequence_list[seq_id]['target_visible'],
                                  'bool')
        target_visible = target_visible & (anno[:, 2] > 0) & (anno[:, 3] > 0)
        if self.filter is not None:
            target_large = (anno[:, 2] * anno[:, 3] > 30 * 30)
            ratio = anno[:, 2] / anno[:, 3]
            target_reasonable_ratio = (10 > ratio) & (ratio > 0.1)
            target_visible = target_visible & target_reasonable_ratio & target_large
        return anno, target_visible

    def _get_frame(self, sequence, frame_id):
        set_name = 'ILSVRC2015_VID_train_{:04d}'.format(sequence['set_id'])
        vid_name = 'ILSVRC2015_train_{:08d}'.format(sequence['vid_id'])
        frame_number = frame_id + sequence['start_frame']

        frame_path = os.path.join(self.root, 'Data', 'VID', 'train', set_name,
                                  vid_name, '{:06d}.JPEG'.format(frame_number))
        # frame_path = os.path.join(self.root, 'Data', 'VID', 'train', vid_name,
        #                           '{:06d}.jpg'.format(frame_number))
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
            'object_class': sequence['class_name'],
            'motion_class': None,
            'major_class': None,
            'root_class': None,
            'motion_adverb': None
        })

        return frame_list, anno_frames, object_meta

    def _process_anno(self, root):
        # Builds individual tracklets
        base_vid_anno_path = os.path.join(root, 'Annotations', 'VID', 'train')

        all_sequences = []
        # for set in sorted(os.listdir(base_vid_anno_path)):
        for set in sorted([
                'ILSVRC2015_VID_train_0000', 'ILSVRC2015_VID_train_0001',
                'ILSVRC2015_VID_train_0002', 'ILSVRC2015_VID_train_0003'
        ]):
            set_id = int(set.split('_')[-1])
            for vid in sorted(
                    os.listdir(os.path.join(base_vid_anno_path, set))):

                vid_id = int(vid.split('_')[-1])
                anno_files = sorted(
                    os.listdir(os.path.join(base_vid_anno_path, set, vid)))

                frame1_anno = ET.parse(
                    os.path.join(base_vid_anno_path, set, vid, anno_files[0]))
                image_size = [
                    int(frame1_anno.find('size/width').text),
                    int(frame1_anno.find('size/height').text)
                ]

                objects = [
                    ET.ElementTree(file=os.path.join(base_vid_anno_path, set,
                                                     vid, f)).findall('object')
                    for f in anno_files
                ]

                tracklets = {}

                # Find all tracklets along with start frame
                for f_id, all_targets in enumerate(objects):
                    for target in all_targets:
                        tracklet_id = target.find('trackid').text
                        if tracklet_id not in tracklets:
                            tracklets[tracklet_id] = f_id

                for tracklet_id, tracklet_start in tracklets.items():
                    tracklet_anno = []
                    target_visible = []
                    class_name = None

                    for f_id in range(tracklet_start, len(objects)):
                        found = False
                        for target in objects[f_id]:
                            if target.find('trackid').text == tracklet_id:
                                if not class_name:
                                    class_name_id = target.find('name').text
                                    class_name = class_name_id
                                    # class_name = self._get_class_name_from_id(class_name_id)
                                x1 = int(target.find('bndbox/xmin').text)
                                y1 = int(target.find('bndbox/ymin').text)
                                x2 = int(target.find('bndbox/xmax').text)
                                y2 = int(target.find('bndbox/ymax').text)

                                tracklet_anno.append([x1, y1, x2 - x1, y2 - y1])
                                target_visible.append(
                                    target.find('occluded').text == '0')

                                found = True
                                break
                        if not found:
                            break

                    new_sequence = {
                        'set_id': set_id,
                        'vid_id': vid_id,
                        'class_name': class_name,
                        'start_frame': tracklet_start,
                        'anno': tracklet_anno,
                        'target_visible': target_visible,
                        'image_size': image_size
                    }
                    all_sequences.append(new_sequence)

        return all_sequences
