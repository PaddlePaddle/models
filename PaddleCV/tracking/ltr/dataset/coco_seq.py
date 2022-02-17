import os
from .base_dataset import BaseDataset
from ltr.data.image_loader import default_image_loader
from pycocotools.coco import COCO
from collections import OrderedDict
from ltr.admin.environment import env_settings
import numpy as np


class MSCOCOSeq(BaseDataset):
    """ The COCO dataset. COCO is an image dataset. Thus, we treat each image as a sequence of length 1.

    Publication:
        Microsoft COCO: Common Objects in Context.
        Tsung-Yi Lin, Michael Maire, Serge J. Belongie, Lubomir D. Bourdev, Ross B. Girshick, James Hays, Pietro Perona,
        Deva Ramanan, Piotr Dollar and C. Lawrence Zitnick
        ECCV, 2014
        https://arxiv.org/pdf/1405.0312.pdf

    Download the images along with annotations from http://cocodataset.org/#download. The root folder should be
    organized as follows.
        - coco_root
            - annotations
                - instances_train2014.json
            - images
                - train2014

    Note: You also have to install the coco pythonAPI from https://github.com/cocodataset/cocoapi.
    """

    def __init__(self,
                 root=None,
                 filter=None,
                 image_loader=default_image_loader):
        root = env_settings().coco_dir if root is None else root
        super().__init__(root, image_loader)
        self.filter = filter

        # self.img_pth = os.path.join(root, 'train2014/')
        self.img_pth = os.path.join(root, 'train2017/')
        # self.anno_path = os.path.join(root, 'annotations/instances_train2014.json')
        self.anno_path = os.path.join(root,
                                      'annotations/instances_train2017.json')

        # Load the COCO set.
        self.coco_set = COCO(self.anno_path)

        self.cats = self.coco_set.cats
        self.sequence_list = self._get_sequence_list()

    def _get_sequence_list(self):
        ann_list = list(self.coco_set.anns.keys())
        seq_list = []
        print('COCO before: {}'.format(len(ann_list)))
        for a in ann_list:
            if self.coco_set.anns[a]['iscrowd'] == 0:
                box = self.coco_set.anns[a]['bbox']
                box = np.reshape(np.array(box), (1, 4))
                target_visible = (box[:, 2] > 0) & (box[:, 3] > 0)
                if self.filter:
                    target_large = (box[:, 2] * box[:, 3] > 30 * 30)
                    ratio = box[:, 2] / box[:, 3]
                    target_reasonable_ratio = (10 > ratio) & (ratio > 0.1)
                    target_visible = target_visible & target_large & target_reasonable_ratio
                if target_visible:
                    seq_list.append(a)
        print('COCO after: {}'.format(len(seq_list)))
        return seq_list

    def is_video_sequence(self):
        return False

    def get_name(self):
        return 'coco'

    def get_num_sequences(self):
        return len(self.sequence_list)

    def get_sequence_info(self, seq_id):
        anno = self._get_anno(seq_id)
        target_visible = (anno[:, 2] > 0) & (anno[:, 3] > 0)
        return anno, target_visible

    def _get_anno(self, seq_id):
        anno = self.coco_set.anns[self.sequence_list[seq_id]]['bbox']
        return np.reshape(np.array(anno), (1, 4))

    def _get_frames(self, seq_id, mask=False):
        path = self.coco_set.loadImgs(
            [self.coco_set.anns[self.sequence_list[seq_id]]['image_id']])[0][
                'file_name']
        img = self.image_loader(os.path.join(self.img_pth, path))

        if mask:
            ann = self.coco_set.anns[self.sequence_list[seq_id]]
            im_mask = (self.coco_set.annToMask(ann).astype(np.float32) > 0.5).astype(np.float32)
            im_mask = np.expand_dims(im_mask, axis=2)
            return img, im_mask
        else:
            return img

    def get_meta_info(self, seq_id):
        try:
            cat_dict_current = self.cats[self.coco_set.anns[self.sequence_list[
                seq_id]]['category_id']]
            object_meta = OrderedDict({
                'object_class': cat_dict_current['name'],
                'motion_class': None,
                'major_class': cat_dict_current['supercategory'],
                'root_class': None,
                'motion_adverb': None
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

    def get_frames(self, seq_id=None, frame_ids=None, anno=None):
        # COCO is an image dataset. Thus we replicate the image denoted by seq_id len(frame_ids) times, and return a
        # list containing these replicated images.
        frame = self._get_frames(seq_id)

        frame_list = [frame.copy() for _ in frame_ids]

        if anno is None:
            anno = self._get_anno(seq_id)

        anno_frames = [anno.copy()[0, :] for _ in frame_ids]

        object_meta = self.get_meta_info(seq_id)

        return frame_list, anno_frames, object_meta

    def get_frames_mask(self, seq_id=None, frame_ids=None, anno=None):
        # COCO is an image dataset. Thus we replicate the image denoted by seq_id len(frame_ids) times, and return a
        # list containing these replicated images.
        frame, mask = self._get_frames(seq_id, mask=True)

        frame_list = [frame.copy() for _ in frame_ids]

        mask_list = [mask.copy() for _ in frame_ids]

        if anno is None:
            anno = self._get_anno(seq_id)

        anno_frames = [anno.copy()[0, :] for _ in frame_ids]

        object_meta = self.get_meta_info(seq_id)

        return frame_list, anno_frames, mask_list, object_meta
