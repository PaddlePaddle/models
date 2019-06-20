# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
from collections import OrderedDict
import inspect

from paddle import fluid

from ppdet.core.workspace import register, serializable
from ppdet.utils.download import get_dataset_path

from ppdet.dataset.reader import Reader
# XXX these are for triggering the decorator
from ppdet.dataset.transform.operator import (
    DecodeImage, MixupImage, NormalizeBox, NormalizeImage,
    RandomDistort, RandomFlipImage, RandomInterpImage,
    ResizeImage, ExpandImage, CropImage, Permute)
from ppdet.dataset.transform.arrange_sample import (
    ArrangeRCNN, ArrangeTestRCNN, ArrangeSSD, ArrangeTestSSD,
    ArrangeYOLO, ArrangeTestYOLO)

__all__ = ['PadBatch', 'MultiScale', 'RandomShape',
           'DataSet', 'CocoDataSet', 'DataFeed', 'TrainFeed', 'EvalFeed',
           'FasterRCNNTrainFeed', 'MaskRCNNTrainFeed',
           'FasterRCNNTestFeed', 'MaskRCNNTestFeed',
           'SSDTrainFeed', 'SSDEvalFeed', 'SSDTestFeed',
           'YolorainFeed', 'YoloEvalFeed', 'YoloTestFeed',
           'make_reader']

feed_var_def = [
    {
        'name': 'im_info',
        'shape': [3],
        'dtype': 'float32',
        'lod_level': 0
    },
    {
        'name': 'im_id',
        'shape': [1],
        'dtype': 'int32',
        'lod_level': 0
    },
    {
        'name': 'gt_box',
        'shape': [4],
        'dtype': 'float32',
        'lod_level': 1
    },
    {
        'name': 'gt_label',
        'shape': [1],
        'dtype': 'int32',
        'lod_level': 1
    },
    {
        'name': 'is_crowd',
        'shape': [1],
        'dtype': 'int32',
        'lod_level': 1
    },
    {
        'name': 'gt_mask',
        'shape': [2],
        'dtype': 'float32',
        'lod_level': 3
    },
    {
        'name': 'is_difficult',
        'shape': [1],
        'dtype': 'int32',
        'lod_level': 1
    },
    {
        'name': 'gt_score',
        'shape': None,
        'dtype': 'float32',
        'lod_level': 0
    },
    {
        'name': 'im_shape',
        'shape': [2],
        'dtype': 'int32',
        'lod_level': 0
    },
]


# XXX batch transforms are only stubs for now, actually handled by `post_map`
@serializable
class PadBatch(object):
    """
    Pad a batch of samples to same dimensions

    Args:
        pad_to_stride (int): pad to multiple of strides, e.g., 32
    """

    def __init__(self, pad_to_stride=0):
        super(PadBatch, self).__init__()
        self.pad_to_stride = pad_to_stride


@serializable
class MultiScale(object):
    """
    Randomly resize image by scale

    Args:
        scales (list): list of int, randomly resize to one of these scales
    """

    def __init__(self, scales=[]):
        super(MultiScale, self).__init__()
        self.scales = scales


@serializable
class RandomShape(object):
    """
    Randomly reshape a batch

    Args:
        sizes (list): list of int, random choose a size from these
    """

    def __init__(self, sizes=[]):
        super(RandomShape, self).__init__()
        self.sizes = sizes


@serializable
class DataSet(object):
    """
    Dataset, e.g., coco, pascal voc

    Args:
        annotation (str): annotation file path
        image_dir (str): directory where image files are stored
        num_classes (int): number of classes
        shuffle (bool): shuffle samples
    """
    __source__ = 'RoiDbSource'

    def __init__(self,
                 annotation,
                 image_dir,
                 dataset_dir=None):
        super(DataSet, self).__init__()
        self.dataset_dir = dataset_dir
        self.annotation = annotation
        self.image_dir = image_dir


COCO_DATASET_DIR = 'coco'
COCO_TRAIN_ANNOTATION = 'annotations/instances_train2017.json'
COCO_TRAIN_IMAGE_DIR = 'train2017'
COCO_VAL_ANNOTATION = 'annotations/instances_val2017.json'
COCO_VAL_IMAGE_DIR = 'val2017'


@serializable
class CocoDataSet(DataSet):
    def __init__(self,
                 dataset_dir=COCO_DATASET_DIR,
                 annotation=COCO_TRAIN_ANNOTATION,
                 image_dir=COCO_TRAIN_IMAGE_DIR):
        super(CocoDataSet, self).__init__(dataset_dir=dataset_dir, 
                                          annotation=annotation, 
                                          image_dir=image_dir)


VOC_DATASET_DIR = 'pascalvoc'
VOC_TRAIN_ANNOTATION = 'VOCdevkit/VOC_all/ImageSets/Main/train.txt'
VOC_VAL_ANNOTATION = 'VOCdevkit/VOC_all/ImageSets/Main/val.txt'
VOC_TEST_ANNOTATION = 'VOCdevkit/VOC_all/ImageSets/Main/test.txt'
VOC_IMAGE_DIR = 'VOCdevkit/VOC_all/JPEGImages'


@serializable
class VocDataSet(DataSet):
    __source__ = 'VOCSource'

    def __init__(self,
                 dataset_dir=VOC_DATASET_DIR,
                 annotation=VOC_TRAIN_ANNOTATION,
                 image_dir=VOC_IMAGE_DIR):
        super(VocDataSet, self).__init__(dataset_dir=dataset_dir, 
                                         annotation=annotation, 
                                         image_dir=image_dir)


@serializable
class SimpleDataSet(DataSet):
    __source__ = 'SimpleSource'

    def __init__(self,
                 dataset_dir=VOC_DATASET_DIR,
                 annotation=VOC_TEST_ANNOTATION,
                 image_dir=VOC_IMAGE_DIR):
        super(SimpleDataSet, self).__init__(dataset_dir=dataset_dir, 
                                            annotation=annotation, 
                                            image_dir=image_dir)


@serializable
class DataFeed(object):
    """
    DataFeed encompasses all data loading related settings

    Args:
        dataset (object): a `Dataset` instance
        fields (list): list of data fields needed
        image_shape (list): list of image dims (C, MAX_DIM, MIN_DIM)
        sample_transforms (list): list of sample transformations to use
        batch_transforms (list): list of batch transformations to use
        batch_size (int): number of images per device
        shuffle (bool): if samples should be shuffled
        drop_last (bool): drop last batch if size is uneven
        num_workers (int): number of workers processes (or threads)
    """
    __category__ = 'data'

    def __init__(self,
                 dataset,
                 fields,
                 image_shape,
                 sample_transforms=None,
                 batch_transforms=None,
                 batch_size=1,
                 shuffle=False,
                 samples=-1,
                 drop_last=False,
                 with_background=True,
                 test_file=None,
                 num_workers=2,
                 use_process=False):
        super(DataFeed, self).__init__()
        self.fields = fields
        self.image_shape = image_shape
        self.sample_transforms = sample_transforms
        self.batch_transforms = batch_transforms
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.samples = samples
        self.drop_last = drop_last
        self.with_background = with_background
        self.test_file = test_file
        self.num_workers = num_workers
        self.use_process = use_process
        self.dataset = dataset
        if isinstance(dataset, dict):
            self.dataset = DataSet(**dataset)


# for custom (i.e., Non-preset) datasets
@register
class TrainFeed(DataFeed):
    __doc__ = DataFeed.__doc__

    def __init__(self,
                 dataset,
                 fields,
                 image_shape,
                 sample_transforms=[],
                 batch_transforms=[],
                 batch_size=1,
                 shuffle=True,
                 samples=-1,
                 drop_last=False,
                 with_background=True,
                 num_workers=2):
        super(TrainFeed, self).__init__(
            dataset, fields, image_shape, sample_transforms, batch_transforms,
            batch_size=batch_size, shuffle=shuffle, samples=samples,
            drop_last=drop_last, with_background=with_background,
            num_workers=num_workers)


@register
class EvalFeed(DataFeed):
    __doc__ = DataFeed.__doc__

    def __init__(self,
                 dataset,
                 fields,
                 image_shape,
                 sample_transforms=[],
                 batch_transforms=[],
                 batch_size=1,
                 shuffle=False,
                 samples=-1,
                 drop_last=False,
                 with_background=True,
                 num_workers=2):
        super(EvalFeed, self).__init__(
            dataset, fields, image_shape, sample_transforms, batch_transforms,
            batch_size=batch_size, shuffle=shuffle, samples=samples,
            drop_last=drop_last,  with_background=with_background,
            num_workers=num_workers)


@register
class TestFeed(DataFeed):
    __doc__ = DataFeed.__doc__

    def __init__(self,
                 dataset,
                 fields,
                 image_shape,
                 sample_transforms=[],
                 batch_transforms=[],
                 batch_size=1,
                 shuffle=False,
                 drop_last=False,
                 with_background=True,
                 test_file=None,
                 num_workers=2):
        super(TestFeed, self).__init__(
            dataset, fields, image_shape, sample_transforms, batch_transforms,
            batch_size=batch_size, shuffle=shuffle,
            drop_last=drop_last,  with_background=with_background,
            test_file=test_file, num_workers=num_workers)


@register
class FasterRCNNTrainFeed(DataFeed):
    __doc__ = DataFeed.__doc__

    def __init__(self,
                 dataset=CocoDataSet().__dict__,
                 fields=[
                     'image', 'im_info', 'im_id', 'gt_box', 'gt_label',
                     'is_crowd'
                 ],
                 image_shape=[3, 1333, 800],
                 sample_transforms=[
                     DecodeImage(to_rgb=True),
                     RandomFlipImage(prob=0.5),
                     NormalizeImage(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225],
                                    is_scale=True,
                                    is_channel_first=False),
                     ResizeImage(target_size=800, max_size=1333, interp=1),
                     Permute(to_bgr=False)
                 ],
                 batch_transforms=[PadBatch()],
                 batch_size=1,
                 shuffle=True,
                 samples=-1,
                 drop_last=False,
                 num_workers=2,
                 use_process=False):
        # XXX this should be handled by the data loader, since `fields` is
        # given, just collect them
        sample_transforms.append(ArrangeRCNN())
        super(FasterRCNNTrainFeed, self).__init__(
            dataset, fields, image_shape, sample_transforms, batch_transforms,
            batch_size=batch_size, shuffle=shuffle, samples=samples,
            drop_last=drop_last, num_workers=num_workers, 
            use_process=use_process)
        # XXX these modes should be unified
        self.mode = 'TRAIN'


# XXX currently use two presets, in the future, these should be combined into a
# single `RCNNTrainFeed`. Mask (and keypoint) should be processed
# automatically if `gt_mask` (or `gt_keypoints`) is in the required fields
@register
class MaskRCNNTrainFeed(DataFeed):
    __doc__ = DataFeed.__doc__

    def __init__(self,
                 dataset=CocoDataSet().__dict__,
                 fields=[
                     'image', 'im_info', 'im_id', 'gt_box', 'gt_label',
                     'is_crowd', 'gt_mask'
                 ],
                 image_shape=[3, 1333, 800],
                 sample_transforms=[
                     DecodeImage(to_rgb=True),
                     RandomFlipImage(prob=0.5, is_mask_flip=True),
                     NormalizeImage(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225],
                                    is_scale=True,
                                    is_channel_first=False),
                     ResizeImage(target_size=800, max_size=1333, interp=1),
                     Permute(to_bgr=False)
                 ],
                 batch_transforms=[PadBatch()],
                 batch_size=1,
                 shuffle=True,
                 samples=-1,
                 drop_last=False,
                 num_workers=2,
                 use_process=False):
        sample_transforms.append(ArrangeRCNN(is_mask=True))
        super(MaskRCNNTrainFeed, self).__init__(
            dataset, fields, image_shape, sample_transforms, batch_transforms,
            batch_size=batch_size, shuffle=shuffle, samples=samples,
            drop_last=drop_last, num_workers=num_workers, 
            use_process=use_process)
        self.mode = 'TRAIN'


@register
class FasterRCNNTestFeed(DataFeed):
    __doc__ = DataFeed.__doc__

    def __init__(self,
                 dataset=CocoDataSet(COCO_VAL_ANNOTATION,
                                     COCO_VAL_IMAGE_DIR).__dict__,
                 fields=['image', 'im_info', 'im_id'],
                 image_shape=[3, 1333, 800],
                 sample_transforms=[
                     DecodeImage(to_rgb=True),
                     NormalizeImage(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225],
                                    is_scale=True,
                                    is_channel_first=False),
                     ResizeImage(target_size=800, max_size=1333, interp=1),
                     Permute(to_bgr=False)
                 ],
                 batch_transforms=[PadBatch()],
                 batch_size=1,
                 shuffle=False,
                 samples=-1,
                 drop_last=False,
                 test_file=None,
                 num_workers=2):
        sample_transforms.append(ArrangeTestRCNN())
        super(FasterRCNNTestFeed, self).__init__(
            dataset, fields, image_shape, sample_transforms, batch_transforms,
            batch_size=batch_size, shuffle=shuffle, samples=samples,
            drop_last=drop_last, test_file=test_file, 
            num_workers=num_workers)
        self.mode = 'VAL'


@register
class MaskRCNNTestFeed(DataFeed):
    __doc__ = DataFeed.__doc__

    def __init__(self,
                 dataset=CocoDataSet(COCO_VAL_ANNOTATION,
                                     COCO_VAL_IMAGE_DIR).__dict__,
                 fields=['image', 'im_info', 'im_id'],
                 image_shape=[3, 1333, 800],
                 sample_transforms=[
                     DecodeImage(to_rgb=True),
                     NormalizeImage(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225],
                                    is_scale=True,
                                    is_channel_first=False),
                     ResizeImage(target_size=800, max_size=1333, interp=1),
                     Permute(to_bgr=False)
                 ],
                 batch_transforms=[PadBatch()],
                 batch_size=1,
                 shuffle=False,
                 samples=-1,
                 drop_last=False,
                 test_file=None,
                 num_workers=2,
                 use_process=False):
        sample_transforms.append(ArrangeTestRCNN(is_mask=True))
        super(MaskRCNNTestFeed, self).__init__(
            dataset, fields, image_shape, sample_transforms, batch_transforms,
            batch_size=batch_size, shuffle=shuffle, samples=samples,
            drop_last=drop_last, test_file=test_file, 
            num_workers=num_workers, use_process=use_process)
        self.mode = 'VAL'


@register
class SSDTrainFeed(DataFeed):
    __doc__ = DataFeed.__doc__

    def __init__(self,
                 dataset=VocDataSet().__dict__,
                 fields=['image', 'gt_box', 'gt_label', 'is_difficult'],
                 image_shape=[3, 300, 300],
                 sample_transforms=[
                     DecodeImage(to_rgb=True),
                     NormalizeBox(),
                     RandomDistort(),
                     ExpandImage(max_ratio=3, prob=0.5),
                     CropImage([[1, 1, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                                [1, 50, 0.3, 1.0, 0.5, 2.0, 0.1, 0.0],
                                [1, 50, 0.3, 1.0, 0.5, 2.0, 0.3, 0.0],
                                [1, 50, 0.3, 1.0, 0.5, 2.0, 0.5, 0.0],
                                [1, 50, 0.3, 1.0, 0.5, 2.0, 0.7, 0.0],
                                [1, 50, 0.3, 1.0, 0.5, 2.0, 0.9, 0.0],
                                [1, 50, 0.3, 1.0, 0.5, 2.0, 0.0, 1.0]]),
                     ResizeImage(target_size=300, use_cv2=False, interp=1),
                     RandomFlipImage(is_normalized=True),
                     Permute(),
                     NormalizeImage(
                         mean=[127.5, 127.5, 127.5],
                         std=[127.502231, 127.502231, 127.502231],
                         is_scale=False)
                 ],
                 batch_transforms=[],
                 batch_size=32,
                 shuffle=True,
                 samples=-1,
                 drop_last=True,
                 num_workers=8,
                 use_process=True):
        sample_transforms.append(ArrangeSSD())
        if isinstance(dataset, dict):
            dataset = VocDataSet(**dataset)
        super(SSDTrainFeed, self).__init__(
            dataset, fields, image_shape, sample_transforms, batch_transforms,
            batch_size=batch_size, shuffle=shuffle, samples=samples,
            drop_last=drop_last, num_workers=num_workers, 
            use_process=use_process)
        self.mode = 'TRAIN'


@register
class SSDEvalFeed(DataFeed):
    __doc__ = DataFeed.__doc__

    def __init__(self,
                 dataset=VocDataSet(VOC_VAL_ANNOTATION).__dict__,
                 fields=['image'],
                 image_shape=[3, 300, 300],
                 sample_transforms=[
                     DecodeImage(to_rgb=True),
                     NormalizeBox(),
                     ResizeImage(target_size=300, use_cv2=False, interp=1),
                     RandomFlipImage(is_normalized=True),
                     Permute(),
                     NormalizeImage(
                         mean=[127.5, 127.5, 127.5],
                         std=[127.502231, 127.502231, 127.502231],
                         is_scale=False)
                 ],
                 batch_transforms=[],
                 batch_size=64,
                 shuffle=False,
                 samples=-1,
                 drop_last=True,
                 num_workers=8,
                 use_process=False):
        sample_transforms.append(ArrangeSSD())
        if isinstance(dataset, dict):
            dataset = VocDataSet(**dataset)
        super(SSDEvalFeed, self).__init__(
            dataset, fields, image_shape, sample_transforms, batch_transforms,
            batch_size=batch_size, shuffle=shuffle, samples=samples,
            drop_last=drop_last, num_workers=num_workers, 
            use_process=use_process)
        self.mode = 'VAL'


@register
class SSDTestFeed(DataFeed):
    __doc__ = DataFeed.__doc__

    def __init__(self,
                 dataset=SimpleDataSet(VOC_TEST_ANNOTATION).__dict__,
                 fields=['image'],
                 image_shape=[3, 300, 300],
                 sample_transforms=[
                     DecodeImage(to_rgb=True),
                     ResizeImage(target_size=300, use_cv2=False, interp=1),
                 ],
                 batch_transforms=[],
                 batch_size=1,
                 shuffle=False,
                 samples=-1,
                 drop_last=False,
                 test_file=None,
                 num_workers=8,
                 use_process=False):
        sample_transforms.append(ArrangeTestSSD())
        if isinstance(dataset, dict):
            dataset = SimpleDataSet(**dataset)
        super(SSDTestFeed, self).__init__(
            dataset, fields, image_shape, sample_transforms, batch_transforms,
            batch_size=batch_size, shuffle=shuffle, samples=samples,
            drop_last=drop_last, test_file=test_file, 
            num_workers=num_workers)
        self.mode = 'TEST'


@register
class YoloTrainFeed(DataFeed):
    __doc__ = DataFeed.__doc__

    def __init__(self,
                 dataset=CocoDataSet().__dict__,
                 fields=['image', 'gt_box', 'gt_label', 'gt_score'],
                 image_shape=[3, 608, 608],
                 sample_transforms=[
                     DecodeImage(to_rgb=True, with_mixup=True),
                     MixupImage(alpha=1.5, beta=1.5),
                     NormalizeBox(),
                     RandomDistort(),
                     ExpandImage(max_ratio=3., prob=.5),
                     CropImage([[1, 1, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                                [1, 50, 0.3, 1.0, 0.5, 2.0, 0.1, 1.0],
                                [1, 50, 0.3, 1.0, 0.5, 2.0, 0.3, 1.0],
                                [1, 50, 0.3, 1.0, 0.5, 2.0, 0.5, 1.0],
                                [1, 50, 0.3, 1.0, 0.5, 2.0, 0.7, 1.0],
                                [1, 50, 0.3, 1.0, 0.5, 2.0, 0.9, 1.0],
                                [1, 50, 0.3, 1.0, 0.5, 2.0, 0.0, 1.0]],
                               satisfy_all=True),
                     RandomInterpImage(target_size=608),
                     RandomFlipImage(is_normalized=True),
                     NormalizeImage(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225],
                                    is_scale=True,
                                    is_channel_first=False),
                     Permute(to_bgr=False),
                 ],
                 batch_transforms=[
                     RandomShape(sizes=[320, 352, 384, 416, 448, 480,
                                        512, 544, 576, 608])
                 ],
                 batch_size=32,
                 shuffle=True,
                 samples=-1,
                 drop_last=True,
                 with_background=False,
                 num_workers=8,
                 num_max_boxes=50,
                 mixup_epoch=250,
                 use_process=True):
        sample_transforms.append(ArrangeYOLO())
        super(YoloTrainFeed, self).__init__(
            dataset, fields, image_shape, sample_transforms, batch_transforms,
            batch_size=batch_size, shuffle=shuffle, samples=samples,
            drop_last=drop_last,  with_background=with_background,
            num_workers=num_workers, use_process=use_process)
        self.num_max_boxes = num_max_boxes
        self.mixup_epoch = mixup_epoch
        self.mode = 'TRAIN'
        self.bufsize = 128


@register
class YoloEvalFeed(DataFeed):
    __doc__ = DataFeed.__doc__

    def __init__(self,
                 dataset=CocoDataSet(COCO_VAL_ANNOTATION,
                                     COCO_VAL_IMAGE_DIR).__dict__,
                 fields=['image', 'im_shape', 'im_id'],
                 image_shape=[3, 608, 608],
                 sample_transforms=[
                     DecodeImage(to_rgb=True),
                     ResizeImage(target_size=608, interp=2),
                     NormalizeImage(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225],
                                    is_scale=True,
                                    is_channel_first=False),
                     Permute(to_bgr=False),
                 ],
                 batch_transforms=[],
                 batch_size=8,
                 shuffle=True,
                 samples=-1,
                 drop_last=True,
                 with_background=False,
                 num_workers=8,
                 num_max_boxes=50,
                 use_process=False):
        sample_transforms.append(ArrangeTestYOLO())
        super(YoloEvalFeed, self).__init__(
            dataset, fields, image_shape, sample_transforms, batch_transforms,
            batch_size=batch_size, shuffle=shuffle, samples=samples,
            drop_last=drop_last,  with_background=with_background,
            num_workers=num_workers, use_process=use_process)
        self.num_max_boxes = num_max_boxes
        self.mode = 'VAL'
        self.bufsize = 128


@register
class YoloTestFeed(DataFeed):
    __doc__ = DataFeed.__doc__

    def __init__(self,
                 dataset=SimpleDataSet(COCO_VAL_ANNOTATION,
                                       COCO_VAL_IMAGE_DIR).__dict__,
                 fields=['image', 'im_shape', 'im_id'],
                 image_shape=[3, 608, 608],
                 sample_transforms=[
                     DecodeImage(to_rgb=True),
                     ResizeImage(target_size=608, interp=2),
                     NormalizeImage(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225],
                                    is_scale=True,
                                    is_channel_first=False),
                     Permute(to_bgr=False),
                 ],
                 batch_transforms=[],
                 batch_size=1,
                 shuffle=True,
                 samples=-1,
                 drop_last=True,
                 with_background=False,
                 test_file=None,
                 num_workers=8,
                 num_max_boxes=50,
                 use_process=False):
        sample_transforms.append(ArrangeTestYOLO())
        if isinstance(dataset, dict):
            dataset = SimpleDataSet(**dataset)
        super(YoloTestFeed, self).__init__(
            dataset, fields, image_shape, sample_transforms, batch_transforms,
            batch_size=batch_size, shuffle=shuffle, samples=samples,
            drop_last=drop_last,  with_background=with_background,
            test_file=test_file, num_workers=num_workers,
            use_process=use_process)
        self.num_max_boxes = num_max_boxes
        self.mode = 'TEST'
        self.bufsize = 128


def make_reader(feed, max_iter=0, use_pyreader=True):
    """this is a adapter for now, some part may be quite hackish

    Args:
        feed (object): a `DataFeed` instance
        max_iter (int): number of iterations, could be removed after some bug fix

    Returns: pyreader, user reader, feed_vars dict
    """

    image_shape = feed.image_shape
    feed_var_map = {var['name']: var for var in feed_var_def}
    feed_var_map['image'] = {
        'name': 'image',
        'shape': image_shape,
        'dtype': 'float32',
        'lod_level': 0
    }

    # YOLO var dim is fixed
    if getattr(feed, 'num_max_boxes', None) is not None:
        feed_var_map['gt_label']['shape'] = [feed.num_max_boxes]
        feed_var_map['gt_score']['shape'] = [feed.num_max_boxes]
        feed_var_map['gt_box']['shape'] = [feed.num_max_boxes, 4]
        feed_var_map['gt_label']['lod_level'] = 0
        feed_var_map['gt_score']['lod_level'] = 0
        feed_var_map['gt_box']['lod_level'] = 0
    mixup_epoch = -1
    if getattr(feed, 'mixup_epoch', None) is not None:
        mixup_epoch = feed.mixup_epoch
    bufsize = 10
    use_process = False
    if getattr(feed, 'bufsize', None) is not None:
        bufsize = feed.bufsize
    if getattr(feed, 'use_process', None) is not None:
        use_process = feed.use_process

    feed_vars = OrderedDict([
        (key, fluid.layers.data(
            name=feed_var_map[key]['name'],
            shape=feed_var_map[key]['shape'],
            dtype=feed_var_map[key]['dtype'],
            lod_level=feed_var_map[key]['lod_level']))
        for key in feed.fields
    ])

    pyreader = None
    if use_pyreader:
        pyreader = fluid.io.PyReader(
            feed_list=list(feed_vars.values()),
            capacity=64,
            use_double_buffer=True,
            iterable=False)

    # XXX temporary hacks below, DO NOT JUDGE!
    mode = feed.mode

    # if DATASET_DIR set and not exists, search dataset under ~/.paddle/dataset
    # if not exists base on DATASET_DIR name (coco or pascal), if not found 
    # under ~/.paddle/dataset, download it.
    if feed.dataset.dataset_dir:
        dataset_dir = get_dataset_path(feed.dataset.dataset_dir)
        feed.dataset.annotation = os.path.join(dataset_dir, feed.dataset.annotation)
        feed.dataset.image_dir = os.path.join(dataset_dir, feed.dataset.image_dir)

    data_config = {
        mode: {
            'ANNO_FILE': feed.dataset.annotation,
            'IMAGE_DIR': feed.dataset.image_dir,
            'IS_SHUFFLE': feed.shuffle,
            'SAMPLES': feed.samples,
            'WITH_BACKGROUND': feed.with_background,
            'MIXUP_EPOCH': mixup_epoch,
            'TYPE': type(feed.dataset).__source__
        }
    }

    if mode == 'TEST':
        data_config[mode]['TEST_FILE'] = feed.test_file

    transform_config = {
        'WORKER_CONF': {
            'bufsize': bufsize,
            'worker_num': feed.num_workers,
            'use_process': use_process
        },
        'BATCH_SIZE': feed.batch_size,
        'DROP_LAST': feed.drop_last
    }

    pad = [t for t in feed.batch_transforms if isinstance(t, PadBatch)]
    random_shape = [t for t in feed.batch_transforms
                    if isinstance(t, RandomShape)]
    multi_scale = [t for t in feed.batch_transforms
                   if isinstance(t, MultiScale)]

    if any(pad):
        transform_config['IS_PADDING'] = True
        if pad[0].pad_to_stride != 0:
            transform_config['COARSEST_STRIDE'] = pad[0].pad_to_stride
    if any(random_shape):
        transform_config['RANDOM_SHAPES'] = random_shape[0].sizes
    if any(multi_scale):
        transform_config['MULTI_SCALES'] = multi_scale[0].scales

    if hasattr(inspect, 'getfullargspec'):
        argspec = inspect.getfullargspec
    else:
        argspec = inspect.getargspec

    ops = []
    for op in feed.sample_transforms:
        op_dict = op.__dict__.copy()
        argnames = [arg for arg in argspec(type(op).__init__).args
                    if arg != 'self']
        op_dict = {k: v for k, v in op_dict.items() if k in argnames}
        op_dict['op'] = op.__class__.__name__
        ops.append(op_dict)
    transform_config['OPS'] = ops

    import yaml
    from ppdet.utils.cli import ColorTTY
    tty = ColorTTY()
    print(tty.green("============ generated data_config ================"))
    print(yaml.dump(data_config, default_flow_style=False, default_style=''))
    print(tty.green("========== generated transform_config =============="))
    print(yaml.dump(transform_config, default_flow_style=False, default_style=''))
    print(tty.green("========== please verify they are correct!!!! =============="))
    print("")

    reader = Reader(data_config, {mode: transform_config}, max_iter)

    return pyreader, reader._make_reader(mode), feed_vars
