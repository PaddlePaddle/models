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

# function:
#    operators to process sample,
#    eg: decode/resize/crop image

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import uuid
import logging
import numpy as np
import cv2
from functools import reduce

logger = logging.getLogger(__name__)

registered_ops = []
def register_op(cls):
    registered_ops.append(cls.__name__)
    return cls


class BboxError(ValueError):
    pass


class ImageError(ValueError):
    pass


@register_op
class BaseOperator(object):
    def __init__(self, name=None):
        if name is None:
            name = self.__class__.__name__
        self._id = name + '_' + str(uuid.uuid4())[-6:]

    def __call__(self, sample, context=None):
        """ Process a sample.
        Args:
            sample (dict): a dict of sample, eg: {'image':xx, 'label': xxx}
            context (dict): info about this sample processing
        Returns:
            result (dict): a processed sample
        """
        return sample

    def __str__(self):
        return '%s' % (self._id)


@register_op
class DecodeImage(BaseOperator):
    def __init__(self, to_rgb=True):
        """ Transform the image data to numpy format.
        Args:
            to_rgb (bool): confirm whether to convert BGR to RGB
        """
        super(DecodeImage, self).__init__()
        self.to_rgb = to_rgb
        if not isinstance(self.to_rgb, bool):
            raise TypeError('{}: the input type is error.'
                            .format(self.__str__))

    def __call__(self, sample, context=None):
        assert 'image' in sample, 'not found image data'
        im = sample['image']
        data = np.frombuffer(im, dtype='uint8')
        im = cv2.imdecode(data, 1)  # BGR mode, but need RGB mode
        if self.to_rgb:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        sample['image'] = im
        return sample


@register_op
class ResizeImage(BaseOperator):
    def __init__(self, target_size=0, max_size=0,
                 interp=cv2.INTER_LINEAR):
        """ 
        Args:
            target_size (int): the taregt size of image's short side
            max_size (int): the max size of image
            interp: the interpolation method
        """
        super(ResizeImage, self).__init__()
        self.target_size = target_size
        self.max_size = max_size
        self.interp = interp
        if not (isinstance(self.target_size, int)
                and isinstance(self.max_size, int)
                and isinstance(self.interp, int)):
            raise TypeError('{}: the input type is error.'
                            .format(self.__str__))

    def __call__(self, sample, context=None):
        """ Resise the image numpy.
        """
        im = sample['image']
        if not isinstance(im, np.ndarray):
            raise TypeError('{}: the image type is not numpy.'
                            .format(self.__str__))
        if len(im.shape) != 3:
            raise ImageError('{}: the image type is not \
                    three-dimensional.'.format(self.__str__))
        im_shape = im.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        if float(im_size_min) == 0:
            raise ZeroDivisionError('{}: the min size of image is \
                    zero.'.format(self.__str__))
        if self.max_size != 0:
            im_scale = float(self.target_size) / float(im_size_min)
            # Prevent the biggest axis from being more than max_size
            if np.round(im_scale * im_size_max) > self.max_size:
                im_scale = float(self.max_size) / float(im_size_max)
            im_scale_x = im_scale
            im_scale_y = im_scale
            sample['im_info'] = np.array([np.round(im_shape[0] * im_scale),
                                          np.round(im_shape[1] * im_scale),
                                          im_scale],
                                         dtype=np.float32)
        else:
            im_scale_x = float(self.target_size) / float(im_shape[1])
            im_scale_y = float(self.target_size) / float(im_shape[0])
        im = cv2.resize(
            im,
            None,
            None,
            fx=im_scale_x,
            fy=im_scale_y,
            interpolation=self.interp)
        sample['image'] = im
        return sample


@register_op
class RandFlipImage(BaseOperator):
    def __init__(self, prob=0.5, is_normalized=False,
                 is_mask_flip=False):
        """ 
        Args:
            prob (float): the probability of flipping image
            is_normalized (bool): whether the bbox scale to [0,1]
            is_mask_flip (bool): whether flip the segmentation
        """
        super(RandFlipImage, self).__init__()
        self.prob = prob
        self.is_normalized = is_normalized
        self.is_mask_flip = is_mask_flip
        if not (isinstance(self.prob, float)
                and isinstance(self.is_normalized, bool)
                and isinstance(self.is_mask_flip, bool)):
            raise TypeError('{}: the input type is error.'
                            .format(self.__str__))

    def flip_segms(self, segms, height, width):
        def _flip_poly(poly, width):
            flipped_poly = np.array(poly)
            flipped_poly[0::2] = width - np.array(poly[0::2]) - 1
            return flipped_poly.tolist()

        def _flip_rle(rle, height, width):
            if 'counts' in rle and type(rle['counts']) == list:
                rle = mask_util.frPyObjects([rle], height, width)
            mask = mask_util.decode(rle)
            mask = mask[:, ::-1, :]
            rle = mask_util.encode(np.array(mask, order='F', dtype=np.uint8))
            return rle

        def is_poly(segm):
            assert isinstance(segm, (list, dict)), \
                'Invalid segm type: {}'.format(type(segm))
            return isinstance(segm, list)

        flipped_segms = []
        for segm in segms:
            if is_poly(segm):
                # Polygon format
                flipped_segms.append([_flip_poly(poly, width)
                                      for poly in segm])
            else:
                # RLE format
                import pycocotools.mask as mask_util
                flipped_segms.append(_flip_rle(segm, height, width))
        return flipped_segms

    def __call__(self, sample, context=None):
        """Filp the image and bounding box.
        Operators:
            1. Flip the image numpy.
            2. Transform the bboxes' x coordinates.
              (Must judge whether the coordinates are normalized!)
            3. Transform the segmentations' x coordinates.
              (Must judge whether the coordinates are normalized!)
        Output:
            sample: the image, bounding box and segmentation part
                    in sample are flipped.
        """
        gt_bbox = sample['gt_bbox']
        im = sample['image']
        if not isinstance(im, np.ndarray):
            raise TypeError('{}: the image type is not numpy.'
                            .format(self.__str__))
        if len(im.shape) != 3:
            raise ImageError('{}: the image type is not \
                    three-dimensional.'.format(self.__str__))
        height, width, _ = im.shape
        if np.random.uniform(0, 1) > self.prob:
            im = im[:, ::-1, :]
            oldx1 = gt_bbox[:, 0].copy()
            oldx2 = gt_bbox[:, 2].copy()
            if self.is_normalized:
                gt_bbox[:, 0] = 1 - oldx2
                gt_bbox[:, 2] = 1 - oldx1
            else:
                gt_bbox[:, 0] = width - oldx2 - 1
                gt_bbox[:, 2] = width - oldx1 - 1
            if gt_bbox.shape[0] != 0 and (gt_bbox[:, 2] < gt_bbox[:, 0]).all():
                raise BboxError('{}: invalid coordinate for bounding box for \
                                the x2 isn\'t more than the x1!'
                                .format(self.__str__))
            sample['gt_bbox'] = gt_bbox
            if self.is_mask_flip and len(sample['gt_poly']) != 0:
                sample['gt_poly'] = self.flip_segms(sample['gt_poly'],
                                                    height, width)
            sample['flipped'] = True
        return sample


@register_op
class NormalizeImage(BaseOperator):
    def __init__(self, mean=[0.485, 0.456, 0.406],
                 std=[1, 1, 1],
                 is_scale=True):
        """ Normalize the image.
        Args:
            mean (list): the pixel mean
            std (list): the pixel variance
        """
        super(NormalizeImage, self).__init__()
        self.mean = mean
        self.std = std
        self.is_scale = is_scale
        if not (isinstance(self.mean, list)
                and isinstance(self.std, list)
                and isinstance(self.is_scale, bool)):
            raise TypeError('{}: the input type is error.'
                            .format(self.__str__))
        from functools import reduce
        if reduce(lambda x,y: x*y, self.std) == 0:
            raise ValueError('{}: the std is wrong!'.format(self.__str__))

    def __call__(self, sample, context=None):
        """
        Operators:
            1.(optional) Scale the image to [0,1]
            2. Each pixel minus mean and is divided by std
        """
        im = sample['image']
        im = im.astype(np.float32, copy=False)
        if self.is_scale:
            im = im / 255.0
        im -= self.mean
        im /= self.std
        sample['image'] = im
        return sample


@register_op
class NormalizeBox(BaseOperator):
    """Transform the bounding box's coornidates to [0,1]."""
    def __init__(self):
        super(NormalizeBox, self).__init__()

    def __call__(self, sample, context):
        gt_bbox = sample['gt_bbox']
        width = sample['w']
        height = sample['h']
        for i in range(gt_bbox.shape[0]):
            gt_bbox[i][0] = gt_bbox[i][0] / width
            gt_bbox[i][1] = gt_bbox[i][1] / height
            gt_bbox[i][2] = gt_bbox[i][2] / width
            gt_bbox[i][3] = gt_bbox[i][3] / height
        sample['gt_bbox'] = gt_bbox
        return sample


@register_op
class Bgr2Rgb(BaseOperator):
    def __init__(self, to_bgr=True,
                 channel_first=True):
        """ Change the channel and color space.
        Args:
            to_bgr (bool): confirm whether to convert RGB to BGR
            channel_first (bool): confirm whether to change channel
        """
        super(Bgr2Rgb, self).__init__()
        self.to_bgr = to_bgr
        self.channel_first = channel_first
        if not (isinstance(self.to_bgr, bool)
                and isinstance(self.channel_first, bool)):
            raise TypeError('{}: the input type is error.'
                            .format(self.__str__))

    def __call__(self, sample, context=None):
        assert 'image' in sample, 'not found image data'
        im = sample['image']
        if self.to_bgr:
            im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        if self.channel_first:
            im = im.transpose((2, 0, 1))
        sample['image'] = im
        return sample


@register_op
class ArrangeSample(BaseOperator):
    """Transform the sample dict to the sample tuple
       which the model need when training.
    """
    def __init__(self, is_mask=False):
        """ Get the standard output.
        Args:
            is_mask (bool): confirm whether to use mask rcnn
        """
        super(ArrangeSample, self).__init__()
        self.is_mask = is_mask
        if not (isinstance(self.is_mask, bool)):
            raise TypeError('{}: the input type is error.'
                            .format(self.__str__))

    def __call__(self, sample, context=None):
        """
        Input:
            sample: a dict which contains image
                    info and annotation info.
            context: a dict which contains additional info.
        Output:
            sample: a tuple which contains the
                    info which training model need.
                    tupe is (image, gt_bbox, gt_class, is_crowd, im_info, gt_masks)
        """
        im = sample['image']
        gt_bbox = sample['gt_bbox']
        gt_class = sample['gt_class']
        outs = (im, gt_bbox, gt_class)
        keys = list(sample.keys())
        if 'is_crowd' in keys:
            is_crowd = sample['is_crowd']
            outs = outs + (is_crowd,)
        if 'im_info' in keys:
            im_info = sample['im_info']
            outs = outs + (im_info,)
        im_id = sample['im_id']
        outs = outs + (im_id,)
        gt_masks = []
        if self.is_mask and len(sample['gt_poly']) != 0 \
                and 'is_crowd' in keys:
            valid = True
            segms = sample['gt_poly']
            assert len(segms) == is_crowd.shape[0]
            for i in range(len(sample['gt_poly'])):
                segm, iscrowd = segms[i], is_crowd[i]
                gt_segm = []
                if iscrowd:
                    gt_segm.append([[0, 0]])
                else:
                    for poly in segm:
                        if len(poly) == 0:
                            valid = False
                            break
                        gt_segm.append(np.array(poly).reshape(-1, 2))
                if (not valid) or len(gt_segm) == 0:
                    break
                gt_masks.append(gt_segm)
            outs = outs + (gt_masks, )
        return outs
