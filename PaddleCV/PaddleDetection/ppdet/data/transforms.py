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

from __future__ import division

try:
    from collections.abc import Sequence
except Exception:
    from collections import Sequence

from numbers import Number
import numpy as np
import cv2


__all__ = ['RandomFlip', 'RandomExpand', 'RandomCrop', 'ColorDistort',
           'MixUp', 'Resize', 'NormalizePermute', 'NormalizeLabels',
           'PadToStride']


class Resize(object):
    def __init__(self,
                 resize_shorter=None,
                 resize_longer=None,
                 target_dim=[],
                 interp=cv2.INTER_LINEAR,
                 scale_box=None,
                 force_pil=False):
        super(Resize, self).__init__()
        self.resize_shorter = resize_shorter
        self.resize_longer = resize_longer
        self.target_dim = target_dim
        self.interp = interp  # 'random' for yolov3
        self.scale_box = scale_box
        self.force_pil = force_pil

    @property
    def batch_seed(self):
        return isinstance(self.target_dim, Sequence)

    def __call__(self, sample):
        w = sample['width']
        h = sample['height']

        interp = self.interp
        if interp == 'random':
            interp = np.random.choice(range(5))

        if self.target_dim:
            assert (self.resize_shorter is None
                    and self.resize_longer is None), \
                "do not set both target_dim and resize_shorter/resize_longer"
            if isinstance(self.target_dim, Sequence):
                assert 'batch_seed' in sample, \
                    "random target_dim requires batch_seed"
                seed = sample['batch_seed']
                dim = np.random.RandomState(seed).choice(self.target_dim)
            else:
                dim = self.target_dim
            resize_w = resize_h = dim
            scale_x = dim / w
            scale_y = dim / h
            # XXX default to scale bbox for YOLO and SSD
            if 'gt_box' in sample and len(sample['gt_box']) > 0:
                if self.scale_box or self.scale_box is None:
                    scale_array = np.array([scale_x, scale_y] * 2,
                                           dtype=np.float32)
                    sample['gt_box'] = np.clip(
                        sample['gt_box'] * scale_array, 0, dim - 1)
        else:
            resize_shorter = self.resize_shorter
            if isinstance(self.resize_shorter, Sequence):
                resize_shorter = np.random.choice(resize_shorter)

            dim_max, dim_min = w > h and (w, h) or (h, w)
            scale = min(self.resize_longer / dim_max, resize_shorter / dim_min)
            resize_w = int(round(w * scale))
            resize_h = int(round(h * scale))
            sample['scale'] = scale
            # XXX this is for RCNN, scaling bbox by default
            # commonly the labels (bboxes and masks) are scaled by the
            # dataloader, but somehow Paddle choose to do it later.
            # This is why we need to pass "scale" around, and this also results
            # in some other caveats, e.g., all transformations that modify
            # bboxes (currently `RandomFlip`) must be applied BEFORE `Resize`.
            if 'gt_box' in sample and len(sample['gt_box']) > 0:
                if self.scale_box:
                    scale_array = np.array([scale, scale] * 2,
                                           dtype=np.float32)
                    sample['gt_box'] = np.clip(
                        sample['gt_box'] * scale_array, 0, dim - 1)

        if self.force_pil:
            from PIL import Image
            img = Image.fromarray(sample['image'])
            img = img.resize((resize_w, resize_h), interp)
            img = np.array(img)
            sample['image'] = img
        else:
            sample['image'] = cv2.resize(
                sample['image'], (resize_w, resize_h), interpolation=interp)
        sample['width'] = resize_w
        sample['height'] = resize_h
        return sample


class RandomFlip(object):
    def __init__(self, prob=.5):
        super(RandomFlip, self).__init__()
        self.prob = prob

    def __call__(self, sample):
        if np.random.uniform(0., 1.) < self.prob:
            return sample

        img = sample['image']
        sample['image'] = img[:, ::-1, :]
        w = sample['width']

        if 'gt_box' in sample and len(sample['gt_box']) > 0:
            swap = sample['gt_box'].copy()
            sample['gt_box'][:, 0] = w - swap[:, 2] - 1
            sample['gt_box'][:, 2] = w - swap[:, 0] - 1

        if 'gt_poly' in sample:
            for poly in sample['gt_poly']:
                for p in poly:
                    p[:, 0] = w - p[:, 0] - 1
        return sample


class ColorDistort(object):
    def __init__(self,
                 hue=[-18, 18, 0.5],
                 saturation=[0.5, 1.5, 0.5],
                 contrast=[0.5, 1.5, 0.5],
                 brightness=[0.5, 1.5, 0.5],
                 random_apply=True):
        super(ColorDistort, self).__init__()
        self.hue = hue
        self.saturation = saturation
        self.contrast = contrast
        self.brightness = brightness
        self.random_apply = random_apply

    def apply_hue(self, img):
        low, high, prob = self.hue
        if np.random.uniform(0., 1.) < prob:
            return img

        img = img.astype(np.float32)

        # XXX works, but result differ from HSV version
        delta = np.random.uniform(low, high)
        u = np.cos(delta * np.pi)
        w = np.sin(delta * np.pi)
        bt = np.array([[1.0, 0.0, 0.0],
                       [0.0, u, -w],
                       [0.0, w, u]])
        tyiq = np.array([[0.299, 0.587, 0.114],
                         [0.596, -0.274, -0.321],
                         [0.211, -0.523, 0.311]])
        ityiq = np.array([[1.0, 0.956, 0.621],
                          [1.0, -0.272, -0.647],
                          [1.0, -1.107, 1.705]])
        t = np.dot(np.dot(ityiq, bt), tyiq).T
        img = np.dot(img, t)
        return img

    def apply_saturation(self, img):
        low, high, prob = self.saturation
        if np.random.uniform(0., 1.) < prob:
            return img
        delta = np.random.uniform(low, high)

        img = img.astype(np.float32)
        gray = img * np.array([[[0.299, 0.587, 0.114]]], dtype=np.float32)
        gray = gray.sum(axis=2, keepdims=True)
        gray *= (1.0 - delta)
        img *= delta
        img += gray
        return img

    def apply_contrast(self, img):
        low, high, prob = self.contrast
        if np.random.uniform(0., 1.) < prob:
            return img
        delta = np.random.uniform(low, high)

        img = img.astype(np.float32)
        img *= delta
        return img

    def apply_brightness(self, img):
        low, high, prob = self.brightness
        if np.random.uniform(0., 1.) < prob:
            return img
        delta = np.random.uniform(low, high)

        img = img.astype(np.float32)
        img += delta
        return img

    def __call__(self, sample):
        img = sample['image']
        if self.random_apply:
            distortions = np.random.permutation([
                self.apply_brightness,
                self.apply_contrast,
                self.apply_saturation,
                self.apply_hue
            ])
            for func in distortions:
                img = func(img)
            sample['image'] = img
            return sample

        img = self.apply_brightness(img)

        if np.random.randint(0, 2):
            img = self.apply_contrast(img)
            img = self.apply_saturation(img)
            img = self.apply_hue(img)
        else:
            img = self.apply_saturation(img)
            img = self.apply_hue(img)
            img = self.apply_contrast(img)
        sample['image'] = img
        return sample


class NormalizePermute(object):
    def __init__(self,
                 mean=[123.675, 116.28, 103.53],
                 std=[58.395, 57.120, 57.375]):
        super(NormalizePermute, self).__init__()
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img = sample['image']
        img = img.astype(np.float32)

        img = img.transpose((2, 0, 1))
        mean = np.array(self.mean, dtype=np.float32)
        std = np.array(self.std, dtype=np.float32)
        invstd = 1. / std
        for v, m, s in zip(img, mean, invstd):
            v.__isub__(m).__imul__(s)
        sample['image'] = img
        return sample


class RandomExpand(object):
    def __init__(self, ratio=4., prob=0.5, fill_value=(127.5,) * 3):
        super(RandomExpand, self).__init__()
        assert ratio > 1.01, "expand ratio must be larger than 1.01"
        self.ratio = ratio
        self.prob = prob
        assert isinstance(fill_value, (Number, Sequence)), \
            "fill value must be either float or sequence"
        if isinstance(fill_value, Number):
            fill_value = (fill_value,) * 3
        if not isinstance(fill_value, tuple):
            fill_value = tuple(fill_value)
        self.fill_value = fill_value

    def __call__(self, sample):
        if np.random.uniform(0., 1.) < self.prob:
            return sample

        img = sample['image']
        height = sample['height']
        width = sample['width']

        expand_ratio = np.random.uniform(1., self.ratio)
        h = int(height * expand_ratio)
        w = int(width * expand_ratio)
        if not h > height or not w > width:
            return sample
        y = np.random.randint(0, h - height)
        x = np.random.randint(0, w - width)
        canvas = np.ones((h, w, 3), dtype=np.uint8)
        canvas *= np.array(self.fill_value, dtype=np.uint8)
        canvas[y:y + height, x:x + width, :] = img.astype(np.uint8)

        sample['height'] = h
        sample['width'] = w
        sample['image'] = canvas
        if 'gt_box' in sample and len(sample['gt_box']) > 0:
            sample['gt_box'] += np.array([x, y] * 2, dtype=np.float32)
        return sample


class RandomCrop(object):
    def __init__(self,
                 aspect_ratio=[.5, 2.],
                 thresholds=[.0, .1, .3, .5, .7, .9],
                 scaling=[.3, 1.],
                 num_attempts=50,
                 allow_no_crop=True,
                 cover_all_box=False):
        super(RandomCrop, self).__init__()
        self.aspect_ratio = aspect_ratio
        self.thresholds = thresholds
        self.scaling = scaling
        self.num_attempts = num_attempts
        self.allow_no_crop = allow_no_crop
        self.cover_all_box = cover_all_box

    def __call__(self, sample):
        if 'gt_box' in sample and len(sample['gt_box']) == 0:
            return sample

        h = sample['height']
        w = sample['width']
        gt_box = sample['gt_box']

        # NOTE Original method attempts to generate one candidate for each
        # threshold then randomly sample one from the resulting list.
        # Here a short circuit approach is taken, i.e., randomly choose a
        # threshold and attempt to find a valid crop, and simply return the
        # first one found.
        # The probability is not exactly the same, kinda resembling the
        # "Monty Hall" problem. Actually carrying out the attempts will affect
        # observability (just like opening doors in the "Monty Hall" game).
        thresholds = list(self.thresholds)
        if self.allow_no_crop:
            thresholds.append('no_crop')
        np.random.shuffle(thresholds)

        for thresh in thresholds:
            if thresh == 'no_crop':
                return sample

            found = False
            for i in range(self.num_attempts):
                scale = np.random.uniform(*self.scaling)
                min_ar, max_ar = self.aspect_ratio
                aspect_ratio = np.random.uniform(max(min_ar, scale**2),
                                                 min(max_ar, scale**-2))
                crop_h = int(h * scale / np.sqrt(aspect_ratio))
                crop_w = int(w * scale * np.sqrt(aspect_ratio))
                crop_y = np.random.randint(0, h - crop_h)
                crop_x = np.random.randint(0, w - crop_w)
                crop_box = [crop_x, crop_y, crop_x + crop_w, crop_y + crop_h]
                iou = self._iou_matrix(gt_box,
                                       np.array([crop_box], dtype=np.float32))
                if iou.max() < thresh:
                    continue

                if self.cover_all_box and iou.min() < thresh:
                    continue

                cropped_box, valid_ids = self._crop_box_with_center_constraint(
                    gt_box, np.array(crop_box, dtype=np.float32))
                if valid_ids.size > 0:
                    found = True
                    break

            if found:
                sample['image'] = self._crop_image(sample['image'], crop_box)
                sample['gt_box'] = np.take(cropped_box, valid_ids, axis=0)
                sample['gt_label'] = np.take(
                    sample['gt_label'], valid_ids, axis=0)
                sample['width'] = crop_box[2] - crop_box[0]
                sample['height'] = crop_box[3] - crop_box[1]
                if 'gt_score' in sample:
                    sample['gt_score'] = np.take(
                        sample['gt_score'], valid_ids, axis=0)
                return sample

        return sample

    def _iou_matrix(self, a, b):
        tl_i = np.maximum(a[:, np.newaxis, :2], b[:, :2])
        br_i = np.maximum(a[:, np.newaxis, 2:], b[:, 2:])

        area_i = np.prod(br_i - tl_i, axis=2) * (tl_i < br_i).all(axis=2)
        area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
        area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
        area_o = (area_a[:, np.newaxis] + area_b - area_i)
        return area_i / (area_o + 1e-10)

    def _crop_box_with_center_constraint(self, box, crop):
        cropped_box = box.copy()

        cropped_box[:, :2] = np.maximum(box[:, :2], crop[:2])
        cropped_box[:, 2:] = np.minimum(box[:, 2:], crop[2:])
        cropped_box[:, :2] -= crop[:2]
        cropped_box[:, 2:] -= crop[:2]

        centers = (box[:, :2] + box[:, 2:]) / 2
        valid = np.logical_and(
            crop[:2] <= centers, centers < crop[2:]).all(axis=1)
        valid = np.logical_and(
            valid, (cropped_box[:, :2] < cropped_box[:, 2:]).all(axis=1))

        return cropped_box, np.where(valid)[0]

    def _crop_image(self, img, crop):
        x1, y1, x2, y2 = crop
        return img[y1:y2, x1:x2, :]


class MixUp(object):
    def __init__(self, alpha=1.5, beta=1.5, steps=-1):
        super(MixUp, self).__init__()
        assert alpha > 0., "alpha should be positive"
        assert beta > 0., "beta should be positive"
        self.alpha = alpha
        self.beta = beta
        self.steps = steps
        self.is_mixup = True

    def __call__(self, sample1, sample2):
        factor = np.clip(np.random.beta(self.alpha, self.beta), 0., 1.)
        if factor == 1.:
            return sample1
        if factor == 0.:
            return sample2

        gt_box1, gt_box2 = sample1['gt_box'], sample2['gt_box']
        gt_label1, gt_label2 = sample1['gt_label'], sample2['gt_label']
        gt_score1, gt_score2 = sample1['gt_score'], sample2['gt_score']
        gt_box = np.concatenate((gt_box1, gt_box2), axis=0)
        gt_label = np.concatenate((gt_label1, gt_label2), axis=0)
        gt_score = np.concatenate((gt_score1 * factor,
                                   gt_score2 * (1. - factor)), axis=0)

        img1, img2 = sample1['image'], sample2['image']
        h1, w1, _ = img1.shape
        h2, w2, _ = img2.shape
        w = max(w1, w2)
        h = max(h1, h2)

        img1 = img1.astype(np.float32)
        img2 = img2.astype(np.float32)

        canvas = np.zeros((h, w, 3), dtype=np.float32)
        canvas[:h1, :w1, :] = img1 * factor
        canvas[:h2, :w2, :] += img2 * (1. - factor)

        sample1['image'] = canvas
        sample1['gt_box'] = gt_box
        sample1['gt_label'] = gt_label
        sample1['gt_score'] = gt_score
        sample1['width'] = w
        sample1['height'] = h

        return sample1


class NormalizeLabels(object):
    def __init__(
            self, num_instances=None, normalize_box=False, to_center=False):
        super(NormalizeLabels, self).__init__()
        self.num_instances = num_instances
        self.normalize_box = normalize_box
        self.to_center = to_center

    def corner_to_center(self, box):
        box[:, 2:] = box[:, 2:] - box[:, :2]
        box[:, :2] = box[:, :2] + box[:, 2:] / 2.
        return box

    def __call__(self, sample):
        if self.num_instances is None:
            sample['gt_label'] = sample['gt_label'].squeeze(-1)
            if 'gt_score' in sample:
                sample['gt_score'] = sample['gt_score'].squeeze(-1)

        if 'gt_box' in sample and len(sample['gt_box']) == 0:
            if self.num_instances is None:
                return sample
            sample['gt_box'] = np.zeros(
                [self.num_instances, 4], dtype=np.float32)
            sample['gt_label'] = np.zeros(
                [self.num_instances, 1], dtype=np.int32)
            if 'gt_score' in sample:
                sample['gt_score'] = np.zeros(
                    [self.num_instances, 1], dtype=np.float32)
            return sample

        if self.normalize_box:
            w = sample['width']
            h = sample['height']
            sample['gt_box'] /= np.array([w, h] * 2, dtype=np.float32)

        if self.to_center:
            sample['gt_box'] = self.corner_to_center(sample['gt_box'])

        if self.num_instances is None:
            return sample

        # cap then pad labels, also squeeze `gt_label` and `gt_score`
        gt_box = sample['gt_box'][:self.num_instances, :]
        gt_label = sample['gt_label'][:self.num_instances, 0]
        pad = self.num_instances - gt_label.size
        gt_box_padded = np.pad(gt_box, ((0, pad), (0, 0)), mode='constant')
        gt_label_padded = np.pad(gt_label, [(0, pad)], mode='constant')
        sample['gt_box'] = gt_box_padded
        sample['gt_label'] = gt_label_padded

        if 'gt_score' in sample:
            gt_score = sample['gt_score'][:self.num_instances, 0]
            gt_score_padded = np.pad(gt_score, [(0, pad)], mode='constant')
            sample['gt_score'] = gt_score_padded

        return sample


class PadToStride(object):
    def __init__(self, stride=1):
        super(PadToStride, self).__init__()
        assert stride > 0, "stride must be greater than zero"
        self.stride = stride

    def __call__(self, batch):
        images = batch['image']
        assert isinstance(images[0], np.ndarray), "images must be ndarrays"

        batch_size = len(images)
        if batch_size == 1 and self.stride == 1:
            batch['padded_height'] = images[0].shape[0]
            batch['padded_width'] = images[0].shape[1]
            return batch
        dims = [i.shape for i in images]
        hs = [dim[1] for dim in dims]
        ws = [dim[2] for dim in dims]
        pad_h = max(hs)
        pad_w = max(ws)
        pad_h = ((pad_h + self.stride - 1) // self.stride) * self.stride
        pad_w = ((pad_w + self.stride - 1) // self.stride) * self.stride
        chan = dims[0][0]

        if all([h == pad_h for h in hs]) and all([w == pad_w for w in ws]):
            batch['padded_height'] = np.array(hs)
            batch['padded_width'] = np.array(ws)
            return batch

        padded = np.zeros((batch_size, chan, pad_h, pad_w), dtype=np.float32)
        for idx, img in enumerate(images):
            padded[idx, :, :hs[idx], :ws[idx]] = img

        batch['image'] = padded
        batch['padded_height'] = np.array([pad_h] * batch_size)
        batch['padded_width'] = np.array([pad_w] * batch_size)
        return batch
