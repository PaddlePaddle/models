#copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

import os
import math
import random
import functools
import numpy as np
import cv2
import logging
import imghdr

import paddle
from paddle import fluid
from utils.autoaugment import ImageNetPolicy
from PIL import Image

policy = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

random.seed(0)
np.random.seed(0)


def rotate_image(img):
    """rotate image

    Args:
        img: image data

    Returns:
        rotated image data
    """
    (h, w) = img.shape[:2]
    center = (w / 2, h / 2)
    angle = np.random.randint(-10, 11)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h))
    return rotated


def random_crop(img, size, settings, scale=None, ratio=None,
                interpolation=None):
    """random crop image
        
    Args:
        img: image data
        size: crop size
        settings: arguments
        scale: scale parameter
        ratio: ratio parameter

    Returns:
        random cropped image data
    """
    lower_scale = settings.lower_scale
    lower_ratio = settings.lower_ratio
    upper_ratio = settings.upper_ratio
    scale = [lower_scale, 1.0] if scale is None else scale
    ratio = [lower_ratio, upper_ratio] if ratio is None else ratio

    aspect_ratio = math.sqrt(np.random.uniform(*ratio))
    w = 1. * aspect_ratio
    h = 1. / aspect_ratio

    bound = min((float(img.shape[0]) / img.shape[1]) / (h**2),
                (float(img.shape[1]) / img.shape[0]) / (w**2))

    scale_max = min(scale[1], bound)
    scale_min = min(scale[0], bound)

    target_area = img.shape[0] * img.shape[1] * np.random.uniform(scale_min,
                                                                  scale_max)
    target_size = math.sqrt(target_area)
    w = int(target_size * w)
    h = int(target_size * h)
    i = np.random.randint(0, img.shape[0] - h + 1)
    j = np.random.randint(0, img.shape[1] - w + 1)

    img = img[i:i + h, j:j + w, :]

    if interpolation:
        resized = cv2.resize(img, (size, size), interpolation=interpolation)
    else:
        resized = cv2.resize(img, (size, size))
    return resized


#NOTE:(2019/08/08) distort color func is not implemented
def distort_color(img):
    """distort image color

    Args:
        img: image data

    Returns:
        distorted color image data
    """
    return img


def resize_short(img, target_size, interpolation=None):
    """resize image
    
    Args:
        img: image data
        target_size: resize short target size
        interpolation: interpolation mode

    Returns:
        resized image data
    """
    percent = float(target_size) / min(img.shape[0], img.shape[1])
    resized_width = int(round(img.shape[1] * percent))
    resized_height = int(round(img.shape[0] * percent))
    if interpolation:
        resized = cv2.resize(
            img, (resized_width, resized_height), interpolation=interpolation)
    else:
        resized = cv2.resize(img, (resized_width, resized_height))
    return resized


def crop_image(img, target_size, center):
    """crop image 
    
    Args:
        img: images data
        target_size: crop target size
        center: crop mode
    
    Returns:
        img: cropped image data
    """
    height, width = img.shape[:2]
    size = target_size
    if center == True:
        w_start = (width - size) // 2
        h_start = (height - size) // 2
    else:
        w_start = np.random.randint(0, width - size + 1)
        h_start = np.random.randint(0, height - size + 1)
    w_end = w_start + size
    h_end = h_start + size
    img = img[h_start:h_end, w_start:w_end, :]
    return img


def create_mixup_reader(settings, rd):
    """
    """

    class context:
        tmp_mix = []
        tmp_l1 = []
        tmp_l2 = []
        tmp_lam = []

    alpha = settings.mixup_alpha

    def fetch_data():
        for item in rd():
            yield item

    def mixup_data():
        for data_list in fetch_data():
            if alpha > 0.:
                lam = np.random.beta(alpha, alpha)
            else:
                lam = 1.
            l1 = np.array(data_list)
            l2 = np.random.permutation(l1)
            mixed_l = [
                l1[i][0] * lam + (1 - lam) * l2[i][0] for i in range(len(l1))
            ]
            yield (mixed_l, l1, l2, lam)

    def mixup_reader():
        for context.tmp_mix, context.tmp_l1, context.tmp_l2, context.tmp_lam in mixup_data(
        ):
            for i in range(len(context.tmp_mix)):
                mixed_l = context.tmp_mix[i]
                l1 = context.tmp_l1[i]
                l2 = context.tmp_l2[i]
                lam = context.tmp_lam
                yield (mixed_l, int(l1[1]), int(l2[1]), float(lam))

    return mixup_reader


def process_image(sample, settings, mode, color_jitter, rotate):
    """ process_image """

    mean = settings.image_mean
    std = settings.image_std
    crop_size = settings.image_shape[1]

    img_path = sample[0]
    img = cv2.imread(img_path)

    if img is None:
        logger.warning("img({0}) is None, pass it.".format(img_path))
        return None

    if mode == 'train':
        if rotate:
            img = rotate_image(img)
        if crop_size > 0:
            img = random_crop(
                img, crop_size, settings, interpolation=settings.interpolation)
        if color_jitter:
            img = distort_color(img)
        if np.random.randint(0, 2) == 1:
            img = img[:, ::-1, :]
    else:
        if crop_size > 0:
            target_size = settings.resize_short_size
            img = resize_short(
                img, target_size, interpolation=settings.interpolation)
            img = crop_image(img, target_size=crop_size, center=True)

    img = img[:, :, ::-1]

    if 'use_aa' in settings and settings.use_aa and mode == 'train':
        img = np.ascontiguousarray(img)
        img = Image.fromarray(img)
        img = policy(img)
        img = np.asarray(img)

    img = img.astype('float32').transpose((2, 0, 1)) / 255
    img_mean = np.array(mean).reshape((3, 1, 1))
    img_std = np.array(std).reshape((3, 1, 1))
    img -= img_mean
    img /= img_std
    # doing training (train.py)
    if mode == 'train' or (mode == 'val' and
                           not hasattr(settings, 'save_json_path')):
        return (img, sample[1])
    #doing testing (eval.py)
    elif mode == 'val' and hasattr(settings, 'save_json_path'):
        return (img, sample[1], sample[0])
    #doing predict (infer.py)
    elif mode == 'test':
        return (img, sample[0])
    else:
        raise Exception("mode not implemented")


def process_batch_data(input_data, settings, mode, color_jitter, rotate):
    batch_data = []
    for sample in input_data:
        if os.path.isfile(sample[0]):
            tmp_data = process_image(sample, settings, mode, color_jitter,
                                     rotate)
            if tmp_data is None:
                continue
            batch_data.append(tmp_data)
        else:
            logger.info("File not exist : {0}".format(sample[0]))
    return batch_data


class ImageNetReader:
    def __init__(self, seed=None):
        self.shuffle_seed = seed

    def set_shuffle_seed(self, seed):
        assert isinstance(seed, int), "shuffle seed must be int"
        self.shuffle_seed = seed

    def _get_single_card_bs(self, settings, mode):
        if settings.use_gpu:
            if mode == "val" and hasattr(settings, "test_batch_size"):
                single_card_bs = int(
                    settings.test_batch_size
                ) // paddle.fluid.core.get_cuda_device_count()
            else:
                single_card_bs = int(
                    settings.
                    batch_size) // paddle.fluid.core.get_cuda_device_count()
        else:
            if mode == "val" and hasattr(settings, "test_batch_size"):
                single_card_bs = int(settings.test_batch_size) // int(
                    os.environ.get('CPU_NUM', 1))
            else:
                single_card_bs = int(settings.batch_size) // int(
                    os.environ.get('CPU_NUM', 1))
        return single_card_bs

    def _reader_creator(self,
                        settings,
                        file_list,
                        mode,
                        shuffle=False,
                        color_jitter=False,
                        rotate=False,
                        data_dir=None):
        num_trainers = int(os.environ.get('PADDLE_TRAINERS_NUM', 1))

        batch_size = self._get_single_card_bs(settings, mode)

        def reader():
            def read_file_list():
                with open(file_list) as flist:
                    full_lines = [line.strip() for line in flist]
                    if mode != "test" and len(full_lines) < settings.batch_size:
                        logger.error(
                            "Error: The number of the whole data ({}) is smaller than the batch_size ({}), and drop_last is turnning on, so nothing  will feed in program, Terminated now. Please reset batch_size to a smaller number or feed more data!".
                            format(len(full_lines), settings.batch_size))
                        os._exit(1)
                    if num_trainers > 1 and mode == "train":
                        assert self.shuffle_seed is not None, "multiprocess train, shuffle seed must be set!"
                        np.random.RandomState(self.shuffle_seed).shuffle(
                            full_lines)
                    elif shuffle:
                        if not settings.enable_ce or not settings.same_feed:
                            np.random.shuffle(full_lines)

                batch_data = []
                if (mode == "train" or mode == "val") and settings.same_feed:
                    temp_file = full_lines[0]
                    logger.info("Same images({},nums:{}) will feed in the net".
                                format(str(temp_file), settings.same_feed))
                    full_lines = []
                    for i in range(settings.same_feed):
                        full_lines.append(temp_file)

                for line in full_lines:
                    img_path, label = line.split()
                    img_path = os.path.join(data_dir, img_path)
                    batch_data.append([img_path, int(label)])
                    if len(batch_data) == batch_size:
                        if mode == 'train' or mode == 'val' or mode == 'test':
                            yield batch_data
                        batch_data = []

            return read_file_list

        data_reader = reader()

        if mode == 'train' and num_trainers > 1:
            assert self.shuffle_seed is not None, \
                "If num_trainers > 1, the shuffle_seed must be set, because " \
                "the order of batch data generated by reader " \
                "must be the same in the respective processes."
            data_reader = paddle.fluid.contrib.reader.distributed_batch_reader(
                data_reader)

        mapper = functools.partial(
            process_batch_data,
            settings=settings,
            mode=mode,
            color_jitter=color_jitter,
            rotate=rotate)

        return fluid.io.xmap_readers(
            mapper,
            data_reader,
            settings.reader_thread,
            settings.reader_buf_size,
            order=False)

    def train(self, settings):
        """Create a reader for trainning

        Args:
            settings: arguments

        Returns:
            train reader
        """
        file_list = os.path.join(settings.data_dir, 'train_list.txt')
        assert os.path.isfile(
            file_list), "{} doesn't exist, please check data list path".format(
                file_list)

        if 'use_aa' in settings and settings.use_aa:
            global policy
            policy = ImageNetPolicy()

        reader = self._reader_creator(
            settings,
            file_list,
            'train',
            shuffle=True,
            color_jitter=False,
            rotate=False,
            data_dir=settings.data_dir)

        if settings.use_mixup == True:
            reader = create_mixup_reader(settings, reader)
            reader = fluid.io.batch(
                reader,
                batch_size=int(settings.batch_size /
                               paddle.fluid.core.get_cuda_device_count()),
                drop_last=True)
        return reader

    def val(self, settings):
        """Create a reader for eval

        Args:
            settings: arguments

        Returns:
            eval reader
        """

        file_list = os.path.join(settings.data_dir, 'val_list.txt')
        assert os.path.isfile(
            file_list), "{} doesn't exist, please check data list path".format(
                file_list)
        return self._reader_creator(
            settings,
            file_list,
            'val',
            shuffle=False,
            data_dir=settings.data_dir)

    def test(self, settings):
        """Create a reader for testing

        Args:
            settings: arguments

        Returns:
            test reader
        """
        file_list = ".tmp.txt"
        imgType_list = {'jpg', 'bmp', 'png', 'jpeg', 'rgb', 'tif', 'tiff'}
        with open(file_list, "w") as fout:
            if settings.image_path:
                fout.write(settings.image_path + " 0" + "\n")
                settings.batch_size = 1
                settings.data_dir = ""
            else:
                tmp_file_list = os.listdir(settings.data_dir)
                for file_name in tmp_file_list:
                    file_path = os.path.join(settings.data_dir, file_name)
                    if imghdr.what(file_path) not in imgType_list:
                        continue
                    fout.write(file_name + " 0" + "\n")
        return self._reader_creator(
            settings,
            file_list,
            'test',
            shuffle=False,
            data_dir=settings.data_dir)
