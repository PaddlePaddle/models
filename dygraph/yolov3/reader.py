# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved
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
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import os
import sys
import random
import time
import copy
import cv2
import box_utils
import image_utils
from pycocotools.coco import COCO
from data_utils import GeneratorEnqueuer
from config import cfg
import paddle.fluid as fluid

class DataSetReader(object):
    """A class for parsing and read COCO dataset"""

    def __init__(self):
        self.has_parsed_categpry = False

    def _parse_dataset_dir(self, mode):
        if 'coco2014' in cfg.dataset:
            cfg.train_file_list = 'annotations/instances_train2014.json'
            cfg.train_data_dir = 'train2014'
            cfg.val_file_list = 'annotations/instances_val2014.json'
            cfg.val_data_dir = 'val2014'
        elif 'coco2017' in cfg.dataset:
            cfg.train_file_list = 'annotations/instances_train2017.json'
            cfg.train_data_dir = 'train2017'
            cfg.val_file_list = 'annotations/instances_val2017.json'
            cfg.val_data_dir = 'val2017'
        else:
            raise NotImplementedError('Dataset {} not supported'.format(
                cfg.dataset))

        if mode == 'train':
            cfg.train_file_list = os.path.join(cfg.data_dir,
                                               cfg.train_file_list)
            cfg.train_data_dir = os.path.join(cfg.data_dir, cfg.train_data_dir)
            self.COCO = COCO(cfg.train_file_list)
            self.img_dir = cfg.train_data_dir
        elif mode == 'test' or mode == 'infer':
            cfg.val_file_list = os.path.join(cfg.data_dir, cfg.val_file_list)
            cfg.val_data_dir = os.path.join(cfg.data_dir, cfg.val_data_dir)
            self.COCO = COCO(cfg.val_file_list)
            self.img_dir = cfg.val_data_dir

    def _parse_dataset_catagory(self):
        self.categories = self.COCO.loadCats(self.COCO.getCatIds())
        self.num_category = len(self.categories)
        self.label_names = []
        self.label_ids = []
        for category in self.categories:
            self.label_names.append(category['name'])
            self.label_ids.append(int(category['id']))
        self.category_to_id_map = {v: i for i, v in enumerate(self.label_ids)}
        print("Load in {} categories.".format(self.num_category))
        if self.num_category != cfg.class_num:
            raise ValueError("category number({}) in your dataset is not equal "
                    "to --class_num={} settting, which may incur errors in "
                    "eval/infer or cause precision loss.".format(
                        self.num_category, cfg.class_num))
        self.has_parsed_categpry = True

    def get_label_infos(self):
        if not self.has_parsed_categpry:
            self._parse_dataset_dir("test")
            self._parse_dataset_catagory()
        return (self.label_names, self.label_ids)

    def _parse_gt_annotations(self, img):
        img_height = img['height']
        img_width = img['width']
        anno = self.COCO.loadAnns(
            self.COCO.getAnnIds(
                imgIds=img['id'], iscrowd=None))
        gt_index = 0
        for target in anno:
            if target['area'] < cfg.gt_min_area:
                continue
            if 'ignore' in target and target['ignore']:
                continue

            box = box_utils.coco_anno_box_to_center_relative(
                target['bbox'], img_height, img_width)
            if box[2] <= 0 and box[3] <= 0:
                continue

            img['gt_boxes'][gt_index] = box
            img['gt_labels'][gt_index] = \
                self.category_to_id_map[target['category_id']]
            gt_index += 1
            if gt_index >= cfg.max_box_num:
                break

    def _parse_images(self, is_train):
        image_ids = self.COCO.getImgIds()
        image_ids.sort()
        imgs = copy.deepcopy(self.COCO.loadImgs(image_ids))
        for img in imgs:
            img['image'] = os.path.join(self.img_dir, img['file_name'])
            assert os.path.exists(img['image']), \
                    "image {} not found.".format(img['image'])
            box_num = cfg.max_box_num
            img['gt_boxes'] = np.zeros((cfg.max_box_num, 4), dtype=np.float32)
            img['gt_labels'] = np.zeros((cfg.max_box_num), dtype=np.int32)
            for k in ['date_captured', 'url', 'license', 'file_name']:
                if k in img:
                    del img[k]
            if is_train:
                self._parse_gt_annotations(img)

        print("Loaded {0} images from {1}.".format(len(imgs), cfg.dataset))

        return imgs

    def _parse_images_by_mode(self, mode):
        if mode == 'infer':
            return []
        else:
            return self._parse_images(is_train=(mode == 'train'))

    def get_reader(self,
                   mode,
                   size=416,
                   batch_size=None,
                   shuffle=False,
                   shuffle_seed=None,
                   mixup_iter=0,
                   max_iter=0,
                   random_sizes=[],
                   image=None):
        assert mode in ['train', 'test', 'infer'], "Unknow mode type!"
        if mode != 'infer':
            assert batch_size is not None, \
                "batch size connot be None in mode {}".format(mode)
            self._parse_dataset_dir(mode)
            self._parse_dataset_catagory()

        def img_reader(img, size, mean, std):
            im_path = img['image']
            im = cv2.imread(im_path).astype('float32')
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

            h, w, _ = im.shape
            im_scale_x = size / float(w)
            im_scale_y = size / float(h)
            out_img = cv2.resize(
                im,
                None,
                None,
                fx=im_scale_x,
                fy=im_scale_y,
                interpolation=cv2.INTER_CUBIC)
            mean = np.array(mean).reshape((1, 1, -1))
            std = np.array(std).reshape((1, 1, -1))
            out_img = (out_img / 255.0 - mean) / std
            out_img = out_img.transpose((2, 0, 1))

            return (out_img, int(img['id']), (h, w))

        def img_reader_with_augment(img, size, mean, std, mixup_img):
            im_path = img['image']
            im = cv2.imread(im_path)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            gt_boxes = img['gt_boxes'].copy()
            gt_labels = img['gt_labels'].copy()
            gt_scores = np.ones_like(gt_labels)

            if mixup_img:
                mixup_im = cv2.imread(mixup_img['image'])
                mixup_im = cv2.cvtColor(mixup_im, cv2.COLOR_BGR2RGB)
                mixup_gt_boxes = np.array(mixup_img['gt_boxes']).copy()
                mixup_gt_labels = np.array(mixup_img['gt_labels']).copy()
                mixup_gt_scores = np.ones_like(mixup_gt_labels)
                im, gt_boxes, gt_labels, gt_scores = \
                    image_utils.image_mixup(im, gt_boxes, gt_labels,
                                            gt_scores, mixup_im, mixup_gt_boxes,
                                            mixup_gt_labels, mixup_gt_scores)


            im, gt_boxes, gt_labels, gt_scores = \
                image_utils.image_augment(im, gt_boxes, gt_labels,
                                          gt_scores, size, mean)

            mean = np.array(mean).reshape((1, 1, -1))
            std = np.array(std).reshape((1, 1, -1))
            out_img = (im / 255.0 - mean) / std
            out_img = out_img.astype('float32').transpose((2, 0, 1))

            return (out_img, gt_boxes, gt_labels, gt_scores)

        def get_img_size(size, random_sizes=[]):
            if len(random_sizes):
                return np.random.choice(random_sizes)
            return size

        def get_mixup_img(imgs, mixup_iter, total_iter, read_cnt):
            if total_iter >= mixup_iter:
                return None

            mixup_idx = np.random.randint(1, len(imgs))
            mixup_img = imgs[(read_cnt + mixup_idx) % len(imgs)]
            return mixup_img

        def reader():
            if mode == 'train':
                imgs = self._parse_images_by_mode(mode)
                if shuffle:
                    if shuffle_seed is not None:
                        np.random.seed(shuffle_seed)
                    np.random.shuffle(imgs)
                read_cnt = 0
                total_iter = 0
                batch_out = []
                img_size = get_img_size(size, random_sizes)
                while True:
                    img = imgs[read_cnt % len(imgs)]
                    mixup_img = get_mixup_img(imgs, mixup_iter, total_iter,
                                              read_cnt)
                    read_cnt += 1
                    if read_cnt % len(imgs) == 0 and shuffle:
                        np.random.shuffle(imgs)
                    im, gt_boxes, gt_labels, gt_scores = \
                        img_reader_with_augment(img, img_size, cfg.pixel_means,
                                                cfg.pixel_stds, mixup_img)
                    batch_out.append([im, gt_boxes, gt_labels, gt_scores])

                    if len(batch_out) == batch_size:
                        yield batch_out
                        batch_out = []
                        total_iter += 1
                        if total_iter >= max_iter:
                            return
                        img_size = get_img_size(size, random_sizes)

            elif mode == 'test':
                imgs = self._parse_images_by_mode(mode)
                batch_out = []
                for img in imgs:
                    im, im_id, im_shape = img_reader(img, size, cfg.pixel_means,
                                                     cfg.pixel_stds)
                    batch_out.append((im, im_id, im_shape))
                    if len(batch_out) == batch_size:
                        yield batch_out
                        batch_out = []
                if len(batch_out) != 0:
                    yield batch_out
            else:
                img = {}
                img['image'] = image
                img['id'] = 0
                im, im_id, im_shape = img_reader(img, size, cfg.pixel_means,
                                                 cfg.pixel_stds)
                batch_out = [(im, im_id, im_shape)]
                yield batch_out

        # NOTE: yolov3 is a special model, if num_trainers > 1, each process 
        # trian the completed dataset.
        # num_trainers = int(os.environ.get('PADDLE_TRAINERS_NUM', 1))
        # if mode == 'train' and num_trainers > 1:
        #     assert shuffle_seed is not None, \
        #         "If num_trainers > 1, the shuffle_seed must be set, because " \
        #         "the order of batch data generated by reader " \
        #         "must be the same in the respective processes."
        #     reader = fluid.contrib.reader.distributed_batch_reader(reader)

        return reader


dsr = DataSetReader()


def train(size=416,
          batch_size=64,
          shuffle=True,
          shuffle_seed=None,
          total_iter=0,
          mixup_iter=0,
          random_sizes=[],
          num_workers=1,
          use_multiprocess_reader=True,
          use_gpu=True):
    generator = dsr.get_reader('train', size, batch_size, shuffle, shuffle_seed,
                               int(mixup_iter / num_workers), total_iter, random_sizes)
    if not use_multiprocess_reader:
        return generator
    else:
        if sys.platform == "win32":
            print("multiprocess is not fully compatible with Windows, "
                    "you can set --use_multiprocess_reader=False if you "
                    "are training on Windows and there are errors incured "
                    "by multiprocess.")
        print("multiprocess reader starting up, it takes a while...")

    def infinite_reader():
        while True:
            for data in generator():
                yield data

    def reader():
        cnt = 0
        data_loader = fluid.io.DataLoader.from_generator(capacity=64,use_multiprocess=True,iterable=True)
        if use_gpu:
            place = fluid.CUDAPlace(fluid.dygraph.parallel.Env().dev_id) if cfg.use_data_parallel else fluid.CUDAPlace(0)
        else:
            place = fluid.CPUPlace()
        data_loader.set_sample_list_generator(infinite_reader,places=place)
        generator_out = []
        for data in data_loader(): 
            im, gt_boxes, gt_labels, gt_scores = data[0].numpy(),data[1].numpy(),data[2].numpy(),data[3].numpy()
            for i in range(batch_size):
                generator_out.append([im[i],gt_boxes[i],gt_labels[i],gt_scores[i]])
            yield generator_out
            generator_out = []
            cnt += 1
            if cnt >= total_iter:
                return
    return reader


def test(size=416, batch_size=1):
    return dsr.get_reader('test', size, batch_size)


def infer(size=416, image=None):
    return dsr.get_reader('infer', size, image=image)


def get_label_infos():
    return dsr.get_label_infos()
