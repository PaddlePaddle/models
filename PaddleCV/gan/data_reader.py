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

from __future__ import print_function
from six.moves import range
from PIL import Image, ImageOps

import gzip
import numpy as np
import argparse
import struct
import os
import paddle
import random


def RandomCrop(img, crop_w, crop_h):
    w, h = img.size[0], img.size[1]
    i = np.random.randint(0, w - crop_w)
    j = np.random.randint(0, h - crop_h)
    return img.crop((i, j, i + crop_w, j + crop_h))


def CentorCrop(img, crop_w, crop_h):
    w, h = img.size[0], img.size[1]
    i = int((w - crop_w) / 2.0)
    j = int((h - crop_h) / 2.0)
    return img.crop((i, j, i + crop_w, j + crop_h))


def RandomHorizonFlip(img):
    i = np.random.rand()
    if i > 0.5:
        img = ImageOps.mirror(img)
    return img


def get_preprocess_param(load_size, crop_size):
    x = np.random.randint(0, np.maximum(0, load_size - crop_size))
    y = np.random.randint(0, np.maximum(0, load_size - crop_size))
    flip = np.random.rand() > 0.5
    return {
        "crop_pos": (x, y),
        "flip": flip,
        "load_size": load_size,
        "crop_size": crop_size
    }


class reader_creator(object):
    ''' read and preprocess dataset'''

    def __init__(self, image_dir, list_filename, batch_size=1, drop_last=False):
        self.image_dir = image_dir
        self.list_filename = list_filename
        self.batch_size = batch_size
        self.drop_last = drop_last

        self.lines = open(self.list_filename).readlines()

    def len(self):
        if self.drop_last or len(self.lines) % self.batch_size == 0:
            return len(self.lines) // self.batch_size
        else:
            return len(self.lines) // self.batch_size + 1

    def get_train_reader(self, args, shuffle=False, return_name=False):
        print(self.image_dir, self.list_filename)

        def reader():
            batch_out = []
            while True:
                if shuffle:
                    np.random.shuffle(self.lines)
                for file in self.lines:
                    file = file.strip('\n\r\t ')
                    img = Image.open(os.path.join(self.image_dir,
                                                  file)).convert('RGB')
                    img = img.resize((args.load_size, args.load_size),
                                     Image.BICUBIC)
                    if args.crop_type == 'Centor':
                        img = CentorCrop(img, args.crop_size, args.crop_size)
                    elif args.crop_type == 'Random':
                        img = RandomCrop(img, args.crop_size, args.crop_size)
                    img = (np.array(img).astype('float32') / 255.0 - 0.5) / 0.5
                    img = img.transpose([2, 0, 1])

                    if return_name:
                        batch_out.append([img, os.path.basename(file)])
                    else:
                        batch_out.append(img)
                    if len(batch_out) == self.batch_size:
                        yield batch_out
                        batch_out = []
                if self.drop_last == False and len(batch_out) != 0:
                    yield batch_out

        return reader

    def get_test_reader(self, args, shuffle=False, return_name=False):
        print(self.image_dir, self.list_filename)

        def reader():
            batch_out = []
            for file in self.lines:
                file = file.strip('\n\r\t ')
                img = Image.open(os.path.join(self.image_dir, file)).convert(
                    'RGB')
                img = img.resize((args.crop_size, args.crop_size),
                                 Image.BICUBIC)
                img = (np.array(img).astype('float32') / 255.0 - 0.5) / 0.5
                img = img.transpose([2, 0, 1])
                if return_name:
                    batch_out.append(
                        [img[np.newaxis, :], os.path.basename(file)])
                else:
                    batch_out.append(img)
                if len(batch_out) == self.batch_size:
                    yield batch_out
                    batch_out = []
            if len(batch_out) != 0:
                yield batch_out

        return reader


class pair_reader_creator(reader_creator):
    ''' read and preprocess dataset'''

    def __init__(self, image_dir, list_filename, batch_size=1, drop_last=False):
        super(pair_reader_creator, self).__init__(
            image_dir, list_filename, batch_size=1, drop_last=drop_last)

    def get_train_reader(self, args, shuffle=False, return_name=False):
        print(self.image_dir, self.list_filename)

        def reader():
            batch_out_1 = []
            batch_out_2 = []
            while True:
                if shuffle:
                    np.random.shuffle(self.lines)
                for line in self.lines:
                    files = line.strip('\n\r\t ').split('\t')
                    img1 = Image.open(os.path.join(self.image_dir, files[
                        0])).convert('RGB')
                    img2 = Image.open(os.path.join(self.image_dir, files[
                        1])).convert('RGB')
                    param = get_preprocess_param(args.load_size, args.crop_size)
                    img1 = img1.resize((args.load_size, args.load_size),
                                       Image.BICUBIC)
                    img2 = img2.resize((args.load_size, args.load_size),
                                       Image.BICUBIC)
                    if args.crop_type == 'Centor':
                        img1 = CentorCrop(img1, args.crop_size, args.crop_size)
                        img2 = CentorCrop(img2, args.crop_size, args.crop_size)
                    elif args.crop_type == 'Random':
                        x = param['crop_pos'][0]
                        y = param['crop_pos'][1]
                        img1 = img1.crop(
                            (x, y, x + args.crop_size, y + args.crop_size))
                        img2 = img2.crop(
                            (x, y, x + args.crop_size, y + args.crop_size))
                    img1 = (
                        np.array(img1).astype('float32') / 255.0 - 0.5) / 0.5
                    img1 = img1.transpose([2, 0, 1])
                    img2 = (
                        np.array(img2).astype('float32') / 255.0 - 0.5) / 0.5
                    img2 = img2.transpose([2, 0, 1])

                    batch_out_1.append(img1)
                    batch_out_2.append(img2)
                    if len(batch_out_1) == self.batch_size:
                        yield batch_out_1, batch_out_2
                        batch_out_1 = []
                        batch_out_2 = []
                if self.drop_last == False and len(batch_out_1) != 0:
                    yield batch_out_1, batch_out_2

        return reader

    def get_test_reader(self, args, shuffle=False, return_name=False):
        print(self.image_dir, self.list_filename)

        def reader():
            batch_out_1 = []
            batch_out_2 = []
            batch_out_3 = []
            for line in self.lines:
                files = line.strip('\n\r\t ').split('\t')
                img1 = Image.open(os.path.join(self.image_dir, files[
                    0])).convert('RGB')
                img2 = Image.open(os.path.join(self.image_dir, files[
                    1])).convert('RGB')
                img1 = img1.resize((args.crop_size, args.crop_size),
                                   Image.BICUBIC)
                img2 = img2.resize((args.crop_size, args.crop_size),
                                   Image.BICUBIC)
                img1 = (np.array(img1).astype('float32') / 255.0 - 0.5) / 0.5
                img1 = img1.transpose([2, 0, 1])
                img2 = (np.array(img2).astype('float32') / 255.0 - 0.5) / 0.5
                img2 = img2.transpose([2, 0, 1])
                if return_name:
                    batch_out_1.append(img1)
                    batch_out_2.append(img2)
                    batch_out_3.append(os.path.basename(files[0]))
                else:
                    batch_out_1.append(img1)
                    batch_out_2.append(img2)
                if len(batch_out_1) == self.batch_size:
                    if return_name:
                        yield batch_out_1, batch_out_2, batch_out_3
                        batch_out_1 = []
                        batch_out_2 = []
                        batch_out_3 = []
                    else:
                        yield batch_out_1, batch_out_2
                        batch_out_1 = []
                        batch_out_2 = []
            if len(batch_out_1) != 0:
                if return_name:
                    yield batch_out_1, batch_out_2, batch_out_3
                else:
                    yield batch_out_1, batch_out_2

        return reader


class celeba_reader_creator(reader_creator):
    ''' read and preprocess dataset'''

    def __init__(self,
                 image_dir,
                 list_filename,
                 args,
                 batch_size=1,
                 drop_last=False):
        self.image_dir = image_dir
        self.list_filename = list_filename
        self.batch_size = batch_size
        self.drop_last = drop_last

        print(self.image_dir, self.list_filename)
        lines = open(self.list_filename).readlines()
        all_attr_names = lines[1].split()
        attr2idx = {}
        for i, attr_name in enumerate(all_attr_names):
            attr2idx[attr_name] = i
        lines = lines[2:]
        self.images = []
        attr_names = args.selected_attrs.split(',')
        for line in lines:
            arr = line.strip().split()
            name = os.path.join('img_align_celeba', arr[0])
            label = []
            for attr_name in attr_names:
                idx = attr2idx[attr_name]
                label.append(arr[idx + 1] == "1")
            self.images.append((name, label))

    def len(self):
        if self.drop_last or len(self.images) % self.batch_size == 0:
            return len(self.images) // self.batch_size
        else:
            return len(self.images) // self.batch_size + 1

    def get_train_reader(self, args, shuffle=False, return_name=False):
        def reader():
            batch_out_1 = []
            batch_out_2 = []
            while True:
                if shuffle:
                    np.random.shuffle(self.images)
                for file, label in self.images:
                    if args.model_net == "StarGAN":
                        img = Image.open(os.path.join(self.image_dir, file))
                        label = np.array(label).astype("float32")
                        img = RandomHorizonFlip(img)
                        img = CentorCrop(img, args.crop_size, args.crop_size)
                        img = img.resize((args.image_size, args.image_size),
                                         Image.BILINEAR)
                    else:
                        img = Image.open(os.path.join(self.image_dir,
                                                      file)).convert('RGB')
                        label = np.array(label).astype("float32")
                        img = CentorCrop(img, args.crop_size, args.crop_size)
                        img = img.resize((args.image_size, args.image_size),
                                         Image.BILINEAR)
                    img = (np.array(img).astype('float32') / 255.0 - 0.5) / 0.5
                    img = img.transpose([2, 0, 1])

                    batch_out_1.append(img)
                    batch_out_2.append(label)
                    if len(batch_out_1) == self.batch_size:
                        yield batch_out_1, batch_out_2
                        batch_out_1 = []
                        batch_out_2 = []
                if self.drop_last == False and len(batch_out_1) != 0:
                    yield batch_out_1, batch_out_2

        return reader

    def get_test_reader(self, args, shuffle=False, return_name=False):
        def reader():
            batch_out_1 = []
            batch_out_2 = []
            batch_out_3 = []
            for file, label in self.images:
                img = Image.open(os.path.join(self.image_dir, file))
                label = np.array(label).astype("float32")
                img = CentorCrop(img, args.crop_size, args.crop_size)
                img = img.resize((args.image_size, args.image_size),
                                 Image.BILINEAR)
                img = (np.array(img).astype('float32') / 255.0 - 0.5) / 0.5
                img = img.transpose([2, 0, 1])
                if return_name:
                    batch_out_1.append(img)
                    batch_out_2.append(label)
                    batch_out_3.append(os.path.basename(file))
                else:
                    batch_out_1.append(img)
                    batch_out_2.append(label)
                if len(batch_out_1) == self.batch_size:
                    if return_name:
                        yield batch_out_1, batch_out_2, batch_out_3
                        batch_out_1 = []
                        batch_out_2 = []
                        batch_out_3 = []
                    else:
                        yield batch_out_1, batch_out_2
                        batch_out_1 = []
                        batch_out_2 = []
            if len(batch_out_1) != 0:
                if return_name:
                    yield batch_out_1, batch_out_2, batch_out_3
                else:
                    yield batch_out_1, batch_out_2

        return reader


def mnist_reader_creator(image_filename, label_filename, buffer_size):
    def reader():
        with gzip.GzipFile(image_filename, 'rb') as image_file:
            img_buf = image_file.read()
            with gzip.GzipFile(label_filename, 'rb') as label_file:
                lab_buf = label_file.read()

                step_label = 0

                offset_img = 0
                # read from Big-endian
                # get file info from magic byte
                # image file : 16B
                magic_byte_img = '>IIII'
                magic_img, image_num, rows, cols = struct.unpack_from(
                    magic_byte_img, img_buf, offset_img)
                offset_img += struct.calcsize(magic_byte_img)

                offset_lab = 0
                # label file : 8B
                magic_byte_lab = '>II'
                magic_lab, label_num = struct.unpack_from(magic_byte_lab,
                                                          lab_buf, offset_lab)
                offset_lab += struct.calcsize(magic_byte_lab)

                while True:
                    if step_label >= label_num:
                        break
                    fmt_label = '>' + str(buffer_size) + 'B'
                    labels = struct.unpack_from(fmt_label, lab_buf, offset_lab)
                    offset_lab += struct.calcsize(fmt_label)
                    step_label += buffer_size

                    fmt_images = '>' + str(buffer_size * rows * cols) + 'B'
                    images_temp = struct.unpack_from(fmt_images, img_buf,
                                                     offset_img)
                    images = np.reshape(images_temp, (buffer_size, rows *
                                                      cols)).astype('float32')
                    offset_img += struct.calcsize(fmt_images)

                    images = images / 255.0 * 2.0 - 1.0
                    for i in range(buffer_size):
                        yield images[i, :], int(
                            labels[i])  # get image and label

    return reader


class data_reader(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.shuffle = self.cfg.shuffle

    def make_data(self):
        if self.cfg.dataset == 'mnist':
            train_images = os.path.join(self.cfg.data_dir, self.cfg.dataset,
                                        "train-images-idx3-ubyte.gz")
            train_labels = os.path.join(self.cfg.data_dir, self.cfg.dataset,
                                        "train-labels-idx1-ubyte.gz")

            train_reader = paddle.batch(
                paddle.reader.shuffle(
                    mnist_reader_creator(train_images, train_labels, 100),
                    buf_size=60000),
                batch_size=self.cfg.batch_size)
            return train_reader
        else:
            if self.cfg.model_net == 'CycleGAN':
                dataset_dir = os.path.join(self.cfg.data_dir, self.cfg.dataset)
                trainA_list = os.path.join(dataset_dir, "trainA.txt")
                trainB_list = os.path.join(dataset_dir, "trainB.txt")
                a_train_reader = reader_creator(
                    image_dir=dataset_dir,
                    list_filename=trainA_list,
                    batch_size=self.cfg.batch_size,
                    drop_last=self.cfg.drop_last)
                b_train_reader = reader_creator(
                    image_dir=dataset_dir,
                    list_filename=trainB_list,
                    batch_size=self.cfg.batch_size,
                    drop_last=self.cfg.drop_last)
                a_reader_test = None
                b_reader_test = None
                if self.cfg.run_test:
                    testA_list = os.path.join(dataset_dir, "testA.txt")
                    testB_list = os.path.join(dataset_dir, "testB.txt")
                    a_test_reader = reader_creator(
                        image_dir=dataset_dir,
                        list_filename=testA_list,
                        batch_size=1,
                        drop_last=self.cfg.drop_last)
                    b_test_reader = reader_creator(
                        image_dir=dataset_dir,
                        list_filename=testB_list,
                        batch_size=1,
                        drop_last=self.cfg.drop_last)
                    a_reader_test = a_test_reader.get_test_reader(
                        self.cfg, shuffle=False, return_name=True)
                    b_reader_test = b_test_reader.get_test_reader(
                        self.cfg, shuffle=False, return_name=True)

                batch_num = max(a_train_reader.len(), b_train_reader.len())
                a_reader = a_train_reader.get_train_reader(
                    self.cfg, shuffle=self.shuffle)
                b_reader = b_train_reader.get_train_reader(
                    self.cfg, shuffle=self.shuffle)

                return a_reader, b_reader, a_reader_test, b_reader_test, batch_num

            elif self.cfg.model_net == 'StarGAN' or self.cfg.model_net == 'STGAN' or self.cfg.model_net == 'AttGAN':
                dataset_dir = os.path.join(self.cfg.data_dir, self.cfg.dataset)
                train_list = os.path.join(dataset_dir, 'train.txt')
                if self.cfg.train_list is not None:
                    train_list = self.cfg.train_list
                train_reader = celeba_reader_creator(
                    image_dir=dataset_dir,
                    list_filename=train_list,
                    batch_size=self.cfg.batch_size,
                    args=self.cfg,
                    drop_last=self.cfg.drop_last)
                reader_test = None
                if self.cfg.run_test:
                    test_list = os.path.join(dataset_dir, "test.txt")
                    if self.cfg.test_list is not None:
                        test_list = self.cfg.test_list
                    test_reader = celeba_reader_creator(
                        image_dir=dataset_dir,
                        list_filename=test_list,
                        batch_size=self.cfg.n_samples,
                        drop_last=self.cfg.drop_last,
                        args=self.cfg)
                    reader_test = test_reader.get_test_reader(
                        self.cfg, shuffle=False, return_name=True)
                batch_num = train_reader.len()
                reader = train_reader.get_train_reader(
                    self.cfg, shuffle=self.shuffle)
                return reader, reader_test, batch_num

            elif self.cfg.model_net == 'Pix2pix':
                dataset_dir = os.path.join(self.cfg.data_dir, self.cfg.dataset)
                train_list = os.path.join(dataset_dir, 'train.txt')
                if self.cfg.train_list is not None:
                    train_list = self.cfg.train_list
                train_reader = pair_reader_creator(
                    image_dir=dataset_dir,
                    list_filename=train_list,
                    batch_size=self.cfg.batch_size,
                    drop_last=self.cfg.drop_last)
                reader_test = None
                if self.cfg.run_test:
                    test_list = os.path.join(dataset_dir, "test.txt")
                    if self.cfg.test_list is not None:
                        test_list = self.cfg.test_list
                    test_reader = pair_reader_creator(
                        image_dir=dataset_dir,
                        list_filename=test_list,
                        batch_size=1,
                        drop_last=self.cfg.drop_last)
                    reader_test = test_reader.get_test_reader(
                        self.cfg, shuffle=False, return_name=True)
                batch_num = train_reader.len()
                reader = train_reader.get_train_reader(
                    self.cfg, shuffle=self.shuffle)
                return reader, reader_test, batch_num
            else:
                dataset_dir = os.path.join(self.cfg.data_dir, self.cfg.dataset)
                train_list = os.path.join(dataset_dir, 'train.txt')
                if self.cfg.train_list is not None:
                    train_list = self.cfg.train_list
                train_reader = reader_creator(
                    image_dir=dataset_dir, list_filename=train_list)
                reader_test = None
                if self.cfg.run_test:
                    test_list = os.path.join(dataset_dir, "test.txt")
                    test_reader = reader_creator(
                        image_dir=dataset_dir,
                        list_filename=test_list,
                        batch_size=1,
                        drop_last=self.cfg.drop_last)
                    reader_test = test_reader.get_test_reader(
                        self.cfg, shuffle=False, return_name=True)
                batch_num = train_reader.len()
                return train_reader, reader_test, batch_num
