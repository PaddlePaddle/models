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
import paddle.fluid as fluid
import random
import sys


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


def get_preprocess_param2(load_size, crop_size):
    x = np.random.randint(0, np.maximum(0, load_size - crop_size))
    y = np.random.randint(0, np.maximum(0, load_size - crop_size))
    flip = np.random.rand() > 0.5
    return {
        "crop_pos": (x, y),
        "flip": flip,
        "load_size": load_size,
        "crop_size": crop_size
    }


def get_preprocess_param4(load_width, load_height, crop_width, crop_height):
    if crop_width == load_width:
        x = 0
        y = 0
    else:
        x = np.random.randint(0, np.maximum(0, load_width - crop_width))
        y = np.random.randint(0, np.maximum(0, load_height - crop_height))
    flip = np.random.rand() > 0.5
    return {"crop_pos": (x, y), "flip": flip}


class reader_creator(object):
    ''' read and preprocess dataset'''

    def __init__(self,
                 image_dir,
                 list_filename,
                 shuffle=False,
                 batch_size=1,
                 mode="TRAIN"):
        self.image_dir = image_dir
        self.list_filename = list_filename
        self.batch_size = batch_size
        self.mode = mode

        self.name2id = {}
        self.id2name = {}

        self.lines = open(self.list_filename).readlines()

        if self.mode == "TRAIN":
            self.shuffle = shuffle
        else:
            self.shuffle = False

    def len(self):
        return len(self.lines) // self.batch_size

    def make_reader(self, args, return_name=False):
        print(self.image_dir, self.list_filename)
        self.with_label = False

        def reader():
            batch_out = []
            batch_out_label = []
            batch_out_name = []

            if self.shuffle:
                np.random.shuffle(self.lines)

            for i, line in enumerate(self.lines):
                line = line.strip('\n\r\t').split(' ')
                if len(line) > 1:
                    self.with_label = True
                    batch_out_label.append(line[1])
                    file = line[0]
                else:
                    file = line[0]
                self.name2id[os.path.basename(file)] = i
                self.id2name[i] = os.path.basename(file)
                img = Image.open(os.path.join(self.image_dir, file)).convert(
                    'RGB')
                if self.mode == "TRAIN":
                    img = img.resize((args.image_size, args.image_size),
                                     Image.BICUBIC)
                    if args.crop_type == 'Centor':
                        img = CentorCrop(img, args.crop_size, args.crop_size)
                    elif args.crop_type == 'Random':
                        img = RandomCrop(img, args.crop_size, args.crop_size)
                else:
                    img = img.resize((args.crop_size, args.crop_size),
                                     Image.BICUBIC)
                img = (np.array(img).astype('float32') / 255.0 - 0.5) / 0.5
                img = img.transpose([2, 0, 1])

                if return_name:
                    batch_out.append(img)
                    batch_out_name.append(i)
                else:
                    batch_out.append(img)
                if len(batch_out) == self.batch_size:
                    if return_name:
                        if self.with_label:
                            yield [[batch_out, batch_out_label, batch_out_name]]
                            batch_out_label = []
                        else:
                            yield batch_out, batch_out_name
                        batch_out_name = []
                    else:
                        if self.with_label:
                            yield [[batch_out, batch_out_label]]
                            batch_out_label = []
                        else:
                            yield [batch_out]
                    batch_out = []

        return reader


class pair_reader_creator(reader_creator):
    ''' read and preprocess dataset'''

    def __init__(self,
                 image_dir,
                 list_filename,
                 shuffle=False,
                 batch_size=1,
                 mode="TRAIN"):
        super(pair_reader_creator, self).__init__(
            image_dir,
            list_filename,
            shuffle=shuffle,
            batch_size=batch_size,
            mode=mode)

    def make_reader(self, args, return_name=False):
        print(self.image_dir, self.list_filename)

        def reader():
            batch_out_1 = []
            batch_out_2 = []
            batch_out_name = []
            if self.shuffle:
                np.random.shuffle(self.lines)
            for i, line in enumerate(self.lines):
                files = line.strip('\n\r\t ').split('\t')
                img1 = Image.open(os.path.join(self.image_dir, files[
                    0])).convert('RGB')
                img2 = Image.open(os.path.join(self.image_dir, files[
                    1])).convert('RGB')

                self.name2id[os.path.basename(files[0])] = i
                self.id2name[i] = os.path.basename(files[0])

                if self.mode == "TRAIN":
                    param = get_preprocess_param2(args.image_size,
                                                  args.crop_size)
                    img1 = img1.resize((args.image_size, args.image_size),
                                       Image.BICUBIC)
                    img2 = img2.resize((args.image_size, args.image_size),
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
                else:
                    img1 = img1.resize((args.crop_size, args.crop_size),
                                       Image.BICUBIC)
                    img2 = img2.resize((args.crop_size, args.crop_size),
                                       Image.BICUBIC)

                img1 = (np.array(img1).astype('float32') / 255.0 - 0.5) / 0.5
                img1 = img1.transpose([2, 0, 1])
                img2 = (np.array(img2).astype('float32') / 255.0 - 0.5) / 0.5
                img2 = img2.transpose([2, 0, 1])

                batch_out_1.append(img1)
                batch_out_2.append(img2)
                if return_name:
                    batch_out_name.append(i)
                if len(batch_out_1) == self.batch_size:
                    if return_name:
                        yield batch_out_1, batch_out_2, batch_out_name
                        batch_out_name = []
                    else:
                        yield batch_out_1, batch_out_2
                    batch_out_1 = []
                    batch_out_2 = []

        return reader


class triplex_reader_creator(reader_creator):
    ''' read and preprocess dataset'''

    def __init__(self,
                 image_dir,
                 list_filename,
                 shuffle=False,
                 batch_size=1,
                 mode="TRAIN"):
        super(triplex_reader_creator, self).__init__(
            image_dir,
            list_filename,
            shuffle=shuffle,
            batch_size=batch_size,
            mode=mode)

        self.name2id = {}
        self.id2name = {}

    def make_reader(self, args, return_name=False):
        print(self.image_dir, self.list_filename)
        print("files length:", len(self.lines))

        def reader():
            batch_out_1 = []
            batch_out_2 = []
            batch_out_3 = []
            batch_out_name = []
            if self.shuffle:
                np.random.shuffle(self.lines)
            for i, line in enumerate(self.lines):
                files = line.strip('\n\r\t ').split('\t')
                if len(files) != 3:
                    print("files is not equal to 3!")
                    sys.exit(-1)
                self.name2id[os.path.basename(files[0])] = i
                self.id2name[i] = os.path.basename(files[0])
                #label image instance
                img1 = Image.open(os.path.join(self.image_dir, files[0]))
                img2 = Image.open(os.path.join(self.image_dir, files[
                    1])).convert('RGB')
                if not args.no_instance:
                    img3 = Image.open(os.path.join(self.image_dir, files[2]))

                if self.mode == "TRAIN":
                    param = get_preprocess_param4(
                        args.load_width, args.load_height, args.crop_width,
                        args.crop_height)
                    img1 = img1.resize((args.load_width, args.load_height),
                                       Image.NEAREST)
                    img2 = img2.resize((args.load_width, args.load_height),
                                       Image.BICUBIC)
                    if not args.no_instance:
                        img3 = img3.resize((args.load_width, args.load_height),
                                           Image.NEAREST)
                    if args.crop_type == 'Centor':
                        img1 = CentorCrop(img1, args.crop_width,
                                          args.crop_height)
                        img2 = CentorCrop(img2, args.crop_width,
                                          args.crop_height)
                        if not args.no_instance:
                            img3 = CentorCrop(img3, args.crop_width,
                                              args.crop_height)
                    elif args.crop_type == 'Random':
                        x = param['crop_pos'][0]
                        y = param['crop_pos'][1]
                        img1 = img1.crop(
                            (x, y, x + args.crop_width, y + args.crop_height))
                        img2 = img2.crop(
                            (x, y, x + args.crop_width, y + args.crop_height))
                        if not args.no_instance:
                            img3 = img3.crop((x, y, x + args.crop_width,
                                              y + args.crop_height))
                else:
                    img1 = img1.resize((args.crop_width, args.crop_height),
                                       Image.NEAREST)
                    img2 = img2.resize((args.crop_width, args.crop_height),
                                       Image.BICUBIC)
                    if not args.no_instance:
                        img3 = img3.resize((args.crop_width, args.crop_height),
                                           Image.NEAREST)

                img1 = np.array(img1)
                index = img1[np.newaxis, :, :]
                input_label = np.zeros(
                    (args.label_nc, index.shape[1], index.shape[2]))
                np.put_along_axis(input_label, index, 1.0, 0)
                img1 = input_label.astype('float32')
                img2 = (np.array(img2).astype('float32') / 255.0 - 0.5) / 0.5
                img2 = img2.transpose([2, 0, 1])
                if not args.no_instance:
                    img3 = np.array(img3)[:, :, np.newaxis]
                    img3 = img3.transpose([2, 0, 1])
                    ###extracte edge from instance
                    edge = np.zeros(img3.shape)
                    edge = edge.astype('int8')
                    edge[:, :, 1:] = edge[:, :, 1:] | (
                        img3[:, :, 1:] != img3[:, :, :-1])
                    edge[:, :, :-1] = edge[:, :, :-1] | (
                        img3[:, :, 1:] != img3[:, :, :-1])
                    edge[:, 1:, :] = edge[:, 1:, :] | (
                        img3[:, 1:, :] != img3[:, :-1, :])
                    edge[:, :-1, :] = edge[:, :-1, :] | (
                        img3[:, 1:, :] != img3[:, :-1, :])
                    img3 = edge.astype('float32')
                    ###end extracte
                batch_out_1.append(img1)
                batch_out_2.append(img2)
                if not args.no_instance:
                    batch_out_3.append(img3)
                if return_name:
                    batch_out_name.append(i)
                if len(batch_out_1) == self.batch_size:
                    if return_name:
                        if not args.no_instance:
                            yield batch_out_1, batch_out_2, batch_out_3, batch_out_name
                        else:
                            yield batch_out_1, batch_out_2, batch_out_name
                        batch_out_name = []
                    else:
                        if not args.no_instance:
                            yield batch_out_1, batch_out_2, batch_out_3
                        else:
                            yield batch_out_1, batch_out_2
                    batch_out_1 = []
                    batch_out_2 = []
                    batch_out_3 = []

        return reader


class celeba_reader_creator(reader_creator):
    ''' read and preprocess dataset'''

    def __init__(self, image_dir, list_filename, args, mode="TRAIN"):
        self.image_dir = image_dir
        self.list_filename = list_filename
        self.mode = mode
        self.args = args

        lines = open(self.list_filename).readlines()

        all_num = int(lines[0])
        train_end = 2 + int(all_num * 0.9)
        test_end = train_end + int(all_num * 0.003)

        all_attr_names = lines[1].split()
        attr2idx = {}
        for i, attr_name in enumerate(all_attr_names):
            attr2idx[attr_name] = i

        if self.mode == "TRAIN":
            self.batch_size = args.batch_size
            self.shuffle = args.shuffle
            lines = lines[2:train_end]
        elif self.mode == 'TEST':
            self.batch_size = args.n_samples
            self.shuffle = False
            lines = lines[train_end:test_end]
        elif self.mode == 'VAL':
            self.batch_size = args.n_samples
            self.shuffle = False
            lines = lines[2:]
        else:
            raise NotImplementedError(
                "Wrong Reader MODE: {}, mode must in [TRAIN|TEST|VAL]".format(
                    self.mode))

        self.images = []
        attr_names = args.selected_attrs.split(',')
        for i, line in enumerate(lines):
            arr = line.strip().split()
            name = os.path.join('img_align_celeba', arr[0])
            label = []
            for attr_name in attr_names:
                idx = attr2idx[attr_name]
                label.append(arr[idx + 1] == "1")
            self.images.append((name, label, arr[0]))

    def len(self):
        return len(self.images) // self.batch_size

    def make_reader(self, return_name=False):
        print(self.image_dir, self.list_filename)

        def reader():
            batch_out_1 = []
            batch_out_2 = []
            batch_out_3 = []
            batch_out_name = []
            if self.shuffle:
                np.random.shuffle(self.images)
            for file, label, f_name in self.images:
                img = Image.open(os.path.join(self.image_dir, file))
                label = np.array(label).astype("float32")
                if self.args.model_net == "StarGAN":
                    img = RandomHorizonFlip(img)
                img = CentorCrop(img, self.args.crop_size, self.args.crop_size)
                img = img.resize((self.args.image_size, self.args.image_size),
                                 Image.BILINEAR)
                img = (np.array(img).astype('float32') / 255.0 - 0.5) / 0.5
                img = img.transpose([2, 0, 1])

                batch_out_1.append(img)
                batch_out_2.append(label)
                if return_name:
                    batch_out_name.append(int(f_name.split('.')[0]))
                if len(batch_out_1) == self.batch_size:
                    batch_out_3 = np.copy(batch_out_2)
                    if self.shuffle:
                        np.random.shuffle(batch_out_3)
                    if return_name:
                        yield batch_out_1, batch_out_2, batch_out_3, batch_out_name
                        batch_out_name = []
                    else:
                        yield batch_out_1, batch_out_2, batch_out_3
                    batch_out_1 = []
                    batch_out_2 = []
                    batch_out_3 = []

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

            train_reader = fluid.io.batch(
                fluid.io.shuffle(
                    mnist_reader_creator(train_images, train_labels, 100),
                    buf_size=60000),
                batch_size=self.cfg.batch_size)
            return train_reader
        else:
            if self.cfg.model_net in ['CycleGAN']:
                dataset_dir = os.path.join(self.cfg.data_dir, self.cfg.dataset)
                trainA_list = os.path.join(dataset_dir, "trainA.txt")
                trainB_list = os.path.join(dataset_dir, "trainB.txt")
                a_train_reader = reader_creator(
                    image_dir=dataset_dir,
                    list_filename=trainA_list,
                    shuffle=self.cfg.shuffle,
                    batch_size=self.cfg.batch_size,
                    mode="TRAIN")
                b_train_reader = reader_creator(
                    image_dir=dataset_dir,
                    list_filename=trainB_list,
                    shuffle=self.cfg.shuffle,
                    batch_size=self.cfg.batch_size,
                    mode="TRAIN")
                a_reader_test = None
                b_reader_test = None
                a_id2name = None
                b_id2name = None
                if self.cfg.run_test:
                    testA_list = os.path.join(dataset_dir, "testA.txt")
                    testB_list = os.path.join(dataset_dir, "testB.txt")
                    a_test_reader = reader_creator(
                        image_dir=dataset_dir,
                        list_filename=testA_list,
                        shuffle=False,
                        batch_size=1,
                        mode="TEST")
                    b_test_reader = reader_creator(
                        image_dir=dataset_dir,
                        list_filename=testB_list,
                        shuffle=False,
                        batch_size=1,
                        mode="TEST")
                    a_reader_test = a_test_reader.make_reader(
                        self.cfg, return_name=True)
                    b_reader_test = b_test_reader.make_reader(
                        self.cfg, return_name=True)
                    a_id2name = a_test_reader.id2name
                    b_id2name = b_test_reader.id2name

                batch_num = max(a_train_reader.len(), b_train_reader.len())
                a_reader = a_train_reader.make_reader(self.cfg)
                b_reader = b_train_reader.make_reader(self.cfg)

                return a_reader, b_reader, a_reader_test, b_reader_test, batch_num, a_id2name, b_id2name

            elif self.cfg.model_net in ['StarGAN', 'STGAN', 'AttGAN']:
                dataset_dir = os.path.join(self.cfg.data_dir, self.cfg.dataset)
                train_list = os.path.join(dataset_dir, 'train.txt')
                if self.cfg.train_list is not None:
                    train_list = self.cfg.train_list
                train_reader = celeba_reader_creator(
                    image_dir=dataset_dir,
                    list_filename=train_list,
                    args=self.cfg,
                    mode="TRAIN")
                reader_test = None
                if self.cfg.run_test:
                    test_list = train_list
                    if self.cfg.test_list is not None:
                        test_list = self.cfg.test_list
                    test_reader = celeba_reader_creator(
                        image_dir=dataset_dir,
                        list_filename=train_list,
                        args=self.cfg,
                        mode="TEST")
                    reader_test = test_reader.make_reader(return_name=True)
                batch_num = train_reader.len()
                reader = train_reader.make_reader()
                return reader, reader_test, batch_num, None

            elif self.cfg.model_net in ['Pix2pix']:
                dataset_dir = os.path.join(self.cfg.data_dir, self.cfg.dataset)
                train_list = os.path.join(dataset_dir, 'train.txt')
                if self.cfg.train_list is not None:
                    train_list = self.cfg.train_list
                train_reader = pair_reader_creator(
                    image_dir=dataset_dir,
                    list_filename=train_list,
                    shuffle=self.cfg.shuffle,
                    batch_size=self.cfg.batch_size,
                    mode="TRAIN")
                reader_test = None
                id2name = None
                if self.cfg.run_test:
                    test_list = os.path.join(dataset_dir, "test.txt")
                    if self.cfg.test_list is not None:
                        test_list = self.cfg.test_list
                    test_reader = pair_reader_creator(
                        image_dir=dataset_dir,
                        list_filename=test_list,
                        shuffle=False,
                        batch_size=1,
                        mode="TEST")
                    reader_test = test_reader.make_reader(
                        self.cfg, return_name=True)
                    id2name = test_reader.id2name
                batch_num = train_reader.len()
                reader = train_reader.make_reader(self.cfg)
                return reader, reader_test, batch_num, id2name
            elif self.cfg.model_net in ['SPADE']:
                dataset_dir = os.path.join(self.cfg.data_dir, self.cfg.dataset)
                train_list = os.path.join(dataset_dir, 'train.txt')
                if self.cfg.train_list is not None:
                    train_list = self.cfg.train_list
                if not os.path.exists(train_list):
                    print(
                        "train_list is NOT EXIST!!! Please prepare train list first"
                    )
                    sys.exit(1)
                train_reader = triplex_reader_creator(
                    image_dir=dataset_dir,
                    list_filename=train_list,
                    shuffle=self.cfg.shuffle,
                    batch_size=self.cfg.batch_size,
                    mode="TRAIN")
                reader_test = None
                id2name = None
                if self.cfg.run_test:
                    test_list = os.path.join(dataset_dir, "test.txt")
                    if self.cfg.test_list is not None:
                        test_list = self.cfg.test_list
                    if not os.path.exists(test_list):
                        print(
                            "test_list is NOT EXIST!!! Please prepare test list first"
                        )
                        sys.exit(1)
                    test_reader = triplex_reader_creator(
                        image_dir=dataset_dir,
                        list_filename=test_list,
                        shuffle=False,
                        batch_size=1,
                        mode="TEST")
                    reader_test = test_reader.make_reader(
                        self.cfg, return_name=True)
                    id2name = test_reader.id2name
                batch_num = train_reader.len()
                reader = train_reader.make_reader(self.cfg)
                return reader, reader_test, batch_num, id2name
            else:
                dataset_dir = os.path.join(self.cfg.data_dir, self.cfg.dataset)
                train_list = os.path.join(dataset_dir, 'train.txt')
                if self.cfg.train_list is not None:
                    train_list = self.cfg.train_list
                train_reader = reader_creator(
                    image_dir=dataset_dir, list_filename=train_list)
                reader_test = None
                id2name = None
                if self.cfg.run_test:
                    test_list = os.path.join(dataset_dir, "test.txt")
                    test_reader = reader_creator(
                        image_dir=dataset_dir,
                        list_filename=test_list,
                        batch_size=self.cfg.n_samples)
                    reader_test = test_reader.make_reader(
                        self.cfg, shuffle=False, return_name=True)
                    id2name = test_reader.id2name
                batch_num = train_reader.len()
                reader = train_reader.make_reader(self.cfg)
                return reader, reader_test, batch_num, id2name
