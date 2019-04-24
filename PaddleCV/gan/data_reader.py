from __future__ import print_function
from six.moves import range
from PIL import Image, ImageOps

import gzip
import numpy as np
import argparse
import struct
import os


def RandomCrop(img, crop_w, crop_h):
    w, h = img.shape[0], img.shape[1]
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
        img = ImageOps.mirror(image)
    return img


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

    def get_reader(self, args, mode='train', shuffle=False, return_name=False):
        print(self.image_dir, self.list_filename)

        def reader():
            if mode == 'train':
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
                            img = CentorCrop(img, args.crop_size,
                                             args.crop_size)
                        elif args.crop_type == 'Random':
                            img = RandomCrop(img, args.crop_size,
                                             args.crop_size)
                        img = (
                            np.array(img).astype('float32') / 255.0 - 0.5) / 0.5
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
            elif mode == 'test':
                batch_out = []
                for file in self.lines:
                    file = file.strip('\n\r\t ')
                    img = Image.open(os.path.join(self.image_dir,
                                                  file)).convert('RGB')
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
            else:
                raise NotImplementedError('mode [%s] is Error, Please check it',
                                          mode)

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
