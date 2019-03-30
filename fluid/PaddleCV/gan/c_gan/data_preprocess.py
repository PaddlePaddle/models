from __future__ import print_function
from six.moves import range

import gzip
import numpy
import argparse
import struct


#TODO add celeA dataset
def reader_creator(image_filename, label_filename, buffer_size):
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
                    images = numpy.reshape(images_temp, (
                        buffer_size, rows * cols)).astype('float32')
                    offset_img += struct.calcsize(fmt_images)

                    images = images / 255.0 * 2.0 - 1.0
                    for i in range(buffer_size):
                        yield images[i, :], int(
                            labels[i])  # get image and label

    return reader
