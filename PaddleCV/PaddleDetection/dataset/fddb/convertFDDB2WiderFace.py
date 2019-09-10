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

import os
from math import *
from PIL import Image
import argparse


def filterCoordinate(c, m):
    if c < 0:
        return 0
    elif c > m:
        return m
    else:
        return c


def list_all_files(rootdir):
    _files = []
    list = os.listdir(rootdir)
    for i in range(len(list)):
        path = os.path.join(rootdir, list[i])
        if os.path.isdir(path):
            _files.extend(list_all_files(path))
        if os.path.isfile(path):
            _files.append(path)
    return _files


def convert(root_dir, ellipse_filename, rect_file):
    with open(ellipse_filename) as fe:
        lines = [line.rstrip('\n') for line in fe]

    i = 0
    while i < len(lines):
        img_file = os.path.join(root_dir, lines[i] + '.jpg')
        img = Image.open(img_file)
        w = img.size[0]
        h = img.size[1]
        num_faces = int(lines[i + 1])
        rect_file.write(lines[i] + '.jpg' + '\n')
        rect_file.write(str(num_faces) + '\n')
        for j in range(num_faces):
            ellipse = lines[i + 2 + j].split()[0:5]
            a = float(ellipse[0])
            b = float(ellipse[1])
            angle = float(ellipse[2])
            centre_x = float(ellipse[3])
            centre_y = float(ellipse[4])

            tan_t = -(b / a) * tan(angle)
            t = atan(tan_t)
            x1 = centre_x + (a * cos(t) * cos(angle) - b * sin(t) * sin(angle))
            x2 = centre_x + (a * cos(t + pi) * cos(angle) - b * sin(t + pi) * sin(angle))
            x_max = filterCoordinate(max(x1, x2), w)
            x_min = filterCoordinate(min(x1, x2), w)

            if tan(angle) != 0:
                tan_t = (b / a) * (1 / tan(angle))
            else:
                tan_t = (b / a) * (1 / (tan(angle) + 0.0001))
            t = atan(tan_t)
            y1 = centre_y + (b * sin(t) * cos(angle) + a * cos(t) * sin(angle))
            y2 = centre_y + (b * sin(t + pi) * cos(angle) + a * cos(t + pi) * sin(angle))
            y_max = filterCoordinate(max(y1, y2), h)
            y_min = filterCoordinate(min(y1, y2), h)

            text = str(x_min) + ' ' + str(y_min) + ' ' + str(x_max - x_min) + ' ' + str(y_max - y_min) + '\n'
            rect_file.write(text)

        i = i + num_faces + 2




def main():
    fddb_fold = FLAGS.fddb_fold
    assert os.path.exists(fddb_fold), 'fddb_fold not found'
    anno_fold = os.path.join(fddb_fold, 'FDDB-folds')
    assert os.path.exists(anno_fold), 'fddb_fold  anno dir not found'
    ellipse_files_list = os.listdir(anno_fold)
    wider_face_dir = os.path.join(fddb_fold, 'wider_face_split')
    if not os.path.exists(wider_face_dir):
        os.makedirs(wider_face_dir)
    gt_txtfile = os.path.join(wider_face_dir, "wider_face_train_bbx_gt.txt")
    rect_filename = open(gt_txtfile, 'wb')
    for file in ellipse_files_list:
        if 'ellipseList' in file:
            ellipse_filename = os.path.join(anno_fold, file)
            convert(fddb_fold, ellipse_filename, rect_filename)

    rect_filename.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fddb_fold",
        type=str,
        default=None,
        help="dataset dir.")
    FLAGS = parser.parse_args()
    main()