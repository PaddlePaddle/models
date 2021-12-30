# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
from __future__ import print_function

import argparse
import os
import numpy as np
from PIL import Image

import paddle
from paddle.io import Dataset
from paddle.vision.transforms import transforms

from paddleslim.quant import quant_post_static


class ImageNetValDataset(Dataset):
    def __init__(self, data_dir, image_size=224, resize_short_size=256):
        super(ImageNetValDataset, self).__init__()
        val_file_list = os.path.join(data_dir, 'val_list.txt')
        test_file_list = os.path.join(data_dir, 'test_list.txt')
        self.data_dir = data_dir

        normalize = transforms.Normalize(
            mean=[123.675, 116.28, 103.53], std=[58.395, 57.120, 57.375])
        self.transform = transforms.Compose([
            transforms.Resize(resize_short_size),
            transforms.CenterCrop(image_size), transforms.Transpose(),
            normalize
        ])

        with open(val_file_list) as flist:
            lines = [line.strip() for line in flist]
            self.data = [line.split() for line in lines]

    def __getitem__(self, index):
        img_path, label = self.data[index]
        img_path = os.path.join(self.data_dir, img_path)
        img = Image.open(img_path).convert('RGB')
        label = np.array([label]).astype(np.int64)
        return self.transform(img), label

    def __len__(self):
        return len(self.data)


def sample_generator(loader):
    def __reader__():
        for indx, data in enumerate(loader):
            images = np.array(data[0])
            yield images

    return __reader__


def main():
    paddle.enable_static()
    place = paddle.CUDAPlace(0) if FLAGS.use_gpu else paddle.CPUPlace()
    val_dataset = ImageNetValDataset(FLAGS.data)
    data_loader = paddle.io.DataLoader(
        val_dataset, places=place, batch_size=FLAGS.batch_size)
    quant_output_dir = os.path.join(FLAGS.output_dir, "mv3_int8_infer")

    exe = paddle.static.Executor(place)
    quant_post_static(
        executor=exe,
        model_dir=FLAGS.model_path,
        quantize_model_path=quant_output_dir,
        sample_generator=sample_generator(data_loader),
        model_filename=FLAGS.model_filename,
        params_filename=FLAGS.params_filename,
        batch_size=FLAGS.batch_size,
        batch_nums=FLAGS.batch_num,
        algo=FLAGS.algo,
        hist_percent=FLAGS.hist_percent)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Quantization on ImageNet")

    parser.add_argument(
        "--model_path", type=str, default=None, help="Inference model path")
    parser.add_argument(
        "--model_filename",
        type=str,
        default=None,
        help="Inference model model_filename")
    parser.add_argument(
        "--params_filename",
        type=str,
        default=None,
        help="Inference model params_filename")
    parser.add_argument(
        "--output_dir", type=str, default='output', help="save dir")
    parser.add_argument(
        '--data',
        default="/dataset/ILSVRC2012",
        help='path to dataset (should have subdirectories named "train" and "val"'
    )
    parser.add_argument(
        '--use_gpu',
        default=True,
        type=bool,
        help='Whether to use GPU or not.')

    # train
    parser.add_argument(
        "--batch_num", default=10, type=int, help="batch num for quant")
    parser.add_argument(
        "--batch_size", default=10, type=int, help="batch size for quant")
    parser.add_argument(
        '--algo', default='hist', type=str, help="calibration algorithm")
    parser.add_argument(
        '--hist_percent',
        default=0.999,
        type=float,
        help="The percentile of algo:hist")

    FLAGS = parser.parse_args()
    assert FLAGS.data, "error: must provide data path"

    main()
