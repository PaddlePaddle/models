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


import random
import numpy as np
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import os
import sys

curPath = os.path.abspath(os.path.dirname(__file__))
parentPath = os.path.split(curPath)[0]
rootPath = os.path.split(parentPath)[0]
sys.path.append(rootPath)


class BaseDataSet:

    def __init__(self, root, split, base_size=1024, crop_size=768, scale=True):
        self.root = root
        support = ['train', 'train_val', 'val', 'test']
        assert split in support, "split= \'{}\' not in {}".format(split, support)
        self.split = split
        self.crop_size = crop_size  # 裁剪大小
        self.base_size = base_size  # 图片最短边
        self.scale = scale
        self.image_path = None
        self.label_path = None

    def sync_transform(self, image, label):
        crop_size = self.crop_size
        if self.scale:
            short_size = random.randint(int(self.base_size * 0.75), int(self.base_size * 2.0))
        else:
            short_size = self.base_size

        # 随机左右翻转
        if random.random() > 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)
        w, h = image.size

        # 同比例缩放
        if h > w:
            out_w = short_size
            out_h = int(1.0 * h / w * out_w)
        else:
            out_h = short_size
            out_w = int(1.0 * w / h * out_h)
        image = image.resize((out_w, out_h), Image.BILINEAR)
        label = label.resize((out_w, out_h), Image.NEAREST)

        # 四周填充
        if short_size < crop_size:
            pad_h = crop_size - out_h if out_h < crop_size else 0
            pad_w = crop_size - out_w if out_w < crop_size else 0
            image = ImageOps.expand(image, border=(pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2),
                                    fill=0)
            label = ImageOps.expand(label, border=(pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2),
                                    fill=255)

        # 随机裁剪
        w, h = image.size
        x = random.randint(0, w - crop_size)
        y = random.randint(0, h - crop_size)
        image = image.crop((x, y, x + crop_size, y + crop_size))
        label = label.crop((x, y, x + crop_size, y + crop_size))

        # # 高斯模糊，可选
        # if random.random() > 0.7:
        #     image = image.filter(ImageFilter.GaussianBlur(radius=random.random()))

        # 可选
        # if random.random() > 0.7:
        #     # 随机亮度
        #     factor = np.random.uniform(0.75, 1.25)
        #     image = ImageEnhance.Brightness(image).enhance(factor)
        #
        #     # 颜色抖动
        #     factor = np.random.uniform(0.75, 1.25)
        #     image = ImageEnhance.Color(image).enhance(factor)
        #
        #     # 随机对比度
        #     factor = np.random.uniform(0.75, 1.25)
        #     image = ImageEnhance.Contrast(image).enhance(factor)
        #
        #     # 随机锐度
        #     factor = np.random.uniform(0.75, 1.25)
        #     image = ImageEnhance.Sharpness(image).enhance(factor)
        return image, label

    def sync_val_transform(self, image, label):
        crop_size = self.crop_size
        short_size = self.base_size

        w, h = image.size

        # 同比例缩放
        if h > w:
            out_w = short_size
            out_h = int(1.0 * h / w * out_w)
        else:
            out_h = short_size
            out_w = int(1.0 * w / h * out_h)
        image = image.resize((out_w, out_h), Image.BILINEAR)
        label = label.resize((out_w, out_h), Image.NEAREST)

        # 中心裁剪
        w, h = image.size
        x1 = int(round((w - crop_size) / 2.))
        y1 = int(round((h - crop_size) / 2.))
        image = image.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        label = label.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        return image, label

    def eval(self, image):
        pass
