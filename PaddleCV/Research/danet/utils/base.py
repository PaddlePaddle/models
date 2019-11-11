# -*- encoding: utf-8 -*-
# Software: PyCharm
# Time    : 2019/9/13 
# Author  : Wang
# File    : base.py
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
    """数据基类，实现数据增强，image、label处理"""

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
        """
        针对train, 含数据增强
        :param image:
        :param label:
        :return: 增强后的image, label
        """
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
        # print(np.array(label))
        # print(image.size)

        # 四周填充
        # 如果短的一边比裁剪的size小，则填充，image补mean,label补255
        if short_size < crop_size:
            pad_h = crop_size - out_h if out_h < crop_size else 0
            pad_w = crop_size - out_w if out_w < crop_size else 0
            # print('pad_size', pad_w, pad_h)
            image = ImageOps.expand(image, border=(pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2),
                                    fill=0)
            label = ImageOps.expand(label, border=(pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2),
                                    fill=255)
            # print(np.array(image).shape)

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
        """
        针对 val, 不含数据增强，只有同比例变换，中心裁剪
        :param image:
        :param label:
        :return: 中心裁剪后的image, label
        """
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
        """针对 test, 不含数据增强，只有同比例变换，中心裁剪"""
        pass


if __name__ == '__main__':
    b = BaseDataSet(r'..\2007_000027.jpg', 'train')
    image = Image.open(r'..\2007_000033.jpg', 'r')
    label = Image.open(r'..\2007_000033.png', 'r')
    image, label = b.sync_transform(image, label)
    print(image.size)
    print(label.size)
    # label = label.convert('P')
    print(np.array(label)[240])
    image.show()
    label.show()
