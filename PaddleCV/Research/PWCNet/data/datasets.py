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
# @FileName: datasets.py reference https://github.com/NVIDIA/flownet2-pytorch/blob/master/datasets.py
import paddle
import paddle.fluid as fluid
import numpy as np
import argparse
import os, math, random
import sys
from os.path import *
import numpy as np
from glob import glob
sys.path.append('../')
import data.utils.frame_utils as frame_utils
from scipy.misc import imsave
from src import flow_vis
from src.read_files import read_txt_to_index


class StaticRandomCrop(object):
    def __init__(self, image_size, crop_size):
        self.th, self.tw = crop_size
        h, w = image_size
        self.h1 = random.randint(0, h - self.th)
        self.w1 = random.randint(0, w - self.tw)

    def __call__(self, img):
        return img[self.h1:(self.h1 + self.th), self.w1:(self.w1 + self.tw), :]


class StaticCenterCrop(object):
    def __init__(self, image_size, crop_size):
        self.th, self.tw = crop_size
        self.h, self.w = image_size

    def __call__(self, img):
        return img[(self.h - self.th) // 2:(self.h + self.th) // 2, (self.w - self.tw) // 2:(self.w + self.tw) // 2, :]


class MpiSintel(object):
    def __init__(self, args, is_cropped=False, root='', dstype='clean', replicates=1):
        self.args = args
        self.is_cropped = is_cropped
        self.crop_size = args.crop_size
        self.render_size = args.inference_size
        self.replicates = replicates

        flow_root = join(root, 'flow')
        image_root = join(root, dstype)

        file_list = sorted(glob(join(flow_root, '*/*.flo')))

        self.flow_list = []
        self.image_list = []

        for file in file_list:
            if 'test' in file:
                # print file
                continue

            fbase = file[len(flow_root) + 1:]
            fprefix = fbase[:-8]
            fnum = int(fbase[-8:-4])

            img1 = join(image_root, fprefix + "%04d" % (fnum + 0) + '.png')
            img2 = join(image_root, fprefix + "%04d" % (fnum + 1) + '.png')

            if not isfile(img1) or not isfile(img2) or not isfile(file):
                continue

            self.image_list += [[img1, img2]]
            self.flow_list += [file]

        self.size = len(self.image_list)

        self.frame_size = frame_utils.read_gen(self.image_list[0][0]).shape

        if (self.render_size[0] < 0) or (self.render_size[1] < 0) or (self.frame_size[0] % 64) or (
                self.frame_size[1] % 64):
            self.render_size[0] = ((self.frame_size[0]) // 64) * 64
            self.render_size[1] = ((self.frame_size[1]) // 64) * 64

        args.inference_size = self.render_size

        assert (len(self.image_list) == len(self.flow_list))

    def __getitem__(self, index):

        index = index % self.size

        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])

        flow = frame_utils.read_gen(self.flow_list[index])

        images = [img1, img2]
        image_size = img1.shape[:2]

        if self.is_cropped:
            cropper = StaticRandomCrop(image_size, self.crop_size)
        else:
            cropper = StaticCenterCrop(image_size, self.render_size)
        images = list(map(cropper, images))
        flow = cropper(flow)

        images = np.array(images).transpose(3, 0, 1, 2)
        flow = flow.transpose(2, 0, 1)
        return [images], [flow]

    def __len__(self):
        return self.size * self.replicates


class MpiSintelClean(MpiSintel):
    def __init__(self, args, is_cropped=False, root='', replicates=1):
        super(MpiSintelClean, self).__init__(args, is_cropped=is_cropped, root=root, dstype='clean',
                                             replicates=replicates)


class MpiSintelFinal(MpiSintel):
    def __init__(self, args, is_cropped=False, root='', replicates=1):
        super(MpiSintelFinal, self).__init__(args, is_cropped=is_cropped, root=root, dstype='final',
                                             replicates=replicates)


class FlyingChairs(object):
    def __init__(self, train_val, args, is_cropped, txt_file, root='/path/to/FlyingChairs_release/data', replicates=1):
        self.args = args
        self.is_cropped = is_cropped
        self.crop_size = args.crop_size
        self.render_size = args.inference_size
        self.replicates = replicates

        images = sorted(glob(join(root, '*.ppm')))

        flow_list = sorted(glob(join(root, '*.flo')))

        assert (len(images) // 2 == len(flow_list))

        image_list = []
        for i in range(len(flow_list)):
            im1 = images[2 * i]
            im2 = images[2 * i + 1]
            image_list += [[im1, im2]]

        assert len(image_list) == len(flow_list)
        if train_val == 'train':
            intindex = np.array(read_txt_to_index(txt_file))
            image_list = np.array(image_list)
            image_list = image_list[intindex == 1]
            image_list = image_list.tolist()
            flow_list = np.array(flow_list)
            flow_list = flow_list[intindex == 1]
            flow_list = flow_list.tolist()
            assert len(image_list) == len(flow_list)
        elif train_val == 'val':
            intindex = np.array(read_txt_to_index(txt_file))
            image_list = np.array(image_list)
            image_list = image_list[intindex == 2]
            image_list = image_list.tolist()
            flow_list = np.array(flow_list)
            flow_list = flow_list[intindex == 2]
            flow_list = flow_list.tolist()
            assert len(image_list) == len(flow_list)
        else:
            raise ValueError('FlyingChairs_train_val.txt not found for txt_file ......')
        self.flow_list = flow_list
        self.image_list = image_list

        self.size = len(self.image_list)

        self.frame_size = frame_utils.read_gen(self.image_list[0][0]).shape

        if (self.render_size[0] < 0) or (self.render_size[1] < 0) or (self.frame_size[0] % 64) or (
                self.frame_size[1] % 64):
            self.render_size[0] = ((self.frame_size[0]) // 64) * 64
            self.render_size[1] = ((self.frame_size[1]) // 64) * 64

        args.inference_size = self.render_size

    def __getitem__(self, index):
        index = index % self.size

        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])

        flow = frame_utils.read_gen(self.flow_list[index])

        images = [img1, img2]
        image_size = img1.shape[:2]
        if self.is_cropped:
            cropper = StaticRandomCrop(image_size, self.crop_size)
        else:
            cropper = StaticCenterCrop(image_size, self.render_size)
        images = list(map(cropper, images))
        flow = cropper(flow)

        images = np.array(images).transpose(3, 0, 1, 2)
        flow = flow.transpose(2, 0, 1)
        return [images], [flow]

    def __len__(self):
        return self.size * self.replicates


def reader_flyingchairs(dataset):
    n = len(dataset)

    def reader():
        for i in range(n):
            a, b = dataset[i]
            yield a[0][:,0,:,:].transpose(1,2,0), a[0][:,1,:,:].transpose(1,2,0), b[0].transpose(1, 2, 0)# a single entry of data is created each time
    return reader


class FlyingThings(object):
    def __init__(self, args, is_cropped, root='/path/to/flyingthings3d', dstype='frames_cleanpass', replicates=1):
        self.args = args
        self.is_cropped = is_cropped
        self.crop_size = args.crop_size
        self.render_size = args.inference_size
        self.replicates = replicates

        image_dirs = sorted(glob(join(root, dstype, 'TRAIN/*/*')))
        image_dirs = sorted([join(f, 'left') for f in image_dirs] + [join(f, 'right') for f in image_dirs])

        flow_dirs = sorted(glob(join(root, 'optical_flow_flo_format/TRAIN/*/*')))
        flow_dirs = sorted(
            [join(f, 'into_future/left') for f in flow_dirs] + [join(f, 'into_future/right') for f in flow_dirs])

        assert (len(image_dirs) == len(flow_dirs))

        self.image_list = []
        self.flow_list = []

        for idir, fdir in zip(image_dirs, flow_dirs):
            images = sorted(glob(join(idir, '*.png')))
            flows = sorted(glob(join(fdir, '*.flo')))
            for i in range(len(flows)):
                self.image_list += [[images[i], images[i + 1]]]
                self.flow_list += [flows[i]]

        assert len(self.image_list) == len(self.flow_list)

        self.size = len(self.image_list)

        self.frame_size = frame_utils.read_gen(self.image_list[0][0]).shape

        if (self.render_size[0] < 0) or (self.render_size[1] < 0) or (self.frame_size[0] % 64) or (
                self.frame_size[1] % 64):
            self.render_size[0] = ((self.frame_size[0]) // 64) * 64
            self.render_size[1] = ((self.frame_size[1]) // 64) * 64

        args.inference_size = self.render_size

    def __getitem__(self, index):
        index = index % self.size

        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])

        flow = frame_utils.read_gen(self.flow_list[index])

        images = [img1, img2]
        image_size = img1.shape[:2]
        if self.is_cropped:
            cropper = StaticRandomCrop(image_size, self.crop_size)
        else:
            cropper = StaticCenterCrop(image_size, self.render_size)
        images = list(map(cropper, images))
        flow = cropper(flow)

        images = np.array(images).transpose(3, 0, 1, 2)
        flow = flow.transpose(2, 0, 1)
        return [images], [flow]

    def __len__(self):
        return self.size * self.replicates


class FlyingThingsClean(FlyingThings):
    def __init__(self, args, is_cropped=False, root='', replicates=1):
        super(FlyingThingsClean, self).__init__(args, is_cropped=is_cropped, root=root, dstype='frames_cleanpass',
                                                replicates=replicates)


class FlyingThingsFinal(FlyingThings):
    def __init__(self, args, is_cropped=False, root='', replicates=1):
        super(FlyingThingsFinal, self).__init__(args, is_cropped=is_cropped, root=root, dstype='frames_finalpass',
                                                replicates=replicates)


class ChairsSDHom(object):
    def __init__(self, args, is_cropped, root='/path/to/chairssdhom/data', dstype='train', replicates=1):
        self.args = args
        self.is_cropped = is_cropped
        self.crop_size = args.crop_size
        self.render_size = args.inference_size
        self.replicates = replicates

        image1 = sorted(glob(join(root, dstype, 't0/*.png')))
        image2 = sorted(glob(join(root, dstype, 't1/*.png')))
        self.flow_list = sorted(glob(join(root, dstype, 'flow/*.flo')))

        assert (len(image1) == len(self.flow_list))

        self.image_list = []
        for i in range(len(self.flow_list)):
            im1 = image1[i]
            im2 = image2[i]
            self.image_list += [[im1, im2]]

        assert len(self.image_list) == len(self.flow_list)

        self.size = len(self.image_list)

        self.frame_size = frame_utils.read_gen(self.image_list[0][0]).shape

        if (self.render_size[0] < 0) or (self.render_size[1] < 0) or (self.frame_size[0] % 64) or (
                self.frame_size[1] % 64):
            self.render_size[0] = ((self.frame_size[0]) // 64) * 64
            self.render_size[1] = ((self.frame_size[1]) // 64) * 64

        args.inference_size = self.render_size

    def __getitem__(self, index):
        index = index % self.size

        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])

        flow = frame_utils.read_gen(self.flow_list[index])
        flow = flow[::-1, :, :]

        images = [img1, img2]
        image_size = img1.shape[:2]
        if self.is_cropped:
            cropper = StaticRandomCrop(image_size, self.crop_size)
        else:
            cropper = StaticCenterCrop(image_size, self.render_size)
        images = list(map(cropper, images))
        flow = cropper(flow)

        images = np.array(images).transpose(3, 0, 1, 2)
        flow = flow.transpose(2, 0, 1)
        return [images], [flow]

    def __len__(self):
        return self.size * self.replicates


class ChairsSDHomTrain(ChairsSDHom):
    def __init__(self, args, is_cropped=False, root='', replicates=1):
        super(ChairsSDHomTrain, self).__init__(args, is_cropped=is_cropped, root=root, dstype='train',
                                               replicates=replicates)


class ChairsSDHomTest(ChairsSDHom):
    def __init__(self, args, is_cropped=False, root='', replicates=1):
        super(ChairsSDHomTest, self).__init__(args, is_cropped=is_cropped, root=root, dstype='test',
                                              replicates=replicates)


class ImagesFromFolder(object):
    def __init__(self, args, is_cropped, root='/path/to/frames/only/folder', iext='png', replicates=1):
        self.args = args
        self.is_cropped = is_cropped
        self.crop_size = args.crop_size
        self.render_size = args.inference_size
        self.replicates = replicates

        images = sorted(glob(join(root, '*.' + iext)))
        self.image_list = []
        for i in range(len(images) - 1):
            im1 = images[i]
            im2 = images[i + 1]
            self.image_list += [[im1, im2]]

        self.size = len(self.image_list)

        self.frame_size = frame_utils.read_gen(self.image_list[0][0]).shape

        if (self.render_size[0] < 0) or (self.render_size[1] < 0) or (self.frame_size[0] % 64) or (
                self.frame_size[1] % 64):
            self.render_size[0] = ((self.frame_size[0]) // 64) * 64
            self.render_size[1] = ((self.frame_size[1]) // 64) * 64

        args.inference_size = self.render_size

    def __getitem__(self, index):
        index = index % self.size

        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])

        images = [img1, img2]
        image_size = img1.shape[:2]
        if self.is_cropped:
            cropper = StaticRandomCrop(image_size, self.crop_size)
        else:
            cropper = StaticCenterCrop(image_size, self.render_size)
        images = list(map(cropper, images))

        images = np.array(images).transpose(3, 0, 1, 2)
        return [images], [np.zeros(images.size()[0:1] + (2,) + images.size()[-2:])]

    def __len__(self):
        return self.size * self.replicates


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.inference_size = [1080, 1920]
    args.crop_size = [384, 512]

    index = 50
    flyingchairs_dataset = FlyingChairs(args, True, root='/ssd2/zhenghe/DATA/FlyingChairs_release/data')
    # a, b = flyingchairs_dataset[index]
    # im1 = a[0][:,0,:,:].transpose(1,2,0)
    # im2 = a[0][:,1,:,:].transpose(1,2,0)
    # flo = b[0].transpose(1, 2, 0) / 20.0
    # flow_color = flow_vis.flow_to_color(flo, convert_to_bgr=False)
    # imsave('./hsv_pd.png', flow_color)
    sample_num = len(flyingchairs_dataset)
    reader = reader_flyingchairs(flyingchairs_dataset)
    BATCH_SIZE = 8
    train_batch_reader = paddle.batch(reader, BATCH_SIZE, drop_last=True)
    epoch_num = 1

    with fluid.dygraph.guard():
        for epoch in range(epoch_num):
            for batch_id, data in enumerate(train_batch_reader()):
                im1_data = np.array(
                    [x[0] for x in data]).astype('float32')
                im2_data = np.array(
                    [x[1] for x in data]).astype('float32')
                flo_data = np.array(
                    [x[2] for x in data]).astype('float32')
                if batch_id % 500 == 0:
                # if batch_id < 10:
                    print(batch_id)
                    print(im1_data.shape)
                    print(im2_data.shape)
                    print(flo_data.shape)
                    im1 = im1_data[0, :, :, :]
                    im2 = im2_data[0, :, :, :]
                    flo = flo_data[0, :, :, :]
                    print(im1.shape)
                    print(im2.shape)
                    print(flo.shape)
                    imsave('./img1.png', im1)
                    imsave('./img2.png', im2)
                    flow_color = flow_vis.flow_to_color(flo, convert_to_bgr=False)
                    imsave('./hsv_pd.png', flow_color)
            print("batch_id:", batch_id)
            print(batch_id * BATCH_SIZE)
            print(sample_num)
        # img = fluid.dygraph.to_variable(dy_x_data)





