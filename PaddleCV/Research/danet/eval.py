# -*- coding: utf-8 -*-
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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

os.environ['FLAGS_eager_delete_tensor_gb'] = "0.0"
os.environ['FLAGS_fraction_of_gpu_memory_to_use'] = "0.99"

import paddle.fluid as fluid
import paddle
import logging
import math
import numpy as np
import shutil
import os

from PIL import ImageOps, Image, ImageEnhance, ImageFilter
from datetime import datetime

from danet import DANet
from options import Options
from utils.cityscapes_data import cityscapes_train
from utils.cityscapes_data import cityscapes_val
from utils.cityscapes_data import cityscapes_test
from utils.lr_scheduler import Lr
from iou import IOUMetric

#  globals
data_mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
data_std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)


def pad_single_image(image, crop_size):
    w, h = image.size
    pad_h = crop_size - h if h < crop_size else 0
    pad_w = crop_size - w if w < crop_size else 0
    image = ImageOps.expand(image, border=(0, 0, pad_w, pad_h), fill=0)
    assert (image.size[0] >= crop_size and image.size[1] >= crop_size)
    return image


def crop_image(image, h0, w0, h1, w1):
    return image.crop((w0, h0, w1, h1))


def flip_left_right_image(image):
    return image.transpose(Image.FLIP_LEFT_RIGHT)


def resize_image(image, out_h, out_w, mode=Image.BILINEAR):
    return image.resize((out_w, out_h), mode)


def mapper_image(image):
    image_array = np.array(image)
    image_array = image_array.transpose((2, 0, 1))
    image_array = image_array / 255.0
    image_array = (image_array - data_mean) / data_std
    image_array = image_array.astype('float32')
    image_array = image_array[np.newaxis, :]
    return image_array


def get_model(args):
    model = DANet('DANet',
                  backbone=args.backbone,
                  num_classes=args.num_classes,
                  batch_size=1,
                  dilated=args.dilated,
                  multi_grid=args.multi_grid,
                  multi_dilation=args.multi_dilation)
    return model


def copy_model(path, new_path):
    shutil.rmtree(new_path, ignore_errors=True)
    shutil.copytree(path, new_path)
    model_path = os.path.join(new_path, '__model__')
    if os.path.exists(model_path):
        os.remove(model_path)


def mean_iou(pred, label, num_classes=19):
    label = fluid.layers.elementwise_min(fluid.layers.cast(label, np.int32),
                                         fluid.layers.assign(np.array([num_classes], dtype=np.int32)))
    label_ig = (label == num_classes).astype('int32')
    label_ng = (label != num_classes).astype('int32')
    pred = fluid.layers.cast(fluid.layers.argmax(pred, axis=1), 'int32')
    pred = pred * label_ng + label_ig * num_classes
    miou, wrong, correct = fluid.layers.mean_iou(pred, label, num_classes + 1)
    label.stop_gradient = True
    return miou, wrong, correct


def change_model_executor_to_dygraph(args):
    temp_image = fluid.layers.data(name='temp_image', shape=[3, 224, 224], dtype='float32')
    model = get_model(args)
    y = model(temp_image)
    if args.cuda:
        gpu_id = int(os.environ.get('FLAGS_selected_gpus', 0))
    place = fluid.CUDAPlace(gpu_id) if args.cuda else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())
    model_path = args.save_model
    assert os.path.exists(model_path), "Please check whether the executor model file address {} exists. " \
                                       "Note: the executor model file is multiple files.".format(model_path)
    fluid.io.load_persistables(exe, model_path, fluid.default_main_program())
    print('load executor train model successful, start change!')
    param_list = fluid.default_main_program().block(0).all_parameters()
    param_name_list = [p.name for p in param_list]
    temp_dict = {}
    for name in param_name_list:
        tensor = fluid.global_scope().find_var(name).get_tensor()
        npt = np.asarray(tensor)
        temp_dict[name] = npt
    del model
    with fluid.dygraph.guard():
        x = np.random.randn(1, 3, 224, 224).astype('float32')
        x = fluid.dygraph.to_variable(x)
        model = get_model(args)
        y = model(x)
        new_param_dict = {}
        for k, v in temp_dict.items():
            value = v
            value_shape = value.shape
            name = k
            tensor = fluid.layers.create_parameter(shape=value_shape,
                                                   name=name,
                                                   dtype='float32',
                                                   default_initializer=fluid.initializer.NumpyArrayInitializer(value))
            new_param_dict[name] = tensor
        assert len(new_param_dict) == len(
            model.state_dict()), "The number of parameters is not equal. Loading parameters failed, " \
                                 "Please check whether the model is consistent!"
        model.set_dict(new_param_dict)
        fluid.save_dygraph(model.state_dict(), model_path)
        del model
        del temp_dict
        print('change executor model to dygraph successful!')


def eval(args):
    if args.change_executor_to_dygraph:
        change_model_executor_to_dygraph(args)
    with fluid.dygraph.guard():
        num_classes = args.num_classes
        base_size = args.base_size
        crop_size = args.crop_size
        multi_scales = args.multi_scales
        flip = args.flip

        if not multi_scales:
            scales = [1.0]
        else:
            # scales = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.2]
            scales = [0.5, 0.75, 1.0, 1.25, 1.35, 1.5, 1.75, 2.0, 2.2]  # It might work better

        if len(scales) == 1:  # single scale
            # stride_rate = 2.0 / 3.0
            stride_rate = 1.0 / 2.0  # It might work better
        else:
            stride_rate = 1.0 / 2.0
        stride = int(crop_size * stride_rate)  # slid stride

        model = get_model(args)
        x = np.random.randn(1, 3, 224, 224).astype('float32')
        x = fluid.dygraph.to_variable(x)
        y = model(x)
        iou = IOUMetric(num_classes)
        model_path = args.save_model
        # load_better_model
        if paddle.__version__ == '1.5.2' and args.load_better_model:
            assert os.path.exists(model_path), "your input save_model: {} ,but '{}' is not exists".format(
                model_path, model_path)
            print('better model exist!')
            new_model_path = 'dygraph/' + model_path
            copy_model(model_path, new_model_path)
            model_param, _ = fluid.dygraph.load_persistables(new_model_path)
            model.load_dict(model_param)
        elif args.load_better_model:
            assert os.path.exists(model_path + '.pdparams'), "your input save_model: {} ,but '{}' is not exists".format(
                model_path, model_path + '.pdparams')
            print('better model exist!')
            model_param, _ = fluid.dygraph.load_dygraph(model_path)
            model.load_dict(model_param)
        else:
            raise ValueError('Please set --load_better_model!')

        assert len(model_param) == len(
            model.state_dict()), "The number of parameters is not equal. Loading parameters failed, " \
                                 "Please check whether the model is consistent!"
        model.eval()

        prev_time = datetime.now()
        # reader = cityscapes_test(split='test', base_size=2048, crop_size=1024, scale=True, xmap=True)
        reader = cityscapes_test(split='val', base_size=2048, crop_size=1024, scale=True, xmap=True)

        print('MultiEvalModule: base_size {}, crop_size {}'.
              format(base_size, crop_size))
        print('scales: {}'.format(scales))
        print('val ing...')
        logging.basicConfig(level=logging.INFO,
                            filename='DANet_{}_eval_dygraph.log'.format(args.backbone),
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        logging.info('DANet')
        logging.info(args)
        palette = pat()
        for data in reader():
            image = data[0]
            label_path = data[1]  # val_label is a picture, test_label is a path
            label = Image.open(label_path, mode='r')  # val_label is a picture, test_label is a path
            save_png_path = label_path.replace('val', '{}_val'.format(args.backbone)).replace('test', '{}_test'.format(
                args.backbone))
            label_np = np.array(label)
            w, h = image.size  # h 1024, w 2048
            scores = np.zeros(shape=[num_classes, h, w], dtype='float32')
            for scale in scales:
                long_size = int(math.ceil(base_size * scale))  # long_size
                if h > w:
                    height = long_size
                    width = int(1.0 * w * long_size / h + 0.5)
                    short_size = width
                else:
                    width = long_size
                    height = int(1.0 * h * long_size / w + 0.5)
                    short_size = height

                cur_img = resize_image(image, height, width)
                # pad
                if long_size <= crop_size:
                    pad_img = pad_single_image(cur_img, crop_size)
                    pad_img = mapper_image(pad_img)
                    pad_img = fluid.dygraph.to_variable(pad_img)
                    pred1, pred2, pred3 = model(pad_img)
                    pred1 = pred1.numpy()
                    outputs = pred1[:, :, :height, :width]
                    if flip:
                        pad_img_filp = flip_left_right_image(cur_img)
                        pad_img_filp = pad_single_image(pad_img_filp, crop_size)  # pad
                        pad_img_filp = mapper_image(pad_img_filp)
                        pad_img_filp = fluid.dygraph.to_variable(pad_img_filp)
                        pred1, pred2, pred3 = model(pad_img_filp)
                        pred1 = fluid.layers.reverse(pred1, axis=3)
                        pred1 = pred1.numpy()
                        outputs += pred1[:, :, :height, :width]
                else:
                    if short_size < crop_size:
                        # pad if needed
                        pad_img = pad_single_image(cur_img, crop_size)
                    else:
                        pad_img = cur_img
                    pw, ph = pad_img.size
                    assert (ph >= height and pw >= width)

                    # slid window
                    h_grids = int(math.ceil(1.0 * (ph - crop_size) / stride)) + 1
                    w_grids = int(math.ceil(1.0 * (pw - crop_size) / stride)) + 1
                    outputs = np.zeros(shape=[1, num_classes, ph, pw], dtype='float32')
                    count_norm = np.zeros(shape=[1, 1, ph, pw], dtype='int32')
                    for idh in range(h_grids):
                        for idw in range(w_grids):
                            h0 = idh * stride
                            w0 = idw * stride
                            h1 = min(h0 + crop_size, ph)
                            w1 = min(w0 + crop_size, pw)
                            crop_img = crop_image(pad_img, h0, w0, h1, w1)
                            pad_crop_img = pad_single_image(crop_img, crop_size)
                            pad_crop_img = mapper_image(pad_crop_img)
                            pad_crop_img = fluid.dygraph.to_variable(pad_crop_img)
                            pred1, pred2, pred3 = model(pad_crop_img)  # shape [1, num_class, h, w]
                            pred = pred1.numpy()  # channel, h, w
                            outputs[:, :, h0:h1, w0:w1] += pred[:, :, 0:h1 - h0, 0:w1 - w0]
                            count_norm[:, :, h0:h1, w0:w1] += 1
                            if flip:
                                pad_img_filp = flip_left_right_image(crop_img)
                                pad_img_filp = pad_single_image(pad_img_filp, crop_size)  # pad
                                pad_img_array = mapper_image(pad_img_filp)
                                pad_img_array = fluid.dygraph.to_variable(pad_img_array)
                                pred1, pred2, pred3 = model(pad_img_array)
                                pred1 = fluid.layers.reverse(pred1, axis=3)
                                pred = pred1.numpy()
                                outputs[:, :, h0:h1, w0:w1] += pred[:, :, 0:h1 - h0, 0:w1 - w0]
                                count_norm[:, :, h0:h1, w0:w1] += 1
                    assert ((count_norm == 0).sum() == 0)
                    outputs = outputs / count_norm
                    outputs = outputs[:, :, :height, :width]
                outputs = fluid.dygraph.to_variable(outputs)
                outputs = fluid.layers.resize_bilinear(outputs, out_shape=[h, w])
                score = outputs.numpy()[0]
                scores += score  # the sum of all scales, shape: [channel, h, w]
                pred = np.argmax(score, axis=0).astype('uint8')
                picture_path = '{}'.format(save_png_path).replace('.png', '_scale_{}'.format(scale))
                save_png(pred, palette, picture_path)
            pred = np.argmax(scores, axis=0).astype('uint8')
            picture_path = '{}'.format(save_png_path).replace('.png', '_scores')
            save_png(pred, palette, picture_path)
            iou.add_batch(pred, label_np)  # cal iou
        print('eval done!')
        logging.info('eval done!')
        acc, acc_cls, iu, mean_iu, fwavacc, kappa = iou.evaluate()
        print('acc = {}'.format(acc))
        logging.info('acc = {}'.format(acc))
        print('acc_cls = {}'.format(acc_cls))
        logging.info('acc_cls = {}'.format(acc_cls))
        print('iu = {}'.format(iu))
        logging.info('iu = {}'.format(iu))
        print('mean_iou -- 255 = {}'.format(mean_iu))
        logging.info('mean_iou --255 = {}'.format(mean_iu))
        print('mean_iou = {}'.format(np.nanmean(iu[:-1])))  # realy iou
        logging.info('mean_iou = {}'.format(np.nanmean(iu[:-1])))
        print('fwavacc = {}'.format(fwavacc))
        logging.info('fwavacc = {}'.format(fwavacc))
        print('kappa = {}'.format(kappa))
        logging.info('kappa = {}'.format(kappa))
        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)
        print('val ' + time_str)
        logging.info('val ' + time_str)


def save_png(pred_value, palette, name):
    if isinstance(pred_value, np.ndarray):
        if pred_value.ndim == 3:
            batch_size = pred_value.shape[0]
            if batch_size == 1:
                pred_value = pred_value.squeeze(axis=0)
                image = Image.fromarray(pred_value).convert('P')
                image.putpalette(palette)
                save_path = '{}.png'.format(name)
                save_dir = os.path.dirname(save_path)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                image.save(save_path)
            else:
                for batch_id in range(batch_size):
                    value = pred_value[batch_id]
                    image = Image.fromarray(value).convert('P')
                    image.putpalette(palette)
                    save_path = '{}.png'.format(name[batch_id])
                    save_dir = os.path.dirname(save_path)
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    image.save(save_path)
        elif pred_value.ndim == 2:
            image = Image.fromarray(pred_value).convert('P')
            image.putpalette(palette)
            save_path = '{}.png'.format(name)
            save_dir = os.path.dirname(save_path)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            image.save(save_path)
    else:
        raise ValueError('Only support nd-array')


def save_png_test(path):
    im = Image.open(path)
    im_array = np.array(im).astype('uint8')
    save_png(im_array, pat(), 'save_png_test')


def pat():
    palette = []
    for i in range(256):
        palette.extend((i, i, i))
    palette[:3 * 19] = np.array([[128, 64, 128],
                                 [244, 35, 232],
                                 [70, 70, 70],
                                 [102, 102, 156],
                                 [190, 153, 153],
                                 [153, 153, 153],
                                 [250, 170, 30],
                                 [220, 220, 0],
                                 [107, 142, 35],
                                 [152, 251, 152],
                                 [70, 130, 180],
                                 [220, 20, 60],
                                 [255, 0, 0],
                                 [0, 0, 142],
                                 [0, 0, 70],
                                 [0, 60, 100],
                                 [0, 80, 100],
                                 [0, 0, 230],
                                 [119, 11, 32]], dtype='uint8').flatten()
    return palette


if __name__ == '__main__':
    options = Options()
    args = options.parse()
    options.print_args()
    eval(args)

