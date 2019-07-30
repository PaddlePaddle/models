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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import paddle.fluid as fluid
import os
import sys
import math
import distutils.util
import numpy as np
import inspect
import matplotlib
import six
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import imageio
import copy

img_dim = 28


def plot(gen_data):
    pad_dim = 1
    paded = pad_dim + img_dim
    gen_data = gen_data.reshape(gen_data.shape[0], img_dim, img_dim)
    n = int(math.ceil(math.sqrt(gen_data.shape[0])))
    gen_data = (np.pad(
        gen_data, [[0, n * n - gen_data.shape[0]], [pad_dim, 0], [pad_dim, 0]],
        'constant').reshape((n, n, paded, paded)).transpose((0, 2, 1, 3))
                .reshape((n * paded, n * paded)))
    fig = plt.figure(figsize=(8, 8))
    plt.axis('off')
    plt.imshow(gen_data, cmap='Greys_r', vmin=-1, vmax=1)
    return fig


def checkpoints(epoch, cfg, exe, trainer, name):
    output_path = os.path.join(cfg.output, 'checkpoints', str(epoch))
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    fluid.io.save_persistables(
        exe, os.path.join(output_path, name), main_program=trainer.program)
    print('save checkpoints {} to {}'.format(name, output_path))
    sys.stdout.flush()


def init_checkpoints(cfg, exe, trainer, name):
    assert os.path.exists(cfg.init_model), "{} cannot be found.".format(
        cfg.init_model)
    fluid.io.load_persistables(
        exe, os.path.join(cfg.init_model, name), main_program=trainer.program)
    print('load checkpoints {} {} DONE'.format(cfg.init_model, name))
    sys.stdout.flush()


def save_test_image(epoch,
                    cfg,
                    exe,
                    place,
                    test_program,
                    g_trainer,
                    A_test_reader,
                    B_test_reader=None):
    out_path = os.path.join(cfg.output, 'test')
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    if cfg.model_net == "Pix2pix":
        for data in zip(A_test_reader()):
            data_A, data_B, name = data[0]
            name = name[0]
            tensor_A = fluid.LoDTensor()
            tensor_B = fluid.LoDTensor()
            tensor_A.set(data_A, place)
            tensor_B.set(data_B, place)
            fake_B_temp = exe.run(
                test_program,
                fetch_list=[g_trainer.fake_B],
                feed={"input_A": tensor_A,
                      "input_B": tensor_B})
            fake_B_temp = np.squeeze(fake_B_temp[0]).transpose([1, 2, 0])
            input_A_temp = np.squeeze(data_A[0]).transpose([1, 2, 0])
            input_B_temp = np.squeeze(data_A[0]).transpose([1, 2, 0])

            imageio.imwrite(out_path + "/fakeB_" + str(epoch) + "_" + name, (
                (fake_B_temp + 1) * 127.5).astype(np.uint8))
            imageio.imwrite(out_path + "/inputA_" + str(epoch) + "_" + name, (
                (input_A_temp + 1) * 127.5).astype(np.uint8))
            imageio.imwrite(out_path + "/inputB_" + str(epoch) + "_" + name, (
                (input_B_temp + 1) * 127.5).astype(np.uint8))
    elif cfg.model_net == "StarGAN":
        for data in zip(A_test_reader()):
            real_img, label_org, name = data[0]
            attr_names = cfg.selected_attrs.split(',')
            tensor_img = fluid.LoDTensor()
            tensor_label_org = fluid.LoDTensor()
            tensor_img.set(real_img, place)
            tensor_label_org.set(label_org, place)
            real_img_temp = save_batch_image(real_img)
            images = [real_img_temp]
            for i in range(cfg.c_dim):
                label_trg_tmp = copy.deepcopy(label_org)
                for j in range(len(label_org)):
                    label_trg_tmp[j][i] = 1.0 - label_trg_tmp[j][i]
                    label_trg = check_attribute_conflict(
                        label_trg_tmp, attr_names[i], attr_names)
                tensor_label_trg = fluid.LoDTensor()
                tensor_label_trg.set(label_trg, place)
                fake_temp, rec_temp = exe.run(
                    test_program,
                    feed={
                        "image_real": tensor_img,
                        "label_org": tensor_label_org,
                        "label_trg": tensor_label_trg
                    },
                    fetch_list=[g_trainer.fake_img, g_trainer.rec_img])
                fake_temp = save_batch_image(fake_temp)
                rec_temp = save_batch_image(rec_temp)
                images.append(fake_temp)
                images.append(rec_temp)
            images_concat = np.concatenate(images, 1)
            if len(label_org) > 1:
                images_concat = np.concatenate(images_concat, 1)
            imageio.imwrite(out_path + "/fake_img" + str(epoch) + "_" + name[0],
                            ((images_concat + 1) * 127.5).astype(np.uint8))
    elif cfg.model_net == 'AttGAN' or cfg.model_net == 'STGAN':
        for data in zip(A_test_reader()):
            real_img, label_org, name = data[0]
            attr_names = cfg.selected_attrs.split(',')
            label_trg = copy.deepcopy(label_org)
            tensor_img = fluid.LoDTensor()
            tensor_label_org = fluid.LoDTensor()
            tensor_label_trg = fluid.LoDTensor()
            tensor_label_org_ = fluid.LoDTensor()
            tensor_label_trg_ = fluid.LoDTensor()
            tensor_img.set(real_img, place)
            tensor_label_org.set(label_org, place)
            real_img_temp = save_batch_image(real_img)
            images = [real_img_temp]
            for i in range(cfg.c_dim):
                label_trg_tmp = copy.deepcopy(label_trg)

                for j in range(len(label_org)):
                    label_trg_tmp[j][i] = 1.0 - label_trg_tmp[j][i]
                    label_trg_tmp = check_attribute_conflict(
                        label_trg_tmp, attr_names[i], attr_names)

                label_org_ = list(map(lambda x: ((x * 2) - 1) * 0.5, label_org))
                label_trg_ = list(
                    map(lambda x: ((x * 2) - 1) * 0.5, label_trg_tmp))

                if cfg.model_net == 'AttGAN':
                    for k in range(len(label_org)):
                        label_trg_[k][i] = label_trg_[k][i] * 2.0
                tensor_label_org_.set(label_org_, place)
                tensor_label_trg.set(label_trg, place)
                tensor_label_trg_.set(label_trg_, place)
                out = exe.run(test_program,
                              feed={
                                  "image_real": tensor_img,
                                  "label_org": tensor_label_org,
                                  "label_org_": tensor_label_org_,
                                  "label_trg": tensor_label_trg,
                                  "label_trg_": tensor_label_trg_
                              },
                              fetch_list=[g_trainer.fake_img])
                fake_temp = save_batch_image(out[0])
                images.append(fake_temp)
            images_concat = np.concatenate(images, 1)
            if len(label_org) > 1:
                images_concat = np.concatenate(images_concat, 1)
            imageio.imwrite(out_path + "/fake_img" + str(epoch) + '_' + name[0],
                            ((images_concat + 1) * 127.5).astype(np.uint8))

    else:
        for data_A, data_B in zip(A_test_reader(), B_test_reader()):
            A_data, A_name = data_A
            B_data, B_name = data_B
            tensor_A = fluid.LoDTensor()
            tensor_B = fluid.LoDTensor()
            tensor_A.set(A_data, place)
            tensor_B.set(B_data, place)
            fake_A_temp, fake_B_temp, cyc_A_temp, cyc_B_temp = exe.run(
                test_program,
                fetch_list=[
                    g_trainer.fake_A, g_trainer.fake_B, g_trainer.cyc_A,
                    g_trainer.cyc_B
                ],
                feed={"input_A": tensor_A,
                      "input_B": tensor_B})
            fake_A_temp = np.squeeze(fake_A_temp[0]).transpose([1, 2, 0])
            fake_B_temp = np.squeeze(fake_B_temp[0]).transpose([1, 2, 0])
            cyc_A_temp = np.squeeze(cyc_A_temp[0]).transpose([1, 2, 0])
            cyc_B_temp = np.squeeze(cyc_B_temp[0]).transpose([1, 2, 0])
            input_A_temp = np.squeeze(data_A[0][0]).transpose([1, 2, 0])
            input_B_temp = np.squeeze(data_B[0][0]).transpose([1, 2, 0])

            imageio.imwrite(out_path + "/fakeB_" + str(epoch) + "_" + A_name[0],
                            ((fake_B_temp + 1) * 127.5).astype(np.uint8))
            imageio.imwrite(out_path + "/fakeA_" + str(epoch) + "_" + B_name[0],
                            ((fake_A_temp + 1) * 127.5).astype(np.uint8))
            imageio.imwrite(out_path + "/cycA_" + str(epoch) + "_" + A_name[0],
                            ((cyc_A_temp + 1) * 127.5).astype(np.uint8))
            imageio.imwrite(out_path + "/cycB_" + str(epoch) + "_" + B_name[0],
                            ((cyc_B_temp + 1) * 127.5).astype(np.uint8))
            imageio.imwrite(
                out_path + "/inputA_" + str(epoch) + "_" + A_name[0], (
                    (input_A_temp + 1) * 127.5).astype(np.uint8))
            imageio.imwrite(
                out_path + "/inputB_" + str(epoch) + "_" + B_name[0], (
                    (input_B_temp + 1) * 127.5).astype(np.uint8))


class ImagePool(object):
    def __init__(self, pool_size=50):
        self.pool = []
        self.count = 0
        self.pool_size = pool_size

    def pool_image(self, image):
        if self.count < self.pool_size:
            self.pool.append(image)
            self.count += 1
            return image
        else:
            p = np.random.rand()
            if p > 0.5:
                random_id = np.random.randint(0, self.pool_size - 1)
                temp = self.pool[random_id]
                self.pool[random_id] = image
                return temp
            else:
                return image


def check_attribute_conflict(label_batch, attr, attrs):
    ''' Based on https://github.com/LynnHo/AttGAN-Tensorflow'''

    def _set(label, value, attr):
        if attr in attrs:
            label[attrs.index(attr)] = value

    attr_id = attrs.index(attr)
    for label in label_batch:
        if attr in ['Bald', 'Receding_Hairline'] and attrs[attr_id] != 0:
            _set(label, 0, 'Bangs')
        elif attr == 'Bangs' and attrs[attr_id] != 0:
            _set(label, 0, 'Bald')
            _set(label, 0, 'Receding_Hairline')
        elif attr in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair'
                      ] and attrs[attr_id] != 0:
            for a in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']:
                if a != attr:
                    _set(label, 0, a)
        elif attr in ['Straight_Hair', 'Wavy_Hair'] and attrs[attr_id] != 0:
            for a in ['Straight_Hair', 'Wavy_Hair']:
                if a != attr:
                    _set(label, 0, a)
    return label_batch


def save_batch_image(img):
    if len(img) == 1:
        res_img = np.squeeze(img).transpose([1, 2, 0])
    else:
        res_img = np.squeeze(img).transpose([0, 2, 3, 1])
    return res_img


def check_gpu(use_gpu):
    """
     Log error and exit when set use_gpu=true in paddlepaddle
     cpu version.
     """
    err = "Config use_gpu cannot be set as true while you are " \
          "using paddlepaddle cpu version ! \nPlease try: \n" \
          "\t1. Install paddlepaddle-gpu to run model on GPU \n" \
          "\t2. Set use_gpu as false in config file to run " \
          "model on CPU"

    try:
        if use_gpu and not fluid.is_compiled_with_cuda():
            print(err)
            sys.exit(1)
    except Exception as e:
        pass
