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
import copy
from PIL import Image

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


def checkpoints(epoch, cfg, trainer, name):
    output_path = os.path.join(cfg.output, 'checkpoints', str(epoch))
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    fluid.save(
        trainer.program, os.path.join(output_path, name))
    print('save checkpoints {} to {}'.format(name, output_path))
    sys.stdout.flush()


def init_checkpoints(cfg, trainer, name):
    assert os.path.exists(cfg.init_model), "{} cannot be found.".format(
        cfg.init_model)
    fluid.load(
        trainer.program, os.path.join(cfg.init_model, name))
    print('load checkpoints {} {} DONE'.format(cfg.init_model, name))
    sys.stdout.flush()


def save_test_image(epoch,
                    cfg,
                    exe,
                    place,
                    test_program,
                    g_trainer,
                    A_test_reader,
                    B_test_reader=None,
                    A_id2name=None,
                    B_id2name=None):
    out_path = os.path.join(cfg.output, 'test')
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    if cfg.model_net == "Pix2pix":
        for data in A_test_reader():
            A_data, B_data, image_name = data[0]['input_A'], data[0][
                'input_B'], data[0]['image_name']
            fake_B_temp = exe.run(test_program,
                                  fetch_list=[g_trainer.fake_B],
                                  feed={"input_A": A_data,
                                        "input_B": B_data})
            fake_B_temp = np.squeeze(fake_B_temp[0]).transpose([1, 2, 0])
            input_A_temp = np.squeeze(np.array(A_data)[0]).transpose([1, 2, 0])
            input_B_temp = np.squeeze(np.array(B_data)[0]).transpose([1, 2, 0])

            fakeB_name = "fakeB_" + str(epoch) + "_" + A_id2name[np.array(
                image_name).astype('int32')[0]]
            inputA_name = "inputA_" + str(epoch) + "_" + A_id2name[np.array(
                image_name).astype('int32')[0]]
            inputB_name = "inputB_" + str(epoch) + "_" + A_id2name[np.array(
                image_name).astype('int32')[0]]

            res_fakeB = Image.fromarray(((fake_B_temp + 1) * 127.5).astype(
                np.uint8))
            res_fakeB.save(os.path.join(out_path, fakeB_name))

            res_inputA = Image.fromarray(((input_A_temp + 1) * 127.5).astype(
                np.uint8))
            res_inputA.save(os.path.join(out_path, inputA_name))

            res_inputB = Image.fromarray(((input_B_temp + 1) * 127.5).astype(
                np.uint8))
            res_inputB.save(os.path.join(out_path, inputB_name))
    elif cfg.model_net == "SPADE":
        for data in A_test_reader():
            data_A, data_B, data_C, name = data[0]['input_label'], data[0][
                'input_img'], data[0]['input_ins'], data[0]['image_name']
            fake_B_temp = exe.run(test_program,
                                  fetch_list=[g_trainer.fake_B],
                                  feed={
                                      "input_label": data_A,
                                      "input_img": data_B,
                                      "input_ins": data_C
                                  })
            fake_B_temp = np.squeeze(fake_B_temp[0]).transpose([1, 2, 0])
            input_B_temp = np.squeeze(data_B[0]).transpose([1, 2, 0])
            image_name = A_id2name[np.array(name).astype('int32')[0]]

            res_fakeB = Image.fromarray(((fake_B_temp + 1) * 127.5).astype(
                np.uint8))
            res_fakeB.save(out_path + "/fakeB_" + str(epoch) + "_" + image_name)
            res_real = Image.fromarray(((input_B_temp + 1) * 127.5).astype(
                np.uint8))
            res_real.save(out_path + "/real_" + str(epoch) + "_" + image_name)
    elif cfg.model_net == "StarGAN":
        for data in A_test_reader():
            real_img, label_org, label_trg, image_name = data[0][
                'image_real'], data[0]['label_org'], data[0]['label_trg'], data[
                    0]['image_name']
            attr_names = cfg.selected_attrs.split(',')
            real_img_temp = save_batch_image(np.array(real_img))
            images = [real_img_temp]
            for i in range(cfg.c_dim):
                label_trg_tmp = copy.deepcopy(np.array(label_org))
                for j in range(len(np.array(label_org))):
                    label_trg_tmp[j][i] = 1.0 - label_trg_tmp[j][i]
                    np_label_trg = check_attribute_conflict(
                        label_trg_tmp, attr_names[i], attr_names)
                label_trg.set(np_label_trg, place)
                fake_temp, rec_temp = exe.run(
                    test_program,
                    feed={
                        "image_real": real_img,
                        "label_org": label_org,
                        "label_trg": label_trg
                    },
                    fetch_list=[g_trainer.fake_img, g_trainer.rec_img])
                fake_temp = save_batch_image(fake_temp)
                rec_temp = save_batch_image(rec_temp)
                images.append(fake_temp)
                images.append(rec_temp)
            images_concat = np.concatenate(images, 1)
            if len(np.array(label_org)) > 1:
                images_concat = np.concatenate(images_concat, 1)
            image_name_save = "fake_img" + str(epoch) + "_" + str(
                np.array(image_name)[0].astype('int32')) + '.jpg'

            res = Image.fromarray(((images_concat + 1) * 127.5).astype(
                np.uint8))
            res.save(os.path.join(out_path, image_name_save))

    elif cfg.model_net == 'AttGAN' or cfg.model_net == 'STGAN':
        for data in A_test_reader():
            real_img, label_org, label_trg, image_name = data[0][
                'image_real'], data[0]['label_org'], data[0]['label_trg'], data[
                    0]['image_name']
            attr_names = cfg.selected_attrs.split(',')
            real_img_temp = save_batch_image(np.array(real_img))
            images = [real_img_temp]
            for i in range(cfg.c_dim):
                label_trg_tmp = copy.deepcopy(np.array(label_trg))

                for j in range(len(label_trg_tmp)):
                    label_trg_tmp[j][i] = 1.0 - label_trg_tmp[j][i]
                    label_trg_tmp = check_attribute_conflict(
                        label_trg_tmp, attr_names[i], attr_names)

                label_org_tmp = list(
                    map(lambda x: ((x * 2) - 1) * 0.5, np.array(label_org)))
                label_trg_tmp = list(
                    map(lambda x: ((x * 2) - 1) * 0.5, label_trg_tmp))

                if cfg.model_net == 'AttGAN':
                    for k in range(len(label_trg_tmp)):
                        label_trg_tmp[k][i] = label_trg_tmp[k][i] * 2.0
                tensor_label_org_ = fluid.LoDTensor()
                tensor_label_org_.set(label_org_tmp, place)
                tensor_label_trg_ = fluid.LoDTensor()
                tensor_label_trg_.set(label_trg_tmp, place)

                out = exe.run(test_program,
                              feed={
                                  "image_real": real_img,
                                  "label_org": label_org,
                                  "label_org_": tensor_label_org_,
                                  "label_trg": label_trg,
                                  "label_trg_": tensor_label_trg_
                              },
                              fetch_list=[g_trainer.fake_img])
                fake_temp = save_batch_image(out[0])
                images.append(fake_temp)
            images_concat = np.concatenate(images, 1)
            if len(label_trg_tmp) > 1:
                images_concat = np.concatenate(images_concat, 1)
            image_name_save = 'fake_img_' + str(epoch) + '_' + str(
                np.array(image_name)[0].astype('int32')) + '.jpg'

            res = Image.fromarray(((images_concat + 1) * 127.5).astype(
                np.uint8))
            res.save(os.path.join(out_path, image_name_save))

    else:
        for data_A, data_B in zip(A_test_reader(), B_test_reader()):
            A_data, A_name = data_A[0]['input_A'], data_A[0]['A_image_name']
            B_data, B_name = data_B[0]['input_B'], data_B[0]['B_image_name']
            fake_A_temp, fake_B_temp, cyc_A_temp, cyc_B_temp = exe.run(
                test_program,
                fetch_list=[
                    g_trainer.fake_A, g_trainer.fake_B, g_trainer.cyc_A,
                    g_trainer.cyc_B
                ],
                feed={"input_A": A_data,
                      "input_B": B_data})
            fake_A_temp = np.squeeze(fake_A_temp[0]).transpose([1, 2, 0])
            fake_B_temp = np.squeeze(fake_B_temp[0]).transpose([1, 2, 0])
            cyc_A_temp = np.squeeze(cyc_A_temp[0]).transpose([1, 2, 0])
            cyc_B_temp = np.squeeze(cyc_B_temp[0]).transpose([1, 2, 0])
            input_A_temp = np.squeeze(np.array(A_data)).transpose([1, 2, 0])
            input_B_temp = np.squeeze(np.array(B_data)).transpose([1, 2, 0])

            fakeA_name = "fakeA_" + str(epoch) + "_" + A_id2name[np.array(
                A_name).astype('int32')[0]]
            fakeB_name = "fakeB_" + str(epoch) + "_" + B_id2name[np.array(
                B_name).astype('int32')[0]]
            inputA_name = "inputA_" + str(epoch) + "_" + A_id2name[np.array(
                A_name).astype('int32')[0]]
            inputB_name = "inputB_" + str(epoch) + "_" + B_id2name[np.array(
                B_name).astype('int32')[0]]
            cycA_name = "cycA_" + str(epoch) + "_" + A_id2name[np.array(
                A_name).astype('int32')[0]]
            cycB_name = "cycB_" + str(epoch) + "_" + B_id2name[np.array(
                B_name).astype('int32')[0]]

            res_fakeB = Image.fromarray(((fake_B_temp + 1) * 127.5).astype(
                np.uint8))
            res_fakeB.save(os.path.join(out_path, fakeB_name))

            res_fakeA = Image.fromarray(((fake_A_temp + 1) * 127.5).astype(
                np.uint8))
            res_fakeA.save(os.path.join(out_path, fakeA_name))

            res_cycA = Image.fromarray(((cyc_A_temp + 1) * 127.5).astype(
                np.uint8))
            res_cycA.save(os.path.join(out_path, cycA_name))

            res_cycB = Image.fromarray(((cyc_B_temp + 1) * 127.5).astype(
                np.uint8))
            res_cycB.save(os.path.join(out_path, cycB_name))

            res_inputA = Image.fromarray(((input_A_temp + 1) * 127.5).astype(
                np.uint8))
            res_inputA.save(os.path.join(out_path, inputA_name))

            res_inputB = Image.fromarray(((input_B_temp + 1) * 127.5).astype(
                np.uint8))
            res_inputB.save(os.path.join(out_path, inputB_name))


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


def check_version():
    """
    Log error and exit when the installed version of paddlepaddle is
    not satisfied.
    """
    err = "PaddlePaddle version 1.7.1 or higher is required, " \
          "or a suitable develop version is satisfied as well. \n" \
          "Please make sure the version is good with your code." \

    try:
        fluid.require_version('1.7.1')
    except Exception as e:
        print(err)
        sys.exit(1)

def get_device_num(args):
    if args.use_gpu:
        gpus = os.environ.get("CUDA_VISIBLE_DEVICES", 1)
        gpu_num = len(gpus.split(','))
        return gpu_num
    else:
        cpu_num = os.environ.get("CPU_NUM", 1)
        return int(cpu_num)
