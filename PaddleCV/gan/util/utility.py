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
    output_path = cfg.output + '/checkpoints/' + str(epoch)
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
    out_path = cfg.output + '/test'
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
            tensor_img = fluid.LoDTensor()
            tensor_label_org = fluid.LoDTensor()
            tensor_img.set(real_img, place)
            tensor_label_org.set(label_org, place)
            real_img_temp = np.squeeze(real_img).transpose([1, 2, 0])
            images = [real_img_temp]
            for i in range(cfg.c_dim):
                label_trg = np.zeros([1, cfg.c_dim]).astype("float32")
                label_trg[0][i] = 1
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
                fake_temp = np.squeeze(fake_temp[0]).transpose([1, 2, 0])
                rec_temp = np.squeeze(rec_temp[0]).transpose([1, 2, 0])
                images.append(fake_temp)
                images.append(rec_temp)
            images_concat = np.concatenate(images, 1)
            imageio.imwrite(out_path + "/fake_img" + str(epoch) + "_" + name[0],
                            ((images_concat + 1) * 127.5).astype(np.uint8))
    elif cfg.model_net == 'AttGAN' or cfg.model_net == 'STGAN':
        for data in zip(A_test_reader()):
            real_img, label_org, name = data[0]
            label_trg = copy.deepcopy(label_org)
            tensor_img = fluid.LoDTensor()
            tensor_label_org = fluid.LoDTensor()
            tensor_label_trg = fluid.LoDTensor()
            tensor_label_org_ = fluid.LoDTensor()
            tensor_label_trg_ = fluid.LoDTensor()
            tensor_img.set(real_img, place)
            tensor_label_org.set(label_org, place)
            real_img_temp = np.squeeze(real_img).transpose([0, 2, 3, 1])
            images = [real_img_temp]
            for i in range(cfg.c_dim):
                label_trg_tmp = copy.deepcopy(label_trg)

                for j in range(len(label_org)):
                    label_trg_tmp[j][i] = 1.0 - label_trg_tmp[j][i]

                label_trg_ = map(lambda x: ((x * 2) - 1) * 0.5, label_trg_tmp)

                for j in range(len(label_org)):
                    label_trg_[j][i] = label_trg_[j][i] * 2.0
                tensor_label_org_.set(label_org, place)
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
                fake_temp = np.squeeze(out[0]).transpose([0, 2, 3, 1])
                images.append(fake_temp)
            images_concat = np.concatenate(images, 1)
            images_concat = np.concatenate(images_concat, 1)
            imageio.imwrite(out_path + "/fake_img" + str(epoch) + '_' + name[0],
                            ((images_concat + 1) * 127.5).astype(np.uint8))

    else:
        for data_A, data_B in zip(A_test_reader(), B_test_reader()):
            A_name = data_A[0][1]
            B_name = data_B[0][1]
            tensor_A = fluid.LoDTensor()
            tensor_B = fluid.LoDTensor()
            tensor_A.set(data_A[0][0], place)
            tensor_B.set(data_B[0][0], place)
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

            imageio.imwrite(out_path + "/fakeB_" + str(epoch) + "_" + A_name, (
                (fake_B_temp + 1) * 127.5).astype(np.uint8))
            imageio.imwrite(out_path + "/fakeA_" + str(epoch) + "_" + B_name, (
                (fake_A_temp + 1) * 127.5).astype(np.uint8))
            imageio.imwrite(out_path + "/cycA_" + str(epoch) + "_" + A_name, (
                (cyc_A_temp + 1) * 127.5).astype(np.uint8))
            imageio.imwrite(out_path + "/cycB_" + str(epoch) + "_" + B_name, (
                (cyc_B_temp + 1) * 127.5).astype(np.uint8))
            imageio.imwrite(out_path + "/inputA_" + str(epoch) + "_" + A_name, (
                (input_A_temp + 1) * 127.5).astype(np.uint8))
            imageio.imwrite(out_path + "/inputB_" + str(epoch) + "_" + B_name, (
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
