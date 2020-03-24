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

import argparse
import functools
import os
from PIL import Image
import paddle.fluid as fluid
import paddle
import numpy as np
import glob
from util.config import add_arguments, print_arguments
from data_reader import celeba_reader_creator, reader_creator, triplex_reader_creator
from util.utility import check_attribute_conflict, check_gpu, save_batch_image, check_version
from util import utility
import copy

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('model_net',         str,   'CGAN',            "The model used")
add_arg('net_G',             str,   "resnet_9block",   "Choose the CycleGAN and Pix2pix generator's network, choose in [resnet_9block|resnet_6block|unet_128|unet_256]")
add_arg('init_model',        str,   None,              "The init model file of directory.")
add_arg('output',            str,   "./infer_result",  "The directory the infer result to be saved to.")
add_arg('input_style',       str,   "A",               "The style of the input, A or B")
add_arg('norm_type',         str,   "batch_norm",      "Which normalization to used")
add_arg('crop_type',         str,   None,      "Which crop type to use")
add_arg('use_gpu',           bool,  True,              "Whether to use GPU to train.")
add_arg('dropout',           bool,  False,             "Whether to use dropout")
add_arg('g_base_dims',       int,   64,                "Base channels in CycleGAN generator")
add_arg('ngf',       int,   64,                "Base channels in SPADE generator")
add_arg('c_dim',             int,   13,                "the size of attrs")
add_arg('use_gru',           bool,  False,             "Whether to use GRU")
add_arg('crop_size',         int,   178,               "crop size")
add_arg('image_size',        int,   128,               "image size")
add_arg('load_height',        int,   128,               "image size")
add_arg('load_width',        int,   128,               "image size")
add_arg('crop_height',        int,   128,               "height of crop size")
add_arg('crop_width',        int,   128,               "width of crop size")
add_arg('selected_attrs',    str,
    "Bald,Bangs,Black_Hair,Blond_Hair,Brown_Hair,Bushy_Eyebrows,Eyeglasses,Male,Mouth_Slightly_Open,Mustache,No_Beard,Pale_Skin,Young",
"the attributes we selected to change")
add_arg('n_samples',        int,   16,                "batch size when test")
add_arg('test_list',         str,   "./data/celeba/list_attr_celeba.txt",                "the test list file")
add_arg('dataset_dir',       str,   "./data/celeba/",                "the dataset directory to be infered")
add_arg('n_layers',          int,   5,                 "default layers in generotor")
add_arg('gru_n_layers',      int,   4,                 "default layers of GRU in generotor")
add_arg('noise_size',        int,   100,               "the noise dimension")
add_arg('label_nc',        int,   36,               "label numbers of SPADE")
add_arg('no_instance', type=bool, default=False, help="Whether to use instance label.")
# yapf: enable


def infer(args):
    data_shape = [None, 3, args.image_size, args.image_size]
    input = fluid.data(name='input', shape=data_shape, dtype='float32')
    label_org_ = fluid.data(
        name='label_org_', shape=[None, args.c_dim], dtype='float32')
    label_trg_ = fluid.data(
        name='label_trg_', shape=[None, args.c_dim], dtype='float32')
    image_name = fluid.data(
        name='image_name', shape=[None, args.n_samples], dtype='int32')

    model_name = 'net_G'

    if args.model_net == 'CycleGAN':
        loader = fluid.io.DataLoader.from_generator(
            feed_list=[input, image_name],
            capacity=4,  ## batch_size * 4
            iterable=True,
            use_double_buffer=True)

        from network.CycleGAN_network import CycleGAN_model
        model = CycleGAN_model()
        if args.input_style == "A":
            fake = model.network_G(input, name="GA", cfg=args)
        elif args.input_style == "B":
            fake = model.network_G(input, name="GB", cfg=args)
        else:
            raise "Input with style [%s] is not supported." % args.input_style
    elif args.model_net == 'Pix2pix':
        loader = fluid.io.DataLoader.from_generator(
            feed_list=[input, image_name],
            capacity=4,  ## batch_size * 4
            iterable=True,
            use_double_buffer=True)

        from network.Pix2pix_network import Pix2pix_model
        model = Pix2pix_model()
        fake = model.network_G(input, "generator", cfg=args)
    elif args.model_net == 'StarGAN':

        loader = fluid.io.DataLoader.from_generator(
            feed_list=[input, label_org_, label_trg_, image_name],
            capacity=32,
            iterable=True,
            use_double_buffer=True)

        from network.StarGAN_network import StarGAN_model
        model = StarGAN_model()
        fake = model.network_G(input, label_trg_, name="g_main", cfg=args)
    elif args.model_net == 'STGAN':
        from network.STGAN_network import STGAN_model

        loader = fluid.io.DataLoader.from_generator(
            feed_list=[input, label_org_, label_trg_, image_name],
            capacity=32,
            iterable=True,
            use_double_buffer=True)

        model = STGAN_model()
        fake, _ = model.network_G(
            input,
            label_org_,
            label_trg_,
            cfg=args,
            name='generator',
            is_test=True)
    elif args.model_net == 'AttGAN':
        from network.AttGAN_network import AttGAN_model

        loader = fluid.io.DataLoader.from_generator(
            feed_list=[input, label_org_, label_trg_, image_name],
            capacity=32,
            iterable=True,
            use_double_buffer=True)

        model = AttGAN_model()
        fake, _ = model.network_G(
            input,
            label_org_,
            label_trg_,
            cfg=args,
            name='generator',
            is_test=True)
    elif args.model_net == 'CGAN':
        noise = fluid.data(
            name='noise', shape=[None, args.noise_size], dtype='float32')
        conditions = fluid.data(
            name='conditions', shape=[None, 1], dtype='float32')

        from network.CGAN_network import CGAN_model
        model = CGAN_model(args.n_samples)
        fake = model.network_G(noise, conditions, name="G")
    elif args.model_net == 'DCGAN':
        noise = fluid.data(
            name='noise', shape=[None, args.noise_size], dtype='float32')

        from network.DCGAN_network import DCGAN_model
        model = DCGAN_model(args.n_samples)
        fake = model.network_G(noise, name="G")
    elif args.model_net == 'SPADE':
        label_shape = [None, args.label_nc, args.crop_height, args.crop_width]
        spade_data_shape = [None, 1, args.crop_height, args.crop_width]
        from network.SPADE_network import SPADE_model
        model = SPADE_model()
        input_label = fluid.data(
            name='input_label', shape=label_shape, dtype='float32')
        input_ins = fluid.data(
            name='input_ins', shape=spade_data_shape, dtype='float32')
        input_ = fluid.layers.concat([input_label, input_ins], 1)
        fake = model.network_G(input_, "generator", cfg=args, is_test=True)
    else:
        raise NotImplementedError("model_net {} is not support".format(
            args.model_net))

    def _compute_start_end(image_name):
        image_name_start = np.array(image_name)[0].astype('int32')
        image_name_end = image_name_start + args.n_samples - 1
        image_name_save = str(np.array(image_name)[0].astype('int32')) + '.jpg'
        print("read {}.jpg ~ {}.jpg".format(image_name_start, image_name_end))
        return image_name_save

    # prepare environment
    place = fluid.CPUPlace()
    if args.use_gpu:
        place = fluid.CUDAPlace(0)
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())
    for var in fluid.default_main_program().all_parameters():
        print(var.name)
    print(args.init_model + '/' + model_name)
    fluid.load(fluid.default_main_program(), os.path.join(args.init_model, model_name))
    print('load params done')
    if not os.path.exists(args.output):
        os.makedirs(args.output)


    attr_names = args.selected_attrs.split(',')

    if args.model_net == 'AttGAN' or args.model_net == 'STGAN':
        test_reader = celeba_reader_creator(
            image_dir=args.dataset_dir,
            list_filename=args.test_list,
            args=args,
            mode="VAL")
        reader_test = test_reader.make_reader(return_name=True)
        loader.set_batch_generator(
            reader_test,
            places=fluid.cuda_places() if args.use_gpu else fluid.cpu_places())
        for data in loader():
            real_img, label_org, label_trg, image_name = data[0]['input'], data[
                0]['label_org_'], data[0]['label_trg_'], data[0]['image_name']
            image_name_save = _compute_start_end(image_name)
            real_img_temp = save_batch_image(np.array(real_img))
            images = [real_img_temp]
            for i in range(args.c_dim):
                label_trg_tmp = copy.deepcopy(np.array(label_trg))
                for j in range(len(label_trg_tmp)):
                    label_trg_tmp[j][i] = 1.0 - label_trg_tmp[j][i]
                    label_trg_tmp = check_attribute_conflict(
                        label_trg_tmp, attr_names[i], attr_names)
                label_org_tmp = list(
                    map(lambda x: ((x * 2) - 1) * 0.5, np.array(label_org)))
                label_trg_tmp = list(
                    map(lambda x: ((x * 2) - 1) * 0.5, label_trg_tmp))
                if args.model_net == 'AttGAN':
                    for k in range(len(label_trg_tmp)):
                        label_trg_tmp[k][i] = label_trg_tmp[k][i] * 2.0
                tensor_label_org_ = fluid.LoDTensor()
                tensor_label_trg_ = fluid.LoDTensor()
                tensor_label_org_.set(label_org_tmp, place)
                tensor_label_trg_.set(label_trg_tmp, place)
                out = exe.run(feed={
                    "input": real_img,
                    "label_org_": tensor_label_org_,
                    "label_trg_": tensor_label_trg_
                },
                              fetch_list=[fake.name])
                fake_temp = save_batch_image(out[0])
                images.append(fake_temp)
            images_concat = np.concatenate(images, 1)
            if len(np.array(label_org)) > 1:
                images_concat = np.concatenate(images_concat, 1)
            fake_image = Image.fromarray(((images_concat + 1) * 127.5).astype(np.uint8))
            fake_image.save(os.path.join(args.output, "fake_image_" + image_name_save))
    elif args.model_net == 'StarGAN':
        test_reader = celeba_reader_creator(
            image_dir=args.dataset_dir,
            list_filename=args.test_list,
            args=args,
            mode="VAL")
        reader_test = test_reader.make_reader(return_name=True)
        loader.set_batch_generator(
            reader_test,
            places=fluid.cuda_places() if args.use_gpu else fluid.cpu_places())
        for data in loader():
            real_img, label_org, label_trg, image_name = data[0]['input'], data[
                0]['label_org_'], data[0]['label_trg_'], data[0]['image_name']
            image_name_save = _compute_start_end(image_name)
            real_img_temp = save_batch_image(np.array(real_img))
            images = [real_img_temp]
            for i in range(args.c_dim):
                label_trg_tmp = copy.deepcopy(np.array(label_org))
                for j in range(len(np.array(label_org))):
                    label_trg_tmp[j][i] = 1.0 - label_trg_tmp[j][i]
                    label_trg_tmp = check_attribute_conflict(
                        label_trg_tmp, attr_names[i], attr_names)
                tensor_label_trg_ = fluid.LoDTensor()
                tensor_label_trg_.set(label_trg_tmp, place)
                out = exe.run(
                    feed={"input": real_img,
                          "label_trg_": tensor_label_trg_},
                    fetch_list=[fake.name])
                fake_temp = save_batch_image(out[0])
                images.append(fake_temp)
            images_concat = np.concatenate(images, 1)
            if len(np.array(label_org)) > 1:
                images_concat = np.concatenate(images_concat, 1)
            fake_image = Image.fromarray(((images_concat + 1) * 127.5).astype(np.uint8))
            fake_image.save(os.path.join(args.output, "fake_image_" + image_name_save))

    elif args.model_net == 'Pix2pix' or args.model_net == 'CycleGAN':
        test_reader = reader_creator(
            image_dir=args.dataset_dir,
            list_filename=args.test_list,
            shuffle=False,
            batch_size=args.n_samples,
            mode="VAL")
        reader_test = test_reader.make_reader(args, return_name=True)
        loader.set_batch_generator(
            reader_test,
            places=fluid.cuda_places() if args.use_gpu else fluid.cpu_places())
        id2name = test_reader.id2name
        for data in loader():
            real_img, image_name = data[0]['input'], data[0]['image_name']
            image_names = []
            for name in image_name:
                image_names.append(id2name[np.array(name).astype('int32')[0]])
            print("read: ", image_names)
            fake_temp = exe.run(fetch_list=[fake.name],
                                feed={"input": real_img})
            fake_temp = save_batch_image(fake_temp[0])
            input_temp = save_batch_image(np.array(real_img))

            if len(image_names) == 1:
                fake_temp = np.expand_dims(fake_temp, axis=0)
                input_temp = np.expand_dims(input_temp, axis=0)
            for i, name in enumerate(image_names):
                fake_image = Image.fromarray(((fake_temp[i] + 1) * 127.5).astype(np.uint8))
                fake_image.save(os.path.join(args.output, "fake_" + name))
                input_image = Image.fromarray(((input_temp[i] + 1) * 127.5).astype(np.uint8))
                input_image.save(os.path.join(args.output, "input_" + name))
    elif args.model_net == 'SPADE':
        test_reader = triplex_reader_creator(
            image_dir=args.dataset_dir,
            list_filename=args.test_list,
            shuffle=False,
            batch_size=1,
            mode="TEST")
        id2name = test_reader.id2name
        reader_test = test_reader.make_reader(args, return_name=True)
        for data in zip(reader_test()):
            data_A, data_B, data_C, name = data[0]
            name = id2name[np.array(name).astype('int32')[0]]
            print("read: ", name)
            tensor_A = fluid.LoDTensor()
            tensor_C = fluid.LoDTensor()
            tensor_A.set(data_A, place)
            tensor_C.set(data_C, place)
            fake_B_temp = exe.run(
                fetch_list=[fake.name],
                feed={"input_label": tensor_A,
                      "input_ins": tensor_C})
            fake_B_temp = np.squeeze(fake_B_temp[0]).transpose([1, 2, 0])
            input_B_temp = np.squeeze(data_B[0]).transpose([1, 2, 0])

            fakeB_image = Image.fromarray(((fake_B_temp + 1) * 127.5).astype(np.uint8))
            fakeB_image.save(os.path.join(args.output, "fakeB_" + name))
            real_image = Image.fromarray(((input_B_temp + 1) * 127.5).astype(np.uint8))
            real_image.save(os.path.join(args.output, "real_" + name))

    elif args.model_net == 'CGAN':
        noise_data = np.random.uniform(
            low=-1.0, high=1.0,
            size=[args.n_samples, args.noise_size]).astype('float32')
        label = np.random.randint(
            0, 9, size=[args.n_samples, 1]).astype('float32')
        noise_tensor = fluid.LoDTensor()
        conditions_tensor = fluid.LoDTensor()
        noise_tensor.set(noise_data, place)
        conditions_tensor.set(label, place)
        fake_temp = exe.run(
            fetch_list=[fake.name],
            feed={"noise": noise_tensor,
                  "conditions": conditions_tensor})[0]
        fake_image = np.reshape(fake_temp, (args.n_samples, -1))

        fig = utility.plot(fake_image)
        plt.savefig(
            os.path.join(args.output, 'fake_cgan.png'), bbox_inches='tight')
        plt.close(fig)

    elif args.model_net == 'DCGAN':
        noise_data = np.random.uniform(
            low=-1.0, high=1.0,
            size=[args.n_samples, args.noise_size]).astype('float32')
        noise_tensor = fluid.LoDTensor()
        noise_tensor.set(noise_data, place)
        fake_temp = exe.run(fetch_list=[fake.name],
                            feed={"noise": noise_tensor})[0]
        fake_image = np.reshape(fake_temp, (args.n_samples, -1))

        fig = utility.plot(fake_image)
        plt.savefig(
            os.path.join(args.output, 'fake_dcgan.png'), bbox_inches='tight')
        plt.close(fig)
    else:
        raise NotImplementedError("model_net {} is not support".format(
            args.model_net))


if __name__ == "__main__":
    args = parser.parse_args()
    print_arguments(args)
    check_gpu(args.use_gpu)
    check_version()
    infer(args)
