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
import sys
import os
import shutil
import logging

import numpy as np
import paddle
import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr

import models

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_ac_tensor(name):
    gamma = fluid.global_scope().find_var(name + '_scale').get_tensor()
    beta = fluid.global_scope().find_var(name + '_offset').get_tensor()
    mean = fluid.global_scope().find_var(name + '_mean').get_tensor()
    var = fluid.global_scope().find_var(name + '_variance').get_tensor()
    return gamma, beta, mean, var


def get_kernel_bn_tensors(name):
    if "conv1" in name:
        bn_name = "bn_" + name
    else:
        bn_name = "bn" + name[3:]

    ac_square = fluid.global_scope().find_var(name +
                                              "_acsquare_weights").get_tensor()
    ac_ver = fluid.global_scope().find_var(name + "_acver_weights").get_tensor()
    ac_hor = fluid.global_scope().find_var(name + "_achor_weights").get_tensor()

    ac_square_bn_gamma, ac_square_bn_beta, ac_square_bn_mean, ac_square_bn_var = \
            get_ac_tensor(bn_name + '_acsquare')
    ac_ver_bn_gamma, ac_ver_bn_beta, ac_ver_bn_mean, ac_ver_bn_var = \
            get_ac_tensor(bn_name + '_acver')
    ac_hor_bn_gamma, ac_hor_bn_beta, ac_hor_bn_mean, ac_hor_bn_var = \
            get_ac_tensor(bn_name + '_achor')

    kernels = [np.array(ac_square), np.array(ac_ver), np.array(ac_hor)]
    gammas = [
        np.array(ac_square_bn_gamma), np.array(ac_ver_bn_gamma),
        np.array(ac_hor_bn_gamma)
    ]
    betas = [
        np.array(ac_square_bn_beta), np.array(ac_ver_bn_beta),
        np.array(ac_hor_bn_beta)
    ]
    means = [
        np.array(ac_square_bn_mean), np.array(ac_ver_bn_mean),
        np.array(ac_hor_bn_mean)
    ]
    var = [
        np.array(ac_square_bn_var), np.array(ac_ver_bn_var),
        np.array(ac_hor_bn_var)
    ]

    return {"kernels": kernels, "bn": (gammas, betas, means, var)}


def kernel_fusion(kernels, gammas, betas, means, var):
    """fuse conv + BN"""
    kernel_size_h, kernel_size_w = kernels[0].shape[2:]

    square = (gammas[0] / (var[0] + 1e-5)
              **0.5).reshape(-1, 1, 1, 1) * kernels[0]
    ver = (gammas[1] / (var[1] + 1e-5)**0.5).reshape(-1, 1, 1, 1) * kernels[1]
    hor = (gammas[2] / (var[2] + 1e-5)**0.5).reshape(-1, 1, 1, 1) * kernels[2]

    b = 0
    for i in range(3):
        b += -((means[i] * gammas[i]) / (var[i] + 1e-5)**0.5) + betas[i]  # eq.7

    square[:, :, :, kernel_size_w // 2:kernel_size_w // 2 + 1] += ver
    square[:, :, kernel_size_h // 2:kernel_size_h // 2 + 1, :] += hor

    return square, b


def convert_main(model_name, input_path, output_path, class_num=1000):
    model = models.__dict__[model_name]()

    main_prog = fluid.Program()
    acnet_prog = fluid.Program()
    startup_prog = fluid.Program()

    with fluid.program_guard(acnet_prog, startup_prog):
        with fluid.unique_name.guard():
            image = fluid.data(
                name="image",
                shape=[-1, 3, 224, 224],
                dtype="float32",
                lod_level=0)
            model_train = models.__dict__[model_name](deploy=False)
            model_train.net(image, class_dim=1000)

    with fluid.program_guard(main_prog, startup_prog):
        with fluid.unique_name.guard():
            image = fluid.data(
                name="image",
                shape=[-1, 3, 224, 224],
                dtype="float32",
                lod_level=0)
            model_infer = models.__dict__[model_name](deploy=True)
            model_infer.net(image, class_dim=1000)

    acnet_prog = acnet_prog.clone(for_test=True)
    main_prog = main_prog.clone(for_test=True)

    place = fluid.CUDAPlace(0)
    exe = fluid.Executor(place)
    exe.run(startup_prog)

    assert os.path.exists(
        input_path), "Pretrained model path {} not exist!".format(input_path)
    fluid.io.load_vars(exe, input_path,
                        main_program=acnet_prog,
                        predicate=lambda var: os.path.exists(os.path.join(input_path, var.name)))

    mapping = {}

    for param in main_prog.blocks[0].all_parameters():
        if "acsquare" in param.name:
            name_root = "_".join(param.name.split("_")[:-2])
            if name_root in mapping.keys():
                mapping[name_root].append(param.name)
            else:
                mapping[name_root] = [param.name]
        else:
            assert param.name not in mapping.keys()
            mapping[param.name] = [param.name]

    for name_root, names in mapping.items():
        if len(names) == 1:
            pass
        else:
            if "bias" in names[0]:
                bias_id = 0
                kernel_id = 1
            else:
                bias_id = 1
                kernel_id = 0

            tensor_bias = fluid.global_scope().find_var(names[
                bias_id]).get_tensor()
            tensor_kernel = fluid.global_scope().find_var(names[
                kernel_id]).get_tensor()

            ret = get_kernel_bn_tensors(name_root)
            kernels = ret['kernels']
            gammas, betas, means, var = ret['bn']

            kernel, bias = kernel_fusion(kernels, gammas, betas, means, var)

            logger.info("Before {}: {}".format(names[
                kernel_id], np.array(tensor_kernel).ravel()[:5]))

            tensor_bias.set(bias, place)
            tensor_kernel.set(kernel, place)

            logger.info("After {}: {}\n".format(names[
                kernel_id], np.array(tensor_kernel).ravel()[:5]))

    if os.path.isdir(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path)
    fluid.io.save_persistables(exe, output_path, main_program=main_prog)


if __name__ == "__main__":
    assert len(
        sys.argv
    ) == 5, "input format: python weights_aggregator.py $model_name $input_path $output_path $class_num"
    model_name = sys.argv[1]
    input_path = sys.argv[2]
    output_path = sys.argv[3]
    class_num = int(sys.argv[4])
    logger.info("model_name: {}".format(model_name))
    logger.info("input_path: {}".format(input_path))
    logger.info("output_path: {}".format(output_path))
    logger.info("class_num: {}".format(class_num))
    convert_main(model_name, input_path, output_path, class_num)
