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

import logging
import os
import six
import sys
import time

import numpy as np
import paddle
import paddle.fluid as fluid

#include palm for easier nlp coding
from palm.toolkit.input_field import InputField
from palm.toolkit.configure import PDConfig

# include task-specific libs
import desc
import reader
from transformer import create_net


def init_from_pretrain_model(args, exe, program):

    assert isinstance(args.init_from_pretrain_model, str)

    if not os.path.exists(args.init_from_pretrain_model):
        raise Warning("The pretrained params do not exist.")
        return False

    def existed_params(var):
        if not isinstance(var, fluid.framework.Parameter):
            return False
        return os.path.exists(
            os.path.join(args.init_from_pretrain_model, var.name))

    fluid.io.load_vars(
        exe,
        args.init_from_pretrain_model,
        main_program=program,
        predicate=existed_params)

    print("finish initing model from pretrained params from %s" %
          (args.init_from_pretrain_model))

    return True


def init_from_params(args, exe, program):

    assert isinstance(args.init_from_params, str)

    if not os.path.exists(args.init_from_params):
        raise Warning("the params path does not exist.")
        return False

    fluid.io.load_params(
        executor=exe,
        dirname=args.init_from_params,
        main_program=program,
        filename="params.pdparams")

    print("finish init model from params from %s" % (args.init_from_params))

    return True


def do_save_inference_model(args):
    if args.use_cuda:
        dev_count = fluid.core.get_cuda_device_count()
        place = fluid.CUDAPlace(0)
    else:
        dev_count = int(os.environ.get('CPU_NUM', 1))
        place = fluid.CPUPlace()

    test_prog = fluid.default_main_program()
    startup_prog = fluid.default_startup_program()

    with fluid.program_guard(test_prog, startup_prog):
        with fluid.unique_name.guard():

            # define input and reader

            input_field_names = desc.encoder_data_input_fields + desc.fast_decoder_data_input_fields
            input_slots = [{
                "name": name,
                "shape": desc.input_descs[name][0],
                "dtype": desc.input_descs[name][1]
            } for name in input_field_names]

            input_field = InputField(input_slots)
            input_field.build(build_pyreader=True)

            # define the network

            predictions = create_net(
                is_training=False, model_input=input_field, args=args)
            out_ids, out_scores = predictions

    # This is used here to set dropout to the test mode.
    test_prog = test_prog.clone(for_test=True)

    # prepare predicting

    ## define the executor and program for training

    exe = fluid.Executor(place)

    exe.run(startup_prog)
    assert (args.init_from_params) or (args.init_from_pretrain_model)

    if args.init_from_params:
        init_from_params(args, exe, test_prog)

    elif args.init_from_pretrain_model:
        init_from_pretrain_model(args, exe, test_prog)

    # saving inference model

    fluid.io.save_inference_model(
        args.inference_model_dir,
        feeded_var_names=input_field_names,
        target_vars=[out_ids, out_scores],
        executor=exe,
        main_program=test_prog,
        model_filename="model.pdmodel",
        params_filename="params.pdparams")

    print("save inference model at %s" % (args.inference_model_dir))


if __name__ == "__main__":
    args = PDConfig(yaml_file="./transformer.yaml")
    args.build()
    args.Print()

    do_save_inference_model(args)
