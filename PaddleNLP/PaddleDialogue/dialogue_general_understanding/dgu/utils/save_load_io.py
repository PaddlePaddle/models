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
"""save or load model api"""

import os
import sys

import paddle
import paddle.fluid as fluid


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


def init_from_checkpoint(args, exe, program):

    assert isinstance(args.init_from_checkpoint, str)

    if not os.path.exists(args.init_from_checkpoint):
        raise Warning("the checkpoint path does not exist.")
        return False

    fluid.io.load_persistables(
        executor=exe,
        dirname=args.init_from_checkpoint,
        main_program=program,
        filename="checkpoint.pdckpt")

    print("finish initing model from checkpoint from %s" %
          (args.init_from_checkpoint))

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


def save_checkpoint(args, exe, program, dirname):

    assert isinstance(args.save_model_path, str)

    checkpoint_dir = os.path.join(args.save_model_path, args.save_checkpoint)

    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    fluid.io.save_persistables(
        exe,
        os.path.join(checkpoint_dir, dirname),
        main_program=program,
        filename="checkpoint.pdckpt")

    print("save checkpoint at %s" % (os.path.join(checkpoint_dir, dirname)))

    return True


def save_param(args, exe, program, dirname):

    assert isinstance(args.save_model_path, str)

    param_dir = os.path.join(args.save_model_path, args.save_param)

    if not os.path.exists(param_dir):
        os.makedirs(param_dir)
    
    fluid.io.save_params(
        exe,
        os.path.join(param_dir, dirname),
        main_program=program,
        filename="params.pdparams")
    print("save parameters at %s" % (os.path.join(param_dir, dirname)))

    return True


