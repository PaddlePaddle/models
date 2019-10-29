#  Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
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

import os
import numpy as np
import paddle.fluid as fluid
import logging
logger = logging.getLogger(__name__)


def is_parameter(var):
    return isinstance(var, fluid.framework.Parameter)


def load_pretrain_params_from_file(exe, prog, pretrained_file, place):
    """
    The pretrined_file stores ResNet50/101 parameters pretrained on ImageNet.
    However, the conv_weights of Nonlocal model is not the same as that in ResNet50/101 because the 
    input shape is [N, C, T, H, W] and the convolution kernels' shape is [Cout, Cin, Kt, Kh, Kw]. It is
    different from the convolution kernels of ResNet whose shape is typically [Cout, Cin, Kh, Kw].
    When loading conv_weights from the pretrained file, shape mismatch error will be raised due to the check 
    in fluid.io. This check on params' shape is newly added in fluid.version==1.6.0. So it is recommendated to 
    treat conv_weights specifically.
    The process is as following:
      1, check the params that will be loaded, those with the same name in the target program and pretrained_file. 
         These params will be called common params in this function.
      2, Create presistable variables in the new_scope with the name of each common params. If it is the weights of 
         convolution, the created varibale's shape will be set to 2D-convolution-kernel type.
      3, load params from the pretrained_file into those persistable variables created in the new_scope
      4, get the value of common params in the new_scope and transform it if it belongs to conv weights.
      5, set the value to params in the target program
    """

    logger.info('load pretrained params from {}'.format(pretrained_file))
    if os.path.isdir(pretrained_file):
        # get params' list in prog
        param_list = filter(is_parameter, prog.list_vars())
        param_name_list = [p.name for p in param_list]

        # get all params' names in pretrained_file
        param_name_from_file = os.listdir(pretrained_file)
        # get common params of prog and pretrained_file
        # only those common params will be loaded from pretrained_file into prog
        common_names = get_common_names(param_name_list, param_name_from_file)

        # get global scope and block for prog
        global_scope = fluid.global_scope()
        global_block = prog.global_block()

        # save details of common params
        common_var_map = {}
        for name in common_names:
            var = global_block.var(name)
            var_type = var.type
            var_dtype = var.dtype
            var_shape = var.shape
            if len(var_shape) == 5:
                # When param is conv_weights, its shape is [Cout, Cin, Kt, Kh, Kw].
                # The corresponding params in ResNet50/101 is [Cout, Cin, Kh, Kw]
                var_shape2d = (var_shape[0], var_shape[1], var_shape[3],
                               var_shape[4])
            else:
                var_shape2d = var_shape[:]
            common_var_map[name] = [var_type, var_dtype, var_shape, var_shape2d]

        # create new_scope and new_prog to create vars
        cpu_place = fluid.CPUPlace()
        exe_cpu = fluid.Executor(cpu_place)
        new_scope = fluid.Scope()
        new_prog = fluid.Program()
        new_start_prog = fluid.Program()
        new_block = new_prog.global_block()

        # create vars in new_scope
        created_vars = []
        with fluid.scope_guard(new_scope):
            with fluid.program_guard(new_prog, new_start_prog):
                for name in common_names:
                    var_type, var_dtype, var_shape, var_shape2d = common_var_map[
                        name]
                    new_var = new_block.create_var(
                        name=name,
                        type=var_type,
                        shape=var_shape2d,
                        dtype=var_dtype,
                        persistable=True)
                    created_vars.append(new_var)

        # load pretrained_file into the persistable vars created in new_scope
        with fluid.scope_guard(new_scope):
            fluid.io.load_vars(
                exe_cpu,
                pretrained_file,
                main_program=new_prog,
                vars=created_vars)

        logger.info('-------- loading params -----------')
        for name in common_names:
            # get the tensor of vars in new_scope
            new_tensor = new_scope.var(name).get_tensor()
            new_value = np.array(new_tensor)

            prog_tensor = global_scope.var(name).get_tensor()
            var_type, var_dtype, var_shape, var_shape2d = common_var_map[name]
            # set the value of loaded vars to those with the same name in the target program
            if len(var_shape) == 5:
                # transform the loaded conv weights into the format of [Cout, Cin, Kt, Kh, Kw]
                num_inflate = var_shape[2]
                stacked_array = np.stack(
                    [new_value] * num_inflate, axis=2) / float(num_inflate)
                prog_tensor.set(stacked_array.astype('float32'), place)
                logger.info("load inflated({}) param {}".format(num_inflate,
                                                                name))
            else:
                prog_tensor.set(new_value, place)
                logger.info("load param {}".format(name))
    else:
        raise TypeError(
            "pretrained file {} is not in a directory, not suitable to load params".
            format(pretrained_file))


def get_common_names(param_name_list, param_name_from_file):
    # name check and return common names both in param_name_list and file
    common_names = []
    paddle_only_names = []
    file_only_names = []
    logger.info('-------- comon params -----------')
    for name in param_name_list:
        if name in param_name_from_file:
            common_names.append(name)
            logger.info(name)
        else:
            paddle_only_names.append(name)
    logger.info('-------- paddle only params ----------')
    for name in paddle_only_names:
        logger.info(name)
    logger.info('-------- file only params -----------')
    for name in param_name_from_file:
        if name in param_name_list:
            assert name in common_names
        else:
            file_only_names.append(name)
            logger.info(name)
    return common_names


def load_weights_params_from_file(exe, prog, weights, place):
    """
    The params of the training process is stored in the file named weights.
    However, the network of the training and test process is slightly different due to the layer 
    named "pred" was fc in trainng but convolution in test. When loading weights of pred (pred_w), 
    from the pretrained file, shape mismatch error will be raised due to the check in fluid.io. 
    This check on params' shape is newly added in fluid.version==1.6.0. So it is recommendated to 
    treat pred_w specifically.
    The process is as following:
      1, get the details of param_list in the target program (prog)
      2, create persistable vars in new_scope with the same name as those in param_list with 
         the details stored in step 1. If the name is 'pred_w', the var shape should be [Cin, Cout].
      3, get the value of vars in the new_scope. 
         If var.name is 'pred_w', transform it from fc-weights type to be consistent with convolution.
      4, set the value to params in prog
    """

    logger.info('Load test weights from {}'.format(weights))

    # get the param_list in prog
    prog_vars = filter(is_parameter, prog.list_vars())

    # save the details of params in prog
    var_map = {}
    for var in prog_vars:
        var_name = var.name
        var_type = var.type
        var_dtype = var.dtype
        var_shape = var.shape
        # For pred_w, get the fc-weights type shape
        if var_name == "pred_w":
            assert len(
                var_shape
            ) == 5, "pred_weights.shape shoud be [Cout, Cin, 1, 1, 1] when test"
            var_shape = (var_shape[1], var_shape[0])
        var_map[var_name] = [var_type, var_dtype, var_shape]

    # create new_scope and new_prog
    cpu_place = fluid.CPUPlace()
    exe_cpu = fluid.Executor(cpu_place)
    new_scope = fluid.Scope()
    new_prog = fluid.Program()
    new_start_prog = fluid.Program()
    new_block = new_prog.global_block()
    created_vars = []
    # create persistable variables in new_scope
    with fluid.scope_guard(new_scope):
        with fluid.program_guard(new_prog, new_start_prog):
            for var_name in var_map.keys():
                var_type, var_dtype, var_shape = var_map[var_name]
                new_var = new_block.create_var(
                    name=var_name,
                    type=var_type,
                    shape=var_shape,
                    dtype=var_dtype,
                    persistable=True)
                created_vars.append(new_var)

    # load params from file into the above vars created in new_scope
    with fluid.scope_guard(new_scope):
        fluid.io.load_vars(
            exe_cpu,
            '',
            main_program=new_prog,
            vars=created_vars,
            filename=weights)

    # get the global scope of prog
    global_scope = fluid.global_scope()

    # set value of vars in new_scope to the params of prog with the same name
    # and specially treat on "pred_w"
    for var_name in var_map.keys():
        global_tensor = global_scope.var(var_name).get_tensor()
        new_tensor = new_scope.var(var_name).get_tensor()
        new_value = np.array(new_tensor)
        if var_name != "pred_w":
            global_tensor.set(new_value, place)
        else:
            pred_array = np.transpose(new_value, (1, 0))
            pred_array = np.reshape(
                pred_array,
                [pred_array.shape[0], pred_array.shape[1], 1, 1, 1])
            global_tensor.set(pred_array.astype('float32'), place)
