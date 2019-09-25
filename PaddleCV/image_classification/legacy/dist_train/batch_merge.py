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

import paddle.fluid as fluid
import numpy as np


def copyback_repeat_bn_params(main_prog):
    repeat_vars = set()
    for op in main_prog.global_block().ops:
        if op.type == "batch_norm":
            repeat_vars.add(op.input("Mean")[0])
            repeat_vars.add(op.input("Variance")[0])
    for vname in repeat_vars:
        real_var = fluid.global_scope().find_var("%s.repeat.0" %
                                                 vname).get_tensor()
        orig_var = fluid.global_scope().find_var(vname).get_tensor()
        orig_var.set(np.array(real_var), fluid.CUDAPlace(0))  # test on GPU0


def append_bn_repeat_init_op(main_prog, startup_prog, num_repeats):
    repeat_vars = set()
    for op in main_prog.global_block().ops:
        if op.type == "batch_norm":
            repeat_vars.add(op.input("Mean")[0])
            repeat_vars.add(op.input("Variance")[0])

    for i in range(num_repeats):
        for op in startup_prog.global_block().ops:
            if op.type == "fill_constant":
                for oname in op.output_arg_names:
                    if oname in repeat_vars:
                        var = startup_prog.global_block().var(oname)
                        repeat_var_name = "%s.repeat.%d" % (oname, i)
                        repeat_var = startup_prog.global_block().create_var(
                            name=repeat_var_name,
                            type=var.type,
                            dtype=var.dtype,
                            shape=var.shape,
                            persistable=var.persistable)
                        main_prog.global_block()._clone_variable(repeat_var)
                        startup_prog.global_block().append_op(
                            type="fill_constant",
                            inputs={},
                            outputs={"Out": repeat_var},
                            attrs=op.all_attrs())
