# -*- encoding: utf8 -*-
import os
import sys
sys.path.append("../")

import paddle
import paddle.fluid as fluid
import numpy as np

from models.model_check import check_cuda
from config import PDConfig
from run_classifier import create_model
import utils


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
            infer_pyreader, probs = create_model(
                args,
                pyreader_name='infer_reader',
                num_labels=args.num_labels,
                is_prediction=True)

    test_prog = test_prog.clone(for_test=True)
    exe = fluid.Executor(place)
    exe.run(startup_prog)

    assert (args.init_from_params) or (args.init_from_pretrain_model)

    if args.init_from_params:
        utils.init_from_params(exe, args.init_from_params, test_prog)
    elif args.init_from_pretrain_model:
        utils.init_from_pretrain_model(exe, args.init_from_pretrain_model, test_prog)

    fluid.io.save_inference_model(
        args.inference_model_dir,
        feeded_var_names=["read_file_0.tmp_0"],
        target_vars=[probs],
        executor=exe,
        main_program=test_prog,
        model_filename="model.pdmodel",
        params_filename="params.pdparams")

    print("save inference model at %s" % (args.inference_model_dir))


if __name__ == "__main__":
    args = PDConfig(json_file="./config.json")
    args.build()
    args.print_arguments()
    check_cuda(args.use_cuda)
    if args.do_save_inference_model:
        do_save_inference_model(args)
    else:
        raise ValueError("`do_save_inference_model` must be True.")