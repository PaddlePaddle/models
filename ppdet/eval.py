"""
  Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
# TODO(dengkaipeng): change comment stype above in github

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import logging
FORMAT='%(asctime)s-%(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)

import paddle.fluid as fluid

from args import parse_args, print_arguments
from ppdet.core.config import load_cfg, merge_cfg
from ppdet.models import Detectors
import ppdet.utils.checkpoint as checkpoint
from ppdet.dataset.reader import Reader
from ppdet.utils.run_utils import parse_fetches, eval_run, eval_results

logger = logging.getLogger(__name__)


def main():
    """
    Main evaluate function
    """
    # 1. load config
    args = parse_args()
    print_arguments(args)
    if args.cfg_file is None:
        raise ValueError("Should specify --cfg_file=configure_file_path.")
    cfg = load_cfg(args.cfg_file)
    merge_cfg(vars(args), cfg)
    merge_cfg({'IS_TRAIN': False}, cfg)

    if cfg.TEST.METRIC_TYPE == 'VOC':
        merge_cfg({'MODE': 'val'}, cfg)

    # 2. build program
    # get detector and losses
    detector = Detectors.get(cfg.MODEL.TYPE)(cfg)
    startup_prog = fluid.Program()
    test_prog = fluid.Program()
    with fluid.program_guard(test_prog, startup_prog):
        with fluid.unique_name.guard():
            if cfg.TEST.METRIC_TYPE == 'COCO':
                fetches = detector.test()
            else:
                fetches = detector.val()
    test_prog = test_prog.clone(True)

    # define executor
    place = fluid.CUDAPlace(0) if cfg.ENV.GPU else fluid.CPUPlace()
    exe = fluid.Executor(place)

    # 3. Compile program for multi-devices
    extra_keys = ['im_info', 'im_id'] if cfg.TEST.METRIC_TYPE == 'COCO' \
                 else []
    keys, values = parse_fetches(fetches, test_prog, extra_keys)

    build_strategy = fluid.BuildStrategy()
    build_strategy.memory_optimize = False
    build_strategy.enable_inplace = False
    compile_program = fluid.compiler.CompiledProgram(
        test_prog).with_data_parallel(build_strategy=build_strategy)

    # 4. Define reader
    reader = Reader(cfg.DATA, cfg.TRANSFORM)
    reader.train()
    test_reader = reader.val()
    pyreader = detector.get_pyreader()
    pyreader.decorate_sample_list_generator(test_reader, place)

    # 5. Load model
    exe.run(startup_prog)
    if cfg.TEST.WEIGHTS:
        checkpoint.load(exe, test_prog, cfg.TEST.WEIGHTS)

    # 6. Run
    results = eval_run(exe, compile_program, pyreader,
                       keys, values)
    # Evaluation
    eval_results(results, cfg, args)


if __name__ == '__main__':
    main()
