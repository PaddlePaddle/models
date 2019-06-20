#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import logging
FORMAT='%(asctime)s-%(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)

import paddle.fluid as fluid

from ppdet.utils.cli import parse_args
import ppdet.utils.checkpoint as checkpoint
from ppdet.core.workspace import load_config, merge_config, create
from ppdet.data_feed import make_reader
from tools.eval_utils import parse_fetches, eval_run, eval_results

logger = logging.getLogger(__name__)


def main():
    """
    Main evaluate function
    """
    args = parse_args()
    cfg = load_config(args.config)

    if 'architecture' in cfg:
        main_arch = cfg['architecture']
    else:
        raise ValueError("The architecture is not specified in config file.")

    merge_config(args.cli_config)

    if cfg['use_gpu']:
        devices_num = fluid.core.get_cuda_device_count()
    else:
        devices_num = os.environ.get('CPU_NUM', multiprocessing.cpu_count())

    if 'eval_feed' not in cfg:
        eval_feed = create(main_arch + 'EvalFeed')
    else:
        eval_feed = create(cfg['eval_feed'])

    # define executor
    place = fluid.CUDAPlace(0) if cfg['use_gpu'] else fluid.CPUPlace()
    exe = fluid.Executor(place)

    # 2. build program
    # get detector and losses
    model = create(main_arch)
    startup_prog = fluid.Program()
    eval_prog = fluid.Program()
    with fluid.program_guard(eval_prog, startup_prog):
        with fluid.unique_name.guard():
            # must be in the same scope as model when initialize feed_vars
            pyreader, reader, feed_vars = make_reader(eval_feed)
            if cfg['metric'] == 'COCO':
                fetches = model.test(feed_vars)
            else:
                fetches = model.val(feed_vars)
    eval_prog = eval_prog.clone(True)
    pyreader.decorate_sample_list_generator(reader, place)

    # 3. Compile program for multi-devices
    build_strategy = fluid.BuildStrategy()
    build_strategy.memory_optimize = False
    build_strategy.enable_inplace = False
    compile_program = fluid.compiler.CompiledProgram(
        eval_prog).with_data_parallel(build_strategy=build_strategy)

    # 5. Load model
    exe.run(startup_prog)
    if cfg['weights']:
        checkpoint.load_checkpoint(exe, eval_prog, cfg['weights'])

    extra_keys = ['im_info', 'im_id', 'im_shape'] if cfg['metric'] == 'COCO' \
                 else []
    keys, values = parse_fetches(fetches, eval_prog, extra_keys)

    # 6. Run
    results = eval_run(exe, compile_program, pyreader,
                       keys, values)
    # Evaluation
    eval_results(results, eval_feed, args, cfg)


if __name__ == '__main__':
    main()
