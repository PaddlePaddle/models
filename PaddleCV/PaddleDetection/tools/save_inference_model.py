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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from paddle import fluid

from ppdet.core.workspace import load_config, merge_config, create

from ppdet.utils.cli import ArgsParser
from ppdet.utils.check import check_gpu
import ppdet.utils.checkpoint as checkpoint

import logging
FORMAT = '%(asctime)s-%(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)


def prune_feed_vars(feeded_var_names, target_vars, prog):
    """
    Filter out feed variables which are not in program,
    pruned feed variables are only used in post processing
    on model output, which are not used in program, such
    as im_id to identify image order, im_shape to clip bbox
    in image.
    """
    exist_var_names = []
    prog = prog.clone()
    prog = prog._prune(feeded_var_names, targets=target_vars)
    global_block = prog.global_block()
    for name in feeded_var_names:
        try:
            v = global_block.var(name)
            exist_var_names.append(str(v.name))
        except Exception:
            logger.info('save_inference_model pruned unused feed '
                        'variables {}'.format(name))
            pass
    return exist_var_names


def save_infer_model(FLAGS, exe, feeded_var_names, test_fetches, infer_prog):
    cfg_name = os.path.basename(FLAGS.config).split('.')[0]
    save_dir = os.path.join(FLAGS.output_dir, cfg_name)
    target_vars = list(test_fetches.values())
    feeded_var_names = prune_feed_vars(feeded_var_names, target_vars,
                                       infer_prog)
    logger.info("Save inference model to {}, input: {}, output: "
                "{}...".format(save_dir, feeded_var_names,
                               [str(var.name) for var in target_vars]))
    fluid.io.save_inference_model(
        save_dir,
        feeded_var_names=feeded_var_names,   # WTF is `feeded`??
        target_vars=target_vars,
        executor=exe,
        main_program=infer_prog,
        params_filename="__params__")


def main():
    cfg = load_config(FLAGS.config)

    merge_config(FLAGS.opt)
    assert 'weights' in cfg, "'weights' is required for inference model"
    assert 'inference_shape' in cfg, \
        "'inference_shape' is required for saving inference model"
    assert 'inference_vars' in cfg, \
        "'inference_vars' is required for saving inference model"

    if 'architecture' in cfg:
        main_arch = cfg.architecture
    else:
        raise ValueError("'architecture' not specified in config file.")

    check_gpu(cfg.use_gpu)

    feed_var_names = [isinstance(v, str) and v or v['name']
                      for v in cfg.inference_vars]

    place = fluid.CUDAPlace(0) if cfg.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)

    model = create(main_arch)

    startup_prog = fluid.Program()
    infer_prog = fluid.Program()
    with fluid.program_guard(infer_prog, startup_prog):
        with fluid.unique_name.guard():
            test_fetches = model.test(image_shape=cfg.inference_shape)
    infer_prog = infer_prog.clone(True)

    exe.run(startup_prog)
    checkpoint.load_params(exe, infer_prog, cfg.weights)
    save_infer_model(FLAGS, exe, feed_var_names, test_fetches, infer_prog)


if __name__ == '__main__':
    parser = ArgsParser()
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory for storing the output model files.")
    FLAGS = parser.parse_args()
    main()
