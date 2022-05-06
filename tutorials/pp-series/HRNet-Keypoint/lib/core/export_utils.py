# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import yaml
from collections import OrderedDict

import paddle
from lib.dataset.category import get_categories

from lib.utils.logger import setup_logger
logger = setup_logger('hrnet')

# Global dictionary
TRT_MIN_SUBGRAPH = {'HRNet': 3, }


def _prune_input_spec(input_spec, program, targets):
    # try to prune static program to figure out pruned input spec
    # so we perform following operations in static mode
    paddle.enable_static()
    pruned_input_spec = [{}]
    program = program.clone()
    program = program._prune(targets=targets)
    global_block = program.global_block()
    for name, spec in input_spec[0].items():
        try:
            v = global_block.var(name)
            pruned_input_spec[0][name] = spec
        except Exception:
            pass
    paddle.disable_static()
    return pruned_input_spec


def _parse_reader(reader_cfg, dataset_cfg, metric, arch, image_shape):
    preprocess_list = []

    anno_file = dataset_cfg.get_anno()

    clsid2catid, catid2name = get_categories(metric, anno_file, arch)

    label_list = [str(cat) for cat in catid2name.values()]

    fuse_normalize = reader_cfg.get('fuse_normalize', False)
    sample_transforms = reader_cfg['sample_transforms']
    for st in sample_transforms[1:]:
        for key, value in st.items():
            p = {'type': key}
            if key == 'Resize':
                if int(image_shape[1]) != -1:
                    value['target_size'] = image_shape[1:]
            if fuse_normalize and key == 'NormalizeImage':
                continue
            p.update(value)
            preprocess_list.append(p)

    return preprocess_list, label_list


def _parse_tracker(tracker_cfg):
    tracker_params = {}
    for k, v in tracker_cfg.items():
        tracker_params.update({k: v})
    return tracker_params


def _dump_infer_config(config, path, image_shape, model):
    arch_state = False
    from lib.utils.config.yaml_helpers import setup_orderdict

    setup_orderdict()
    use_dynamic_shape = True if image_shape[2] == -1 else False
    infer_cfg = OrderedDict({
        'mode': 'fluid',
        'draw_threshold': 0.5,
        'metric': config['metric'],
        'use_dynamic_shape': use_dynamic_shape
    })
    infer_arch = config['architecture']

    for arch, min_subgraph_size in TRT_MIN_SUBGRAPH.items():
        if arch in infer_arch:
            infer_cfg['arch'] = arch
            infer_cfg['min_subgraph_size'] = min_subgraph_size
            arch_state = True
            break
    if not arch_state:
        logger.error(
            'Architecture: {} is not supported for exporting model now.\n'.
            format(infer_arch) +
            'Please set TRT_MIN_SUBGRAPH in ppdet/engine/export_utils.py')
        os._exit(0)
    label_arch = 'keypoint_arch'

    reader_cfg = config['TestReader']
    dataset_cfg = config['TestDataset']

    infer_cfg['Preprocess'], infer_cfg['label_list'] = _parse_reader(
        reader_cfg, dataset_cfg, config['metric'], label_arch, image_shape[1:])

    yaml.dump(infer_cfg, open(path, 'w'))
    logger.info("Export inference config file to {}".format(
        os.path.join(path)))
