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

import six
import numpy as np
import unittest
import paddle.fluid as fluid
import paddle.compat as cpt


from ppdet.core.config import load_cfg, merge_cfg
from ppdet.models.backbones.mobilenet import MobileNetV1Backbone


def conv_norm_pnames(name):
    """
    The naming rules are same as them in
    https://github.com/PaddlePaddle/models/blob/develop/PaddleCV/image_classification/models/mobilenet.py
    """
    pnames = [name + '_weights']
    pnames += [name + "_bn" + '_scale']
    pnames += [name + "_bn" + '_offset']
    pnames += [name + "_bn" + '_mean']
    pnames += [name + "_bn" + '_variance']
    return pnames

def depthwise_separable_names(name):
    """
    The naming rules are same as them in
    https://github.com/PaddlePaddle/models/blob/develop/PaddleCV/image_classification/models/mobilenet.py
    """

    pnames = conv_norm_pnames(name + "_dw")
    pnames.extend(conv_norm_pnames(name + "_seq"))
    return pnames


class TestMobileNet(unittest.TestCase):
    def setUp(self):
        self.image_shape = [3, 300, 300]
        cfg_file = 'configs/ssd_MobileNet_1x.yml'
        self.cfg = load_cfg(cfg_file)

    def get_mobilenet_params(self):
        params_names = conv_norm_pnames('conv1')
        params_names.extend(depthwise_separable_names('conv2_1'))
        params_names.extend(depthwise_separable_names('conv2_2'))
        params_names.extend(depthwise_separable_names('conv3_1'))
        params_names.extend(depthwise_separable_names('conv3_2'))
        params_names.extend(depthwise_separable_names('conv4_1'))
        params_names.extend(depthwise_separable_names('conv4_2'))
        for i in range(5):
            params_names.extend(depthwise_separable_names("conv5" + "_" + str(i + 1)))
        params_names.extend(depthwise_separable_names('conv5_6'))
        params_names.extend(depthwise_separable_names('conv6'))
        return params_names

    def compare_mobilenet(self):
        prog = fluid.Program()
        startup_prog = fluid.Program()
        with fluid.program_guard(prog, startup_prog):
            image = fluid.layers.data(
                name='image', shape=self.image_shape, dtype='float32')
            out = MobileNetV1Backbone(self.cfg)(image)
            # actual names
            parameters = prog.global_block().all_parameters()
            actual_pnames = [p.name for p in parameters]
        # expected names
        expect_pnames = self.get_mobilenet_params()

        actual_pnames.sort()
        expect_pnames.sort()

        self.assertEqual(len(actual_pnames), len(expect_pnames))
        # check parameter names
        for a, e in zip(actual_pnames, expect_pnames):
            self.assertTrue(a == e, "Parameter names have diff: \n" +
                            " Actual: " + str(a) + "\n Expect: " + str(e))
        # check decay of batch_norm
        for p in parameters:
            if 'bn' in p.name and ('scale' in p.name or 'offset' in p.name):
                self.assertTrue(p.regularizer is None)

    def test_mobilenetv1(self):
        self.compare_mobilenet()
        
        
if __name__ == '__main__':
    unittest.main()
