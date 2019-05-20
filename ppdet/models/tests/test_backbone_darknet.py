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
from ppdet.models.backbones.darknet import DarkNet53Backbone


def conv_norm_pnames(name):
    pnames = [name + '.conv.weights']
    pnames += [name + '.bn.scale']
    pnames += [name + '.bn.offset']
    pnames += [name + '.bn.mean']
    pnames += [name + '.bn.var']
    return pnames

def basicblock_names(name):
    """
    The naming rules are same as them in
    https://github.com/PaddlePaddle/models/blob/develop/PaddleCV/yolov3/models/darknet.py
    """

    pnames = conv_norm_pnames(name + ".0")
    pnames.extend(conv_norm_pnames(name + ".1"))
    return pnames


class TestDarkNet(unittest.TestCase):
    def setUp(self):
        self.image_shape = [3, 608, 608]
        self.depth_cfg = {
                53: ([1,2,8,8,4], DarkNet53Backbone)
        }
        self.depth= 53

    def get_darknet_params(self, depth):
        """
        The naming rules are same as them in
        https://github.com/PaddlePaddle/models/blob/develop/PaddleCV/yolov3/models/darknet.py

        Return:
            list[string]: all parameter names
        """
        cfg_stages, _ = self.depth_cfg[depth]

        params_names = conv_norm_pnames('yolo_input')
        params_names.extend(conv_norm_pnames('yolo_input.downsample'))
        for i, stage in enumerate(cfg_stages):
            for s in six.moves.xrange(stage):
                params_names.extend(basicblock_names('stage.{}.{}'.format(i, s)))
            if i < len(cfg_stages) - 1:
                params_names.extend(conv_norm_pnames('stage.{}.downsample'.format(i)))

        return params_names

    def compare_darknet(self, depth, bn_decay):
        prog = fluid.Program()
        startup_prog = fluid.Program()
        _, backbone_func = self.depth_cfg[depth]
        with fluid.program_guard(prog, startup_prog):
            image = fluid.layers.data(
                name='image', shape=self.image_shape, dtype='float32')
            feat = backbone_func(image, bn_decay=bn_decay)
            # actual names
            parameters = prog.global_block().all_parameters()
            actual_pnames = [cpt.to_bytes(p.name) for p in parameters]
        # expected names
        expect_pnames = self.get_darknet_params(depth=depth)

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
                self.assertTrue(p.regularizer._regularization_coeff == float(bn_decay))

    def test_darknet53_without_bndecay(self):
        self.compare_darknet(depth=53, bn_decay=False)

    def test_darknet53_with_bndecay(self):
        self.compare_darknet(depth=53, bn_decay=True)



if __name__ == '__main__':
    unittest.main()
