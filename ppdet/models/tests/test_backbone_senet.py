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
import unittest
import paddle.fluid as fluid
import paddle.compat as cpt
from ppdet.core.config import load_cfg, merge_cfg
from ppdet.models.backbones.senet import SENet, SENet154Backbone, SENet154C5


def bottleneck_names(name, bn_affine, short_conv=True):
    def conv_norm(name):
        pnames = [name + '_weights']
        bn_name = name + "_bn"
        pnames += [bn_name + '_scale']
        pnames += [bn_name + '_offset']
        if not bn_affine:
            pnames += [bn_name + '_mean']
            pnames += [bn_name + '_variance']
        return pnames

    def sename(name):
        pnames = [name + '_sqz_weights']
        pnames += [name + '_sqz_offset']
        pnames += [name + '_exc_weights']
        pnames += [name + '_exc_offset']
        return pnames

    names = conv_norm('conv' + name + '_x1')
    names.extend(conv_norm('conv' + name + '_x2'))
    names.extend(conv_norm('conv' + name + '_x3'))
    names.extend(sename('fc' + name))
    if short_conv:
        names.extend(conv_norm(name))
    return names


class TestSENet(unittest.TestCase):
    def setUp(self):
        self.dshape = [3, 224, 224]
        res_cfg = {50: [3, 4, 6, 3], 101: [3, 4, 23, 3], 152: [3, 8, 36, 3]}
        self.layers = 152
        self.depth = res_cfg[self.layers]
        self.stage_filters = [256, 512, 1024, 2048]
        self.fixbn = True
        cfg_file = 'configs/faster-rcnn_SENet154-C4_1x.yml'
        self.cfg = load_cfg(cfg_file)

    def get_C1ToC4_params(self, bn_affine):
        """
        Args:
            bn_affine (bool): meaning use affine_channel
        Return:
            list[string]: all parameter names
        """
        param_names = [
            'conv1_1_weights', 'conv1_2_weights', 'conv1_3_weights',
            'conv1_1_bn_scale', 'conv1_2_bn_scale', 'conv1_3_bn_scale',
            'conv1_1_bn_offset', 'conv1_2_bn_offset', 'conv1_3_bn_offset'
        ]
        if not bn_affine:
            param_names += [
                'conv1_1_bn_mean',
                'conv1_2_bn_mean',
                'conv1_3_bn_mean',
                'conv1_1_bn_variance',
                'conv1_2_bn_variance',
                'conv1_3_bn_variance',
            ]
        n = 3
        for b in range(len(self.depth) - 1):
            n += 1
            for i in range(self.depth[b]):
                name = str(n) + '_' + str(i + 1)
                param_names.extend(bottleneck_names(name, bn_affine, i == 0))
        return param_names

    def get_C5_params(self, bn_affine):
        """
        Args:
            bn_affine (bool): meaning use affine_channel
        Return:
            list[string]: all parameter names
        """
        param_names = []
        for i in range(self.depth[-1]):
            name = str(7) + '_' + str(i + 1)
            param_names.extend(bottleneck_names(name, bn_affine, i == 0))
        return param_names

    def compare_C1ToC4(self, bn_affine):
        prog = fluid.Program()
        startup_prog = fluid.Program()
        with fluid.program_guard(prog, startup_prog):
            data = fluid.layers.data(
                name='input', shape=self.dshape, dtype='float32')
            backbone = SENet154Backbone(self.cfg)
            feat = backbone(data)
            # actual names
            parameters = prog.global_block().all_parameters()
            actual_pnames = [cpt.to_bytes(p.name) for p in parameters]
        # expected names
        expect_pnames = self.get_C1ToC4_params(bn_affine)
        actual_pnames.sort()
        expect_pnames.sort()
        self.assertEqual(len(actual_pnames), len(expect_pnames))
        # check parameter names
        for a, e in zip(actual_pnames, expect_pnames):
            if isinstance(a, bytes):
                a = a.decode()
            if isinstance(e, bytes):
                e = e.decode()
            self.assertTrue(a == e, "Parameter names have diff: \n" +
                            " Actual: " + str(a) + "\n Expect: " + str(e))
        # check learning rate of batch_norm
        for p in parameters:
            if 'bn' in p.name and ('scale' in p.name or 'offset' in p.name):
                self.assertTrue(p.optimize_attr['learning_rate'] == 0.)
                self.assertTrue(p.stop_gradient)

    def test_C1ToC4_bn(self):
        merge_cfg({
            'AFFINE_CHANNEL': False,
            'FREEZE_BN': True,
            'FREEZE_AT': 2,
            'GROUPS': 64
        }, self.cfg.MODEL)
        assert not self.cfg.MODEL.AFFINE_CHANNEL
        self.compare_C1ToC4(False)

    def test_C1ToC4_affine(self):
        merge_cfg({
            'AFFINE_CHANNEL': True,
            'FREEZE_BN': True,
            'FREEZE_AT': 2,
            'GROUPS': 64
        }, self.cfg.MODEL)
        assert self.cfg.MODEL.AFFINE_CHANNEL
        self.compare_C1ToC4(True)

    def compare_C5(self, bn_affine):
        prog = fluid.Program()
        startup_prog = fluid.Program()
        with fluid.program_guard(prog, startup_prog):
            data = fluid.layers.data(
                name='input', shape=[2048, 14, 14], dtype='float32')
            #feat = ResNet50C5(data, True, bn_affine)
            c5_stage = SENet154C5(self.cfg)
            feat = c5_stage(data)
            # actual names
            parameters = prog.global_block().all_parameters()
            actual_pnames = [cpt.to_bytes(p.name) for p in parameters]
        # expected names
        expect_pnames = self.get_C5_params(bn_affine)
        actual_pnames.sort()
        expect_pnames.sort()
        self.assertEqual(len(actual_pnames), len(expect_pnames))
        for a, e in zip(actual_pnames, expect_pnames):
            if isinstance(a, bytes):
                a = a.decode()
            if isinstance(e, bytes):
                e = e.decode()
            self.assertTrue(a == e, "Parameter names have diff: \n" +
                            " Actual: " + str(a) + "\n Expect: " + str(e))
        # check learning rate of batch_norm
        for p in parameters:
            if 'bn' in p.name and ('scale' in p.name or 'offset' in p.name):
                self.assertTrue(p.optimize_attr['learning_rate'] == 0.)
                self.assertTrue(p.stop_gradient)

    def test_C5_bn(self):
        merge_cfg({
            'AFFINE_CHANNEL': False,
            'FREEZE_BN': True,
            'GROUPS': 64
        }, self.cfg.MODEL)
        assert not self.cfg.MODEL.AFFINE_CHANNEL
        self.compare_C5(False)

    def test_C5_affine(self):
        merge_cfg({
            'AFFINE_CHANNEL': True,
            'FREEZE_BN': True,
            'GROUPS': 64
        }, self.cfg.MODEL)
        assert self.cfg.MODEL.AFFINE_CHANNEL
        self.compare_C5(True)


if __name__ == '__main__':
    unittest.main()
