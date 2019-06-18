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
from ppdet.models.backbones.resnet_vd import ResNetVd, ResNetVd50Backbone, ResNetVd50C5


def bottleneck_names(name, bn_affine, short_conv=True):
    def conv_norm(name):
        pnames = [name + '_weights']
        pnames += ['bn' + name[3:] + '_scale']
        pnames += ['bn' + name[3:] + '_offset']
        if not bn_affine:
            pnames += ['bn' + name[3:] + '_mean']
            pnames += ['bn' + name[3:] + '_variance']
        return pnames

    names = conv_norm(name + "_branch2a")
    names.extend(conv_norm(name + "_branch2b"))
    names.extend(conv_norm(name + "_branch2c"))
    if short_conv:
        names.extend(conv_norm(name + "_branch1"))
    return names


class TestResNetVd(unittest.TestCase):
    def setUp(self):
        self.dshape = [3, 224, 224]
        res_cfg = {
            18: [2, 2, 2, 2],
            34: [3, 4, 6, 3],
            50: [3, 4, 6, 3],
            101: [3, 4, 23, 3],
            152: [3, 8, 36, 3]
        }
        self.layers = 50
        self.depth = res_cfg[self.layers]
        self.fixbn = True

        cfg_file = 'configs/faster-rcnn_ResNetVd50-C4_1x.yml'
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
            'bnv1_1_scale', 'bnv1_1_offset', 'bnv1_2_scale', 'bnv1_2_offset',
            'bnv1_3_scale', 'bnv1_3_offset'
        ]
        if not bn_affine:
            param_names += [
                'bnv1_1_mean', 'bnv1_1_variance', 'bnv1_2_mean',
                'bnv1_2_variance', 'bnv1_3_mean', 'bnv1_3_variance'
            ]

        for b in range(len(self.depth) - 1):
            for i in range(self.depth[b]):
                if self.layers in [101, 152] and b == 2:
                    if i == 0:
                        name = "res" + str(b + 2) + "a"
                    else:
                        name = "res" + str(b + 2) + "b" + str(i)
                else:
                    name = "res" + str(b + 2) + chr(97 + i)
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
            name = "res" + str(5) + chr(97 + i)
            param_names.extend(bottleneck_names(name, bn_affine, i == 0))
        return param_names

    def compare_C1ToC4(self, bn_affine):
        prog = fluid.Program()
        startup_prog = fluid.Program()
        with fluid.program_guard(prog, startup_prog):
            data = fluid.layers.data(
                name='input', shape=self.dshape, dtype='float32')
            backbone = ResNetVd50Backbone(self.cfg)
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
            'FREEZE_AT': 2
        }, self.cfg.MODEL)
        assert not self.cfg.MODEL.AFFINE_CHANNEL
        self.compare_C1ToC4(False)

    def test_C1ToC4_affine(self):
        merge_cfg({
            'AFFINE_CHANNEL': True,
            'FREEZE_BN': True,
            'FREEZE_AT': 2
        }, self.cfg.MODEL)
        assert self.cfg.MODEL.AFFINE_CHANNEL
        self.compare_C1ToC4(True)

    def compare_C5(self, bn_affine):
        prog = fluid.Program()
        startup_prog = fluid.Program()
        with fluid.program_guard(prog, startup_prog):
            data = fluid.layers.data(
                name='input', shape=[1024, 14, 14], dtype='float32')
            #feat = ResNet50C5(data, True, bn_affine)
            c5_stage = ResNetVd50C5(self.cfg)
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
        }, self.cfg.MODEL)
        assert not self.cfg.MODEL.AFFINE_CHANNEL
        self.compare_C5(False)

    def test_C5_affine(self):
        merge_cfg({
            'AFFINE_CHANNEL': True,
            'FREEZE_BN': True,
        }, self.cfg.MODEL)
        assert self.cfg.MODEL.AFFINE_CHANNEL
        self.compare_C5(True)


if __name__ == '__main__':
    unittest.main()
