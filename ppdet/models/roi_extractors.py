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

import paddle.fluid as fluid


class RoIExtractor(object):
    def __init__(self, cfg):
        self.cfg = cfg

    def get_roi_feat(self, head_input, rois):
        raise NotImplementedError


class RoIAlign(RoIExtractor):
    def __init__(self, cfg):
        super(RoIAlign, self).__init__(cfg)

    def get_roi_feat(self, head_input, rois):
        roi_feat = fluid.layers.roi_align(
            input=head_input,
            rois=rois,
            pooled_height=self.cfg.ROI_EXTRACTOR.ROI_RESOLUTION,
            pooled_width=self.cfg.ROI_EXTRACTOR.ROI_RESOLUTION,
            spatial_scale=self.cfg.ROI_EXTRACTOR.SPATIAL_SCALE,
            sampling_ratio=self.cfg.ROI_EXTRACTOR.SAMPLING_RATIO)

        return roi_feat


class RoIPool(RoIExtractor):
    def __init__(self, cfg):
        super(RoIPool, self).__init__(cfg)

    def get_roi_feat(self, head_input, rois):
        roi_feat = fluid.layers.roi_pool(
            input=head_input,
            rois=rois,
            pooled_height=self.cfg.ROI_EXTRACTOR.ROI_RESOLUTION,
            pooled_width=self.cfg.ROI_EXTRACTOR.ROI_RESOLUTION,
            spatial_scale=self.cfg.ROI_EXTRACTOR.SPATIAL_SCALE)

        return roi_feat


class FPNRoIAlign(RoIExtractor):
    def __init__(self, cfg):
        super(FPNRoIAlign, self).__init__(cfg)

    def get_roi_feat(self, head_inputs, rois):
        k_min = self.cfg.ROI_EXTRACTOR.FPN_ROI_MIN_LEVEL
        k_max = self.cfg.ROI_EXTRACTOR.FPN_ROI_MAX_LEVEL
        num_roi_lvls = k_max - k_min + 1
        rois_dist, restore_index = fluid.layers.distribute_fpn_proposals(
            rois,
            k_min,
            k_max,
            self.cfg.ROI_EXTRACTOR.FPN_ROI_CANCONICAL_LEVEL,
            self.cfg.ROI_EXTRACTOR.FPN_ROI_CANONICAL_SCALE,
            name='distribute')
        # head_inputs is in descend order
        # rois_dist is in ascend order
        roi_out_list = []
        for lvl in range(num_roi_lvls):
            rois = rois_dist[lvl]
            head_input = head_inputs[num_roi_lvls - lvl - 1]
            sc = eval(self.cfg.ROI_EXTRACTOR.SPATIAL_SCALE[num_roi_lvls - lvl -
                                                           1])
            roi_out = fluid.layers.roi_align(
                input=head_input,
                rois=rois,
                pooled_height=self.cfg.ROI_EXTRACTOR.ROI_RESOLUTION,
                pooled_width=self.cfg.ROI_EXTRACTOR.ROI_RESOLUTION,
                spatial_scale=sc,
                sampling_ratio=self.cfg.ROI_EXTRACTOR.SAMPLING_RATIO,
                name='roi_align_lvl_' + str(lvl))
        roi_out_list.append(roi_out)
        roi_feat_shuffle = fluid.layers.concat(roi_out_list)
        roi_feat = fluid.layers.gather(roi_feat_shuffle, restore_index)
        roi_feat = fluid.layers.lod_reset(roi_feat, rois)

        return roi_feat
