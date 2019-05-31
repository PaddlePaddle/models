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

from ..registry import RoIExtractors

__all__ = ['RoIPool', 'RoIAlign', 'FPNRoIAlign']


class RoIExtractor(object):
    """
    RoIExtractor class

    Args:
        cfg(Dict): All parameters in dictionary.
    """

    def __init__(self, cfg):
        self.cfg = cfg

    def get_roi_feat(self, head_inputs, rois):
        """
        Get feature after RoIExtractor.
 
        Args:
            head_inputs(Variable): The inputs of RoIExtractor.
                The format of input tensor is NCHW. N is batch size, C is the 
                number of input channels, H is the height of the feature, 
                and W is the width of the feature.
            rois(Variable): RoIs to pool over. Should be a 2-D LoDTensor of 
                shape (num_rois, 4) given as [[x1, y1, x2, y2], ...]. (x1, y1) 
                is the top left coordinates, and (x2, y2) is the bottom right 
                coordinates.
        """
        raise NotImplementedError


@RoIExtractors.register
class RoIAlign(RoIExtractor):
    """
        RoIAlign class
    """

    def __init__(self, cfg):
        super(RoIAlign, self).__init__(cfg)

    def get_roi_feat(self, head_input, rois):
        """
        Adopt RoI align to get RoI features

        Returns:
            roi_feat(Variable): RoI features with shape of [M, C, R, R], where 
                M is the number of RoIs and R is RoI resolution
                
        """
        roi_feat = fluid.layers.roi_align(
            input=head_input,
            rois=rois,
            pooled_height=self.cfg.ROI_EXTRACTOR.ROI_RESOLUTION,
            pooled_width=self.cfg.ROI_EXTRACTOR.ROI_RESOLUTION,
            spatial_scale=self.cfg.ROI_EXTRACTOR.SPATIAL_SCALE,
            sampling_ratio=self.cfg.ROI_EXTRACTOR.SAMPLING_RATIO)

        return roi_feat


@RoIExtractors.register
class RoIPool(RoIExtractor):
    """
        RoIPool class
    """

    def __init__(self, cfg):
        super(RoIPool, self).__init__(cfg)

    def get_roi_feat(self, head_inputs, rois):
        """
        Adopt RoI pooling to get RoI features

        Returns:
            roi_feat(Variable): RoI features with shape of [M, C, R, R], where 
                M is the number of RoIs and R is RoI resolution

        """
        roi_feat = fluid.layers.roi_pool(
            input=head_input,
            rois=rois,
            pooled_height=self.cfg.ROI_EXTRACTOR.ROI_RESOLUTION,
            pooled_width=self.cfg.ROI_EXTRACTOR.ROI_RESOLUTION,
            spatial_scale=self.cfg.ROI_EXTRACTOR.SPATIAL_SCALE)

        return roi_feat


@RoIExtractors.register
class FPNRoIAlign(object):
    """
        FPNRoIAlign class
    """

    def __init__(self, cfg):
        self.cfg = cfg

    def get_roi_feat(self, head_inputs, rois, name_list, spatial_scale):
        """
        Adopt RoI align onto several level of feature maps to get RoI features.
        Distribute RoIs to different levels by area and get a list of RoI 
        features by distributed RoIs and their corresponding feature maps.

        Returns:
            roi_feat(Variable): RoI features with shape of [M, C, R, R], 
                where M is the number of RoIs and R is RoI resolution

        """
        k_min = self.cfg.ROI_EXTRACTOR.FPN_ROI_MIN_LEVEL
        k_max = self.cfg.ROI_EXTRACTOR.FPN_ROI_MAX_LEVEL
        num_roi_lvls = k_max - k_min + 1
        input_name_list = name_list[-num_roi_lvls:]
        spatial_scale = spatial_scale[-num_roi_lvls:]
        rois_dist, restore_index = fluid.layers.distribute_fpn_proposals(
            rois,
            k_min,
            k_max,
            self.cfg.ROI_EXTRACTOR.FPN_ROI_CANCONICAL_LEVEL,
            self.cfg.ROI_EXTRACTOR.FPN_ROI_CANONICAL_SCALE,
            name='distribute')
        # rois_dist is in ascend order
        roi_out_list = []
        for lvl in range(num_roi_lvls):
            name_index = num_roi_lvls - lvl - 1
            rois_input = rois_dist[lvl]
            head_input = head_inputs[input_name_list[name_index]]
            sc = spatial_scale[name_index]
            roi_out = fluid.layers.roi_align(
                input=head_input,
                rois=rois_input,
                pooled_height=self.cfg.ROI_EXTRACTOR.ROI_RESOLUTION,
                pooled_width=self.cfg.ROI_EXTRACTOR.ROI_RESOLUTION,
                spatial_scale=sc,
                sampling_ratio=self.cfg.ROI_EXTRACTOR.SAMPLING_RATIO,
                name='roi_align_lvl_' + str(lvl))
            roi_out_list.append(roi_out)
        roi_feat_shuffle = fluid.layers.concat(roi_out_list)
        roi_feat_ = fluid.layers.gather(roi_feat_shuffle, restore_index)
        roi_feat = fluid.layers.lod_reset(roi_feat_, rois)

        return roi_feat
