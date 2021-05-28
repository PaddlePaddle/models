# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from smoke.ops import gather_op


def sigmoid_hm(hm_features):
    """sigmoid to headmap

    Args:
        hm_features (paddle.Tensor): heatmap

    Returns:
        paddle.Tensor: sigmoid heatmap
    """
    x = F.sigmoid(hm_features)
    x = x.clip(min=1e-4, max=1 - 1e-4)

    return x

def nms_hm(heat_map, kernel=3):
    """Do max_pooling for nms

    Args:
        heat_map (paddle.Tensor): pred cls heatmap
        kernel (int, optional): max_pool kernel size. Defaults to 3.

    Returns:
        heatmap after nms
    """
    pad = (kernel - 1) // 2

    hmax = F.max_pool2d(heat_map,
                        kernel_size=(kernel, kernel),
                        stride=1,
                        padding=pad)
    eq_index = (hmax == heat_map).astype("float32")

    return heat_map * eq_index


def select_topk(heat_map, K=100):
    """
    Args:
        heat_map: heat_map in [N, C, H, W]
        K: top k samples to be selected
        score: detection threshold

    Returns:

    """
    
    #batch, c, height, width = paddle.shape(heat_map)

    batch, c = heat_map.shape[:2]
    height = paddle.shape(heat_map)[2]
    width = paddle.shape(heat_map)[3]
    
    # First select topk scores in all classes and batchs
    # [N, C, H, W] -----> [N, C, H*W]
    heat_map = paddle.reshape(heat_map, (batch, c, -1))
    # Both in [N, C, K]
    topk_scores_all, topk_inds_all = paddle.topk(heat_map, K)
    

    # topk_inds_all = topk_inds_all % (height * width) # todo: this seems redudant
    topk_ys = (topk_inds_all // width).astype("float32")
    topk_xs = (topk_inds_all % width).astype("float32")
    

    # Select topK examples across channel
    # [N, C, K] -----> [N, C*K]
    topk_scores_all = paddle.reshape(topk_scores_all, (batch, -1))
    # Both in [N, K]
    topk_scores, topk_inds = paddle.topk(topk_scores_all, K)
    topk_clses = (topk_inds // K).astype("float32")

    # First expand it as 3 dimension
    topk_inds_all = paddle.reshape(_gather_feat(paddle.reshape(topk_inds_all, (batch, -1, 1)), topk_inds), (batch, K))
    topk_ys = paddle.reshape(_gather_feat(paddle.reshape(topk_ys, (batch, -1, 1)), topk_inds), (batch, K))
    topk_xs = paddle.reshape(_gather_feat(paddle.reshape(topk_xs, (batch, -1, 1)), topk_inds), (batch, K))
    
    return dict({"topk_score": topk_scores, "topk_inds_all": topk_inds_all,
                "topk_clses": topk_clses, "topk_ys": topk_ys, "topk_xs": topk_xs})


def _gather_feat(feat, ind, mask=None):
    """
    Select specific indexs on featuremap
    Args:
        feat: all results in 3 dimensions
        ind: positive index

    Returns:

    """
    channel = feat.shape[-1]
    ind = ind.unsqueeze(-1).expand((ind.shape[0], ind.shape[1], channel))

    feat = gather_op(feat, 1, ind)

    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, channel)
    return feat

def select_point_of_interest(batch, index, feature_maps):
    """
    Select POI(point of interest) on feature map
    Args:
        batch: batch size
        index: in point format or index format
        feature_maps: regression feature map in [N, C, H, W]

    Returns:

    """
    w = feature_maps.shape[3]
    index_length = len(index.shape)
    if index_length == 3:
        index = index[:, :, 1] * w + index[:, :, 0]
    index = paddle.reshape(index, (batch, -1))
    # [N, C, H, W] -----> [N, H, W, C]
    feature_maps = paddle.transpose(feature_maps, (0, 2, 3, 1))
    channel = feature_maps.shape[-1]
    # [N, H, W, C] -----> [N, H*W, C]
    feature_maps = paddle.reshape(feature_maps, (batch, -1, channel))
    # expand index in channels
    index = index.unsqueeze(-1).tile((1, 1, channel))
    # select specific features bases on POIs
    feature_maps = gather_op(feature_maps, 1, index)

    return feature_maps