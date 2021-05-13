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
from paddle import nn

from smoke.models.layers  import nms_hm, select_topk, select_point_of_interest
from smoke.cvlibs import manager
from smoke.models.heads import SMOKECoder

@manager.POSTPROCESSORS.add_component
class PostProcessor(nn.Layer):
    def __init__(self,
                 depth_ref,
                 dim_ref,
                 reg_head=10,
                 det_threshold=0.25,
                 max_detection=50,
                 pred_2d=True):
        super().__init__()

        self.smoke_coder = SMOKECoder(depth_ref, dim_ref)
        self.reg_head = reg_head
        self.max_detection = max_detection
        self.det_threshold = det_threshold
        self.pred_2d = pred_2d

    def forward(self, predictions, targets):

        pred_heatmap, pred_regression = predictions[0], predictions[1]
        batch = pred_heatmap.shape[0]

        heatmap = nms_hm(pred_heatmap)

        topk_dict = select_topk(
            heatmap,
            K=self.max_detection,
        )
        scores, indexs = topk_dict["topk_score"], topk_dict["topk_inds_all"]
        clses, ys = topk_dict["topk_clses"], topk_dict["topk_ys"]
        xs = topk_dict["topk_xs"]

        pred_regression = select_point_of_interest(
            batch, indexs, pred_regression
        )
        
        pred_regression_pois = paddle.reshape(pred_regression, (-1, self.reg_head))
        
        pred_proj_points = paddle.concat([paddle.reshape(xs, (-1, 1)), paddle.reshape(ys, (-1, 1))], axis=1)
 
        # FIXME: fix hard code here
        pred_depths_offset = pred_regression_pois[:, 0]
        pred_proj_offsets = pred_regression_pois[:, 1:3]
        pred_dimensions_offsets = pred_regression_pois[:, 3:6]
        pred_orientation = pred_regression_pois[:, 6:8]
        pred_bbox_size = pred_regression_pois[:, 8:10]

        pred_depths = self.smoke_coder.decode_depth(pred_depths_offset)
        pred_locations = self.smoke_coder.decode_location(
            pred_proj_points,
            pred_proj_offsets,
            pred_depths,
            targets["K"], 
            targets["trans_mat"])
        pred_dimensions = self.smoke_coder.decode_dimension(
            clses,
            pred_dimensions_offsets
        )
        # we need to change center location to bottom location
        pred_locations[:, 1] += pred_dimensions[:, 1] / 2

        pred_rotys, pred_alphas = self.smoke_coder.decode_orientation(
            pred_orientation,
            pred_locations
        )

        if self.pred_2d:
            box2d = self.smoke_coder.decode_bbox_2d(pred_proj_points, pred_bbox_size, 
                                                    targets["trans_mat"],
                                                    targets["image_size"])
        else:
            box2d = paddle.to_tensor([0, 0, 0, 0])

        # change variables to the same dimension
        clses = paddle.reshape(clses, (-1, 1))
        pred_alphas = paddle.reshape(pred_alphas, (-1, 1))
        pred_rotys = paddle.reshape(pred_rotys, (-1, 1))
        scores = paddle.reshape(scores, (-1, 1))

        l, h, w = pred_dimensions.chunk(3, 1)
        pred_dimensions = paddle.concat([h, w, l], axis=1)

     
        result = paddle.concat([
            clses, pred_alphas, box2d, pred_dimensions, pred_locations, pred_rotys, scores
        ], axis=1)

        keep_idx = result[:, -1] > self.det_threshold

        if paddle.sum(keep_idx.astype("int32")) >= 1:
            keep_idx = paddle.nonzero(result[:, -1] > self.det_threshold)
            result = paddle.gather(result, keep_idx)
        else:
            result = paddle.to_tensor([])


        return result