#  Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.
"""
Contains proposal functions
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import paddle.fluid as fluid

import utils.box_utils as box_utils
from utils.config import cfg

__all__ = ["get_proposal_func"]


def get_proposal_func(cfg, mode='TRAIN'):
    def decode_bbox_target(roi_box3d, pred_reg, anchor_size, loc_scope,
                           loc_bin_size, num_head_bin, get_xz_fine=True,
                           loc_y_scope=0.5, loc_y_bin_size=0.25,
                           get_y_by_bin=False, get_ry_fine=False):
        per_loc_bin_num = int(loc_scope / loc_bin_size) * 2
        loc_y_bin_num = int(loc_y_scope / loc_y_bin_size) * 2
        
        # recover xz localization
        x_bin_l, x_bin_r = 0, per_loc_bin_num
        z_bin_l, z_bin_r = per_loc_bin_num, per_loc_bin_num * 2
        start_offset = z_bin_r
        
        x_bin = np.argmax(pred_reg[:, x_bin_l: x_bin_r], axis=1)
        z_bin = np.argmax(pred_reg[:, z_bin_l: z_bin_r], axis=1)
        
        pos_x = x_bin.astype('float32') * loc_bin_size + loc_bin_size / 2 - loc_scope
        pos_z = z_bin.astype('float32') * loc_bin_size + loc_bin_size / 2 - loc_scope
        if get_xz_fine:
            x_res_l, x_res_r = per_loc_bin_num * 2, per_loc_bin_num * 3
            z_res_l, z_res_r = per_loc_bin_num * 3, per_loc_bin_num * 4
            start_offset = z_res_r
            
            x_res_norm = pred_reg[:, x_res_l:x_res_r][np.arange(len(x_bin)), x_bin]
            z_res_norm = pred_reg[:, z_res_l:z_res_r][np.arange(len(z_bin)), z_bin]

            x_res = x_res_norm * loc_bin_size
            z_res = z_res_norm * loc_bin_size
            pos_x += x_res
            pos_z += z_res

        # recover y localization
        if get_y_by_bin:
            y_bin_l, y_bin_r = start_offset, start_offset + loc_y_bin_num
            y_res_l, y_res_r = y_bin_r, y_bin_r + loc_y_bin_num
            start_offset = y_res_r

            y_bin = np.argmax(pred_reg[:, y_bin_l: y_bin_r], axis=1)
            y_res_norm = pred_reg[:, y_res_l:y_res_r][np.arange(len(y_bin)), y_bin]
            y_res = y_res_norm * loc_y_bin_size
            pos_y = y_bin.astype('float32') * loc_y_bin_size + loc_y_bin_size / 2 - loc_y_scope + y_res
            pos_y = pos_y + np.array(roi_box3d[:, 1]).reshape(-1)
        else:
            y_offset_l, y_offset_r = start_offset, start_offset + 1
            start_offset = y_offset_r
            
            pos_y = np.array(roi_box3d[:, 1]) + np.array(pred_reg[:, y_offset_l])
            pos_y = pos_y.reshape(-1)

        # recover ry rotation
        ry_bin_l, ry_bin_r = start_offset, start_offset + num_head_bin
        ry_res_l, ry_res_r = ry_bin_r, ry_bin_r + num_head_bin
        
        ry_bin = np.argmax(pred_reg[:, ry_bin_l: ry_bin_r], axis=1)
        ry_res_norm = pred_reg[:, ry_res_l:ry_res_r][np.arange(len(ry_bin)), ry_bin]
        if get_ry_fine:
            # divide pi/2 into several bins
            angle_per_class = (np.pi / 2) / num_head_bin
            ry_res = ry_res_norm * (angle_per_class / 2)
            ry = (ry_bin.astype('float32') * angle_per_class + angle_per_class / 2) + ry_res - np.pi / 4
        else:
            angle_per_class = (2 * np.pi) / num_head_bin
            ry_res = ry_res_norm * (angle_per_class / 2)
            
            # bin_center is (0, 30, 60, 90, 120, ..., 270, 300, 330)
            ry = np.fmod(ry_bin.astype('float32') * angle_per_class + ry_res, 2 * np.pi)
            ry[ry > np.pi] -= 2 * np.pi
        
        # recover size
        size_res_l, size_res_r = ry_res_r, ry_res_r + 3
        assert size_res_r == pred_reg.shape[1]
        
        size_res_norm = pred_reg[:, size_res_l: size_res_r]
        hwl = size_res_norm * anchor_size + anchor_size
        
        def rotate_pc_along_y(pc, angle):
            cosa = np.cos(angle).reshape(-1, 1)
            sina = np.sin(angle).reshape(-1, 1)
            
            R = np.concatenate([cosa, -sina, sina, cosa], axis=-1).reshape(-1, 2, 2)
            pc_temp = pc[:, [0, 2]].reshape(-1, 1, 2)
            pc[:, [0, 2]] = np.matmul(pc_temp, R.transpose(0, 2, 1)).reshape(-1, 2)
            
            return pc
        
        # shift to original coords
        roi_center = np.array(roi_box3d[:, 0:3])
        shift_ret_box3d = np.concatenate((
            pos_x.reshape(-1, 1),
            pos_y.reshape(-1, 1),
            pos_z.reshape(-1, 1),
            hwl, ry.reshape(-1, 1)), axis=1)
        ret_box3d = shift_ret_box3d
        if roi_box3d.shape[1] == 7:
            roi_ry = np.array(roi_box3d[:, 6]).reshape(-1)
            ret_box3d = rotate_pc_along_y(np.array(shift_ret_box3d), -roi_ry)
            ret_box3d[:, 6] += roi_ry
        ret_box3d[:, [0, 2]] += roi_center[:, [0, 2]]
        return ret_box3d

    def distance_based_proposal(scores, proposals, sorted_idxs):
        nms_range_list = [0, 40.0, 80.0]
        pre_tot_top_n = cfg[mode].RPN_PRE_NMS_TOP_N
        pre_top_n_list = [0, int(pre_tot_top_n * 0.7), pre_tot_top_n - int(pre_tot_top_n * 0.7)]
        post_tot_top_n = cfg[mode].RPN_POST_NMS_TOP_N
        post_top_n_list = [0, int(post_tot_top_n * 0.7), post_tot_top_n - int(post_tot_top_n * 0.7)]

        batch_size = scores.shape[0]
        ret_proposals = np.zeros((batch_size, cfg[mode].RPN_POST_NMS_TOP_N, 7), dtype='float32')
        ret_scores= np.zeros((batch_size, cfg[mode].RPN_POST_NMS_TOP_N, 1), dtype='float32')

        for b, (score, proposal, sorted_idx) in enumerate(zip(scores, proposals, sorted_idxs)):
            # sort by score
            score_ord = score[sorted_idx]
            proposal_ord = proposal[sorted_idx]

            dist = proposal_ord[:, 2]
            first_mask = (dist > nms_range_list[0]) & (dist <= nms_range_list[1])

            scores_single_list, proposals_single_list = [], []
            for i in range(1, len(nms_range_list)):
                # get proposal distance mask
                dist_mask = ((dist > nms_range_list[i - 1]) & (dist <= nms_range_list[i]))

                if dist_mask.sum() != 0:
                    # this area has points, reduce by mask
                    cur_scores = score_ord[dist_mask]
                    cur_proposals = proposal_ord[dist_mask]

                    # fetch pre nms top K
                    cur_scores = cur_scores[:pre_top_n_list[i]]
                    cur_proposals = cur_proposals[:pre_top_n_list[i]]
                else:
                    assert i == 2, '%d' % i
                    # this area doesn't have any points, so use rois of first area
                    cur_scores = score_ord[first_mask]
                    cur_proposals = proposal_ord[first_mask]

                    # fetch top K of first area
                    cur_scores = cur_scores[pre_top_n_list[i - 1]:][:pre_top_n_list[i]]
                    cur_proposals = cur_proposals[pre_top_n_list[i - 1]:][:pre_top_n_list[i]]

                # oriented nms
                boxes_bev = box_utils.boxes3d_to_bev(cur_proposals)
                s_scores, s_proposals = box_utils.box_nms(
                        boxes_bev, cur_scores, cur_proposals,
                        cfg[mode].RPN_NMS_THRESH, post_top_n_list[i],
                        cfg.RPN.NMS_TYPE)
                if len(s_scores) > 0:
                    scores_single_list.append(s_scores)
                    proposals_single_list.append(s_proposals)

            scores_single = np.concatenate(scores_single_list, axis=0)
            proposals_single = np.concatenate(proposals_single_list, axis=0)

            prop_num = proposals_single.shape[0]
            ret_scores[b, :prop_num, 0] = scores_single
            ret_proposals[b, :prop_num] = proposals_single 
        # ret_proposals.tofile("proposal.data")
        # ret_scores.tofile("score.data")
        return np.concatenate([ret_proposals, ret_scores], axis=-1)

    def score_based_proposal(scores, proposals, sorted_idxs):
        batch_size = scores.shape[0]
        ret_proposals = np.zeros((batch_size, cfg[mode].RPN_POST_NMS_TOP_N, 7), dtype='float32')
        ret_scores= np.zeros((batch_size, cfg[mode].RPN_POST_NMS_TOP_N, 1), dtype='float32')
        for b, (score, proposal, sorted_idx) in enumerate(zip(scores, proposals, sorted_idxs)):
            # sort by score
            score_ord = score[sorted_idx]
            proposal_ord = proposal[sorted_idx]

            # pre nms top K
            cur_scores = score_ord[:cfg[mode].RPN_PRE_NMS_TOP_N]
            cur_proposals = proposal_ord[:cfg[mode].RPN_PRE_NMS_TOP_N]

            boxes_bev = box_utils.boxes3d_to_bev(cur_proposals)
            s_scores, s_proposals = box_utils.box_nms(
                    boxes_bev, cur_scores, cur_proposals,
                    cfg[mode].RPN_NMS_THRESH,
                    cfg[mode].RPN_POST_NMS_TOP_N,
                    'rotate')
            prop_num = len(s_proposals)
            ret_scores[b, :prop_num, 0] = s_scores 
            ret_proposals[b, :prop_num] = s_proposals 
        # ret_proposals.tofile("proposal.data")
        # ret_scores.tofile("score.data")
        return np.concatenate([ret_proposals, ret_scores], axis=-1)

    def generate_proposal(x):
        rpn_scores = np.array(x[:, :, 0])[:, :, 0]
        roi_box3d = x[:, :, 1:4]
        pred_reg = x[:, :, 4:]

        proposals = decode_bbox_target(
                np.array(roi_box3d).reshape(-1, roi_box3d.shape()[-1]), 
                np.array(pred_reg).reshape(-1, pred_reg.shape()[-1]), 
                anchor_size=np.array(cfg.CLS_MEAN_SIZE[0], dtype='float32'),
	       	loc_scope=cfg.RPN.LOC_SCOPE,
	       	loc_bin_size=cfg.RPN.LOC_BIN_SIZE,
	       	num_head_bin=cfg.RPN.NUM_HEAD_BIN,
	       	get_xz_fine=cfg.RPN.LOC_XZ_FINE,
	       	get_y_by_bin=False,
	       	get_ry_fine=False)
        proposals[:, 1] += proposals[:, 3] / 2
        proposals = proposals.reshape(rpn_scores.shape[0], -1, proposals.shape[-1])

        sorted_idxs = np.argsort(-rpn_scores, axis=-1)

        if cfg.TEST.RPN_DISTANCE_BASED_PROPOSE:
            ret = distance_based_proposal(rpn_scores, proposals, sorted_idxs)
        else:
            ret = score_based_proposal(rpn_scores, proposals, sorted_idxs)

        return ret


    return generate_proposal 


if __name__ == "__main__":
    np.random.seed(3333)
    x_np = np.random.random((4, 256, 84)).astype('float32')

    from config import cfg
    cfg.RPN.LOC_XZ_FINE = True
    # cfg.TEST.RPN_DISTANCE_BASED_PROPOSE = False
    # cfg.RPN.NMS_TYPE = 'rotate'
    proposal_func = get_proposal_func(cfg)

    x = fluid.data(name="x", shape=[None, 256, 84], dtype='float32')
    proposal = fluid.default_main_program().current_block().create_var(
                    name="proposal", dtype='float32', shape=[256, 7])
    fluid.layers.py_func(proposal_func, x, proposal)
    loss = fluid.layers.reduce_mean(proposal)

    place = fluid.CUDAPlace(0)
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())
    ret = exe.run(fetch_list=[proposal.name, loss.name], feed={'x': x_np})
    print(ret)
