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

import cython
import numpy as np
cimport numpy as np

@cython.boundscheck(False)
@cython.wraparound(False)
def pts_in_boxes3d(np.ndarray pts_rect, np.ndarray boxes3d):
    """
    :param pts: (N, 3) in rect-camera coords
    :param boxes3d: (M, 7)
    :return: boxes_pts_mask_list: (M), list with [(N), (N), ..]
    """
    cdef float MAX_DIS = 10.0
    cdef np.ndarray boxes_pts_mask_list = np.zeros((boxes3d.shape[0], pts_rect.shape[0]), dtype='int32')
    cdef int boxes3d_num = boxes3d.shape[0]
    cdef int pts_rect_num = pts_rect.shape[0]
    cdef float cx, by, cz, h, w, l, angle, cy, cosa, sina, x_rot, z_rot
    cdef int x, y, z

    for i in range(boxes3d_num):
        cx, by, cz, h, w, l, angle = boxes3d[i, :]
        cy = by - h / 2.
        cosa = np.cos(angle)
        sina = np.sin(angle)
        for j in range(pts_rect_num):
            x, y, z = pts_rect[j, :]

            if np.abs(x - cx) > MAX_DIS or np.abs(y - cy) > h / 2. or np.abs(z - cz) > MAX_DIS:
                continue

            x_rot = (x - cx) * cosa + (z - cz) * (-sina)
            z_rot = (x - cx) * sina + (z - cz) * cosa
            boxes_pts_mask_list[i, j] = int(x_rot >= -l / 2. and x_rot <= l / 2. and
                                            z_rot >= -w / 2. and z_rot <= w / 2.)
    return boxes_pts_mask_list


@cython.boundscheck(False)
@cython.wraparound(False)
def rotate_pc_along_y(np.ndarray pc, float rot_angle):
    """
    params pc: (N, 3+C), (N, 3) is in the rectified camera coordinate
    params rot_angle: rad scalar
    Output pc: updated pc with XYZ rotated
    """
    cosval = np.cos(rot_angle)
    sinval = np.sin(rot_angle)
    rotmat = np.array([[cosval, -sinval], [sinval, cosval]])
    pc[:, [0, 2]] = np.dot(pc[:, [0, 2]], np.transpose(rotmat))
    return pc


@cython.boundscheck(False)
@cython.wraparound(False)
def rotate_pc_along_y_np(np.ndarray pc, np.ndarray rot_angle):
    """
    :param pc: (N, 512, 3 + C)
    :param rot_angle: (N)
    :return:
    TODO: merge with rotate_pc_along_y_torch in bbox_transform.py
    """
    cdef np.ndarray cosa, sina, raw_1, raw_2, R, pc_temp
    cosa = np.cos(rot_angle).reshape(-1, 1)
    sina = np.sin(rot_angle).reshape(-1, 1)
    raw_1 = np.concatenate([cosa, -sina], axis=1)
    raw_2 = np.concatenate([sina, cosa], axis=1)
    # # (N, 2, 2)
    R = np.concatenate((np.expand_dims(raw_1, axis=1), np.expand_dims(raw_2, axis=1)), axis=1)
    pc_temp = pc[:, :, [0, 2]]
    pc[:, :, [0, 2]] = np.matmul(pc_temp, R.transpose(0, 2, 1))
    
    return pc


@cython.boundscheck(False)
@cython.wraparound(False)
def enlarge_box3d(np.ndarray boxes3d, float extra_width):
    """
    :param boxes3d: (N, 7) [x, y, z, h, w, l, ry]
    """
    cdef np.ndarray large_boxes3d
    if isinstance(boxes3d, np.ndarray):
        large_boxes3d = boxes3d.copy()
    else:
        large_boxes3d = boxes3d.clone()
    large_boxes3d[:, 3:6] += extra_width * 2
    large_boxes3d[:, 1] += extra_width

    return large_boxes3d


@cython.boundscheck(False)
@cython.wraparound(False)
def boxes3d_to_corners3d(np.ndarray boxes3d, bint rotate=True):
    """
    :param boxes3d: (N, 7) [x, y, z, h, w, l, ry]
    :param rotate:
    :return: corners3d: (N, 8, 3)
    """
    cdef int boxes_num = boxes3d.shape[0]
    cdef np.ndarray h, w, l
    h, w, l = boxes3d[:, 3], boxes3d[:, 4], boxes3d[:, 5]
    cdef np.ndarray x_corners, y_corners
    x_corners = np.array([l / 2., l / 2., -l / 2., -l / 2., l / 2., l / 2., -l / 2., -l / 2.], dtype=np.float32).T  # (N, 8)
    z_corners = np.array([w / 2., -w / 2., -w / 2., w / 2., w / 2., -w / 2., -w / 2., w / 2.], dtype=np.float32).T  # (N, 8)

    y_corners = np.zeros((boxes_num, 8), dtype=np.float32)
    y_corners[:, 4:8] = -h.reshape(boxes_num, 1).repeat(4, axis=1)  # (N, 8)
    
    cdef np.ndarray ry, zeros, ones, rot_list, R_list, temp_corners, rotated_corners
    if rotate:
        ry = boxes3d[:, 6]
        zeros, ones = np.zeros(ry.size, dtype=np.float32), np.ones(ry.size, dtype=np.float32)
        rot_list = np.array([[np.cos(ry), zeros, -np.sin(ry)],
                             [zeros,       ones,       zeros],
                             [np.sin(ry), zeros,  np.cos(ry)]])  # (3, 3, N)
        R_list = np.transpose(rot_list, (2, 0, 1))  # (N, 3, 3)

        temp_corners = np.concatenate((x_corners.reshape(-1, 8, 1), y_corners.reshape(-1, 8, 1),
                                       z_corners.reshape(-1, 8, 1)), axis=2)  # (N, 8, 3)
        rotated_corners = np.matmul(temp_corners, R_list)  # (N, 8, 3)
        x_corners, y_corners, z_corners = rotated_corners[:, :, 0], rotated_corners[:, :, 1], rotated_corners[:, :, 2]

    cdef np.ndarray x_loc, y_loc, z_loc
    x_loc, y_loc, z_loc = boxes3d[:, 0], boxes3d[:, 1], boxes3d[:, 2]

    cdef np.ndarray x, y, z, corners 
    x = x_loc.reshape(-1, 1) + x_corners.reshape(-1, 8)
    y = y_loc.reshape(-1, 1) + y_corners.reshape(-1, 8)
    z = z_loc.reshape(-1, 1) + z_corners.reshape(-1, 8)

    corners = np.concatenate((x.reshape(-1, 8, 1), y.reshape(-1, 8, 1), z.reshape(-1, 8, 1)), axis=2).astype(np.float32)

    return corners


@cython.boundscheck(False)
@cython.wraparound(False)
def objs_to_boxes3d(obj_list):
    cdef np.ndarray boxes3d = np.zeros((obj_list.__len__(), 7), dtype=np.float32)
    cdef int k 
    for k, obj in enumerate(obj_list):
        boxes3d[k, 0:3], boxes3d[k, 3], boxes3d[k, 4], boxes3d[k, 5], boxes3d[k, 6] \
            = obj.pos, obj.h, obj.w, obj.l, obj.ry
    return boxes3d


@cython.boundscheck(False)
@cython.wraparound(False)
def objs_to_scores(obj_list):
    cdef np.ndarray scores = np.zeros((obj_list.__len__()), dtype=np.float32)
    cdef int k 
    for k, obj in enumerate(obj_list):
        scores[k] = obj.score
    return scores


def get_iou3d(np.ndarray corners3d, np.ndarray query_corners3d, bint need_bev=False):
    """
    :param corners3d: (N, 8, 3) in rect coords
    :param query_corners3d: (M, 8, 3)
    :return:
    """
    from shapely.geometry import Polygon
    A, B = corners3d, query_corners3d
    N, M = A.shape[0], B.shape[0]
    iou3d = np.zeros((N, M), dtype=np.float32)
    iou_bev = np.zeros((N, M), dtype=np.float32)

    # for height overlap, since y face down, use the negative y
    min_h_a = -A[:, 0:4, 1].sum(axis=1) / 4.0
    max_h_a = -A[:, 4:8, 1].sum(axis=1) / 4.0
    min_h_b = -B[:, 0:4, 1].sum(axis=1) / 4.0
    max_h_b = -B[:, 4:8, 1].sum(axis=1) / 4.0

    for i in range(N):
        for j in range(M):
            max_of_min = np.max([min_h_a[i], min_h_b[j]])
            min_of_max = np.min([max_h_a[i], max_h_b[j]])
            h_overlap = np.max([0, min_of_max - max_of_min])
            if h_overlap == 0:
                continue

            bottom_a, bottom_b = Polygon(A[i, 0:4, [0, 2]].T), Polygon(B[j, 0:4, [0, 2]].T)
            if bottom_a.is_valid and bottom_b.is_valid:
                # check is valid,  A valid Polygon may not possess any overlapping exterior or interior rings.
                bottom_overlap = bottom_a.intersection(bottom_b).area
            else:
                bottom_overlap = 0.
            overlap3d = bottom_overlap * h_overlap
            union3d = bottom_a.area * (max_h_a[i] - min_h_a[i]) + bottom_b.area * (max_h_b[j] - min_h_b[j]) - overlap3d
            iou3d[i][j] = overlap3d / union3d
            iou_bev[i][j] = bottom_overlap / (bottom_a.area + bottom_b.area - bottom_overlap)

    if need_bev:
        return iou3d, iou_bev

    return iou3d


def get_objects_from_label(label_file):
    import utils.object3d as object3d

    with open(label_file, 'r') as f:
        lines = f.readlines()
    objects = [object3d.Object3d(line) for line in lines]
    return objects


@cython.boundscheck(False)
@cython.wraparound(False)
def _rotate_pc_along_y(np.ndarray pc, np.ndarray angle):
    cdef np.ndarray cosa = np.cos(angle)
    cosa=cosa.reshape(-1, 1)
    cdef np.ndarray sina = np.sin(angle)
    sina = sina.reshape(-1, 1)

    cdef np.ndarray R = np.concatenate([cosa, -sina, sina, cosa], axis=-1)
    R = R.reshape(-1, 2, 2)
    cdef np.ndarray pc_temp = pc[:, [0, 2]]
    pc_temp = pc_temp.reshape(-1, 1, 2)
    cdef np.ndarray pc_temp_1 = np.matmul(pc_temp, R.transpose(0, 2, 1))
    pc_temp_1 = pc_temp_1.reshape(-1, 2)
    pc[:,[0,2]] = pc_temp_1 

    return pc

@cython.boundscheck(False)
@cython.wraparound(False)
def decode_bbox_target(
    np.ndarray roi_box3d, 
    np.ndarray pred_reg, 
    np.ndarray anchor_size, 
    float loc_scope,
    float loc_bin_size, 
    int num_head_bin, 
    bint get_xz_fine=True,
    float loc_y_scope=0.5, 
    float loc_y_bin_size=0.25,
    bint get_y_by_bin=False, 
    bint get_ry_fine=False):
    
    cdef int per_loc_bin_num = int(loc_scope / loc_bin_size) * 2
    cdef int loc_y_bin_num = int(loc_y_scope / loc_y_bin_size) * 2

    # recover xz localization
    cdef int x_bin_l = 0
    cdef int x_bin_r = per_loc_bin_num
    cdef int z_bin_l = per_loc_bin_num, 
    cdef int z_bin_r = per_loc_bin_num * 2
    cdef int start_offset = z_bin_r
    cdef np.ndarray x_bin = np.argmax(pred_reg[:, x_bin_l: x_bin_r], axis=1)
    cdef np.ndarray z_bin = np.argmax(pred_reg[:, z_bin_l: z_bin_r], axis=1)

    cdef np.ndarray pos_x = x_bin.astype('float32') * loc_bin_size + loc_bin_size / 2 - loc_scope
    cdef np.ndarray pos_z = z_bin.astype('float32') * loc_bin_size + loc_bin_size / 2 - loc_scope

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
    cdef int  ry_bin_l = start_offset, 
    cdef int ry_bin_r = start_offset + num_head_bin
    cdef int ry_res_l = ry_bin_r, 
    cdef int ry_res_r = ry_bin_r + num_head_bin

    cdef np.ndarray ry_bin = np.argmax(pred_reg[:, ry_bin_l: ry_bin_r], axis=1)
    cdef np.ndarray ry_res_norm = pred_reg[:, ry_res_l:ry_res_r][np.arange(len(ry_bin)), ry_bin]
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
    cdef int size_res_l = ry_res_r 
    cdef int size_res_r = ry_res_r + 3
    assert size_res_r == pred_reg.shape[1]

    cdef np.ndarray size_res_norm = pred_reg[:, size_res_l: size_res_r]
    cdef np.ndarray hwl = size_res_norm * anchor_size + anchor_size

    # shift to original coords
    cdef np.ndarray roi_center = np.array(roi_box3d[:, 0:3])
    cdef np.ndarray shift_ret_box3d = np.concatenate((
        pos_x.reshape(-1, 1),
        pos_y.reshape(-1, 1),
        pos_z.reshape(-1, 1),
        hwl, ry.reshape(-1, 1)), axis=1)
    ret_box3d = shift_ret_box3d
    if roi_box3d.shape[1] == 7:
        roi_ry = np.array(roi_box3d[:, 6]).reshape(-1)
        ret_box3d = _rotate_pc_along_y(np.array(shift_ret_box3d), -roi_ry)
        ret_box3d[:, 6] += roi_ry
    ret_box3d[:, [0, 2]] += roi_center[:, [0, 2]]
    
    return ret_box3d
