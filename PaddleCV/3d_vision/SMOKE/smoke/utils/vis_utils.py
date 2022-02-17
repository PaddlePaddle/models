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

import cv2
import numpy as np

from smoke.ops import gather_op


def get_ratio(ori_img_size, output_size, down_ratio=(4, 4)):
    return np.array([[down_ratio[1] * ori_img_size[1] / output_size[1], 
                     down_ratio[0] * ori_img_size[0] / output_size[0]]], np.float32)

def get_img(img_path):
    img = cv2.imread(img_path)
    ori_img_size = img.shape
    img = cv2.resize(img, (960, 640))
    output_size = img.shape
    img = img/255.0
    img = np.subtract(img, np.array([0.485, 0.456, 0.406]))
    img = np.true_divide(img, np.array([0.229, 0.224, 0.225]))
    img = np.array(img, np.float32)
    img = img.transpose(2, 0, 1)
    img = img[None,:,:,:]
    img = paddle.to_tensor(img)
    return img, ori_img_size, output_size

def encode_box3d(rotys, dims, locs, K, image_size):
    '''
    construct 3d bounding box for each object.
    Args:
        rotys: rotation in shape N
        dims: dimensions of objects
        locs: locations of objects

    Returns:
        box_3d in camera frame, shape(b, 2, 8)
    '''
    if len(rotys.shape) == 2:
        rotys = rotys.flatten()
    if len(dims.shape) == 3:
        dims = paddle.reshape(dims, (-1, 3))
    if len(locs.shape) == 3:
        locs = paddle.reshape(locs, (-1, 3))

    N = rotys.shape[0]
    ry = rad_to_matrix(rotys, N)

    dims = paddle.reshape(dims, (-1, 1)).tile((1, 8))
    dims[::3, :4], dims[2::3, :4] = 0.5 * dims[::3, :4], 0.5 * dims[2::3, :4]
    dims[::3, 4:], dims[2::3, 4:] = -0.5 * dims[::3, 4:], -0.5 * dims[2::3, 4:]
    dims[1::3, :4], dims[1::3, 4:] = 0., -dims[1::3, 4:]
    index = paddle.to_tensor([[4, 0, 1, 2, 3, 5, 6, 7],
                            [4, 5, 0, 1, 6, 7, 2, 3],
                            [4, 5, 6, 0, 1, 2, 3, 7]]).tile((N, 1))

    box_3d_object = gather_op(dims, 1, index)
    box_3d = paddle.matmul(ry, paddle.reshape(box_3d_object, (N, 3, -1)))
    box_3d += locs.unsqueeze(-1).tile((1, 1, 8))

    box3d_image = paddle.matmul(K, box_3d)
    box3d_image = box3d_image[:, :2, :] / paddle.reshape(box3d_image[:, 2, :], (box_3d.shape[0], 1, box_3d.shape[2]))
    box3d_image = box3d_image.astype("int32")
    box3d_image = box3d_image.astype("float32")

    box3d_image[:, 0] = box3d_image[:, 0].clip(0, image_size[1])
    box3d_image[:, 1] = box3d_image[:, 1].clip(0, image_size[0])

    return box3d_image

def rad_to_matrix(rotys, N):

    cos, sin = rotys.cos(), rotys.sin()

    i_temp = paddle.to_tensor([[1, 0, 1],
                            [0, 1, 0],
                            [-1, 0, 1]]).astype("float32")

    ry = paddle.reshape(i_temp.tile((N, 1)), (N, -1, 3))

    ry[:, 0, 0] *= cos
    ry[:, 0, 2] *= sin
    ry[:, 2, 0] *= sin
    ry[:, 2, 2] *= cos

    return ry


def draw_box_3d(image, corners, color=None):
    ''' Draw 3d bounding box in image
        corners: (8,2) array of vertices for the 3d box in following order:
    '''

    # face_idx = [[0, 1, 5, 4],
    #             [1, 2, 6, 5],
    #             [2, 3, 7, 6],
    #             [3, 0, 4, 7]]
    if color is None:
        color = (0, 0, 255)
    face_idx = [[5, 4, 3, 6],
                [1, 2, 3, 4],
                [1, 0, 7, 2],
                [0, 5, 6, 7]]
    for ind_f in range(3, -1, -1):
        f = face_idx[ind_f]
        for j in range(4):
            cv2.line(image, (corners[f[j], 0], corners[f[j], 1]),
                     (corners[f[(j + 1) % 4], 0], corners[f[(j + 1) % 4], 1]), color, 2, lineType=cv2.LINE_AA)
        if ind_f == 0:
            cv2.line(image, (corners[f[0], 0], corners[f[0], 1]),
                     (corners[f[2], 0], corners[f[2], 1]), color, 1, lineType=cv2.LINE_AA)
            cv2.line(image, (corners[f[1], 0], corners[f[1], 1]),
                     (corners[f[3], 0], corners[f[3], 1]), color, 1, lineType=cv2.LINE_AA)

    return image