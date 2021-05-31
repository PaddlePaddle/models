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

import numpy as np
import paddle

from smoke.ops import gather_op


class SMOKECoder(paddle.nn.Layer):
    """SMOKE Coder class
    """
    def __init__(self, depth_ref, dim_ref):
        super().__init__()
        
        # self.depth_ref = paddle.to_tensor(depth_ref)
        # self.dim_ref = paddle.to_tensor(dim_ref)
        
        self.depth_decoder = DepthDecoder(depth_ref)
        self.dimension_decoder = DimensionDecoder(dim_ref)

    

    @staticmethod
    def rad_to_matrix(rotys, N):
        """decode rotys to R_matrix

        Args:
            rotys (Tensor): roty of objects
            N (int): num of batch

        Returns:
            Tensor: R matrix with shape (N, 3, 3)
            R = [[cos(r), 0, sin(r)], [0, 1, 0], [-cos(r), 0, sin(r)]]
        """
        
        cos, sin = rotys.cos(), rotys.sin()

        i_temp = paddle.to_tensor([[1, 0, 1], [0, 1, 0], [-1, 0, 1]], dtype="float32")

        # ry = paddle.reshape(i_temp.tile([N, 1]), (N, -1, 3))
        
        # ry[:, 0, 0] *= cos
        # ry[:, 0, 2] *= sin
        # ry[:, 2, 0] *= sin
        # ry[:, 2, 2] *= cos

        # slice bug, so use concat
        pos1 = (paddle.ones([N], dtype="float32") * cos).unsqueeze(-1)
        pos2 = (paddle.zeros([N], dtype="float32")).unsqueeze(-1)
        pos3 = (paddle.ones([N], dtype="float32") * sin).unsqueeze(-1)
        pos4 = (paddle.zeros([N], dtype="float32")).unsqueeze(-1)
        pos5 = (paddle.ones([N], dtype="float32")).unsqueeze(-1)
        pos6 = (paddle.zeros([N], dtype="float32")).unsqueeze(-1)
        pos7 = (paddle.ones([N], dtype="float32") * (-sin)).unsqueeze(-1)
        pos8 = (paddle.zeros([N], dtype="float32")).unsqueeze(-1)
        pos9 = (paddle.ones([N], dtype="float32") * cos).unsqueeze(-1)

        ry = paddle.concat([pos1, pos2, pos3, pos4, pos5, pos6, pos7, pos8, pos9], axis=1)
        
        ry = paddle.reshape(ry, [N, 3, 3])
        
        return ry

    def encode_box3d(self, rotys, dims, locs):
        """
        construct 3d bounding box for each object.
        Args:
            rotys: rotation in shape N
            dims: dimensions of objects
            locs: locations of objects

        Returns:

        """
        if len(rotys.shape) == 2:
            rotys = rotys.flatten()
        if len(dims.shape) == 3:
            dims = paddle.reshape(dims, (-1, 3))
        if len(locs.shape) == 3:
            locs = paddle.reshape(locs, (-1, 3))

        N = rotys.shape[0]
        ry = self.rad_to_matrix(rotys, N)
        
        # if test:
        #     dims.register_hook(lambda grad: print('dims grad', grad.sum()))
        # dims = paddle.reshape(dims, (-1, 1)).tile([1, 8])

        # dims[::3, :4] = 0.5 * dims[::3, :4]
        # dims[1::3, :4] = 0.
        # dims[2::3, :4] = 0.5 * dims[2::3, :4]

        # dims[::3, 4:] = -0.5 * dims[::3, 4:]
        # dims[1::3, 4:] = -dims[1::3, 4:]
        # dims[2::3, 4:] = -0.5 * dims[2::3, 4:]

        
        dim_left_1 = (0.5 * dims[:, 0]).unsqueeze(-1)
        dim_left_2 = paddle.zeros([dims.shape[0], 1]).astype("float32") #(paddle.zeros_like(dims[:, 1])).unsqueeze(-1)
        dim_left_3 = (0.5 * dims[:, 2]).unsqueeze(-1)
        dim_left = paddle.concat([dim_left_1, dim_left_2, dim_left_3], axis=1)
        dim_left = paddle.reshape(dim_left, (-1, 1)).tile([1, 4])

        dim_right_1 = (-0.5 * dims[:, 0]).unsqueeze(-1)
        dim_right_2 = (-dims[:, 1]).unsqueeze(-1)
        dim_right_3 = (-0.5 * dims[:, 2]).unsqueeze(-1)
        dim_right = paddle.concat([dim_right_1, dim_right_2, dim_right_3], axis=1)
        dim_right = paddle.reshape(dim_right, (-1, 1)).tile([1, 4])
        

        dims = paddle.concat([dim_left, dim_right], axis=1)
        


        index = paddle.to_tensor([[4, 0, 1, 2, 3, 5, 6, 7],
                                 [4, 5, 0, 1, 6, 7, 2, 3],
                                 [4, 5, 6, 0, 1, 2, 3, 7]]).tile([N, 1])

        box_3d_object = gather_op(dims, 1, index)
        
        box_3d = paddle.matmul(ry, paddle.reshape(box_3d_object, (N, 3, -1)))
        # box_3d += locs.unsqueeze(-1).repeat(1, 1, 8)
        box_3d += locs.unsqueeze(-1).tile((1, 1, 8))

        return box_3d

    def decode_depth(self, depths_offset):
        """
        Transform depth offset to depth
        """
        #depth = depths_offset * self.depth_ref[1] + self.depth_ref[0]

        #return depth
        return self.depth_decoder(depths_offset)

    def decode_location(self,
                        points,
                        points_offset,
                        depths,
                        Ks,
                        trans_mats):
        """
        retrieve objects location in camera coordinate based on projected points
        Args:
            points: projected points on feature map in (x, y)
            points_offset: project points offset in (delata_x, delta_y)
            depths: object depth z
            Ks: camera intrinsic matrix, shape = [N, 3, 3]
            trans_mats: transformation matrix from image to feature map, shape = [N, 3, 3]

        Returns:
            locations: objects location, shape = [N, 3]
        """

        # number of points
        N = points_offset.shape[0]
        # batch size
        N_batch = Ks.shape[0]
        batch_id = paddle.arange(N_batch).unsqueeze(1)
        # obj_id = batch_id.repeat(1, N // N_batch).flatten()
        obj_id = batch_id.tile([1, N // N_batch]).flatten()
        
        # trans_mats_inv = trans_mats.inverse()[obj_id]
        # Ks_inv = Ks.inverse()[obj_id]
        inv = trans_mats.inverse()
        trans_mats_inv = paddle.concat([inv[int(obj_id[i])].unsqueeze(0) for i in range(len(obj_id))])
        
        inv = Ks.inverse()
        Ks_inv = paddle.concat([inv[int(obj_id[i])].unsqueeze(0) for i in range(len(obj_id))])

        points = paddle.reshape(points, (-1, 2))
        assert points.shape[0] == N
        
        # int + float -> int, but float + int -> float
        # proj_points = points + points_offset
        proj_points =  points_offset + points
        
        # transform project points in homogeneous form.
        proj_points_extend = paddle.concat(
            (proj_points.astype("float32"), paddle.ones((N, 1))), axis=1)
        # expand project points as [N, 3, 1]
        proj_points_extend = proj_points_extend.unsqueeze(-1)
        # transform project points back on image
        proj_points_img = paddle.matmul(trans_mats_inv, proj_points_extend)
        # with depth
        proj_points_img = proj_points_img * paddle.reshape(depths, (N, -1, 1))
        # transform image coordinates back to object locations
        locations = paddle.matmul(Ks_inv, proj_points_img)

        return locations.squeeze(2)

    def decode_location_without_transmat(self,
                        points, points_offset,
                        depths, Ks, down_ratios=None):
        """
        retrieve objects location in camera coordinate based on projected points
        Args:
            points: projected points on feature map in (x, y)
            points_offset: project points offset in (delata_x, delta_y)
            depths: object depth z
            Ks: camera intrinsic matrix, shape = [N, 3, 3]
            trans_mats: transformation matrix from image to feature map, shape = [N, 3, 3]

        Returns:
            locations: objects location, shape = [N, 3]
        """

        if down_ratios is None:
            down_ratios = [(1, 1)]


        # number of points
        N = points_offset.shape[0]
        # batch size
        N_batch = Ks.shape[0]
        #batch_id = paddle.arange(N_batch).unsqueeze(1)
        batch_id = paddle.arange(N_batch).reshape((N_batch, 1))
        
        
        # obj_id = batch_id.repeat(1, N // N_batch).flatten()
        obj_id = batch_id.tile([1, N // N_batch]).flatten()
        
        # Ks_inv = Ks[obj_id] pytorch
        
        # Ks_inv = paddle.concat([Ks[int(obj_id[i])].unsqueeze(0) for i in range(len(obj_id))])
        length = int(obj_id.shape[0])
        ks_v = []
        for i in range(length):
            ks_v.append(Ks[int(obj_id[i])].unsqueeze(0))
        Ks_inv = paddle.concat(ks_v)
   
        down_ratio = down_ratios[0]
        points = paddle.reshape(points, (numel_t(points)//2, 2))
        proj_points = points + points_offset
        
        # trans point from heatmap to ori image, down_sample * resize_scale
        proj_points[:, 0] = down_ratio[0] * proj_points[:, 0]
        proj_points[:, 1] = down_ratio[1] * proj_points[:, 1]
        # transform project points in homogeneous form.
     
        
        proj_points_extend = paddle.concat(
            [proj_points, paddle.ones((N, 1))], axis=1)
        # expand project points as [N, 3, 1]
        proj_points_extend = proj_points_extend.unsqueeze(-1)
        # with depth
        proj_points_img = proj_points_extend * paddle.reshape(depths, (N, numel_t(depths)//N, 1))
        # transform image coordinates back to object locations
        locations = paddle.matmul(Ks_inv, proj_points_img)

        return locations.squeeze(2)

    def decode_bbox_2d_without_transmat(self, points, bbox_size, down_ratios=None):
        """get bbox 2d

        Args:
            points (paddle.Tensor, (50, 2)): 2d center
            bbox_size (paddle.Tensor, (50, 2)): 2d bbox height and width
            trans_mats (paddle.Tensor, (1, 3, 3)): transformation coord from img to feature map
        """

        if down_ratios is None:
            down_ratios = [(1, 1)]
        # number of points
        N = bbox_size.shape[0]
        points = paddle.reshape(points, (-1, 2))
        assert points.shape[0] == N

        box2d = paddle.zeros((N, 4))
        down_ratio = down_ratios[0]
        box2d[:, 0] = (points[:, 0] - bbox_size[:, 0] / 2)
        box2d[:, 1] = (points[:, 1] - bbox_size[:, 1] / 2)
        box2d[:, 2] = (points[:, 0] + bbox_size[:, 0] / 2)
        box2d[:, 3] = (points[:, 1] + bbox_size[:, 1] / 2)

        box2d[:, 0] = down_ratio[0] * box2d[:, 0]
        box2d[:, 1] = down_ratio[1] * box2d[:, 1]
        box2d[:, 2] = down_ratio[0] * box2d[:, 2]
        box2d[:, 3] = down_ratio[1] * box2d[:, 3]   

        return box2d

    def decode_dimension(self, cls_id, dims_offset):
        """
        retrieve object dimensions
        Args:
            cls_id: each object id
            dims_offset: dimension offsets, shape = (N, 3)

        Returns:

        """
        # cls_id = cls_id.flatten().long()
        # dims_select = self.dim_ref[cls_id, :]
        # cls_id = cls_id.flatten()
        # dims_select = paddle.concat([self.dim_ref[int(cls_id[i])].unsqueeze(0) for i in range(len(cls_id))])
        # dimensions = dims_offset.exp() * dims_select

        # return dimensions
        return self.dimension_decoder(cls_id, dims_offset)

    def decode_orientation(self, vector_ori, locations, flip_mask=None):
        """
        retrieve object orientation
        Args:
            vector_ori: local orientation in [sin, cos] format
            locations: object location

        Returns: for training we only need roty
                 for testing we need both alpha and roty

        """

        locations = paddle.reshape(locations, (-1, 3))
        rays = paddle.atan(locations[:, 0] / (locations[:, 2] + 1e-7))
        alphas = paddle.atan(vector_ori[:, 0] / (vector_ori[:, 1] + 1e-7))

        # get cosine value positive and negtive index.
        cos_pos_idx = (vector_ori[:, 1] >= 0).nonzero()
        cos_neg_idx = (vector_ori[:, 1] < 0).nonzero()
        
        PI = 3.14159
        for i in range(cos_pos_idx.shape[0]):
            ind = int(cos_pos_idx[i,0])
            alphas[ind] = alphas[ind] - PI / 2
        for i in range(cos_neg_idx.shape[0]):
            ind = int(cos_neg_idx[i,0])
            alphas[ind] = alphas[ind] + PI / 2

        # alphas[cos_pos_idx] -= PI / 2
        # alphas[cos_neg_idx] += PI / 2

        # retrieve object rotation y angle.
        rotys = alphas + rays

        # in training time, it does not matter if angle lies in [-PI, PI]
        # it matters at inference time? todo: does it really matter if it exceeds.
        larger_idx = (rotys > PI).nonzero()
        small_idx = (rotys < -PI).nonzero()

        if len(larger_idx) != 0:
            for i in range(larger_idx.shape[0]):
                ind = int(larger_idx[i,0])
                rotys[ind] -= 2 * PI
        if len(small_idx) != 0:
            for i in range(small_idx.shape[0]):
                ind = int(small_idx[i,0])
                rotys[ind] += 2 * PI

        if flip_mask is not None:
            
            fm = flip_mask.astype("float32").flatten()
            rotys_flip = fm * rotys

            # rotys_flip_pos_idx = rotys_flip > 0
            # rotys_flip_neg_idx = rotys_flip < 0
            # rotys_flip[rotys_flip_pos_idx] -= PI
            # rotys_flip[rotys_flip_neg_idx] += PI
            
            rotys_flip_pos_idx = (rotys_flip > 0).nonzero()
            rotys_flip_neg_idx = (rotys_flip < 0).nonzero()

            for i in range(rotys_flip_pos_idx.shape[0]):
                ind = int(rotys_flip_pos_idx[i, 0])
                rotys_flip[ind] -= PI
            for i in range(rotys_flip_neg_idx.shape[0]):
                ind = int(rotys_flip_neg_idx[i, 0])
                rotys_flip[ind] += PI
            

            rotys_all = fm * rotys_flip + (1 - fm) * rotys

            return rotys_all

        else:
            return rotys, alphas

    def decode_bbox_2d(self, points, bbox_size, trans_mats, img_size):
        """get bbox 2d

        Args:
            points (paddle.Tensor, (50, 2)): 2d center
            bbox_size (paddle.Tensor, (50, 2)): 2d bbox height and width
            trans_mats (paddle.Tensor, (1, 3, 3)): transformation coord from img to feature map
        """
     

        img_size = img_size.flatten()

        # number of points
        N = bbox_size.shape[0]
        # batch size
        N_batch = trans_mats.shape[0]
        batch_id = paddle.arange(N_batch).unsqueeze(1)
        # obj_id = batch_id.repeat(1, N // N_batch).flatten()
        obj_id = batch_id.tile([1, N // N_batch]).flatten()

        inv = trans_mats.inverse()
        trans_mats_inv = paddle.concat([inv[int(obj_id[i])].unsqueeze(0) for i in range(len(obj_id))])

        #trans_mats_inv = trans_mats.inverse()[obj_id]
        points = paddle.reshape(points, (-1, 2))
        assert points.shape[0] == N

        box2d = paddle.zeros([N, 4])
        box2d[:, 0] = (points[:, 0] - bbox_size[:, 0] / 2)
        box2d[:, 1] = (points[:, 1] - bbox_size[:, 1] / 2)
        box2d[:, 2] = (points[:, 0] + bbox_size[:, 0] / 2)
        box2d[:, 3] = (points[:, 1] + bbox_size[:, 1] / 2)      
        # transform project points in homogeneous form.
        proj_points_extend_top = paddle.concat(
            (box2d[:, :2], paddle.ones([N, 1])), axis=1)
        proj_points_extend_bot = paddle.concat(
            (box2d[:, 2:], paddle.ones([N, 1])), axis=1)

        # expand project points as [N, 3, 1]
        proj_points_extend_top = proj_points_extend_top.unsqueeze(-1)
        proj_points_extend_bot = proj_points_extend_bot.unsqueeze(-1)

        # transform project points back on image
        proj_points_img_top = paddle.matmul(trans_mats_inv, proj_points_extend_top)
        proj_points_img_bot = paddle.matmul(trans_mats_inv, proj_points_extend_bot)
        box2d[:, :2] = proj_points_img_top.squeeze(2)[:, :2]
        box2d[:, 2:] = proj_points_img_bot.squeeze(2)[:, :2]

        box2d[:, ::2] = box2d[:, ::2].clip(0, img_size[0])
        box2d[:, 1::2] = box2d[:, 1::2].clip(0, img_size[1])
        return box2d

class DepthDecoder(paddle.nn.Layer):
    def __init__(self, depth_ref):
        super().__init__()
        self.depth_ref = paddle.to_tensor(depth_ref)
    def forward(self, depths_offset):
        """
        Transform depth offset to depth
        """
        depth = depths_offset * self.depth_ref[1] + self.depth_ref[0]

        return depth

class DimensionDecoder(paddle.nn.Layer):
    def __init__(self, dim_ref):
        super().__init__()
        self.dim_ref = paddle.to_tensor(dim_ref)

    def forward(self, cls_id, dims_offset):
        """
        retrieve object dimensions
        Args:
            cls_id: each object id
            dims_offset: dimension offsets, shape = (N, 3)

        Returns:

        """
        # cls_id = cls_id.flatten().long()
        # dims_select = self.dim_ref[cls_id, :]
        cls_id = cls_id.flatten()
        
        #dims_select = paddle.concat([self.dim_ref[int(cls_id[i])].unsqueeze(0) for i in range(len(cls_id))])
        length = int(cls_id.shape[0])
        list_v = []
        for i in range(length):
            list_v.append(self.dim_ref[int(cls_id[i])].unsqueeze(0))
        dims_select = paddle.concat(list_v)
        
        dimensions = dims_offset.exp() * dims_select

        return dimensions

def numel_t(var):
    from numpy import prod
    assert -1 not in var.shape
    return prod(var.shape)