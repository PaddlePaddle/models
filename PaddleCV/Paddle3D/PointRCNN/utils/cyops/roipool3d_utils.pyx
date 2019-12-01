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
 
import numpy as np 
cimport numpy as np 
cimport cython 
from libc.math cimport sin, cos 

@cython.boundscheck(False)
@cython.wraparound(False)
cdef enlarge_box3d(np.ndarray boxes3d, int extra_width):
    """
    :param boxes3d: (N, 7) [x, y, z, h, w, l, ry]
    """
    if isinstance(boxes3d, np.ndarray):
        large_boxes3d = boxes3d.copy()
    else:
        large_boxes3d = boxes3d.clone()
    large_boxes3d[:, 3:6] += extra_width * 2
    large_boxes3d[:, 1] += extra_width
    return large_boxes3d

@cython.boundscheck(False)
@cython.wraparound(False)
cdef pt_in_box(float x, float y, float z, float cx, float bottom_y, float cz, float h, float w, float l, float angle):
    cdef float max_ids = 10.0
    cdef float cy = bottom_y - h / 2.0
    if ((abs(x - cx) > max_ids) or (abs(y - cy) > h / 2.0) or (abs(z - cz) > max_ids)):
        return 0
    cdef float cosa = cos(angle)
    cdef float sina = sin(angle)
    cdef float x_rot = (x - cx) * cosa + (z - cz) * (-sina)

    cdef float z_rot = (x - cx) * sina + (z - cz) * cosa

    cdef float flag = (x_rot >= -l / 2.0) and (x_rot <= l / 2.0) and (z_rot >= -w / 2.0) and (z_rot <= w / 2.0)
    return flag

@cython.boundscheck(False)
@cython.wraparound(False)
cdef _rotate_pc_along_y(np.ndarray pc, float rot_angle):
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
def roipool3d_cpu(
        np.ndarray[float, ndim=2] pts, 
        np.ndarray[float, ndim=2] pts_feature, 
        np.ndarray[float, ndim=2] boxes3d, 
        np.ndarray[float, ndim=2] pts_extra_input, 
        int pool_extra_width, int sampled_pt_num, int batch_size=1, bint canonical_transform=False):
    cdef np.ndarray pts_feature_all = np.concatenate((pts_extra_input, pts_feature), axis=1)

    cdef np.ndarray larged_boxes3d = enlarge_box3d(boxes3d.reshape(-1, 7), pool_extra_width).reshape(batch_size, -1, 7)

    cdef int pts_num  = pts.shape[0], 
    cdef int boxes_num = boxes3d.shape[0] 
    cdef int feature_len = pts_feature_all.shape[1]
    cdef np.ndarray pts_data = np.zeros((batch_size, boxes_num, sampled_pt_num, 3))
    cdef np.ndarray features_data = np.zeros((batch_size, boxes_num, sampled_pt_num, feature_len))
    cdef np.ndarray empty_flag_data = np.zeros((batch_size, boxes_num))
    
    cdef int cnt = 0
    cdef float cx = 0.
    cdef float bottom_y = 0.
    cdef float cz = 0.
    cdef float h = 0.
    cdef float w = 0.
    cdef float l = 0.
    cdef float ry = 0.
    cdef float x = 0. 
    cdef float y = 0.
    cdef float z = 0.
    cdef np.ndarray x_i
    cdef np.ndarray feat_i 
    cdef int bs
    cdef int i 
    cdef int j 
    for bs in range(batch_size):
        # boxes: 64,7
        for i in range(boxes_num):
            cnt = 0
            # box
            box = larged_boxes3d[bs][i]
            cx = box[0]
            bottom_y = box[1]
            cz = box[2]
            h = box[3]
            w = box[4]
            l = box[5]
            ry = box[6]
            # points: 16384,3  
            x_i = pts
            # features: 16384, 128 
            feat_i = pts_feature_all
            
            for j in range(pts_num):
                x = x_i[j][0]
                y = x_i[j][1]
                z = x_i[j][2]
                cur_in_flag = pt_in_box(x,y,z,cx,bottom_y,cz,h,w,l,ry)
                if cur_in_flag:
                    if cnt < sampled_pt_num:
                        pts_data[bs][i][cnt][:] = x_i[j]
                        features_data[bs][i][cnt][:] = feat_i[j]
                        cnt += 1
                    else:
                        break 

            if cnt == 0:
                empty_flag_data[bs][i] = 1
            elif (cnt < sampled_pt_num):
                for k in range(cnt, sampled_pt_num):
                    pts_data[bs][i][k] = pts_data[bs][i][k % cnt]
                    features_data[bs][i][k] = features_data[bs][i][k % cnt]


    pooled_pts = pts_data.astype("float32")[0]
    pooled_features = features_data.astype('float32')[0]
    pooled_empty_flag = empty_flag_data.astype('int64')[0]

    cdef int extra_input_len = pts_extra_input.shape[1]
    pooled_pts = np.concatenate((pooled_pts, pooled_features[:,:,0:extra_input_len]),axis=2)
    pooled_features = pooled_features[:,:,extra_input_len:]
    
    if canonical_transform:
        # Translate to the roi coordinates
        roi_ry = boxes3d[:, 6] % (2 * np.pi)  # 0~2pi
        roi_center = boxes3d[:, 0:3]
        # shift to center
        pooled_pts[:, :, 0:3] = pooled_pts[:, :, 0:3] - roi_center[:, np.newaxis, :]
        for k in range(pooled_pts.shape[0]):
            pooled_pts[k] = _rotate_pc_along_y(pooled_pts[k], roi_ry[k])
        return pooled_pts, pooled_features, pooled_empty_flag

    return pooled_pts, pooled_features, pooled_empty_flag


#def roipool3d_cpu(pts, pts_feature, boxes3d, pts_extra_input, pool_extra_width, sampled_pt_num=512, batch_size=1):
#    return _roipool3d_cpu(pts, pts_feature, boxes3d, pts_extra_input, pool_extra_width, sampled_pt_num, batch_size)
