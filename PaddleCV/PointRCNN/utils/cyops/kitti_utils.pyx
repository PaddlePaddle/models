"""
This code is borrow from https://github.com/sshaoshuai/PointRCNN/blob/master/lib/utils/calibration.py
"""
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
    raw_1 = np.concatenate((cosa, -sina), axis=1)
    raw_2 = np.concatenate((sina, cosa), axis=1)
    # # (N, 2, 2)
    R = np.concatenate((np.expand_dims(raw_1, axis=1), np.expand_dims(raw_2, axis=1)), axis=1)
    pc_temp = pc[:, :, [0, 2]]
    pc[:, :, [0, 2]] = np.squeeze(np.matmul(pc_temp, R.transpose(0, 2, 1)), axis=1)
    
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
