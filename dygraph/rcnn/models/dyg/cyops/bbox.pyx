cimport cython
import numpy as np 
cimport numpy as np 


@cython.boundscheck(False)
def bbox_overlaps(
    np.ndarray roi_boxes, 
    np.ndarray gt_boxes):
    
    cdef np.ndarray  w1 = np.maximum(roi_boxes[:, 2] - roi_boxes[:, 0] + 1, 0.0)
    cdef np.ndarray  h1 = np.maximum(roi_boxes[:, 3] - roi_boxes[:, 1] + 1, 0.0)
    cdef np.ndarray  w2 = np.maximum(gt_boxes[:, 2] - gt_boxes[:, 0] + 1, 0.0)
    cdef np.ndarray  h2 = np.maximum(gt_boxes[:, 3] - gt_boxes[:, 1] + 1, 0.0)
    cdef np.ndarray area1 = w1 * h1
    cdef np.ndarray area2 = w2 * h2

    cdef np.ndarray overlaps = np.zeros((roi_boxes.shape[0], gt_boxes.shape[0]))
    for ind1 in range(roi_boxes.shape[0]):
        for ind2 in range(gt_boxes.shape[0]):
            inter_x1 = np.maximum(roi_boxes[ind1, 0], gt_boxes[ind2, 0])
            inter_y1 = np.maximum(roi_boxes[ind1, 1], gt_boxes[ind2, 1])
            inter_x2 = np.minimum(roi_boxes[ind1, 2], gt_boxes[ind2, 2])
            inter_y2 = np.minimum(roi_boxes[ind1, 3], gt_boxes[ind2, 3])
            inter_w = np.maximum(inter_x2 - inter_x1 + 1, 0)
            inter_h = np.maximum(inter_y2 - inter_y1 + 1, 0)
            inter_area = inter_w * inter_h
            iou = inter_area / (area1[ind1] + area2[ind2] - inter_area)
            overlaps[ind1, ind2] = iou
            #if iou > 0:
            #    print("iou({},{}): {}".format(ind1,ind2,iou))
    return overlaps


@cython.boundscheck(False)
def box_to_delta(
    np.ndarray ex_boxes, 
    np.ndarray gt_boxes, 
    np.ndarray weights):
    
    ex_w = ex_boxes[:, 2] - ex_boxes[:, 0] + 1
    ex_h = ex_boxes[:, 3] - ex_boxes[:, 1] + 1
    ex_ctr_x = ex_boxes[:, 0] + 0.5 * ex_w
    ex_ctr_y = ex_boxes[:, 1] + 0.5 * ex_h

    gt_w = gt_boxes[:, 2] - gt_boxes[:, 0] + 1
    gt_h = gt_boxes[:, 3] - gt_boxes[:, 1] + 1
    gt_ctr_x = gt_boxes[:, 0] + 0.5 * gt_w
    gt_ctr_y = gt_boxes[:, 1] + 0.5 * gt_h

    dx = (gt_ctr_x - ex_ctr_x) / ex_w / weights[0]
    dy = (gt_ctr_y - ex_ctr_y) / ex_h / weights[1]
    dw = (np.log(gt_w / ex_w)) / weights[2]
    dh = (np.log(gt_h / ex_h)) / weights[3]

    targets = np.vstack([dx, dy, dw, dh]).transpose()
    return targets


@cython.boundscheck(False)
def compute_targets(
    np.ndarray roi_boxes, 
    np.ndarray gt_boxes, 
    np.ndarray labels, 
    np.ndarray bbox_reg_weights):

    cdef np.ndarray targets = np.zeros((roi_boxes.shape[0], roi_boxes.shape[1]), dtype='i4')
    #bbox_reg_weights = np.asarray(bbox_reg_weights)
    targets = box_to_delta(
        ex_boxes=roi_boxes, gt_boxes=gt_boxes, weights=bbox_reg_weights)

    return np.hstack([labels[:, np.newaxis], targets]).astype(np.float32, copy=False)


@cython.boundscheck(False)
def expand_bbox_targets(
    np.ndarray bbox_targets_input, 
    int class_nums, 
    bint is_cls_agnostic):

    cdef np.ndarray class_labels = bbox_targets_input[:, 0]
    cdef np.ndarray fg_inds = np.where(class_labels > 0)[0]
    #if is_cls_agnostic:
    #    class_labels = [1 if ll > 0 else 0 for ll in class_labels]
    #    class_labels = np.array(class_labels, dtype=np.int32)
    #    class_nums = 2
    cdef np.ndarray bbox_targets = np.zeros((class_labels.shape[0], 4 * class_nums if not is_cls_agnostic else 4 * 2))
    cdef np.ndarray bbox_inside_weights = np.zeros((bbox_targets.shape[0], bbox_targets.shape[1]))

    cdef int class_label, start_ind, end_ind, ind 
    for ind in fg_inds:
        class_label = int(class_labels[ind]) if not is_cls_agnostic else 1
        start_ind = class_label * 4
        end_ind = class_label * 4 + 4
        bbox_targets[ind, start_ind:end_ind] = bbox_targets_input[ind, 1:]
        bbox_inside_weights[ind, start_ind:end_ind] = (1.0, 1.0, 1.0, 1.0)
    return bbox_targets, bbox_inside_weights

