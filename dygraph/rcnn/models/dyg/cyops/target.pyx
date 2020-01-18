cimport cython
import numpy as np
cimport numpy as np


@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
cpdef rpn_target_assign(
    np.ndarray anchor_box,
    np.ndarray gt_boxes,
    np.ndarray is_crowd,
    np.ndarray im_info,
    float rpn_straddle_thresh,
    int rpn_batch_size_per_im,
    float rpn_positive_overlap,
    float rpn_negative_overlap,
    float rpn_fg_fraction,
    bint use_random=False):

    cdef int anchor_num = anchor_box.shape[0]
    cdef int batch_size = gt_boxes.shape[0]
    cdef int im_height, im_width, i  
    cdef float im_scale 
    cdef np.ndarray inds_inside, inside_anchors, gt_boxes_slice, is_crowd_slice, not_crowd_inds, iou  
    cdef np.ndarray loc_inds, score_inds, labels, gt_inds, bbox_inside_weight
    cdef np.ndarray loc_indexes, score_indexes, tgt_labels, tgt_bboxes, bbox_inside_weights 
    
    for i in range(batch_size):
        im_height = im_info[i][0]
        im_width = im_info[i][1]
        im_scale = im_info[i][2]
        if rpn_straddle_thresh >= 0:
            # Only keep anchors inside the image by a margin of straddle_thresh
            inds_inside = np.where(
                (anchor_box[:, 0] >= -rpn_straddle_thresh) &
                (anchor_box[:, 1] >= -rpn_straddle_thresh) & (
                    anchor_box[:, 2] < im_width + rpn_straddle_thresh) & (
                        anchor_box[:, 3] < im_height + rpn_straddle_thresh))[0]
            # keep only inside anchors
            inside_anchors = anchor_box[inds_inside, :]
        else:
            inds_inside = np.arange(anchor_box.shape[0])
            inside_anchors = anchor_box
        gt_boxes_slice = gt_boxes[i] * im_scale
        is_crowd_slice = is_crowd[i]

        not_crowd_inds = np.where(is_crowd_slice == 0)[0]
        gt_boxes_slice = gt_boxes_slice[not_crowd_inds]
        iou = bbox_overlaps(inside_anchors, gt_boxes_slice)

        loc_inds, score_inds, labels, gt_inds, bbox_inside_weight = sample_anchor_cy(
            iou, 
            rpn_batch_size_per_im,
            rpn_positive_overlap,
            rpn_negative_overlap,
            rpn_fg_fraction,
            use_random
        )
        # unmap to all anchor 
        sampled_loc_inds = inds_inside[loc_inds]
        sampled_score_inds = inds_inside[score_inds]
        
        sampled_anchor = anchor_box[sampled_loc_inds]
        sampled_gt = gt_boxes_slice[gt_inds]
        box_deltas = box_to_delta(
            sampled_anchor, sampled_gt, 
            np.asarray([1., 1., 1., 1.])
        )
        
        if i == 0:
            loc_indexes = sampled_loc_inds
            score_indexes = sampled_score_inds
            tgt_labels = labels
            tgt_bboxes = box_deltas
            bbox_inside_weights = bbox_inside_weight
        else:
            loc_indexes = np.concatenate(
                [loc_indexes, sampled_loc_inds + i * anchor_num])
            score_indexes = np.concatenate(
                [score_indexes, sampled_score_inds + i * anchor_num])
            tgt_labels = np.concatenate([tgt_labels, labels])
            tgt_bboxes = np.vstack([tgt_bboxes, box_deltas])
            bbox_inside_weights = np.vstack([bbox_inside_weights, \
                                             bbox_inside_weight])
    
    return loc_indexes, score_indexes, tgt_labels, tgt_bboxes, bbox_inside_weights 


@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
cdef sample_anchor_cy(
    np.ndarray anchor_by_gt_overlap,
    int rpn_batch_size_per_im,
    float rpn_positive_overlap,
    float rpn_negative_overlap,
    float rpn_fg_fraction,
    bint use_random=False):
    
    cdef np.ndarray anchor_to_gt_argmax = anchor_by_gt_overlap.argmax(axis=1)
    cdef np.ndarray anchor_to_gt_max = anchor_by_gt_overlap[np.arange(
        anchor_by_gt_overlap.shape[0]), anchor_to_gt_argmax]

    cdef np.ndarray gt_to_anchor_argmax = anchor_by_gt_overlap.argmax(axis=0)
    cdef np.ndarray gt_to_anchor_max = anchor_by_gt_overlap[gt_to_anchor_argmax, np.arange(
        anchor_by_gt_overlap.shape[1])]
    cdef np.ndarray anchors_with_max_overlap = np.where(
        anchor_by_gt_overlap == gt_to_anchor_max)[0]

    cdef np.ndarray labels = np.ones((anchor_by_gt_overlap.shape[0], ), dtype=np.int32) * -1
    labels[anchors_with_max_overlap] = 1
    labels[anchor_to_gt_max >= rpn_positive_overlap] = 1
   
    cdef int num_fg = int(rpn_fg_fraction * rpn_batch_size_per_im)
    cdef np.ndarray fg_inds = np.where(labels == 1)[0]
    if len(fg_inds) > num_fg and use_random:
        disable_inds = np.random.choice(
            fg_inds, size=(len(fg_inds) - num_fg), replace=False)
    else:
        disable_inds = fg_inds[num_fg:]

    labels[disable_inds] = -1
    fg_inds = np.where(labels == 1)[0]
    
    cdef int num_bg = rpn_batch_size_per_im - np.sum(labels == 1)
    cdef np.ndarray bg_inds = np.where(anchor_to_gt_max < rpn_negative_overlap)[0]
    if len(bg_inds) > num_bg and use_random:
        enable_inds = bg_inds[np.random.randint(len(bg_inds), size=num_bg)]
    else:
        enable_inds = bg_inds[:num_bg]

    cdef np.ndarray fg_fake_inds = np.array([], np.int32)
    cdef np.ndarray fg_value = np.array([fg_inds[0]], np.int32)
    cdef int fake_num = 0
    for bg_id in enable_inds:
        if bg_id in fg_inds:
            fake_num += 1
            fg_fake_inds = np.hstack([fg_fake_inds, fg_value])
    labels[enable_inds] = 0

    #bbox_inside_weight[fake_num:, :] = 1
    
    fg_inds = np.where(labels == 1)[0]
    bg_inds = np.where(labels == 0)[0]
    
    cdef np.ndarray loc_index = np.hstack([fg_fake_inds, fg_inds])
    cdef np.ndarray score_index = np.hstack([fg_inds, bg_inds])
    labels = labels[score_index]
    #assert not np.any(labels == -1), "Wrong labels with -1"

    cdef np.ndarray gt_inds = anchor_to_gt_argmax[loc_index]
   
    cdef np.ndarray bbox_inside_weight = np.zeros((len(loc_index), 4), dtype=np.float32)
    bbox_inside_weight[fake_num:, :] = 1
    
    return loc_index, score_index, labels, gt_inds, bbox_inside_weight


@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
cpdef generate_proposal_labels(
    np.ndarray rpn_rois, 
    np.ndarray rpn_rois_lod, 
    np.ndarray gt_classes, 
    np.ndarray is_crowd, 
    np.ndarray gt_boxes, 
    np.ndarray im_info,
    int batch_size_per_im,
    float fg_fraction, 
    float fg_thresh, 
    float bg_thresh_hi, 
    float bg_thresh_lo, 
    np.ndarray bbox_reg_weights,
    int class_nums, 
    bint use_random=False, 
    bint is_cls_agnostic=False, 
    bint is_cascade_rcnn=False):
    
    cdef list rois = []
    cdef list labels_int32 = []
    cdef list bbox_targets = []
    cdef list bbox_inside_weights = []
    cdef list bbox_outside_weights = []
    cdef list lod = []
    cdef int batch_size = gt_boxes.shape[0]
    # TODO: modify here
    # rpn_rois = rpn_rois.reshape(batch_size, -1, 4)
    cdef int st_num = 0
    cdef int im_i, rpn_rois_num  
    cdef np.ndarray sampled_rois, sampled_labels, s_bbox_targets, s_bbox_inside_weights, s_bbox_outside_weights

    for im_i, rpn_rois_num in enumerate(rpn_rois_lod):
        #sampled_rois, sampled_labels, bbox_targets, bbox_inside_weights, bbox_outside_weights 
        sampled_rois, sampled_labels, s_bbox_targets, s_bbox_inside_weights, s_bbox_outside_weights = sample_rois_cy(
            rpn_rois[st_num:rpn_rois_num], 
            gt_classes[im_i], is_crowd[im_i], gt_boxes[im_i], im_info[im_i],
            batch_size_per_im, fg_fraction, fg_thresh,
            bg_thresh_hi, bg_thresh_lo, bbox_reg_weights,
            class_nums, use_random, is_cls_agnostic, is_cascade_rcnn)

        st_num = rpn_rois_num
        rois.append(sampled_rois)
        labels_int32.append(sampled_labels)
        bbox_targets.append(s_bbox_targets)
        bbox_inside_weights.append(s_bbox_inside_weights)
        bbox_outside_weights.append(s_bbox_outside_weights)
        lod.append(sampled_rois.shape[0])
    
    cdef np.ndarray o_rois = np.concatenate(rois, axis=0).astype(np.float32) 
    cdef np.ndarray o_labels =  np.concatenate(labels_int32, axis=0).astype(np.int32).reshape(-1, 1) 
    cdef np.ndarray o_bbox_targets = np.concatenate(bbox_targets, axis=0).astype(np.float32)
    cdef np.ndarray o_bbox_inside_weights = np.concatenate(bbox_inside_weights, axis=0).astype(np.float32)
    cdef np.ndarray o_bbox_outside_weights = np.concatenate(bbox_outside_weights, axis=0).astype(np.float32)
    cdef np.ndarray o_lod = np.asarray(lod, np.int32)
    
    return o_rois, o_labels, o_bbox_targets, o_bbox_inside_weights, o_bbox_outside_weights, o_lod 
  

@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
cdef sample_rois_cy(
    np.ndarray rpn_rois, 
    np.ndarray gt_classes, 
    np.ndarray is_crowd, 
    np.ndarray gt_boxes, 
    np.ndarray im_info,
    int batch_size_per_im, 
    float fg_fraction, 
    float fg_thresh, 
    float bg_thresh_hi,
    float bg_thresh_lo, 
    np.ndarray bbox_reg_weights, 
    int class_nums, 
    bint use_random, 
    bint is_cls_agnostic,
    bint is_cascade_rcnn):
    
    cdef int rois_per_image = int(batch_size_per_im)
    cdef int fg_rois_per_im = int(np.round(fg_fraction * rois_per_image))

    # Roidb
    cdef float im_scale = im_info[2]
    cdef float inv_im_scale = 1. / im_scale
    rpn_rois = rpn_rois * inv_im_scale
    if is_cascade_rcnn:
        rpn_rois = rpn_rois[gt_boxes.shape[0]:, :]
    cdef np.ndarray boxes = np.vstack([gt_boxes, rpn_rois])
    cdef np.ndarray gt_overlaps = np.zeros((boxes.shape[0], class_nums))
    cdef np.ndarray box_to_gt_ind_map = np.zeros((boxes.shape[0]), dtype=np.int32)

    cdef np.ndarray proposal_to_gt_overlaps, overlaps_argmax, overlaps_max, overlapped_boxes_ind, overlapped_boxes_gt_classes
    if len(gt_boxes) > 0:
        proposal_to_gt_overlaps = bbox_overlaps(boxes, gt_boxes)
        overlaps_argmax = proposal_to_gt_overlaps.argmax(axis=1)
        overlaps_max = proposal_to_gt_overlaps.max(axis=1)
        # Boxes which with non-zero overlap with gt boxes
        overlapped_boxes_ind = np.where(overlaps_max > 0)[0]
        overlapped_boxes_gt_classes = gt_classes[overlaps_argmax[overlapped_boxes_ind]]
        gt_overlaps[overlapped_boxes_ind, overlapped_boxes_gt_classes] = overlaps_max[overlapped_boxes_ind]
        box_to_gt_ind_map[overlapped_boxes_ind] = overlaps_argmax[overlapped_boxes_ind]

    cdef np.ndarray crowd_ind = np.where(is_crowd)[0]
    gt_overlaps[crowd_ind] = -1

    cdef np.ndarray max_overlaps = gt_overlaps.max(axis=1)
    cdef np.ndarray max_classes = gt_overlaps.argmax(axis=1)
    
    cdef np.ndarray fg_inds, bg_inds 
    cdef int fg_rois_per_this_image, bg_rois_per_this_image 
    # Cascade RCNN Decode Filter
    if is_cascade_rcnn:
        ws = boxes[:, 2] - boxes[:, 0] + 1
        hs = boxes[:, 3] - boxes[:, 1] + 1
        keep = np.where((ws > 0) & (hs > 0))[0]
        boxes = boxes[keep]
        fg_inds = np.where(max_overlaps >= fg_thresh)[0]
        bg_inds = np.where((max_overlaps < bg_thresh_hi) & (max_overlaps >=
                                                            bg_thresh_lo))[0]
        fg_rois_per_this_image = fg_inds.shape[0]
        bg_rois_per_this_image = bg_inds.shape[0]
    else:
        # Foreground
        fg_inds = np.where(max_overlaps >= fg_thresh)[0]
        fg_rois_per_this_image = np.minimum(fg_rois_per_im, fg_inds.shape[0])
        # Sample foreground if there are too many
        if (fg_inds.shape[0] > fg_rois_per_this_image) and use_random:
            fg_inds = np.random.choice(
                fg_inds, size=fg_rois_per_this_image, replace=False)
        fg_inds = fg_inds[:fg_rois_per_this_image]
        # Background
        bg_inds = np.where((max_overlaps < bg_thresh_hi) & (max_overlaps >=bg_thresh_lo))[0]
        bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
        bg_rois_per_this_image = np.minimum(bg_rois_per_this_image, bg_inds.shape[0])
        # Sample background if there are too many
        if (bg_inds.shape[0] > bg_rois_per_this_image) and use_random:
            bg_inds = np.random.choice(bg_inds, size=bg_rois_per_this_image, replace=False)
        bg_inds = bg_inds[:bg_rois_per_this_image]

    cdef np.ndarray keep_inds = np.append(fg_inds, bg_inds)
    cdef np.ndarray sampled_labels = max_classes[keep_inds]
    sampled_labels[fg_rois_per_this_image:] = 0
    cdef np.ndarray sampled_boxes = boxes[keep_inds]
    cdef np.ndarray sampled_gts = gt_boxes[box_to_gt_ind_map[keep_inds]]
    sampled_gts[fg_rois_per_this_image:, :] = gt_boxes[0]

    cdef np.ndarray bbox_label_targets 
    cdef np.ndarray bbox_targets 
    cdef np.ndarray bbox_inside_weights 
    cdef np.ndarray bbox_outside_weights 
    cdef np.ndarray sampled_rois 
    bbox_label_targets = compute_targets(sampled_boxes, sampled_gts, sampled_labels, bbox_reg_weights)
    bbox_targets, bbox_inside_weights = expand_bbox_targets(bbox_label_targets, class_nums, is_cls_agnostic)
    bbox_outside_weights = np.array(bbox_inside_weights > 0, dtype=bbox_inside_weights.dtype)
    # Scale rois
    sampled_rois = sampled_boxes * im_scale
    
    return sampled_rois, sampled_labels, bbox_targets, bbox_inside_weights, bbox_outside_weights 


@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
cdef bbox_overlaps(
    np.ndarray roi_boxes, 
    np.ndarray gt_boxes):
    
    cdef np.ndarray  w1 = np.maximum(roi_boxes[:, 2] - roi_boxes[:, 0] + 1, 0.0)
    cdef np.ndarray  h1 = np.maximum(roi_boxes[:, 3] - roi_boxes[:, 1] + 1, 0.0)
    cdef np.ndarray  w2 = np.maximum(gt_boxes[:, 2] - gt_boxes[:, 0] + 1, 0.0)
    cdef np.ndarray  h2 = np.maximum(gt_boxes[:, 3] - gt_boxes[:, 1] + 1, 0.0)
    cdef np.ndarray area1 = w1 * h1
    cdef np.ndarray area2 = w2 * h2

    cdef np.ndarray overlaps = np.zeros((roi_boxes.shape[0], gt_boxes.shape[0]))
    cdef float inter_x1, inter_y1, inter_x2, inter_y2, inter_w, inter_h, inter_area, iou 

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


@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
cdef box_to_delta(
    np.ndarray ex_boxes, 
    np.ndarray gt_boxes, 
    np.ndarray weights):
   
    cdef np.ndarray ex_w, ex_h, ex_ctr_x, ex_ctr_y, gt_w, gt_h, gt_ctr_x, gt_ctr_y, dx, dy, dw, dh, targets 
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


@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
cdef compute_targets(
    np.ndarray roi_boxes, 
    np.ndarray gt_boxes, 
    np.ndarray labels, 
    np.ndarray bbox_reg_weights):

    cdef np.ndarray targets = np.zeros((roi_boxes.shape[0], roi_boxes.shape[1]), dtype='i4')
    #bbox_reg_weights = np.asarray(bbox_reg_weights)
    targets = box_to_delta(
        ex_boxes=roi_boxes, gt_boxes=gt_boxes, weights=bbox_reg_weights)

    return np.hstack([labels[:, np.newaxis], targets]).astype(np.float32, copy=False)


@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
cdef expand_bbox_targets(
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
