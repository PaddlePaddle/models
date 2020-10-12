"""
This code is based on https://github.com/garrickbrazil/M3D-RPN/blob/master/lib/rpn_util.py
This file is meant to contain functions which are
specific to region proposal networks.
"""

#import matplotlib.pyplot as plt
import subprocess
#import torch
import math
import os
import re
import gc
from lib.util import *
from lib.core import *
from data.augmentations import *
from lib.nms.gpu_nms import gpu_nms

from copy import deepcopy
import numpy as np
import pdb
import cv2
import paddle
from paddle.fluid import framework

import paddle.fluid as fluid
from paddle.fluid.dygraph import to_variable


def generate_anchors(conf, imdb, cache_folder):
    """
    Generates the anchors according to the configuration and
    (optionally) based on the imdb properties.
    """
    #
    # use cache?
    if (cache_folder is not None
        ) and os.path.exists(os.path.join(cache_folder, 'anchors.pkl')):

        anchors = pickle_read(os.path.join(cache_folder, 'anchors.pkl'))
#
# generate anchors
    else:
        #
        anchors = np.zeros(
            [len(conf.anchor_scales) * len(conf.anchor_ratios), 4],
            dtype=np.float32)
        #
        aind = 0
        #
        # compute simple anchors based on scale/ratios
        for scale in conf.anchor_scales:
            #
            for ratio in conf.anchor_ratios:
                #
                h = scale
                w = scale * ratio
                #
                anchors[aind, 0:4] = anchor_center(w, h, conf.feat_stride)
                aind += 1
#
#
# optionally cluster anchors
        if conf.cluster_anchors:
            anchors = cluster_anchors(
                conf.feat_stride, anchors, conf.test_scale, imdb, conf.lbls,
                conf.ilbls, conf.anchor_ratios, conf.min_gt_vis, conf.min_gt_h,
                conf.max_gt_h, conf.even_anchors, conf.expand_anchors)
#
#
# has 3d? then need to compute stats for each new dimension
# presuming that anchors are initialized in "2d"
        elif conf.has_3d:
            #
            # compute the default stats for each anchor
            normalized_gts = []
            #
            # check all images
            for imind, imobj in enumerate(imdb):
                #
                # has ground truths?
                if len(imobj.gts) > 0:
                    #
                    scale = imobj.scale * conf.test_scale / imobj.imH
                    #
                    # determine ignores
                    igns, rmvs = determine_ignores(imobj.gts, conf.lbls,
                                                   conf.ilbls, conf.min_gt_vis,
                                                   conf.min_gt_h, np.inf, scale)
                    #
                    # accumulate boxes
                    gts_all = bbXYWH2Coords(
                        np.array([gt.bbox_full * scale for gt in imobj.gts]))
                    gts_val = gts_all[(rmvs == False) & (igns == False), :]
                    #
                    gts_3d = np.array([gt.bbox_3d for gt in imobj.gts])
                    gts_3d = gts_3d[(rmvs == False) & (igns == False), :]
                    #
                    if gts_val.shape[0] > 0:
                        #
                        # center all 2D ground truths
                        for gtind in range(0, gts_val.shape[0]):
                            w = gts_val[gtind, 2] - gts_val[gtind, 0] + 1
                            h = gts_val[gtind, 3] - gts_val[gtind, 1] + 1
                            #
                            gts_val[gtind, 0:4] = anchor_center(
                                w, h, conf.feat_stride)
#
                    if gts_val.shape[0] > 0:
                        normalized_gts += np.concatenate(
                            (gts_val, gts_3d), axis=1).tolist()
#
# convert to np
            normalized_gts = np.array(normalized_gts)
            #
            # expand dimensions
            anchors = np.concatenate(
                (anchors, np.zeros([anchors.shape[0], 5])), axis=1)
            #
            # bbox_3d order --> [cx3d, cy3d, cz3d, w3d, h3d, l3d, rotY]
            anchors_z3d = [[] for x in range(anchors.shape[0])]
            anchors_w3d = [[] for x in range(anchors.shape[0])]
            anchors_h3d = [[] for x in range(anchors.shape[0])]
            anchors_l3d = [[] for x in range(anchors.shape[0])]
            anchors_rotY = [[] for x in range(anchors.shape[0])]
            #
            # find best matches for each ground truth
            ols = iou(anchors[:, 0:4], normalized_gts[:, 0:4])
            gt_target_ols = np.amax(ols, axis=0)
            gt_target_anchor = np.argmax(ols, axis=0)
            #
            # assign each box to an anchor
            for gtind, gt in enumerate(normalized_gts):
                #
                anum = gt_target_anchor[gtind]
                #
                if gt_target_ols[gtind] > 0.2:
                    anchors_z3d[anum].append(gt[6])
                    anchors_w3d[anum].append(gt[7])
                    anchors_h3d[anum].append(gt[8])
                    anchors_l3d[anum].append(gt[9])
                    anchors_rotY[anum].append(gt[10])
#
# compute global means
            anchors_z3d_gl = np.empty(0)
            anchors_w3d_gl = np.empty(0)
            anchors_h3d_gl = np.empty(0)
            anchors_l3d_gl = np.empty(0)
            anchors_rotY_gl = np.empty(0)
            #
            # update anchors
            for aind in range(0, anchors.shape[0]):
                #
                if len(np.array(anchors_z3d[aind])) > 0:
                    #
                    if conf.has_3d:
                        #
                        anchors_z3d_gl = np.hstack(
                            (anchors_z3d_gl, np.array(anchors_z3d[aind])))
                        anchors_w3d_gl = np.hstack(
                            (anchors_w3d_gl, np.array(anchors_w3d[aind])))
                        anchors_h3d_gl = np.hstack(
                            (anchors_h3d_gl, np.array(anchors_h3d[aind])))
                        anchors_l3d_gl = np.hstack(
                            (anchors_l3d_gl, np.array(anchors_l3d[aind])))
                        anchors_rotY_gl = np.hstack(
                            (anchors_rotY_gl, np.array(anchors_rotY[aind])))
                        #
                        anchors[aind, 4] = np.mean(np.array(anchors_z3d[aind]))
                        anchors[aind, 5] = np.mean(np.array(anchors_w3d[aind]))
                        anchors[aind, 6] = np.mean(np.array(anchors_h3d[aind]))
                        anchors[aind, 7] = np.mean(np.array(anchors_l3d[aind]))
                        anchors[aind, 8] = np.mean(np.array(anchors_rotY[aind]))
#
                else:
                    raise ValueError('Non-used anchor #{} found'.format(aind))
#
        if (cache_folder is not None):
            pickle_write(os.path.join(cache_folder, 'anchors.pkl'), anchors)

#
    conf.anchors = anchors


def anchor_center(w, h, stride):
    """
    Centers an anchor based on a stride and the anchor shape (w, h).

    center ground truths with steps of half stride
    hence box 0 is centered at (7.5, 7.5) rather than (0, 0)
    for a feature stride of 16 px.
    """

    anchor = np.zeros([4], dtype=np.float32)
    #
    anchor[0] = -w / 2 + (stride - 1) / 2
    anchor[1] = -h / 2 + (stride - 1) / 2
    anchor[2] = w / 2 + (stride - 1) / 2
    anchor[3] = h / 2 + (stride - 1) / 2
    #
    return anchor


def cluster_anchors(feat_stride,
                    anchors,
                    test_scale,
                    imdb,
                    lbls,
                    ilbls,
                    anchor_ratios,
                    min_gt_vis=0.99,
                    min_gt_h=0,
                    max_gt_h=10e10,
                    even_anchor_distribution=False,
                    expand_anchors=False,
                    expand_stop_dt=0.0025):
    """
    Clusters the anchors based on the imdb boxes (in 2D and/or 3D).
#
    Generally, this method does a custom k-means clustering using 2D IoU
    as a distance metric.
    """

    normalized_gts = []

    # keep track if using 3d
    has_3d = False

    # check all images
    for imind, imobj in enumerate(imdb):

        # has ground truths?
        if len(imobj.gts) > 0:

            scale = imobj.scale * test_scale / imobj.imH

            # determine ignores
            igns, rmvs = determine_ignores(imobj.gts, lbls, ilbls, min_gt_vis,
                                           min_gt_h, np.inf, scale)

            # check for 3d box
            has_3d = 'bbox_3d' in imobj.gts[0]

            # accumulate boxes
            gts_all = bbXYWH2Coords(
                np.array([gt.bbox_full * scale for gt in imobj.gts]))
            gts_val = gts_all[(rmvs == False) & (igns == False), :]

            if has_3d:
                gts_3d = np.array([gt.bbox_3d for gt in imobj.gts])
                gts_3d = gts_3d[(rmvs == False) & (igns == False), :]

            if gts_val.shape[0] > 0:

                # center all 2D ground truths
                for gtind in range(0, gts_val.shape[0]):

                    w = gts_val[gtind, 2] - gts_val[gtind, 0] + 1
                    h = gts_val[gtind, 3] - gts_val[gtind, 1] + 1

                    gts_val[gtind, 0:4] = anchor_center(w, h, feat_stride)

            if gts_val.shape[0] > 0:

                # add normalized gts given 3d or 2d boxes
                if has_3d:
                    normalized_gts += np.concatenate(
                        (gts_val, gts_3d), axis=1).tolist()
                else:
                    normalized_gts += gts_val.tolist()

    # convert to np
    normalized_gts = np.array(normalized_gts)

    # sort by height
    sorted_inds = np.argsort((normalized_gts[:, 3] - normalized_gts[:, 1] + 1))
    normalized_gts = normalized_gts[sorted_inds, :]

    min_h = normalized_gts[0, 3] - normalized_gts[0, 1] + 1
    max_h = normalized_gts[-1, 3] - normalized_gts[-1, 1] + 1

    # for 3d, expand dimensions
    if has_3d:
        anchors = np.concatenate(
            (anchors, np.zeros([anchors.shape[0], 5])), axis=1)

    # init expand
    best_anchors = anchors
    expand_last_iou = 0
    expand_dif = 1
    best_iou = 0
    best_cov = 0

    while np.round(expand_dif, 5) > expand_stop_dt:

        # init cluster
        max_rounds = 1000
        round = 0
        last_iou = 0
        dif = 1

        if even_anchor_distribution:

            sample_num = int(
                np.floor(normalized_gts.shape[0] / anchors.shape[0]))

            # evenly distribute the anchors
            for aind in range(0, anchors.shape[0]):

                x1 = normalized_gts[aind * sample_num:(aind * sample_num +
                                                       sample_num), 0]
                y1 = normalized_gts[aind * sample_num:(aind * sample_num +
                                                       sample_num), 1]
                x2 = normalized_gts[aind * sample_num:(aind * sample_num +
                                                       sample_num), 2]
                y2 = normalized_gts[aind * sample_num:(aind * sample_num +
                                                       sample_num), 3]

                w = np.mean(x2 - x1 + 1)
                h = np.mean(y2 - y1 + 1)

                anchors[aind, 0:4] = anchor_center(w, h, feat_stride)

        else:

            base = ((max_gt_h) / (min_gt_h))**(1 / (anchors.shape[0] - 1))
            anchor_scales = np.array(
                [(min_gt_h) * (base**i) for i in range(0, anchors.shape[0])])

            aind = 0

            # compute new anchors
            for scale in anchor_scales:

                for ratio in anchor_ratios:

                    h = scale
                    w = scale * ratio

                    anchors[aind, 0:4] = anchor_center(w, h, feat_stride)

                    aind += 1

        while round < max_rounds and dif > -0.0:

            # make empty arrays for each anchor
            anchors_h = [[] for x in range(anchors.shape[0])]
            anchors_w = [[] for x in range(anchors.shape[0])]

            if has_3d:

                # bbox_3d order --> [cx3d, cy3d, cz3d, w3d, h3d, l3d, rotY]
                anchors_z3d = [[] for x in range(anchors.shape[0])]
                anchors_w3d = [[] for x in range(anchors.shape[0])]
                anchors_h3d = [[] for x in range(anchors.shape[0])]
                anchors_l3d = [[] for x in range(anchors.shape[0])]
                anchors_rotY = [[] for x in range(anchors.shape[0])]

            round_ious = []

            # find best matches for each ground truth
            ols = iou(anchors[:, 0:4], normalized_gts[:, 0:4])
            gt_target_ols = np.amax(ols, axis=0)
            gt_target_anchor = np.argmax(ols, axis=0)

            # assign each box to an anchor
            for gtind, gt in enumerate(normalized_gts):

                anum = gt_target_anchor[gtind]

                w = gt[2] - gt[0] + 1
                h = gt[3] - gt[1] + 1

                anchors_h[anum].append(h)
                anchors_w[anum].append(w)

                if has_3d:
                    anchors_z3d[anum].append(gt[6])
                    anchors_w3d[anum].append(gt[7])
                    anchors_h3d[anum].append(gt[8])
                    anchors_l3d[anum].append(gt[9])
                    anchors_rotY[anum].append(gt[10])

                round_ious.append(gt_target_ols[gtind])

            # compute current iou
            cur_iou = np.mean(np.array(round_ious))

            # update anchors
            for aind in range(0, anchors.shape[0]):

                # compute mean h/w
                if len(np.array(anchors_h[aind])) > 0:

                    mean_h = np.mean(np.array(anchors_h[aind]))
                    mean_w = np.mean(np.array(anchors_w[aind]))

                    anchors[aind, 0:4] = anchor_center(mean_w, mean_h,
                                                       feat_stride)

                    if has_3d:
                        anchors[aind, 4] = np.mean(np.array(anchors_z3d[aind]))
                        anchors[aind, 5] = np.mean(np.array(anchors_w3d[aind]))
                        anchors[aind, 6] = np.mean(np.array(anchors_h3d[aind]))
                        anchors[aind, 7] = np.mean(np.array(anchors_l3d[aind]))
                        anchors[aind, 8] = np.mean(np.array(anchors_rotY[aind]))

                else:

                    # anchor not used
                    anchors[aind, :] = 0

            anchors = np.nan_to_num(anchors)
            valid_anchors = np.invert(np.all(anchors == 0, axis=1))

            # redistribute non-valid anchors
            valid_anchors_inds = np.flatnonzero(valid_anchors)

            # determine most heavy anchors (to be split up)
            valid_multi = np.array([len(x) for x in anchors_h])
            valid_multi = valid_multi[valid_anchors_inds]
            valid_multi = valid_multi / np.sum(valid_multi)

            # store best configuration
            if cur_iou > best_iou:
                best_iou = cur_iou
                best_anchors = anchors[valid_anchors, :]
                best_cov = np.mean(np.array(round_ious) > 0.5)

            # add random new anchors for any not used
            for aind in range(0, anchors.shape[0]):

                # make a new anchor
                if not valid_anchors[aind]:
                    randomness = 0.5
                    multi = randomness * np.random.rand(len(valid_anchors_inds))
                    multi += valid_multi
                    multi /= np.sum(multi)
                    anchors[aind, :] = np.dot(anchors[valid_anchors_inds, :].T,
                                              multi.T)

            if not all(valid_anchors):
                logging.info(
                    'warning: round {} some anchors not used during clustering'.
                    format(round))

            dif = cur_iou - last_iou
            last_iou = cur_iou

            round += 1

        logging.info(
            'anchors={}, rounds={}, mean_iou={:.4f}, gt_coverage={:.4f}'.format(
                anchors.shape[0], round, best_iou, best_cov))

        expand_dif = best_iou - expand_last_iou
        expand_last_iou = best_iou

        # expand anchors to next size
        if anchors.shape[0] < expand_anchors and expand_dif > expand_stop_dt:

            # append blank anchor
            if has_3d:
                anchors = np.vstack((anchors, [0, 0, 0, 0, 0, 0, 0, 0, 0]))
            else:
                anchors = np.vstack((anchors, [0, 0, 0, 0]))

        # force stop
        else:
            expand_dif = -1

    logging.info('final_iou={:.4f}, final_coverage={:.4f}'.format(best_iou,
                                                                  best_cov))

    return best_anchors


def compute_targets(gts_val,
                    gts_ign,
                    box_lbls,
                    rois,
                    fg_thresh,
                    ign_thresh,
                    bg_thresh_lo,
                    bg_thresh_hi,
                    best_thresh,
                    gts_3d=None,
                    anchors=[],
                    tracker=[]):
    """
     Computes the bbox targets of a set of rois and a set
     of ground truth boxes, provided various ignore
     settings in configuration
     """

    ols = None
    has_3d = gts_3d is not None

    # init transforms which respectively hold [dx, dy, dw, dh, label]
    # for labels bg=-1, ign=0, fg>=1
    transforms = np.zeros([len(rois), 5], dtype=np.float32)
    raw_gt = np.zeros([len(rois), 5], dtype=np.float32)

    # if 3d, then init other terms after
    if has_3d:
        transforms = np.pad(transforms, [(0, 0), (0, gts_3d.shape[1])],
                            'constant')
        raw_gt = np.pad(raw_gt, [(0, 0), (0, gts_3d.shape[1])], 'constant')

    if gts_val.shape[0] > 0 or gts_ign.shape[0] > 0:

        if gts_ign.shape[0] > 0:

            # compute overlaps ign
            ols_ign = iou_ign(rois, gts_ign)
            ols_ign_max = np.amax(ols_ign, axis=1)

        else:
            ols_ign_max = np.zeros([rois.shape[0]], dtype=np.float32)

        if gts_val.shape[0] > 0:

            # compute overlaps valid
            ols = iou(rois, gts_val)
            ols_max = np.amax(ols, axis=1)
            targets = np.argmax(ols, axis=1)

            # find best matches for each ground truth
            gt_best_rois = np.argmax(ols, axis=0)
            gt_best_ols = np.amax(ols, axis=0)

            gt_best_rois = gt_best_rois[gt_best_ols >= best_thresh]
            gt_best_ols = gt_best_ols[gt_best_ols >= best_thresh]

            fg_inds = np.flatnonzero(ols_max >= fg_thresh)
            fg_inds = np.concatenate((fg_inds, gt_best_rois))
            fg_inds = np.unique(fg_inds)

            target_rois = gts_val[targets[fg_inds], :]
            src_rois = rois[fg_inds, :]

            if len(fg_inds) > 0:

                # compute 2d transform
                transforms[fg_inds, 0:4] = bbox_transform(src_rois, target_rois)

                raw_gt[fg_inds, 0:4] = target_rois

                if has_3d:

                    tracker = tracker.astype(np.int64)
                    src_3d = anchors[tracker[fg_inds], 4:]
                    target_3d = gts_3d[targets[fg_inds]]

                    raw_gt[fg_inds, 5:] = target_3d

                    # compute 3d transform
                    transforms[fg_inds, 5:] = bbox_transform_3d(
                        src_rois, src_3d, target_3d)

                # store labels
                transforms[fg_inds, 4] = [box_lbls[x] for x in targets[fg_inds]]
                assert (all(transforms[fg_inds, 4] >= 1))

        else:

            ols_max = np.zeros(rois.shape[0], dtype=int)
            fg_inds = np.empty(shape=[0])
            gt_best_rois = np.empty(shape=[0])

        # determine ignores
        ign_inds = np.flatnonzero(ols_ign_max >= ign_thresh)

        # determine background
        bg_inds = np.flatnonzero((ols_max >= bg_thresh_lo) & (ols_max <
                                                              bg_thresh_hi))

        # subtract fg and igns from background
        bg_inds = np.setdiff1d(bg_inds, ign_inds)
        bg_inds = np.setdiff1d(bg_inds, fg_inds)
        bg_inds = np.setdiff1d(bg_inds, gt_best_rois)

        # mark background
        transforms[bg_inds, 4] = -1

    else:

        # all background
        transforms[:, 4] = -1

    return transforms, ols, raw_gt


def hill_climb(p2,
               p2_inv,
               box_2d,
               x2d,
               y2d,
               z2d,
               w3d,
               h3d,
               l3d,
               ry3d,
               step_z_init=0,
               step_r_init=0,
               z_lim=0,
               r_lim=0,
               min_ol_dif=0.0):
    """hill climb"""

    step_z = step_z_init
    step_r = step_r_init

    ol_best, verts_best, _, invalid = test_projection(
        p2, p2_inv, box_2d, x2d, y2d, z2d, w3d, h3d, l3d, ry3d)

    if invalid: return z2d, ry3d, verts_best

    # attempt to fit z/rot more properly
    while (step_z > z_lim or step_r > r_lim):

        if step_z > z_lim:

            ol_neg, verts_neg, _, invalid_neg = test_projection(
                p2, p2_inv, box_2d, x2d, y2d, z2d - step_z, w3d, h3d, l3d, ry3d)
            ol_pos, verts_pos, _, invalid_pos = test_projection(
                p2, p2_inv, box_2d, x2d, y2d, z2d + step_z, w3d, h3d, l3d, ry3d)

            invalid = ((ol_pos - ol_best) <= min_ol_dif) and (
                (ol_neg - ol_best) <= min_ol_dif)

            if invalid:
                step_z = step_z * 0.5

            elif (ol_pos - ol_best
                  ) > min_ol_dif and ol_pos > ol_neg and not invalid_pos:
                z2d += step_z
                ol_best = ol_pos
                verts_best = verts_pos
            elif (ol_neg - ol_best) > min_ol_dif and not invalid_neg:
                z2d -= step_z
                ol_best = ol_neg
                verts_best = verts_neg
            else:
                step_z = step_z * 0.5

        if step_r > r_lim:

            ol_neg, verts_neg, _, invalid_neg = test_projection(
                p2, p2_inv, box_2d, x2d, y2d, z2d, w3d, h3d, l3d, ry3d - step_r)
            ol_pos, verts_pos, _, invalid_pos = test_projection(
                p2, p2_inv, box_2d, x2d, y2d, z2d, w3d, h3d, l3d, ry3d + step_r)

            invalid = ((ol_pos - ol_best) <= min_ol_dif) and (
                (ol_neg - ol_best) <= min_ol_dif)

            if invalid:
                step_r = step_r * 0.5

            elif (ol_pos - ol_best
                  ) > min_ol_dif and ol_pos > ol_neg and not invalid_pos:
                ry3d += step_r
                ol_best = ol_pos
                verts_best = verts_pos
            elif (ol_neg - ol_best) > min_ol_dif and not invalid_neg:
                ry3d -= step_r
                ol_best = ol_neg
                verts_best = verts_neg
            else:
                step_r = step_r * 0.5

    while ry3d > math.pi:
        ry3d -= math.pi * 2
    while ry3d < (-math.pi):
        ry3d += math.pi * 2

    return z2d, ry3d, verts_best


# def clsInd2Name(lbls, ind):
#     """
#     Converts a cls ind to string name
#     """

#     if ind>=0 and ind<len(lbls):
#         return lbls[ind]
#     else:
#         raise ValueError('unknown class')


def clsName2Ind(lbls, cls):
    """
    Converts a cls name to an ind
    """
    if cls in lbls:
        return lbls.index(cls) + 1
    else:
        raise ValueError('unknown class')


def compute_bbox_stats(conf, imdb, cache_folder=''):
    """
    Computes the mean and standard deviation for each regression
    parameter (usually pertaining to [dx, dy, sw, sh] but sometimes
    for 3d parameters too).

    Once these stats are known we normalize the regression targets
    to have 0 mean and 1 variance, to hypothetically ease training.
    """

    if (cache_folder is not None) and os.path.exists(os.path.join(cache_folder, 'bbox_means.pkl')) \
            and os.path.exists(os.path.join(cache_folder, 'bbox_stds.pkl')):

        means = pickle_read(os.path.join(cache_folder, 'bbox_means.pkl'))
        stds = pickle_read(os.path.join(cache_folder, 'bbox_stds.pkl'))

    else:

        if conf.has_3d:
            squared_sums = np.zeros([1, 11], dtype=np.float128)
            sums = np.zeros([1, 11], dtype=np.float128)
        else:
            squared_sums = np.zeros([1, 4], dtype=np.float128)
            sums = np.zeros([1, 4], dtype=np.float128)

        class_counts = np.zeros([1], dtype=np.float128) + 1e-10

        # compute the mean first
        logging.info('Computing bbox regression mean..')

        for imind, imobj in enumerate(imdb):

            if len(imobj.gts) > 0:

                scale_factor = imobj.scale * conf.test_scale / imobj.imH
                feat_size = calc_output_size(
                    np.array([imobj.imH, imobj.imW]) * scale_factor,
                    conf.feat_stride)
                rois = locate_anchors(conf.anchors, feat_size, conf.feat_stride)

                # determine ignores
                igns, rmvs = determine_ignores(imobj.gts, conf.lbls, conf.ilbls,
                                               conf.min_gt_vis, conf.min_gt_h,
                                               np.inf, scale_factor)

                # accumulate boxes
                gts_all = bbXYWH2Coords(
                    np.array([gt.bbox_full * scale_factor for gt in imobj.gts]))

                # filter out irrelevant cls, and ignore cls
                gts_val = gts_all[(rmvs == False) & (igns == False), :]
                gts_ign = gts_all[(rmvs == False) & (igns == True), :]

                # accumulate labels
                box_lbls = np.array([gt.cls for gt in imobj.gts])
                box_lbls = box_lbls[(rmvs == False) & (igns == False)]
                box_lbls = np.array(
                    [clsName2Ind(conf.lbls, cls) for cls in box_lbls])

                if conf.has_3d:

                    # accumulate 3d boxes
                    gts_3d = np.array([gt.bbox_3d for gt in imobj.gts])
                    gts_3d = gts_3d[(rmvs == False) & (igns == False), :]

                    # rescale centers (in 2d)
                    for gtind, gt in enumerate(gts_3d):
                        gts_3d[gtind, 0:2] *= scale_factor

                    # compute transforms for all 3d
                    transforms, _, _ = compute_targets(
                        gts_val,
                        gts_ign,
                        box_lbls,
                        rois,
                        conf.fg_thresh,
                        conf.ign_thresh,
                        conf.bg_thresh_lo,
                        conf.bg_thresh_hi,
                        conf.best_thresh,
                        gts_3d=gts_3d,
                        anchors=conf.anchors,
                        tracker=rois[:, 4])
                else:

                    # compute transforms for 2d
                    transforms, _, _ = compute_targets(
                        gts_val, gts_ign, box_lbls, rois, conf.fg_thresh,
                        conf.ign_thresh, conf.bg_thresh_lo, conf.bg_thresh_hi,
                        conf.best_thresh)

                gt_inds = np.flatnonzero(transforms[:, 4] > 0)

                if len(gt_inds) > 0:

                    if conf.has_3d:
                        sums[:, 0:4] += np.sum(transforms[gt_inds, 0:4], axis=0)
                        sums[:, 4:] += np.sum(transforms[gt_inds, 5:12], axis=0)
                    else:
                        sums += np.sum(transforms[gt_inds, 0:4], axis=0)

                    class_counts += len(gt_inds)

        means = sums / class_counts

        logging.info('Computing bbox regression stds..')

        for imobj in imdb:

            if len(imobj.gts) > 0:

                scale_factor = imobj.scale * conf.test_scale / imobj.imH
                feat_size = calc_output_size(
                    np.array([imobj.imH, imobj.imW]) * scale_factor,
                    conf.feat_stride)
                rois = locate_anchors(conf.anchors, feat_size, conf.feat_stride)

                # determine ignores
                igns, rmvs = determine_ignores(imobj.gts, conf.lbls, conf.ilbls,
                                               conf.min_gt_vis, conf.min_gt_h,
                                               np.inf, scale_factor)

                # accumulate boxes
                gts_all = bbXYWH2Coords(
                    np.array([gt.bbox_full * scale_factor for gt in imobj.gts]))

                # filter out irrelevant cls, and ignore cls
                gts_val = gts_all[(rmvs == False) & (igns == False), :]
                gts_ign = gts_all[(rmvs == False) & (igns == True), :]

                # accumulate labels
                box_lbls = np.array([gt.cls for gt in imobj.gts])
                box_lbls = box_lbls[(rmvs == False) & (igns == False)]
                box_lbls = np.array(
                    [clsName2Ind(conf.lbls, cls) for cls in box_lbls])

                if conf.has_3d:

                    # accumulate 3d boxes
                    gts_3d = np.array([gt.bbox_3d for gt in imobj.gts])
                    gts_3d = gts_3d[(rmvs == False) & (igns == False), :]

                    # rescale centers (in 2d)
                    for gtind, gt in enumerate(gts_3d):
                        gts_3d[gtind, 0:2] *= scale_factor

                    # compute transforms for all 3d
                    transforms, _, _ = compute_targets(
                        gts_val,
                        gts_ign,
                        box_lbls,
                        rois,
                        conf.fg_thresh,
                        conf.ign_thresh,
                        conf.bg_thresh_lo,
                        conf.bg_thresh_hi,
                        conf.best_thresh,
                        gts_3d=gts_3d,
                        anchors=conf.anchors,
                        tracker=rois[:, 4])
                else:

                    # compute transforms for 2d
                    transforms, _, _ = compute_targets(
                        gts_val, gts_ign, box_lbls, rois, conf.fg_thresh,
                        conf.ign_thresh, conf.bg_thresh_lo, conf.bg_thresh_hi,
                        conf.best_thresh)

                gt_inds = np.flatnonzero(transforms[:, 4] > 0)

                if len(gt_inds) > 0:

                    if conf.has_3d:
                        squared_sums[:, 0:4] += np.sum(np.power(
                            transforms[gt_inds, 0:4] - means[:, 0:4], 2),
                                                       axis=0)
                        squared_sums[:, 4:] += np.sum(np.power(
                            transforms[gt_inds, 5:12] - means[:, 4:], 2),
                                                      axis=0)

                    else:
                        squared_sums += np.sum(np.power(
                            transforms[gt_inds, 0:4] - means, 2),
                                               axis=0)

        stds = np.sqrt((squared_sums / class_counts))

        means = means.astype(float)
        stds = stds.astype(float)

        logging.info('used {:d} boxes with avg std {:.4f}'.format(
            int(class_counts[0]), np.mean(stds)))

        if (cache_folder is not None):
            pickle_write(os.path.join(cache_folder, 'bbox_means.pkl'), means)
            pickle_write(os.path.join(cache_folder, 'bbox_stds.pkl'), stds)

    conf.bbox_means = means
    conf.bbox_stds = stds


def flatten_tensor(input):
    """
    Flattens and permutes a tensor from size
    [B x C x W x H] --> [B x (W x H) x C]
    [B x C x H x W] --> [B x (W x H) x C]
    """

    bsize, csize, h, w = input.shape
    input_trans = fluid.layers.transpose(input, [0, 2, 3, 1])
    output = fluid.layers.reshape(input_trans, [bsize, h * w, csize])
    return output


# def unflatten_tensor(input, feat_size, anchors):
#     """
#     Un-flattens and un-permutes a tensor from size
#     [B x (W x H) x C] --> [B x C x W x H]
#     """

#     bsize = input.shape[0]

#     if len(input.shape) >= 3: csize = input.shape[2]
#     else: csize = 1

#     input = input.view(bsize, feat_size[0] * anchors.shape[0], feat_size[1], csize)
#     input = input.permute(0, 3, 1, 2).contiguous()

#     return input


def project_3d(p2, x3d, y3d, z3d, w3d, h3d, l3d, ry3d, return_3d=False):
    """
	Projects a 3D box into 2D vertices

	Args:
		p2 (nparray): projection matrix of size 4x3
		x3d: x-coordinate of center of object
		y3d: y-coordinate of center of object
		z3d: z-cordinate of center of object
		w3d: width of object
		h3d: height of object
		l3d: length of object
		ry3d: rotation w.r.t y-axis
	"""

    # compute rotational matrix around yaw axis
    R = np.array([[+math.cos(ry3d), 0, +math.sin(ry3d)], [0, 1, 0],
                  [-math.sin(ry3d), 0, +math.cos(ry3d)]])

    # 3D bounding box corners
    x_corners = np.array([0, l3d, l3d, l3d, l3d, 0, 0, 0])
    y_corners = np.array([0, 0, h3d, h3d, 0, 0, h3d, h3d])
    z_corners = np.array([0, 0, 0, w3d, w3d, w3d, w3d, 0])

    x_corners += -l3d / 2
    y_corners += -h3d / 2
    z_corners += -w3d / 2

    # bounding box in object co-ordinate
    corners_3d = np.array([x_corners, y_corners, z_corners])

    # rotate
    corners_3d = R.dot(corners_3d)

    # translate
    corners_3d += np.array([x3d, y3d, z3d]).reshape((3, 1))

    corners_3D_1 = np.vstack((corners_3d, np.ones((corners_3d.shape[-1]))))
    corners_2D = p2.dot(corners_3D_1)
    corners_2D = corners_2D / corners_2D[2]

    bb3d_lines_verts_idx = [0, 1, 2, 3, 4, 5, 6, 7, 0, 5, 4, 1, 2, 7, 6, 3]

    verts3d = (corners_2D[:, bb3d_lines_verts_idx][:2]).astype(float).T

    if return_3d:
        return verts3d, corners_3d
    else:
        return verts3d


def project_3d_corners(p2, x3d, y3d, z3d, w3d, h3d, l3d, ry3d):
    """
	Projects a 3D box into 2D vertices

	Args:
		p2 (nparray): projection matrix of size 4x3
		x3d: x-coordinate of center of object
		y3d: y-coordinate of center of object
		z3d: z-cordinate of center of object
		w3d: width of object
		h3d: height of object
		l3d: length of object
		ry3d: rotation w.r.t y-axis
	"""

    # compute rotational matrix around yaw axis
    R = np.array([[+math.cos(ry3d), 0, +math.sin(ry3d)], [0, 1, 0],
                  [-math.sin(ry3d), 0, +math.cos(ry3d)]])

    # 3D bounding box corners
    x_corners = np.array([0, l3d, l3d, l3d, l3d, 0, 0, 0])
    y_corners = np.array([0, 0, h3d, h3d, 0, 0, h3d, h3d])
    z_corners = np.array([0, 0, 0, w3d, w3d, w3d, w3d, 0])
    '''
	order of vertices
	0  upper back right
	1  upper front right
	2  bottom front right
	3  bottom front left
	4  upper front left
	5  upper back left
	6  bottom back left
	7  bottom back right

	bot_inds = np.array([2,3,6,7])
	top_inds = np.array([0,1,4,5])
	'''

    x_corners += -l3d / 2
    y_corners += -h3d / 2
    z_corners += -w3d / 2

    # bounding box in object co-ordinate
    corners_3d = np.array([x_corners, y_corners, z_corners])

    # rotate
    corners_3d = R.dot(corners_3d)

    # translate
    corners_3d += np.array([x3d, y3d, z3d]).reshape((3, 1))

    corners_3D_1 = np.vstack((corners_3d, np.ones((corners_3d.shape[-1]))))
    corners_2D = p2.dot(corners_3D_1)
    corners_2D = corners_2D / corners_2D[2]

    return corners_2D, corners_3D_1


# def bbCoords2XYWH(box):
#     """
#     Convert from [x1, y1, x2, y2] to [x,y,w,h]
#     """

#     if box.shape[0] == 0: return np.empty([0, 4], dtype=float)

#     box[:, 2] -= box[:, 0] + 1
#     box[:, 3] -= box[:, 1] + 1

#     return box


def bbXYWH2Coords(box):
    """
    Convert from [x,y,w,h] to [x1, y1, x2, y2]
    """

    if box.shape[0] == 0: return np.empty([0, 4], dtype=float)

    box[:, 2] += box[:, 0] - 1
    box[:, 3] += box[:, 1] - 1

    return box


def bbox_transform_3d(ex_rois_2d, ex_rois_3d, gt_rois):
    """
    Compute the bbox target transforms in 3D.

    Translations are done as simple difference, whereas others involving
    scaling are done in log space (hence, log(1) = 0, log(0.8) < 0 and
    log(1.2) > 0 which is a good property).
    """

    ex_widths = ex_rois_2d[:, 2] - ex_rois_2d[:, 0] + 1.0
    ex_heights = ex_rois_2d[:, 3] - ex_rois_2d[:, 1] + 1.0
    ex_ctr_x = ex_rois_2d[:, 0] + 0.5 * (ex_widths - 1)
    ex_ctr_y = ex_rois_2d[:, 1] + 0.5 * (ex_heights - 1)

    gt_ctr_x = gt_rois[:, 0]
    gt_ctr_y = gt_rois[:, 1]

    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights

    delta_z = gt_rois[:, 2] - ex_rois_3d[:, 0]
    scale_w = np.log(gt_rois[:, 3] / ex_rois_3d[:, 1])
    scale_h = np.log(gt_rois[:, 4] / ex_rois_3d[:, 2])
    scale_l = np.log(gt_rois[:, 5] / ex_rois_3d[:, 3])
    deltaRotY = gt_rois[:, 6] - ex_rois_3d[:, 4]

    targets = np.vstack((targets_dx, targets_dy, delta_z, scale_w, scale_h,
                         scale_l, deltaRotY)).transpose()
    targets = np.hstack((targets, gt_rois[:, 7:]))

    return targets


def bbox_transform(ex_rois, gt_rois):
    """
    Compute the bbox target transforms in 2D.

    Translations are done as simple difference, whereas others involving
    scaling are done in log space (hence, log(1) = 0, log(0.8) < 0 and
    log(1.2) > 0 which is a good property).
    """

    ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
    ex_ctr_x = ex_rois[:, 0] + 0.5 * (ex_widths - 1)
    ex_ctr_y = ex_rois[:, 1] + 0.5 * (ex_heights - 1)

    gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + 1.0
    gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + 1.0
    gt_ctr_x = gt_rois[:, 0] + 0.5 * (gt_widths - 1.0)
    gt_ctr_y = gt_rois[:, 1] + 0.5 * (gt_heights - 1.0)

    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = np.log(gt_widths / ex_widths)
    targets_dh = np.log(gt_heights / ex_heights)

    targets = np.vstack(
        (targets_dx, targets_dy, targets_dw, targets_dh)).transpose()

    return targets


def bbox_transform_inv(boxes, deltas, means=None, stds=None):
    """
    Compute the bbox target transforms in 3D.
    Translations are done as simple difference, whereas others involving
    scaling are done in log space (hence, log(1) = 0, log(0.8) < 0 and
    log(1.2) > 0 which is a good property).
 
    Args:
        bboxes (nparray): N x 5 array describing [x1, y1, x2, y2, anchor_index]
        deltas (nparray): N x 4 array describing [dx, dy, dw, dh]
    return: bbox target transforms in 3D (nparray) 
    """

    if boxes.shape[0] == 0:
        return np.zeros((0, deltas.shape[1]), dtype=deltas.dtype)

    # boxes = boxes.astype(deltas.dtype, copy=False)
    data_type = type(deltas)
    if data_type == paddle.fluid.core_avx.VarBase:
        boxes = to_variable(boxes)

    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    dx = deltas[:, 0]
    dy = deltas[:, 1]
    dw = deltas[:, 2]
    dh = deltas[:, 3]

    if stds is not None:
        dx *= stds[0]
        dy *= stds[1]
        dw *= stds[2]
        dh *= stds[3]

    if means is not None:
        dx += means[0]
        dy += means[1]
        dw += means[2]
        dh += means[3]

    if data_type == np.ndarray:
        pred_ctr_x = dx * widths + ctr_x
        pred_ctr_y = dy * heights + ctr_y
        pred_w = np.exp(dw) * widths
        pred_h = np.exp(dh) * heights

        pred_boxes = np.zeros(deltas.shape)

        # x1, y1, x2, y2
        pred_boxes[:, 0] = pred_ctr_x - 0.5 * pred_w
        pred_boxes[:, 1] = pred_ctr_y - 0.5 * pred_h
        pred_boxes[:, 2] = pred_ctr_x + 0.5 * pred_w
        pred_boxes[:, 3] = pred_ctr_y + 0.5 * pred_h

        return pred_boxes
    elif data_type == paddle.fluid.core_avx.VarBase:
        pred_ctr_x = dx * widths + ctr_x
        pred_ctr_y = dy * heights + ctr_y
        pred_w = fluid.layers.exp(dw) * widths
        pred_h = fluid.layers.exp(dh) * heights

        pred_x1 = fluid.layers.unsqueeze(pred_ctr_x - 0.5 * pred_w, 1)
        pred_y1 = fluid.layers.unsqueeze(pred_ctr_y - 0.5 * pred_h, 1)
        pred_x2 = fluid.layers.unsqueeze(pred_ctr_x + 0.5 * pred_w, 1)
        pred_y2 = fluid.layers.unsqueeze(pred_ctr_y + 0.5 * pred_h, 1)
        pred_boxes = fluid.layers.concat(
            input=[pred_x1, pred_y1, pred_x2, pred_y2], axis=1)
        return pred_boxes
    else:
        raise ValueError('unknown data type {}'.format(data_type))


def determine_ignores(gts,
                      lbls,
                      ilbls,
                      min_gt_vis=0.99,
                      min_gt_h=0,
                      max_gt_h=10e10,
                      scale_factor=1):
    """
    Given various configuration settings, determine which ground truths
    are ignored and which are relevant.
    """

    igns = np.zeros([len(gts)], dtype=bool)
    rmvs = np.zeros([len(gts)], dtype=bool)

    for gtind, gt in enumerate(gts):

        ign = gt.ign
        ign |= gt.visibility < min_gt_vis
        ign |= gt.bbox_full[3] * scale_factor < min_gt_h
        ign |= gt.bbox_full[3] * scale_factor > max_gt_h
        ign |= gt.cls in ilbls

        rmv = not gt.cls in (lbls + ilbls)

        igns[gtind] = ign
        rmvs[gtind] = rmv

    return igns, rmvs


def locate_anchors(anchors, feat_size, stride):
    """
    Spreads each anchor shape across a feature map of size feat_size spaced by a known stride.

    Args:
        anchors (ndarray): N x 4 array describing [x1, y1, x2, y2] displacements for N anchors
        feat_size (ndarray): the downsampled resolution W x H to spread anchors across
        stride (int): stride of a network
        

    Returns:
         ndarray: 2D array = [(W x H) x 5] array consisting of [x1, y1, x2, y2, anchor_index]
    """

    # compute rois
    shift_x = np.array(range(0, feat_size[1], 1)) * float(stride)
    shift_y = np.array(range(0, feat_size[0], 1)) * float(stride)
    [shift_x, shift_y] = np.meshgrid(shift_x, shift_y)

    rois = np.expand_dims(anchors[:, 0:4], axis=1)
    shift_x = np.expand_dims(shift_x, axis=0)
    shift_y = np.expand_dims(shift_y, axis=0)

    shift_x1 = shift_x + np.expand_dims(rois[:, :, 0], axis=2)
    shift_y1 = shift_y + np.expand_dims(rois[:, :, 1], axis=2)
    shift_x2 = shift_x + np.expand_dims(rois[:, :, 2], axis=2)
    shift_y2 = shift_y + np.expand_dims(rois[:, :, 3], axis=2)

    # compute anchor tracker
    anchor_tracker = np.zeros(shift_x1.shape, dtype=float)
    for aind in range(0, rois.shape[0]):
        anchor_tracker[aind, :, :] = aind

    stack_size = feat_size[0] * anchors.shape[0]

    shift_x1 = shift_x1.reshape(1, stack_size, feat_size[1]).reshape(-1, 1)
    shift_y1 = shift_y1.reshape(1, stack_size, feat_size[1]).reshape(-1, 1)
    shift_x2 = shift_x2.reshape(1, stack_size, feat_size[1]).reshape(-1, 1)
    shift_y2 = shift_y2.reshape(1, stack_size, feat_size[1]).reshape(-1, 1)
    anchor_tracker = anchor_tracker.reshape(1, stack_size,
                                            feat_size[1]).reshape(-1, 1)

    rois = np.concatenate(
        (shift_x1, shift_y1, shift_x2, shift_y2, anchor_tracker), 1)

    return rois


def calc_output_size(res, stride):
    """
    Approximate the output size of a network

    Args:
        res (ndarray): input resolution
        stride (int): stride of a network

    Returns:
         ndarray: output resolution
    """

    return np.ceil(np.array(res) / stride).astype(int)


def im_detect_3d(im, net, rpn_conf, preprocess, p2, gpu=0, synced=False):
    """
    Object detection in 3D
    """

    imH_orig = im.shape[0]
    imW_orig = im.shape[1]

    im = preprocess(im)
    im = im[np.newaxis, :, :, :]
    imH = im.shape[2]
    imW = im.shape[3]
    # move to GPU
    im = to_variable(im)

    scale_factor = imH / imH_orig

    cls, prob, bbox_2d, bbox_3d, feat_size, rois = net(im)

    # compute feature resolution
    num_anchors = rpn_conf.anchors.shape[0]

    bbox_x = bbox_2d[:, :, 0]
    bbox_y = bbox_2d[:, :, 1]
    bbox_w = bbox_2d[:, :, 2]
    bbox_h = bbox_2d[:, :, 3]

    bbox_x3d = bbox_3d[:, :, 0]
    bbox_y3d = bbox_3d[:, :, 1]
    bbox_z3d = bbox_3d[:, :, 2]
    bbox_w3d = bbox_3d[:, :, 3]
    bbox_h3d = bbox_3d[:, :, 4]
    bbox_l3d = bbox_3d[:, :, 5]
    bbox_ry3d = bbox_3d[:, :, 6]

    # detransform 3d
    bbox_x3d = bbox_x3d * rpn_conf.bbox_stds[:, 4][
        0] + rpn_conf.bbox_means[:, 4][0]
    bbox_y3d = bbox_y3d * rpn_conf.bbox_stds[:, 5][
        0] + rpn_conf.bbox_means[:, 5][0]
    bbox_z3d = bbox_z3d * rpn_conf.bbox_stds[:, 6][
        0] + rpn_conf.bbox_means[:, 6][0]
    bbox_w3d = bbox_w3d * rpn_conf.bbox_stds[:, 7][
        0] + rpn_conf.bbox_means[:, 7][0]
    bbox_h3d = bbox_h3d * rpn_conf.bbox_stds[:, 8][
        0] + rpn_conf.bbox_means[:, 8][0]
    bbox_l3d = bbox_l3d * rpn_conf.bbox_stds[:, 9][
        0] + rpn_conf.bbox_means[:, 9][0]
    bbox_ry3d = bbox_ry3d * rpn_conf.bbox_stds[:, 10][
        0] + rpn_conf.bbox_means[:, 10][0]

    # find 3d source

    #tracker = rois[:, 4].cpu().detach().numpy().astype(np.int64)
    #src_3d = torch.from_numpy(rpn_conf.anchors[tracker, 4:]).cuda().type(torch.cuda.FloatTensor)
    tracker = rois[:, 4].astype(np.int64)
    src_3d = rpn_conf.anchors[tracker, 4:]

    #tracker_sca = rois_sca[:, 4].cpu().detach().numpy().astype(np.int64)
    #src_3d_sca = torch.from_numpy(rpn_conf.anchors[tracker_sca, 4:]).cuda().type(torch.cuda.FloatTensor)

    # compute 3d transform
    widths = rois[:, 2] - rois[:, 0] + 1.0
    heights = rois[:, 3] - rois[:, 1] + 1.0
    ctr_x = rois[:, 0] + 0.5 * widths
    ctr_y = rois[:, 1] + 0.5 * heights

    bbox_x3d_np = bbox_x3d.numpy()
    bbox_y3d_np = bbox_y3d.numpy()  #(1, N)
    bbox_z3d_np = bbox_z3d.numpy()
    bbox_w3d_np = bbox_w3d.numpy()
    bbox_l3d_np = bbox_l3d.numpy()
    bbox_h3d_np = bbox_h3d.numpy()
    bbox_ry3d_np = bbox_ry3d.numpy()

    bbox_x3d_np = bbox_x3d_np[0, :] * widths + ctr_x
    bbox_y3d_np = bbox_y3d_np[0, :] * heights + ctr_y

    bbox_x_np = bbox_x.numpy()
    bbox_y_np = bbox_y.numpy()
    bbox_w_np = bbox_w.numpy()
    bbox_h_np = bbox_h.numpy()

    bbox_z3d_np = src_3d[:, 0] + bbox_z3d_np[0, :]  #(N, 5), (N2, 1)
    bbox_w3d_np = np.exp(bbox_w3d_np[0, :]) * src_3d[:, 1]
    bbox_h3d_np = np.exp(bbox_h3d_np[0, :]) * src_3d[:, 2]
    bbox_l3d_np = np.exp(bbox_l3d_np[0, :]) * src_3d[:, 3]
    bbox_ry3d_np = src_3d[:, 4] + bbox_ry3d_np[0, :]

    # bundle
    coords_3d = np.stack((bbox_x3d_np, bbox_y3d_np, bbox_z3d_np[:bbox_x3d_np.shape[0]], bbox_w3d_np[:bbox_x3d_np.shape[0]], bbox_h3d_np[:bbox_x3d_np.shape[0]], \
        bbox_l3d_np[:bbox_x3d_np.shape[0]], bbox_ry3d_np[:bbox_x3d_np.shape[0]]), axis=1)#[N, 7]

    # compile deltas pred

    deltas_2d = np.concatenate(
        (bbox_x_np[0, :, np.newaxis], bbox_y_np[0, :, np.newaxis],
         bbox_w_np[0, :, np.newaxis], bbox_h_np[0, :, np.newaxis]),
        axis=1)  #N,4
    coords_2d = bbox_transform_inv(
        rois,
        deltas_2d,
        means=rpn_conf.bbox_means[0, :],
        stds=rpn_conf.bbox_stds[0, :])  #[N,4]

    # detach onto cpu
    #coords_2d = coords_2d.cpu().detach().numpy()
    #coords_3d = coords_3d.cpu().detach().numpy()
    prob_np = prob[0, :, :].numpy()  #.cpu().detach().numpy()

    # scale coords
    coords_2d[:, 0:4] /= scale_factor
    coords_3d[:, 0:2] /= scale_factor

    cls_pred = np.argmax(prob_np[:, 1:], axis=1) + 1
    scores = np.amax(prob_np[:, 1:], axis=1)

    aboxes = np.hstack((coords_2d, scores[:, np.newaxis]))

    sorted_inds = (-aboxes[:, 4]).argsort()
    original_inds = (sorted_inds).argsort()
    aboxes = aboxes[sorted_inds, :]
    coords_3d = coords_3d[sorted_inds, :]
    cls_pred = cls_pred[sorted_inds]
    tracker = tracker[sorted_inds]

    if synced:

        # nms
        keep_inds = gpu_nms(
            aboxes[:, 0:5].astype(np.float32),
            rpn_conf.nms_thres,
            device_id=gpu)

        # convert to bool
        keep = np.zeros([aboxes.shape[0], 1], dtype=bool)
        keep[keep_inds, :] = True

        # stack the keep array,
        # sync to the original order
        aboxes = np.hstack((aboxes, keep))
        aboxes[original_inds, :]

    else:

        # pre-nms
        cls_pred = cls_pred[0:min(rpn_conf.nms_topN_pre, cls_pred.shape[0])]
        tracker = tracker[0:min(rpn_conf.nms_topN_pre, tracker.shape[0])]
        aboxes = aboxes[0:min(rpn_conf.nms_topN_pre, aboxes.shape[0]), :]
        coords_3d = coords_3d[0:min(rpn_conf.nms_topN_pre, coords_3d.shape[0])]

        # nms
        keep_inds = gpu_nms(
            aboxes[:, 0:5].astype(np.float32),
            rpn_conf.nms_thres,
            device_id=gpu)

        # stack cls prediction
        aboxes = np.hstack((aboxes, cls_pred[:, np.newaxis], coords_3d,
                            tracker[:, np.newaxis]))

        # suppress boxes
        aboxes = aboxes[keep_inds, :]

    # clip boxes
    if rpn_conf.clip_boxes:
        aboxes[:, 0] = np.clip(aboxes[:, 0], 0, imW_orig - 1)
        aboxes[:, 1] = np.clip(aboxes[:, 1], 0, imH_orig - 1)
        aboxes[:, 2] = np.clip(aboxes[:, 2], 0, imW_orig - 1)
        aboxes[:, 3] = np.clip(aboxes[:, 3], 0, imH_orig - 1)
    return aboxes


def get_2D_from_3D(p2, cx3d, cy3d, cz3d, w3d, h3d, l3d, rotY):

    verts3d, corners_3d = project_3d(
        p2, cx3d, cy3d, cz3d, w3d, h3d, l3d, rotY, return_3d=True)

    # any boxes behind camera plane?
    if np.any(corners_3d[2, :] <= 0):
        ign = True

    else:
        x = min(verts3d[:, 0])
        y = min(verts3d[:, 1])
        x2 = max(verts3d[:, 0])
        y2 = max(verts3d[:, 1])

    return np.array([x, y, x2, y2])


def test_kitti_3d(dataset_test,
                  net,
                  rpn_conf,
                  results_path,
                  test_path,
                  use_log=True):
    """
    Test the KITTI framework for object detection in 3D
    """

    # import read_kitti_cal
    from data.m3drpn_reader import read_kitti_cal

    imlist = list_files(
        os.path.join(test_path, dataset_test, 'validation', 'image_2', ''),
        '*.png')

    preprocess = Preprocess([rpn_conf.test_scale], rpn_conf.image_means,
                            rpn_conf.image_stds)

    # fix paths slightly
    _, test_iter, _ = file_parts(results_path.replace('/data', ''))
    test_iter = test_iter.replace('results_', '')

    # init
    test_start = time()

    for imind, impath in enumerate(imlist):

        im = cv2.imread(impath)

        base_path, name, ext = file_parts(impath)

        # read in calib
        p2 = read_kitti_cal(
            os.path.join(test_path, dataset_test, 'validation', 'calib', name +
                         '.txt'))
        p2_inv = np.linalg.inv(p2)

        # forward test batch
        aboxes = im_detect_3d(im, net, rpn_conf, preprocess, p2)

        base_path, name, ext = file_parts(impath)

        file = open(os.path.join(results_path, name + '.txt'), 'w')
        text_to_write = ''

        for boxind in range(0, min(rpn_conf.nms_topN_post, aboxes.shape[0])):

            box = aboxes[boxind, :]
            score = box[4]
            cls = rpn_conf.lbls[int(box[5] - 1)]

            #if score >= 0.75:
            if score >= 0.5:  #TODO yexiaoqing

                x1 = box[0]
                y1 = box[1]
                x2 = box[2]
                y2 = box[3]
                width = (x2 - x1 + 1)
                height = (y2 - y1 + 1)

                # plot 3D box
                x3d = box[6]
                y3d = box[7]
                z3d = box[8]
                w3d = box[9]
                h3d = box[10]
                l3d = box[11]
                ry3d = box[12]

                # convert alpha into ry3d
                coord3d = np.linalg.inv(p2).dot(
                    np.array([x3d * z3d, y3d * z3d, 1 * z3d, 1]))
                ry3d = convertAlpha2Rot(ry3d, coord3d[2], coord3d[0])

                step_r = 0.3 * math.pi
                r_lim = 0.01
                box_2d = np.array([x1, y1, width, height])

                z3d, ry3d, verts_best = hill_climb(
                    p2,
                    p2_inv,
                    box_2d,
                    x3d,
                    y3d,
                    z3d,
                    w3d,
                    h3d,
                    l3d,
                    ry3d,
                    step_r_init=step_r,
                    r_lim=r_lim)

                # predict a more accurate projection
                coord3d = np.linalg.inv(p2).dot(
                    np.array([x3d * z3d, y3d * z3d, 1 * z3d, 1]))
                alpha = convertRot2Alpha(ry3d, coord3d[2], coord3d[0])

                x3d = coord3d[0]
                y3d = coord3d[1]
                z3d = coord3d[2]

                y3d += h3d / 2

                text_to_write += (
                    '{} -1 -1 {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} '
                    + '{:.6f} {:.6f}\n').format(cls, alpha, x1, y1, x2, y2, h3d,
                                                w3d, l3d, x3d, y3d, z3d, ry3d,
                                                score)

        file.write(text_to_write)
        file.close()

        # display stats
        if (imind + 1) % 1000 == 0:
            time_str, dt = compute_eta(test_start, imind + 1, len(imlist))

            print_str = 'testing {}/{}, dt: {:0.3f}, eta: {}'.format(
                imind + 1, len(imlist), dt, time_str)

            if use_log: logging.info(print_str)
            else: print(print_str)

    # evaluate
    script = os.path.join(test_path, dataset_test, 'devkit', 'cpp',
                          'evaluate_object')
    with open(os.devnull, 'w') as devnull:
        out = subprocess.check_output(
            [script, results_path.replace('/data', '')], stderr=devnull)

    for lbl in rpn_conf.lbls:

        lbl = lbl.lower()

        respath_2d = os.path.join(
            results_path.replace('/data', ''),
            'stats_{}_detection.txt'.format(lbl))
        respath_gr = os.path.join(
            results_path.replace('/data', ''),
            'stats_{}_detection_ground.txt'.format(lbl))
        respath_3d = os.path.join(
            results_path.replace('/data', ''),
            'stats_{}_detection_3d.txt'.format(lbl))

        if os.path.exists(respath_2d):
            easy, mod, hard = parse_kitti_result(respath_2d)

            print_str = 'test_iter {} 2d {} --> easy: {:0.4f}, mod: {:0.4f}, hard: {:0.4f}'.format(
                test_iter, lbl, easy, mod, hard)
            if use_log: logging.info(print_str)
            else: print(print_str)

        if os.path.exists(respath_gr):
            easy, mod, hard = parse_kitti_result(respath_gr)

            print_str = 'test_iter {} gr {} --> easy: {:0.4f}, mod: {:0.4f}, hard: {:0.4f}'.format(
                test_iter, lbl, easy, mod, hard)

            if use_log: logging.info(print_str)
            else: print(print_str)

        if os.path.exists(respath_3d):
            easy, mod, hard = parse_kitti_result(respath_3d)

            print_str = 'test_iter {} 3d {} --> easy: {:0.4f}, mod: {:0.4f}, hard: {:0.4f}'.format(
                test_iter, lbl, easy, mod, hard)

            if use_log: logging.info(print_str)
            else: print(print_str)


def parse_kitti_result(respath):

    text_file = open(respath, 'r')

    acc = np.zeros([3, 41], dtype=float)

    lind = 0
    for line in text_file:

        parsed = re.findall('([\d]+\.?[\d]*)', line)

        for i, num in enumerate(parsed):
            acc[lind, i] = float(num)

        lind += 1

    text_file.close()

    easy = np.mean(acc[0, 0:41:4])
    mod = np.mean(acc[1, 0:41:4])
    hard = np.mean(acc[2, 0:41:4])

    #easy = np.mean(acc[0, 1:41:1])
    #mod = np.mean(acc[1, 1:41:1])
    #hard = np.mean(acc[2, 1:41:1])

    return easy, mod, hard


# def parse_kitti_vo(respath):

#     text_file = open(respath, 'r')

#     acc = np.zeros([1, 2], dtype=float)

#     lind = 0
#     for line in text_file:

#         parsed = re.findall('([\d]+\.?[\d]*)', line)

#         for i, num in enumerate(parsed):
#             acc[lind, i] = float(num)

#         lind += 1

#     text_file.close()

#     t = acc[0, 0]*100
#     r = acc[0, 1]

#     return t, r


def test_projection(p2, p2_inv, box_2d, cx, cy, z, w3d, h3d, l3d, rotY):
    """
    Tests the consistency of a 3D projection compared to a 2D box
    """

    x = box_2d[0]
    y = box_2d[1]
    x2 = x + box_2d[2] - 1
    y2 = y + box_2d[3] - 1

    coord3d = p2_inv.dot(np.array([cx * z, cy * z, z, 1]))

    cx3d = coord3d[0]
    cy3d = coord3d[1]
    cz3d = coord3d[2]

    # put back on ground first
    #cy3d += h3d/2

    # re-compute the 2D box using 3D (finally, avoids clipped boxes)
    verts3d, corners_3d = project_3d(
        p2, cx3d, cy3d, cz3d, w3d, h3d, l3d, rotY, return_3d=True)

    invalid = np.any(corners_3d[2, :] <= 0)

    x_new = min(verts3d[:, 0])
    y_new = min(verts3d[:, 1])
    x2_new = max(verts3d[:, 0])
    y2_new = max(verts3d[:, 1])

    b1 = np.array([x, y, x2, y2])[np.newaxis, :]
    b2 = np.array([x_new, y_new, x2_new, y2_new])[np.newaxis, :]

    #ol = iou(b1, b2)[0][0]
    ol = -(np.abs(x - x_new) + np.abs(y - y_new) + np.abs(x2 - x2_new) +
           np.abs(y2 - y2_new))

    return ol, verts3d, b2, invalid
