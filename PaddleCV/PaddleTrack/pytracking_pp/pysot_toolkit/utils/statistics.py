"""
    @author fangyi.zhang@vipl.ict.ac.cn
"""

import numpy as np
from numba import jit
from . import region

def calculate_failures(trajectory):
    """ Calculate number of failures
    Args:
        trajectory: list of bbox
    Returns:
        num_failures: number of failures
        failures: failures point in trajectory, start with 0
    """
    failures = [i for i, x in zip(range(len(trajectory)), trajectory)
            if len(x) == 1 and x[0] == 2]
    num_failures = len(failures)
    return num_failures, failures

def calculate_accuracy(pred_trajectory, gt_trajectory,
        burnin=0, ignore_unknown=True, bound=None):
    """Caculate accuracy socre as average overlap over the entire sequence
    Args:
        trajectory: list of bbox
        gt_trajectory: list of bbox
        burnin: number of frames that have to be ignored after the failure
        ignore_unknown: ignore frames where the overlap is unknown
        bound: bounding region
    Return:
        acc: average overlap
        overlaps: per frame overlaps
    """
    pred_trajectory_ = pred_trajectory
    if not ignore_unknown:
        unkown = [len(x)==1 and x[0] == 0 for x in pred_trajectory]
    
    if burnin > 0:
        pred_trajectory_ = pred_trajectory[:]
        mask = [len(x)==1 and x[0] == 1 for x in pred_trajectory]
        for i in range(len(mask)):
            if mask[i]:
                for j in range(burnin):
                    if i + j < len(mask):
                        pred_trajectory_[i+j] = [0]
    min_len = min(len(pred_trajectory_), len(gt_trajectory))
    overlaps = region.vot_overlap_traj(pred_trajectory_[:min_len],
            gt_trajectory[:min_len], bound)

    if not ignore_unknown:
        overlaps = [u if u else 0 for u in unkown]

    acc = 0
    if len(overlaps) > 0:
        acc = np.nanmean(overlaps)
    return acc, overlaps

# def caculate_expected_overlap(pred_trajectorys, gt_trajectorys, skip_init, traj_length=None,
#         weights=None, tags=['all']):
#     """ Caculate expected overlap
#     Args:
#         pred_trajectory: list of bbox
#         gt_trajectory: list of bbox
#         traj_length: a list of sequence length for which the overlap should be evaluated
#         weights: a list of per-sequence weights that indicate how much does each sequence
#                 contribute to the estimate
#         tags:  set list of tags for which to perform calculation
#     """
#     overlaps = [calculate_accuracy(pred, gt)[1]
#             for pred, gt in zip(pred_trajectorys, gt_trajectorys)]
#     failures = [calculate_accuracy(pred, gt)[1]
#             for pred, gt in zip(pred_trajectorys, gt_trajectorys)]
# 
#     if traj_length is None:
#         traj_length = range(1, max([len(x) for x in gt_trajectorys])+1)
#     traj_length = list(set(traj_length))

@jit(nopython=True)
def overlap_ratio(rect1, rect2):
    '''Compute overlap ratio between two rects
    Args
        rect:2d array of N x [x,y,w,h]
    Return:
        iou
    '''
    # if rect1.ndim==1:
    #     rect1 = rect1[np.newaxis, :]
    # if rect2.ndim==1:
    #     rect2 = rect2[np.newaxis, :]
    left = np.maximum(rect1[:,0], rect2[:,0])
    right = np.minimum(rect1[:,0]+rect1[:,2], rect2[:,0]+rect2[:,2])
    top = np.maximum(rect1[:,1], rect2[:,1])
    bottom = np.minimum(rect1[:,1]+rect1[:,3], rect2[:,1]+rect2[:,3])

    intersect = np.maximum(0,right - left) * np.maximum(0,bottom - top)
    union = rect1[:,2]*rect1[:,3] + rect2[:,2]*rect2[:,3] - intersect
    iou = intersect / union
    iou = np.maximum(np.minimum(1, iou), 0)
    return iou

@jit(nopython=True)
def success_overlap(gt_bb, result_bb, n_frame):
    thresholds_overlap = np.arange(0, 1.05, 0.05)
    success = np.zeros(len(thresholds_overlap))
    iou = np.ones(len(gt_bb)) * (-1)
    # mask = np.sum(gt_bb > 0, axis=1) == 4
    mask = np.sum(gt_bb[:, 2:] > 0, axis=1) == 2
    iou[mask] = overlap_ratio(gt_bb[mask], result_bb[mask])
    for i in range(len(thresholds_overlap)):
        success[i] = np.sum(iou > thresholds_overlap[i]) / float(n_frame)
    return success

@jit(nopython=True)
def success_error(gt_center, result_center, thresholds, n_frame):
    # n_frame = len(gt_center)
    success = np.zeros(len(thresholds))
    dist = np.ones(len(gt_center)) * (-1)
    mask = np.sum(gt_center > 0, axis=1) == 2
    dist[mask] = np.sqrt(np.sum(
        np.power(gt_center[mask] - result_center[mask], 2), axis=1))
    for i in range(len(thresholds)):
        success[i] = np.sum(dist <= thresholds[i]) / float(n_frame)
    return success

@jit(nopython=True)
def determine_thresholds(scores, resolution=100):
    """
    Args:
        scores: 1d array of score
    """
    scores = np.sort(scores[np.logical_not(np.isnan(scores))])
    delta = np.floor(len(scores) / (resolution - 2))
    idxs = np.floor(np.linspace(delta-1, len(scores)-delta, resolution-2)+0.5).astype(np.int32)
    thresholds = np.zeros((resolution))
    thresholds[0] = - np.inf
    thresholds[-1] = np.inf
    thresholds[1:-1] = scores[idxs]
    return thresholds

@jit(nopython=True)
def calculate_f1(overlaps, score, bound, thresholds, N):
    overlaps = np.array(overlaps)
    overlaps[np.isnan(overlaps)] = 0
    score = np.array(score)
    score[np.isnan(score)] = 0
    precision = np.zeros(len(thresholds))
    recall = np.zeros(len(thresholds))
    for i, th in enumerate(thresholds):
        if th == - np.inf:
            idx = score > 0
        else:
            idx = score >= th
        if np.sum(idx) == 0:
            precision[i] = 1
            recall[i] = 0
        else:
            precision[i] = np.mean(overlaps[idx])
            recall[i] = np.sum(overlaps[idx]) / N
    f1 = 2 * precision * recall / (precision + recall)
    return f1, precision, recall

@jit(nopython=True)
def calculate_expected_overlap(fragments, fweights):
    max_len = fragments.shape[1]
    expected_overlaps = np.zeros((max_len), np.float32)
    expected_overlaps[0] = 1

    # TODO Speed Up 
    for i in range(1, max_len):
        mask = np.logical_not(np.isnan(fragments[:, i]))
        if np.any(mask):
            fragment = fragments[mask, 1:i+1]
            seq_mean = np.sum(fragment, 1) / fragment.shape[1]
            expected_overlaps[i] = np.sum(seq_mean *
                fweights[mask]) / np.sum(fweights[mask])
    return expected_overlaps
