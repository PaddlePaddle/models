# -*- coding: utf-8 -*-
# @Author : W
# @Time : 2019/6/21
# @File : iou.py
# @Software: PyCharm

import numpy as np


class IOUMetric:
    """
    Class to calculate mean-iou using fast_hist method
    调用方式：
    iou = IOUMetric(num_classes=21)
    for ...test_loader:
        iou.add_batch(pred_np,label_np)
    iou.evaluate()
    """

    def __init__(self, num_classes):
        self.num_classes = num_classes+1
        self.hist = np.zeros((num_classes+1, num_classes+1))
        # 混淆矩阵：
        """
            row: 真值
            col: 预测值
            对角线即预测正确的个数
            iou = 重叠 /（行和 + 列和 - 重叠）
            
            例如：
            样例一：
                pred_np = np.array([[[0, 2, 2, 2],  [2, 1, 1, 2]]])
                label_np = np.array([[[2, 1, 2, 255], [2, 1, 1, 255]]])
                
               [[0. 0. 0. 0.]       label中有0个像素点（行和）是第0类，预测有0个像素的是第0类
                [0. 2. 1. 0.]       label中有3个像素点（行和）是第1类；预测有2个像素的是第1类，有一个像素点是第2类
                [1. 0. 2. 0.]       label中有3个像素点（行和）是第2类；预测有2个像素的是第2类，有一个像素点是第0类
                [0. 0. 0. 2.]]      label中有2个像素点（行和）是第3类，预测有2个像素的是第3类，注意该类是忽略类别，不参与iou
            
            iou = [0,  0.66666667,  0.5  , 1]
                  0/1,     2/3   ,  2/4  , 2/2
            
            样例二：
                pred_np = np.array([[[0, 0, 2, 2],  [2, 1, 1, 2]]])
                label_np = np.array([[[0, 1, 2, 255], [2, 1, 1, 255]]])
               [[1. 0. 0. 0.]       label中有1个像素点（行和）是第0类；预测有1个像素的是第0类
                [1. 2. 0. 0.]       label中有3个像素点（行和）是第1类；预测有2个像素的是第1类，有一个像素点是第0类
                [0. 0. 2. 0.]       label中有2个像素点（行和）是第2类；预测有2个像素的是第2类
                [0. 0. 0. 2.]]      label中有2个像素点（行和）是第3类，预测有2个像素的是第3类，注意该类是忽略类别，不参与iou
            
            iou = [0.5,  0.66666667,  1  ,  1]
                   1/2,     2/3    , 2/2 ,  2/2    
        """

    def _fast_hist(self, label_pred, label_true):
        mask = (label_true >= 0) & (label_true < self.num_classes)
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) +
            label_pred[mask], minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        return hist

    def add_batch(self, predictions, gts):
        # gts = BHW
        # predictions = BHW
        if isinstance(gts, np.ndarray):
            gts_ig = (gts == 255).astype(np.int32)
            gts_nig = (gts != 255).astype(np.int32)
            # print(predictions)
            gts[gts == 255] = self.num_classes-1  # 19
            predictions = gts_nig * predictions + gts_ig * (self.num_classes-1)
            # print(predictions)
        for lp, lt in zip(predictions, gts):
            self.hist += self._fast_hist(lp.flatten(), lt.flatten())

    def evaluate(self):
        # diag = np.diag(self.hist)
        # total = self.hist.sum()
        # col = self.hist.sum(axis=0)
        # row = self.hist.sum(axis=1)

        acc = np.diag(self.hist).sum() / self.hist.sum()
        acc_cls = np.nanmean(np.diag(self.hist) / self.hist.sum(axis=1))
        iu = np.diag(self.hist) / (self.hist.sum(axis=1) + self.hist.sum(axis=0) - np.diag(self.hist))
        mean_iu = np.nanmean(iu)
        freq = self.hist.sum(axis=1) / self.hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        kappa = (self.hist.sum() * np.diag(self.hist).sum() - (self.hist.sum(axis=0) * self.hist.sum(axis=1)).sum()) / (
                self.hist.sum() ** 2 - (self.hist.sum(axis=0) * self.hist.sum(axis=1)).sum())
        return acc, acc_cls, iu, mean_iu, fwavacc, kappa

    def evaluate_kappa(self):
        kappa = (self.hist.sum() * np.diag(self.hist).sum() - (self.hist.sum(axis=0) * self.hist.sum(axis=1)).sum()) / (
                self.hist.sum() ** 2 - (self.hist.sum(axis=0) * self.hist.sum(axis=1)).sum())
        return kappa

    def evaluate_iou_kappa(self):
        iu = np.diag(self.hist) / (self.hist.sum(axis=1) + self.hist.sum(axis=0) - np.diag(self.hist))
        mean_iu = np.nanmean(iu)
        kappa = (self.hist.sum() * np.diag(self.hist).sum() - (self.hist.sum(axis=0) * self.hist.sum(axis=1)).sum()) / (
                self.hist.sum() ** 2 - (self.hist.sum(axis=0) * self.hist.sum(axis=1)).sum())
        return mean_iu, kappa

    def evaluate_iu(self):
        iu = np.diag(self.hist) / (self.hist.sum(axis=1) + self.hist.sum(axis=0) - np.diag(self.hist))
        return iu


if __name__ == '__main__':
    pred_np = np.array([[[0, 0, 2, 2],  [2, 1, 1, 2]]])
    label_np = np.array([[[0, 1, 2, 255], [2, 1, 1, 255]]])
    # label_np = np.array([[[2, 0, 0, 255], [2, 1, 1, 255]]])
    print(pred_np.shape)
    iou = IOUMetric(3)
    iou.add_batch(pred_np, label_np)
    print(iou.hist)
    acc, acc_cls, iu, mean_iu, fwavacc, kappa = iou.evaluate()
    print('acc = {}'.format(acc))
    print('acc_cls = {}'.format(acc_cls))
    print('iu = {}'.format(iu))
    print('mean_iu = {}'.format(mean_iu))
    print('mean_iu = {}'.format(np.nanmean(iu[:-1])))  # 真正的iou
    print('fwavacc = {}'.format(fwavacc))
    print('kappa = {}'.format(kappa))
