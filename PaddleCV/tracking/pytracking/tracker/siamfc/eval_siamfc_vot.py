import os
import numpy as np

from PIL import Image

import os.path as osp
import sys
CURRENT_DIR = osp.dirname(__file__)
sys.path.append(osp.join(CURRENT_DIR, '..', '..', '..'))

import json
from pytracking.tracker.siamfc.siamfc import SiamFC

from tqdm import tqdm

from pytracking.parameter.siamfc.default import parameters


class ValidVOT(SiamFC):
    def __init__(self, dataset_root, dataset_name, params):
        super(ValidVOT, self).__init__(params)
        """
        dataset_root: the root directory of dataset
        dataset_name: the name of VOT dataste, [VOt2015, VOT2018, ...]
        """
        self.params = self.params
        self.root_path = dataset_root
        if not os.path.exists(self.root_path):
            raise Exception("'{}' does not exists".format(self.root_path))

        dataset_list = ['VOT2015', 'VOT2018']
        if dataset_name not in dataset_list:
            raise Exception("ValidVOT's dataset_name can only be one of {}".
                            format(dataset_list))

        self.dataset_name = dataset_name
        self.vot2013_json = os.path.join(self.root_path, dataset_name + '.json')
        # self.otb2013_json = "/paddle/Datasets/OTB100/CVPR13.json"

        self.meta_data = json.load(open(self.vot2013_json, 'rb'))
        self.video_name = list(self.meta_data.keys())

    def inference_reinit(self, epoch, start_frame=0):

        # video-wised
        vid_num = len(self.video_name)
        vid_ious = np.zeros(vid_num)
        vid_length = np.zeros(vid_num)
        fail_num = np.zeros(vid_num)

        burn_in_period = 5
        pbar = tqdm(
            self.meta_data.keys(),
            desc='loading ' + self.dataset_name,
            ncols=100)

        for idx, vid in enumerate(pbar):
            pbar.set_postfix_str(vid)

            gt_boxs = self.meta_data[vid]['gt_rect']
            img_list = self.meta_data[vid]['img_names']
            imgs_num = len(img_list)

            gt_box_list = []
            pre_box_list = []

            valid_frames_num = imgs_num - start_frame
            step = start_frame
            reinit = True
            re_init_frame = step
            while step < imgs_num:
                img = Image.open(os.path.join(self.root_path, img_list[step]))

                gt_box = list(map(float, self.region_to_bbox(gt_boxs[step])))

                if reinit:
                    # the tracker was initialized
                    # five frames after the failure
                    self.initialize(img, gt_box)
                    reinit = False
                    # print("reinit, vid: {}, step: {}, failnum: {}".format(vid, step, fail_num[idx]))
                    continue

                pre_box = self.track(img)
                if step - re_init_frame < 10:
                    # burn in period
                    step += 1
                    valid_frames_num -= 1
                    continue

                pre_box_list.append(list(pre_box))
                gt_box_list.append(gt_box)

                iou = self._compute_iou(pre_box, gt_box)
                vid_ious[idx] += iou

                if iou == 0.:
                    reinit = True

                    fail_num[idx] += 1
                    # the tracker was initialized
                    # five frames after the failure
                    step += burn_in_period
                    re_init_frame = step
                    valid_frames_num -= burn_in_period
                step += 1

            vid_length[idx] = valid_frames_num
            #print("idx: {}, vid: {}, failure: {}, miou: {}\n".format(idx, vid, fail_num[idx],
            #                                                         vid_ious[idx]/valid_frames_num))

        acc = np.sum(vid_ious) / np.sum(vid_length)
        print("##########Evaluation##########")
        print("##acc = {}".format(acc))
        print("##failure = {}".format(np.sum(fail_num)))

        return acc, np.sum(fail_num)

    def _compute_iou(self, box1, box2):
        """
        computing IoU
        print("acc shape", acc.shape, "vid_length shape: ", vid_length.shape)
        print("acc shape", acc.shape, "vid_length shape: ", vid_length.shape)
        :param rec1: (x0, y0, w, h), which reflects
                (top, left, bottom, right)
        :param rec2: (x0, y0, w, h)
        :return: scala value of IoU
        """
        rec1 = box1
        rec2 = box2
        # computing area of each rectangles
        S_rec1 = (rec1[2] + 1) * (rec1[3] + 1)
        S_rec2 = (rec2[2] + 1) * (rec2[3] + 1)

        # computing the sum_area
        sum_area = S_rec1 + S_rec2

        # find the each edge of intersect rectangle
        left_line = max(rec1[1], rec2[1])
        right_line = min(rec1[3] + rec1[1], rec2[3] + rec2[1])
        top_line = max(rec1[0], rec2[0])
        bottom_line = min(rec1[2] + rec2[0], rec2[2] + rec2[0])

        # judge if there is an intersect
        if left_line >= right_line or top_line >= bottom_line:
            iou = 0.
        else:
            intersect = (right_line - left_line + 1) * (
                bottom_line - top_line + 1)
            iou = (intersect / (sum_area - intersect)) * 1.0
        assert iou >= 0
        assert iou <= 1.01
        return iou

    def region_to_bbox(self, region, center=False):

        n = len(region)
        region = np.array(region)
        assert n == 4 or n == 8, (
            'GT region format is invalid, should have 4 or 8 entries.')

        # we assume the grountruth bounding boxes are saved with 0-indexing
        def _rect(region, center):

            if center:
                x = region[0]
                y = region[1]
                w = region[2]
                h = region[3]
                cx = x + w / 2
                cy = y + h / 2
                return cx, cy, w, h
            else:
                region[0] -= 1
                region[1] -= 1
                return region

        def _poly(region, center):
            cx = np.mean(region[::2])
            cy = np.mean(region[1::2])
            x1 = np.min(region[::2])
            x2 = np.max(region[::2])
            y1 = np.min(region[1::2])
            y2 = np.max(region[1::2])
            A1 = np.linalg.norm(region[0:2] - region[2:4]) * np.linalg.norm(
                region[2:4] - region[4:6])
            A2 = (x2 - x1) * (y2 - y1)
            s = np.sqrt(A1 / A2)
            w = s * (x2 - x1) + 1
            h = s * (y2 - y1) + 1

            if center:
                return cx, cy, w, h
            else:
                return cx - w / 2, cy - h / 2, w, h

        if n == 4:
            return _rect(region, center)
        else:
            return _poly(region, center)


import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
    '--checkpoint',
    type=str,
    default="./checkpoint/",
    help="the path of saved siamfc params file")
parser.add_argument(
    '--dataset_dir',
    type=str,
    default="/paddle/Datasets/VOT2015",
    help="the path of VOT dataset")
parser.add_argument(
    '--dataset_name',
    type=str,
    default="VOT2015",
    help="can only be one of [VOT2015, VOT2018]")

parser.add_argument(
    '--start_epoch',
    type=int,
    default=1,
    help="evaluate from start_epoch epoch, greater than 1")
parser.add_argument(
    '--end_epoch',
    type=int,
    default=50,
    help="evaluate ends at end_epoch epoch, smaller than 50 ")

args = parser.parse_args()

if __name__ == '__main__':

    params = parameters()
    params.net_path = args.checkpoint
    start_epoch = args.start_epoch
    end_epoch = args.end_epoch

    assert start_epoch >= 1 and end_epoch <= 50 and start_epoch < end_epoch

    best_acc, best_failure, best_epoch = 0, 100, start_epoch

    for i in range(start_epoch, end_epoch, 2):
        params.net_path = os.path.join(args.checkpoint, "SiamNet_ep%004d" % i)
        valid = ValidVOT(
            dataset_root=args.dataset_dir,
            dataset_name=args.dataset_name,
            params=params)

        acc, failure = valid.inference_reinit(epoch=i)
        print("####Epoch: {}, ACC: {}, Failure: {}".format(i, acc, failure))
        if acc > best_acc and failure <= 84:
            best_acc = acc
            best_epoch = i
            print("####Best ACC: {}, Failure: {}, corresponding epoch: {}".
                  format(best_acc, failure, best_epoch))
