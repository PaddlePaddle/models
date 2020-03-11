import os
import numpy as np

from PIL import Image

import os.path as osp
import sys
CURRENT_DIR = osp.dirname(__file__)
sys.path.append(osp.join(CURRENT_DIR, '..', '..', '..'))

from pytracking.pysot_toolkit.utils import success_overlap, success_error
import json
from pytracking.tracker.siamfc.siamfc import SiamFC

from tqdm import tqdm

from pytracking.parameter.siamfc.default import parameters


class ValidOTB(SiamFC):
    def __init__(self, dataset_root, dataset_name, params):
        super(ValidOTB, self).__init__(params)
        """
        dataset_root: the root directory of dataset
        dataset_name: the name of OTB dataste, [CVPR2013, OTB50, OTB100]
        """
        self.params = self.params
        self.root_path = dataset_root
        if not os.path.exists(self.root_path):
            raise Exception("'{}' does not exists".format(self.root_path))

        dataset_list = ['CVPR13', 'OTB2013', 'OTB100', 'OTB50']
        if dataset_name not in dataset_list:
            raise Exception("ValidOTB's dataset_name can only be one of {}".
                            format(dataset_list))
        if dataset_name == 'OTB2013':
            dataset_name = 'CVPR13'
        self.dataset_name = dataset_name
        self.otb2013_json = os.path.join(self.root_path, dataset_name + '.json')

        self.meta_data = json.load(open(self.otb2013_json, 'rb'))
        self.video_name = list(self.meta_data.keys())

    def inference(self, epoch):

        gtbb = []
        prebb = []
        """ add save dir """
        save_dir = "./eval_otb13/epoch_" + str(epoch)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # load videos
        pbar = tqdm(
            self.meta_data.keys(),
            desc='loading ' + self.dataset_name,
            ncols=100)
        for idx, vid in enumerate(pbar):
            pbar.set_postfix_str(vid)

            gt_boxs = self.meta_data[vid]['gt_rect']
            start_frame, end_frame = 0, len(gt_boxs)
            img_list = self.meta_data[vid]['img_names']
            assert len(img_list) == len(gt_boxs)

            gt_box_list = []
            pre_box_list = []
            for i in range(start_frame, end_frame):
                img = Image.open(os.path.join(self.root_path, img_list[i]))
                if len(img.size) < 3 or img.size[-1] == 1:
                    img = img.convert('RGB')

                gt_box = gt_boxs[i - start_frame]

                if i == start_frame:
                    self.initialize(image=img, state=gt_box)
                    pre_box_list.append(gt_box)
                    gt_box_list.append(gt_box)
                    continue
                else:
                    pre_box = self.track(img)

                    pre_box_list.append(list(pre_box))
                    gt_box_list.append(gt_box)

            gtbb += gt_box_list
            prebb += pre_box_list
            """ add save_dir"""
            vid_save_dir = os.path.join(save_dir, vid + '.txt')
            with open(vid_save_dir, 'w') as f:
                outputs = []
                for res in pre_box_list:
                    outputs.append('{},{},{},{}'.format(res[0], res[1], res[2],
                                                        res[3]))
                f.write('\n'.join(outputs))

        auc = success_overlap(np.array(gtbb), np.array(prebb), len(gtbb))

        thresholds = np.arange(0, 51, 1)
        gt_center = self.convert_bb_to_center(np.array(gtbb))
        tracker_center = self.convert_bb_to_center(np.array(prebb))
        precision = success_error(
            np.array(gt_center),
            np.array(tracker_center), thresholds, len(gtbb))
        print("####AUC:{}, Precision:{}".format(
            np.mean(auc), np.mean(precision)))

        return np.mean(auc), np.mean(precision)

    def convert_bb_to_center(self, bboxes):
        return np.array([(bboxes[:, 0] + (bboxes[:, 2] - 1) / 2),
                         (bboxes[:, 1] + (bboxes[:, 3] - 1) / 2)]).T


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
    default="/paddle/Datasets/OTB100",
    help="the path of OTB dataset")
parser.add_argument(
    '--dataset_name',
    type=str,
    default="CVPR13",
    help="can only be one of [CVPR13, OTB2013, OTB50, OTB100]")

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

    best_auc, best_epoch = 0, start_epoch

    for i in range(start_epoch, end_epoch, 1):
        params.net_path = os.path.join(args.checkpoint, "SiamNet_ep%004d" % i)
        valid = ValidOTB(
            dataset_root=args.dataset_dir,
            dataset_name=args.dataset_name,
            params=params)

        auc, precision = valid.inference(epoch=i)

        if auc > best_auc:
            best_auc = auc
            best_epoch = i
        print("####Best AUC is {}, corresponding epoch is {}".format(
            best_auc, best_epoch))
