from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import paddle.fluid
import argparse
import importlib
import os
import os.path as osp
import pickle
import sys
from glob import glob

import cv2 as cv
import numpy as np
from tqdm import tqdm

CURRENT_DIR = osp.dirname(__file__)
sys.path.append(osp.join(CURRENT_DIR, '..'))

from pytracking.admin.environment import env_settings
from pytracking.pysot_toolkit.pysot.datasets import DatasetFactory
from pytracking.pysot_toolkit.pysot.evaluation import EAOBenchmark, AccuracyRobustnessBenchmark, OPEBenchmark
from pytracking.pysot_toolkit.pysot.utils.region import vot_overlap

parser = argparse.ArgumentParser(description='tracking evaluation')

parser.add_argument('--dataset', '-d', type=str, help='dataset name')
parser.add_argument(
    '--training_base_param', '-tr', type=str, help='training base params name')
parser.add_argument('--epoch', '-e', type=str, help='epoch specifications')
parser.add_argument(
    '--tracking_base_param', '-te', type=str, help='tracking base params name')
parser.add_argument(
    '--num_repeat', '-n', default=1, type=int, help='number of repeat')
parser.add_argument(
    '--exp_id', '-ex', default='', type=str, help='experiment id')

args = parser.parse_args()


def read_image(x):
    if isinstance(x, str):
        img = cv.imread(x)
    else:
        img = x
    return cv.cvtColor(img, cv.COLOR_BGR2RGB)


def get_tracker_params(param_module, params):
    tracker_params = param_module.parameters()
    tracker_params.debug = 0  # disable debug
    # change checkpoint path
    tracker_params.features.features[0].net_path = params['checkpoint']
    return tracker_params


def create_tracker(params):
    base_param = params['tracking_base_param']
    base_tracker = base_param.split('.')[0]
    param_module = importlib.import_module('pytracking.parameter.{}'.format(
        base_param))
    tracker_params = get_tracker_params(param_module, params)
    tracker_module = importlib.import_module('pytracking.tracker.{}'.format(
        base_tracker))
    tracker_class = tracker_module.get_tracker_class()
    return tracker_class(tracker_params)


def get_axis_aligned_bbox(region):
    region = np.array(region)
    if len(region.shape) == 3:
        # region (1,4,2)
        region = np.array([
            region[0][0][0], region[0][0][1], region[0][1][0], region[0][1][1],
            region[0][2][0], region[0][2][1], region[0][3][0], region[0][3][1]
        ])

    cx = np.mean(region[0::2])
    cy = np.mean(region[1::2])
    x1 = min(region[0::2])

    x2 = max(region[0::2])
    y1 = min(region[1::2])
    y2 = max(region[1::2])

    A1 = np.linalg.norm(region[0:2] - region[2:4]) * np.linalg.norm(region[
        2:4] - region[4:6])
    A2 = (x2 - x1) * (y2 - y1)
    s = np.sqrt(A1 / A2)
    w = s * (x2 - x1) + 1
    h = s * (y2 - y1) + 1

    x11 = cx - w // 2
    y11 = cy - h // 2

    return x11, y11, w, h


def run_tracker(tracker, video, reset=False):
    if reset:
        frame_counter = 0
        pred_bboxes = []
        for idx, (img_p, gt_bbox) in enumerate(video):
            if idx == frame_counter:
                # init your tracker here
                image = read_image(img_p)
                if len(gt_bbox) == 8:
                    init_bbox = get_axis_aligned_bbox(gt_bbox)
                else:
                    init_bbox = gt_bbox
                tracker.initialize(image, init_bbox)
                pred_bboxes.append(1)
            elif idx > frame_counter:
                # get tracking result here
                image = read_image(img_p)
                pred_bbox = tracker.track(image)
                overlap = vot_overlap(pred_bbox, gt_bbox,
                                      (image.shape[1], image.shape[0]))
                if overlap > 0:
                    # continue tracking
                    pred_bboxes.append(pred_bbox)
                else:
                    # lost target, restart
                    pred_bboxes.append(2)
                    frame_counter = idx + 5
            else:
                pred_bboxes.append(0)
    else:
        pred_bboxes = []
        for idx, (img_p, gt_bbox) in enumerate(video):
            if idx == 0:
                # init your tracker here
                image = read_image(img_p)
                if len(gt_bbox) == 8:
                    init_bbox = get_axis_aligned_bbox(gt_bbox)
                else:
                    init_bbox = gt_bbox
                tracker.initialize(image, init_bbox)
                pred_bboxes.append(init_bbox)
            else:
                # get tracking result here
                image = read_image(img_p)
                pred_bbox = tracker.track(image)
                pred_bboxes.append(pred_bbox)
    return pred_bboxes


def run_one_sequence(video, params, tracker=None):
    # idt = multiprocessing.current_process()._identity[0]
    # os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(idt % 4)
    save_dir = osp.join(params['result_dir'], params['save_dataset_name'],
                        params['tracking_base_param'], params['exp_id'])

    if tracker is None:
        tracker = create_tracker(params)

    if 'VOT' in params['dataset_name']:
        save_sub_dir = osp.join(save_dir, 'baseline', video.name)
        os.makedirs(save_sub_dir, exist_ok=True)
        num_repeat = params.get('num_repeat', 1)
        for repeat_idx in range(1, num_repeat + 1):
            save_path = osp.join(save_sub_dir,
                                 video.name + '_{:03d}.txt'.format(repeat_idx))
            if osp.exists(save_path): continue
            pred_bboxes = run_tracker(tracker, video, reset=True)

            # Save tracking results
            with open(save_path, 'w') as f:
                outputs = []
                for res in pred_bboxes:
                    if isinstance(res, int):
                        outputs.append('{}'.format(res))
                    else:
                        if len(res) is 8:
                            outputs.append('{},{},{},{},{},{},{},{}'.format(
                                res[0], res[1], res[2], res[3], res[4], res[5], res[6], res[7]))
                        else:
                            outputs.append('{},{},{},{}'.format(
                                res[0], res[1], res[2], res[3]))
                f.write('\n'.join(outputs))
    else:
        os.makedirs(save_dir, exist_ok=True)
        save_path = osp.join(save_dir, video.name + '.txt')
        if osp.exists(save_path): return
        pred_bboxes = run_tracker(tracker, video, reset=False)

        # Save tracking results
        with open(save_path, 'w') as f:
            outputs = []
            for res in pred_bboxes:
                outputs.append('{},{},{},{}'.format(res[0], res[1], res[2], res[
                    3]))
            f.write('\n'.join(outputs))


def run_one_dataset(dataset, params):
    # use the same tracker for all sequences
    tracker = create_tracker(params)
    # create new tracker for each sequence
    # tracker = None
    for video in tqdm(list(dataset.videos.values())):
        run_one_sequence(video, params, tracker=tracker)


def compute_evaluation_metrics(dataset, params):
    result_dir = osp.join(params['result_dir'], params['save_dataset_name'],
                          params['tracking_base_param'])
    tracker_name = params['exp_id']
    trackers = [tracker_name]
    dataset.set_tracker(result_dir, trackers)

    if 'VOT' in params['dataset_name']:
        ar_benchmark = AccuracyRobustnessBenchmark(dataset)
        ar_result = {}
        ar_result.update(ar_benchmark.eval(trackers))

        eao_benchmark = EAOBenchmark(dataset)
        eao_result = {}
        eao_result.update(eao_benchmark.eval(trackers))

        ar_benchmark.show_result(ar_result, eao_result)
        metrics = {'ar': ar_result, 'eao': eao_result}
    else:
        benchmark = OPEBenchmark(dataset)
        success_result = {}
        precision_result = {}
        success_result.update(benchmark.eval_success(trackers))
        precision_result.update(benchmark.eval_precision(trackers))
        benchmark.show_result(success_result, precision_result)
        metrics = {'success': success_result, 'precision': precision_result}
    return metrics


def save_info(params, metrics):
    save_dir = osp.join(params['result_dir'], params['save_dataset_name'],
                        params['tracking_base_param'], params['exp_id'])
    with open(osp.join(save_dir, 'params.pickle'), 'wb') as f:
        pickle.dump(params, f)

    with open(osp.join(save_dir, 'metrics.txt'), 'w') as f:
        f.write('{}'.format(metrics))


def run_tracking_and_evaluate(params):
    """Receive hyperparameters and return the evaluation metric"""
    # load dataset
    root = os.path.abspath(
        osp.join(env_settings().dataset_path, params['save_dataset_name']))
    dataset = DatasetFactory.create_dataset(
        name=params['dataset_name'], dataset_root=root)

    run_one_dataset(dataset, params)
    metrics = compute_evaluation_metrics(dataset, params)

    return metrics


def get_checkpoint_path(training_base_param, epoch):
    model_dir = osp.abspath(
        osp.join(env_settings().network_path, *training_base_param.split('.')))
    model_names = glob(model_dir + '/*.pdparams')
    prefix = '_'.join(model_names[0].split('_')[:-1])
    return osp.join(model_dir, '{}_ep{:04d}'.format(prefix, epoch))


def parse_epoch(epoch_str):
    epochs = eval(epoch_str)
    try:
        iterator = iter(epochs)
    except:
        if isinstance(epochs, int):
            iterator = [epochs]
        else:
            raise NotImplementedError
    return iterator


def main():
    for epoch in parse_epoch(args.epoch):
        # get checkpoint
        checkpoint_pth = get_checkpoint_path(args.training_base_param, epoch)

        if args.exp_id == '':
            exp_id = args.training_base_param + '.epoch{}'.format(epoch)
        else:
            exp_id = args.exp_id
        print('=> Evaluating: {}'.format(exp_id))

        if args.dataset in ['CVPR13', 'OTB50', 'OTB100']:
            # for OTB datasets, we save results into the same directory
            save_dataset_name = 'OTB100'
        else:
            save_dataset_name = args.dataset

        # set up parameters
        params = {
            'dataset_name': args.dataset,
            'checkpoint': checkpoint_pth,
            'tracking_base_param': args.tracking_base_param,
            'num_repeat': args.num_repeat,
            'exp_id': exp_id,
            'result_dir': env_settings().results_path,
            'save_dataset_name': save_dataset_name,
        }

        metrics = run_tracking_and_evaluate(params)
        save_info(params, metrics)


if __name__ == '__main__':
    main()
