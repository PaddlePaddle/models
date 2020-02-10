import os
import numpy as np

from glob import glob
from tqdm import tqdm
from colorama import Style, Fore

from ..utils import determine_thresholds, calculate_accuracy, calculate_f1

class F1Benchmark:
    def __init__(self, dataset):
        """
        Args:
            result_path:
        """
        self.dataset = dataset

    def eval(self, eval_trackers=None):
        """
        Args:
            eval_tags: list of tag
            eval_trackers: list of tracker name
        Returns:
            eao: dict of results
        """
        if eval_trackers is None:
            eval_trackers = self.dataset.tracker_names
        if isinstance(eval_trackers, str):
            eval_trackers = [eval_trackers]

        ret = {}
        for tracker_name in eval_trackers:
            precision, recall, f1 = self._cal_precision_reall(tracker_name)
            ret[tracker_name] = {"precision": precision,
                                 "recall": recall,
                                 "f1": f1
                                }
        return ret

    def _cal_precision_reall(self, tracker_name):
        score = []
        # for i in range(len(self.dataset)):
        #     video = self.dataset[i]
        for video in self.dataset:
            if tracker_name not in video.confidence:
                score += video.load_tracker(self.dataset.tracker_path, tracker_name, False)[1]
            else:
                score += video.confidence[tracker_name]
        score = np.array(score)
        thresholds = determine_thresholds(score)[::-1]

        precision = {}
        recall = {}
        f1 = {}
        for i in range(len(self.dataset)):
            video = self.dataset[i]
            gt_traj = video.gt_traj
            N = sum([1 for x in gt_traj if len(x) > 1])
            if tracker_name not in video.pred_trajs:
                tracker_traj, score = video.load_tracker(self.dataset.tracker_path, tracker_name, False)
            else:
                tracker_traj = video.pred_trajs[tracker_name]
                score = video.confidence[tracker_name]
            overlaps = calculate_accuracy(tracker_traj, gt_traj, \
                    bound=(video.width,video.height))[1]
            f1[video.name], precision[video.name], recall[video.name] = \
                    calculate_f1(overlaps, score, (video.width,video.height),thresholds, N)
        return precision, recall, f1

    def show_result(self, result, show_video_level=False, helight_threshold=0.5):
        """pretty print result
        Args:
            result: returned dict from function eval
        """
        # sort tracker according to f1
        sorted_tracker = {}
        for tracker_name, ret in result.items():
            precision = np.mean(list(ret['precision'].values()), axis=0)
            recall = np.mean(list(ret['recall'].values()), axis=0)
            f1 = 2 * precision * recall / (precision + recall)
            max_idx = np.argmax(f1)
            sorted_tracker[tracker_name] = (precision[max_idx], recall[max_idx],
                    f1[max_idx])
        sorted_tracker_ = sorted(sorted_tracker.items(),
                                 key=lambda x:x[1][2],
                                 reverse=True)[:20]
        tracker_names = [x[0] for x in sorted_tracker_]

        tracker_name_len = max((max([len(x) for x in result.keys()])+2), 12)
        header = "|{:^"+str(tracker_name_len)+"}|{:^11}|{:^8}|{:^7}|"
        header = header.format('Tracker Name',
                'Precision', 'Recall', 'F1')
        bar = '-' * len(header)
        formatter = "|{:^"+str(tracker_name_len)+"}|{:^11.3f}|{:^8.3f}|{:^7.3f}|"
        print(bar)
        print(header)
        print(bar)
        # for tracker_name, ret in result.items():
        #     precision = np.mean(list(ret['precision'].values()), axis=0)
        #     recall = np.mean(list(ret['recall'].values()), axis=0)
        #     f1 = 2 * precision * recall / (precision + recall)
        #     max_idx = np.argmax(f1)
        for tracker_name in tracker_names:
            precision = sorted_tracker[tracker_name][0]
            recall = sorted_tracker[tracker_name][1]
            f1 = sorted_tracker[tracker_name][2]
            print(formatter.format(tracker_name, precision, recall, f1))
        print(bar)

        if show_video_level and len(result) < 10:
            print('\n\n')
            header1 = "|{:^14}|".format("Tracker name")
            header2 = "|{:^14}|".format("Video name")
            for tracker_name in result.keys():
                # col_len = max(20, len(tracker_name))
                header1 += ("{:^28}|").format(tracker_name)
                header2 += "{:^11}|{:^8}|{:^7}|".format("Precision", "Recall", "F1")
            print('-'*len(header1))
            print(header1)
            print('-'*len(header1))
            print(header2)
            print('-'*len(header1))
            videos = list(result[tracker_name]['precision'].keys())
            for video in videos:
                row = "|{:^14}|".format(video)
                for tracker_name in result.keys():
                    precision = result[tracker_name]['precision'][video]
                    recall = result[tracker_name]['recall'][video]
                    f1 = result[tracker_name]['f1'][video]
                    max_idx = np.argmax(f1)
                    precision_str = "{:^11.3f}".format(precision[max_idx])
                    if precision[max_idx] < helight_threshold:
                        row += f'{Fore.RED}{precision_str}{Style.RESET_ALL}|'
                    else:
                        row += precision_str+'|'
                    recall_str = "{:^8.3f}".format(recall[max_idx])
                    if recall[max_idx] < helight_threshold:
                        row += f'{Fore.RED}{recall_str}{Style.RESET_ALL}|'
                    else:
                        row += recall_str+'|'
                    f1_str = "{:^7.3f}".format(f1[max_idx])
                    if f1[max_idx] < helight_threshold:
                        row += f'{Fore.RED}{f1_str}{Style.RESET_ALL}|'
                    else:
                        row += f1_str+'|'
                print(row)
            print('-'*len(header1))
