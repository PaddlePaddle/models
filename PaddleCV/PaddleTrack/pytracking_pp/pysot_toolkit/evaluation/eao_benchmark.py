import os
import time
import numpy as np

from glob import glob

from ..utils import calculate_failures, calculate_accuracy, calculate_expected_overlap

class EAOBenchmark:
    """
    Args:
        dataset:
    """
    def __init__(self, dataset, skipping=5, tags=['all']):
        self.dataset = dataset
        self.skipping = skipping
        self.tags = tags
        # NOTE we not use gmm to generate low, high, peak value
        if dataset.name == 'VOT2019':
            self.low = 46
            self.high = 291
            self.peak = 128
        elif dataset.name == 'VOT2018' or dataset.name == 'VOT2017':
            self.low = 100
            self.high = 356
            self.peak = 160
        elif dataset.name == 'VOT2016':
            self.low = 108 #TODO
            self.high = 371
            self.peak = 168
        elif dataset.name == 'VOT2015':
            self.low = 114
            self.high = 378
            self.peak = 176

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
            eao = self._calculate_eao(tracker_name, self.tags)
            ret[tracker_name] = eao
        return ret

    def show_result(self, result, topk=10):
        """pretty print result
        Args:
            result: returned dict from function eval
        """
        if len(self.tags) == 1:
            tracker_name_len = max((max([len(x) for x in result.keys()])+2), 12)
            header = ("|{:^"+str(tracker_name_len)+"}|{:^10}|").format('Tracker Name', 'EAO')
            bar = '-'*len(header)
            formatter = "|{:^20}|{:^10.3f}|"
            print(bar)
            print(header)
            print(bar)
            tracker_eao = sorted(result.items(), 
                                 key=lambda x: x[1]['all'], 
                                 reverse=True)[:topk]
            for tracker_name, eao in tracker_eao:
            # for tracker_name, ret in result.items():
                print(formatter.format(tracker_name, eao['all']))
            print(bar)
        else:
            header = "|{:^20}|".format('Tracker Name')
            header += "{:^7}|{:^15}|{:^14}|{:^15}|{:^13}|{:^11}|{:^7}|".format(*self.tags)
            bar = '-'*len(header)
            formatter = "{:^7.3f}|{:^15.3f}|{:^14.3f}|{:^15.3f}|{:^13.3f}|{:^11.3f}|{:^7.3f}|"
            print(bar)
            print(header)
            print(bar)
            sorted_tacker = sorted(result.items(), 
                                   key=lambda x: x[1]['all'],
                                   reverse=True)[:topk]
            sorted_tacker = [x[0] for x in sorted_tacker]
            for tracker_name in sorted_tacker:
            # for tracker_name, ret in result.items():
                print("|{:^20}|".format(tracker_name)+formatter.format(
                    *[result[tracker_name][x] for x in self.tags]))
            print(bar)

    def _calculate_eao(self, tracker_name, tags):
        eao = {}
        for traj_idx in range(16):
            all_overlaps = []
            all_failures = []
            video_names = []
            gt_traj_length = []
            # for i in range(len(self.dataset)):
            for video in self.dataset:
                # video = self.dataset[i]
                gt_traj = video.gt_traj
                if tracker_name not in video.pred_trajs:
                    tracker_trajs = video.load_tracker(self.dataset.tracker_path, tracker_name, False)
                else:
                    tracker_trajs = video.pred_trajs[tracker_name]

                if traj_idx < len(tracker_trajs):
                    suffix = '_{}'.format(traj_idx)
                    tracker_trajs = tracker_trajs[traj_idx:traj_idx+1]
                elif traj_idx == len(tracker_trajs):
                    suffix = ''
                    tracker_trajs = tracker_trajs
                else:
                    return eao

                for tracker_traj in tracker_trajs:
                    gt_traj_length.append(len(gt_traj))
                    video_names.append(video.name)
                    overlaps = calculate_accuracy(tracker_traj, gt_traj, bound=(video.width-1, video.height-1))[1]
                    failures = calculate_failures(tracker_traj)[1]
                    all_overlaps.append(overlaps)
                    all_failures.append(failures)
            fragment_num = sum([len(x)+1 for x in all_failures])
            max_len = max([len(x) for x in all_overlaps])
            if len(tracker_trajs) == 0:
                print('Warning: some seqs in {}.{} not found'.format(tracker_name, tags))
            seq_weight = 1 / (len(tracker_trajs) + 1e-10)  # division by zero
            for tag in tags:
                # prepare segments
                fweights = np.ones((fragment_num)) * np.nan
                fragments = np.ones((fragment_num, max_len)) * np.nan
                seg_counter = 0
                for name, traj_len, failures, overlaps in zip(video_names, gt_traj_length,
                        all_failures, all_overlaps):
                    if len(failures) > 0:
                        points = [x+self.skipping for x in failures if
                                x+self.skipping <= len(overlaps)]
                        points.insert(0, 0)
                        for i in range(len(points)):
                            if i != len(points) - 1:
                                fragment = np.array(overlaps[points[i]:points[i+1]+1])
                                fragments[seg_counter, :] = 0
                            else:
                                fragment = np.array(overlaps[points[i]:])
                            fragment[np.isnan(fragment)] = 0
                            fragments[seg_counter, :len(fragment)] = fragment
                            if i != len(points) - 1:
                                # tag_value = self.dataset[name].tags[tag][points[i]:points[i+1]+1]
                                tag_value = self.dataset[name].select_tag(tag, points[i], points[i+1]+1)
                                w = sum(tag_value) / (points[i+1] - points[i]+1)
                                fweights[seg_counter] = seq_weight * w
                            else:
                                # tag_value = self.dataset[name].tags[tag][points[i]:len(overlaps)]
                                tag_value = self.dataset[name].select_tag(tag, points[i], len(overlaps))
                                w = sum(tag_value) / (traj_len - points[i]+1e-16)
                                fweights[seg_counter] = seq_weight * w  # (len(fragment) / (traj_len-points[i]))
                            seg_counter += 1
                    else:
                        # no failure
                        max_idx = min(len(overlaps), max_len)
                        fragments[seg_counter, :max_idx] = overlaps[:max_idx]
                        # tag_value = self.dataset[name].tags[tag][:max_idx]
                        tag_value = self.dataset[name].select_tag(tag, 0, max_idx)
                        w = sum(tag_value) / max_idx
                        fweights[seg_counter] = seq_weight * w
                        seg_counter += 1

                expected_overlaps = calculate_expected_overlap(fragments, fweights)
                # caculate eao
                weight = np.zeros((len(expected_overlaps)))
                weight[self.low-1:self.high-1+1] = 1
                is_valid = np.logical_not(np.isnan(expected_overlaps))
                eao_ = np.sum(expected_overlaps[is_valid] * weight[is_valid]) / np.sum(weight[is_valid])
                eao[tag + suffix] = eao_
        return eao
