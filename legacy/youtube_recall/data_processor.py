#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import argparse
import os
import cPickle

from utils import logger
"""
This script will output 2 files:
1. feature_dict.pkl
2. item_freq.pkl
"""


class FeatureGenerator(object):
    """
    Encode feature values with low-frequency filtering.
    """

    def __init__(self, feat_appear_limit=20):
        """
        @feat_appear_limit: int
        """
        self._dic = None  # feature value --> id
        self._count = None  # numbers of appearances of feature values
        self._feat_appear_limit = feat_appear_limit

    def add_feat_val(self, feat_val):
        """
        Add feature values and count numbers of its appearance. 
        """
        if self._count is None:
            self._count = {'<unk>': 0}
        if feat_val == "NULL":
            feat_val = '<unk>'
        if feat_val not in self._count:
            self._count[feat_val] = 1
        else:
            self._count[feat_val] += 1
            self._count['<unk>'] += 1

    def _filter_feat(self):
        """
        Filter low-frequency feature values.
        """
        self._items = filter(lambda x: x[1] > self._feat_appear_limit,
                             self._count.items())
        self._items.sort(key=lambda x: x[1], reverse=True)

    def _build_dict(self):
        """
        Build feature values --> ids dict.
        """
        self._dic = {}
        self._filter_feat()
        for i in xrange(len(self._items)):
            self._dic[self._items[i][0]] = i
        self.dim = len(self._dic)

    def get_feat_id(self, feat_val):
        """
        Get id of feature value after encoding.
        """
        # build dict
        if self._dic is None:
            self._build_dict()

        # find id
        if feat_val in self._dic:
            return self._dic[feat_val]
        else:
            return self._dic['<unk>']

    def get_dim(self):
        """
        Get dim.
        """
        # build dict
        if self._dic is None:
            self._build_dict()
        return len(self._dic)

    def get_dict(self):
        """
        Get dict.
        """
        # build dict
        if self._dic is None:
            self._build_dict()
        return self._dic

    def get_total_count(self):
        """
        Compute total num of count.
        """
        total_count = 0
        for i in xrange(len(self._items)):
            feat_val = self._items[i][0]
            c = self._items[i][1]
            total_count += c
        return total_count

    def count_iterator(self):
        """
        Iterate feature values and its num of appearance.
        """
        for i in xrange(len(self._items)):
            yield self._items[i][0], self._items[i][1]

    def __repr__(self):
        """
        """
        return '<FeatureGenerator %d>' % self._dim


def scan_build_dict(data_path, features_dict):
    """
    Scan the raw data and add all feature values.
    """
    logger.info('scan data set')

    with open(data_path, 'r') as f:
        for (line_id, line) in enumerate(f):
            fields = line.strip('\n').split('\t')
            user_id = fields[0]
            province = fields[1]
            features_dict['province'].add_feat_val(province)
            city = fields[2]
            features_dict['city'].add_feat_val(city)
            item_infos = fields[3]
            phone = fields[4]
            features_dict['phone'].add_feat_val(phone)
            for item_info in item_infos.split(";"):
                item_info_array = item_info.split(":")
                item = item_info_array[0]
                features_dict['history_clicked_items'].add_feat_val(item)
                features_dict['user_id'].add_feat_val(user_id)
                category = item_info_array[1]
                features_dict['history_clicked_categories'].add_feat_val(
                    category)
                tags = item_info_array[2]
                for tag in tags.split("_"):
                    features_dict['history_clicked_tags'].add_feat_val(tag)


def parse_args():
    """
    parse arguments
    """
    parser = argparse.ArgumentParser(
        description="PaddlePaddle Youtube Recall Model Example")
    parser.add_argument(
        '--train_set_path',
        type=str,
        required=True,
        help="path of the train set")
    parser.add_argument(
        '--output_dir', type=str, required=True, help="directory to output")
    parser.add_argument(
        '--feat_appear_limit',
        type=int,
        default=20,
        help="the minimum number of feature values appears (default: 20)")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # check argument
    assert os.path.exists(
        args.train_set_path), 'The train set path does not exist.'

    # features used
    features = [
        'user_id', 'province', 'city', 'phone', 'history_clicked_items',
        'history_clicked_tags', 'history_clicked_categories'
    ]

    # init feature generators
    features_dict = {}
    for feature in features:
        features_dict[feature] = FeatureGenerator(
            feat_appear_limit=args.feat_appear_limit)

    # scan data for building dict
    scan_build_dict(args.train_set_path, features_dict)

    # generate feature_dict.pkl
    feature_encoding_dict = {}
    for feature in features:
        d = features_dict[feature].get_dict()
        feature_encoding_dict[feature] = d
        logger.info('Feature:%s, dimension is %d' % (feature, len(d)))
    output_dict_path = os.path.join(args.output_dir, 'feature_dict.pkl')
    with open(output_dict_path, "w") as f:
        cPickle.dump(feature_encoding_dict, f, -1)

    # generate item_freq.pkl
    item_freq_list = []
    g = features_dict['history_clicked_items']
    total_count = g.get_total_count()
    for feat_val, feat_count in g.count_iterator():
        item_freq_list.append(float(feat_count) / total_count)
    logger.info('item_freq, dimension is %d' % (len(item_freq_list)))
    output_item_freq_path = os.path.join(args.output_dir, 'item_freq.pkl')
    with open(output_item_freq_path, "w") as f:
        cPickle.dump(item_freq_list, f, -1)

    logger.info('Complete!')
