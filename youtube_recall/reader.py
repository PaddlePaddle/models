#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys

from utils import logger
from utils import TaskMode


class Reader(object):
    """
    Reader
    """

    def __init__(self, feature_dict=None, window_size=20):
        """
        init
        @window_size: window_size
        """
        self._feature_dict = feature_dict
        self._window_size = window_size

    def train(self, path):
        """
        load train set
        @path: train set path
        """
        logger.info("start train reader from %s" % path)
        mode = TaskMode.create_train()
        return self._reader(path, mode)

    def test(self, path):
        """
        load test set
        @path: test set path
        """
        logger.info("start test reader from %s" % path)
        mode = TaskMode.create_test()
        return self._reader(path, mode)

    def infer(self, path):
        """
        load infer set
        @path: infer set path
        """
        logger.info("start infer reader from %s" % path)
        mode = TaskMode.create_infer()
        return self._reader(path, mode)

    def infer_user(self, user_list):
        """
        load user set to infer
        @user_list: user list
        """
        return self._reader_user(user_list)

    def _reader(self, path, mode):
        """
        parse data set
        """
        USER_ID_UNK = self._feature_dict['user_id'].get('<unk>')
        PROVINCE_UNK = self._feature_dict['province'].get('<unk>')
        CITY_UNK = self._feature_dict['city'].get('<unk>')
        ITEM_UNK = self._feature_dict['history_clicked_items'].get('<unk>')
        CATEGORY_UNK = self._feature_dict['history_clicked_categories'].get(
            '<unk>')
        TAG_UNK = self._feature_dict['history_clicked_tags'].get('<unk>')
        PHONE_UNK = self._feature_dict['phone'].get('<unk>')
        with open(path) as f:
            for line in f:
                fields = line.strip('\n').split('\t')
                user_id = self._feature_dict['user_id'].get(fields[0],
                                                            USER_ID_UNK)
                province = self._feature_dict['province'].get(fields[1],
                                                              PROVINCE_UNK)
                city = self._feature_dict['city'].get(fields[2], CITY_UNK)
                item_infos = fields[3]
                phone = self._feature_dict['phone'].get(fields[4], PHONE_UNK)
                history_clicked_items_all = []
                history_clicked_tags_all = []
                history_clicked_categories_all = []
                for item_info in item_infos.split(';'):
                    item_info_array = item_info.split(':')
                    item = item_info_array[0]
                    item_encoded_id = self._feature_dict['history_clicked_items'].get(\
                            item, ITEM_UNK)
                    if item_encoded_id != ITEM_UNK:
                        history_clicked_items_all.append(item_encoded_id)
                        category = item_info_array[1]
                        history_clicked_categories_all.append(
                                self._feature_dict['history_clicked_categories'].get(\
                                        category, CATEGORY_UNK))
                        tags = item_info_array[2]
                        tag_split = map(str, [self._feature_dict['history_clicked_tags'].get(\
                                tag, TAG_UNK) \
                                for tag in tags.strip().split("_")])
                        history_clicked_tags_all.append("_".join(tag_split))

                if not mode.is_infer():
                    history_clicked_items_all.insert(0, 0)
                    history_clicked_tags_all.insert(0, "0")
                    history_clicked_categories_all.insert(0, 0)

                    for i in range(1, len(history_clicked_items_all)):
                        start = max(0, i - self._window_size)
                        history_clicked_items = history_clicked_items_all[start:
                                                                          i]
                        history_clicked_categories = history_clicked_categories_all[
                            start:i]
                        history_clicked_tags_str = history_clicked_tags_all[
                            start:i]
                        history_clicked_tags = []
                        for tags_a in history_clicked_tags_str:
                            for tag in tags_a.split("_"):
                                history_clicked_tags.append(int(tag))
                        target_item = history_clicked_items_all[i]
                        yield user_id, province, city, \
                              history_clicked_items, history_clicked_categories, \
                              history_clicked_tags, phone, target_item
                else:
                    history_clicked_items = history_clicked_items_all
                    history_clicked_categories = history_clicked_categories_all
                    history_clicked_tags_str = history_clicked_tags_all
                    history_clicked_tags = []
                    for tags_a in history_clicked_tags_str:
                        for tag in tags_a.split("_"):
                            history_clicked_tags.append(int(tag))
                    yield user_id, province, city, \
                          history_clicked_items, history_clicked_categories, \
                          history_clicked_tags, phone

    def _reader_user(self, user_list):
        """
        parse user list
        """
        USER_ID_UNK = self._feature_dict['user_id'].get('<unk>')
        for user in user_list:
            user_id = self._feature_dict['user_id'].get(user, USER_ID_UNK)
            yield user_id, 0, 0, [0], [0], [0], 0


if __name__ == "__main__":
    # this is to test and debug reader function
    train_data = sys.argv[1]
    feature_dict = sys.argv[2]
    window_size = int(sys.argv[3])

    import cPickle
    with open(feature_dict) as f:
        feature_dict = cPickle.load(f)

    r = Reader(feature_dict, window_size)

    for dat in r.train(train_data):
        print dat
