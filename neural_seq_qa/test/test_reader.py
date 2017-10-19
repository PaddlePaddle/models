import unittest
import os
import itertools
import math
import logging

# set up python path
topdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
import sys
sys.path += [topdir, os.path.join(topdir, "data", "evaluation")]

import reader
import utils

formatter = logging.Formatter(
    "[%(levelname)s %(asctime)s.%(msecs)d %(filename)s:%(lineno)d] %(message)s",
    datefmt='%Y-%m-%d %I:%M:%S')
ch = logging.StreamHandler()
ch.setFormatter(formatter)
utils.logger.addHandler(ch)


class Vocab(object):
    @property
    def data(self):
        word_dict_path = os.path.join(topdir, "data", "embedding",
                                      "wordvecs.vcb")
        return utils.load_dict(word_dict_path)


class NegativeSampleRatioTest(unittest.TestCase):
    def check_ratio(self, negative_sample_ratio):
        for keep_first_b in [True, False]:
            settings = reader.Settings(
                vocab=Vocab().data,
                is_training=True,
                label_schema="BIO2",
                negative_sample_ratio=negative_sample_ratio,
                hit_ans_negative_sample_ratio=0.25,
                keep_first_b=keep_first_b)

            filename = os.path.join(topdir, "test", "trn_data.gz")
            data_stream = reader.create_reader(filename, settings)
            total, negative_num = 5000, 0
            for _, d in itertools.izip(xrange(total), data_stream()):
                labels = d[reader.LABELS]
                if labels.count(0) == 0:
                    negative_num += 1

            ratio = negative_num / float(total)
            self.assertLessEqual(math.fabs(ratio - negative_sample_ratio), 0.01)

    def runTest(self):
        for ratio in [1., 0.25, 0.]:
            self.check_ratio(ratio)


class KeepFirstBTest(unittest.TestCase):
    def runTest(self):
        for keep_first_b in [True, False]:
            for label_schema in ["BIO", "BIO2"]:
                settings = reader.Settings(
                    vocab=Vocab().data,
                    is_training=True,
                    label_schema=label_schema,
                    negative_sample_ratio=0.2,
                    hit_ans_negative_sample_ratio=0.25,
                    keep_first_b=keep_first_b)

                filename = os.path.join(topdir, "test", "trn_data.gz")
                data_stream = reader.create_reader(filename, settings)
                total, at_least_one, one = 1000, 0, 0
                for _, d in itertools.izip(xrange(total), data_stream()):
                    labels = d[reader.LABELS]
                    b_num = labels.count(0)
                    if b_num >= 1:
                        at_least_one += 1
                    if b_num == 1:
                        one += 1

                self.assertLess(at_least_one, total)
                if keep_first_b:
                    self.assertEqual(one, at_least_one)
                else:
                    self.assertLess(one, at_least_one)


class DictTest(unittest.TestCase):
    def runTest(self):
        settings = reader.Settings(
            vocab=Vocab().data,
            is_training=True,
            label_schema="BIO2",
            negative_sample_ratio=0.2,
            hit_ans_negative_sample_ratio=0.25,
            keep_first_b=True)

        filename = os.path.join(topdir, "test", "trn_data.gz")
        data_stream = reader.create_reader(filename, settings)
        q_uniq_ids, e_uniq_ids = set(), set()
        for _, d in itertools.izip(xrange(1000), data_stream()):
            q_uniq_ids.update(d[reader.Q_IDS])
            e_uniq_ids.update(d[reader.E_IDS])

        self.assertGreater(len(q_uniq_ids), 50)
        self.assertGreater(len(e_uniq_ids), 50)


if __name__ == '__main__':
    unittest.main()
