from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import unittest
import numpy as np
import data_utils.augmentor.trans_mean_variance_norm as trans_mean_variance_norm
import data_utils.augmentor.trans_add_delta as trans_add_delta
import data_utils.augmentor.trans_splice as trans_splice
import data_utils.augmentor.trans_delay as trans_delay


class TestTransMeanVarianceNorm(unittest.TestCase):
    """unit test for TransMeanVarianceNorm
    """

    def setUp(self):
        self._file_path = "./data_utils/augmentor/tests/data/" \
                          "global_mean_var_search26kHr"

    def test(self):
        feature = np.zeros((2, 120), dtype="float32")
        feature.fill(1)
        trans = trans_mean_variance_norm.TransMeanVarianceNorm(self._file_path)
        (feature1, label1, name) = trans.perform_trans((feature, None, None))
        (mean, var) = trans.get_mean_var()
        feature_flat1 = feature1.flatten()
        feature_flat = feature.flatten()
        one = np.ones((1), dtype="float32")
        for idx, val in enumerate(feature_flat1):
            cur_idx = idx % 120
            self.assertAlmostEqual(val, (one[0] - mean[cur_idx]) * var[cur_idx])


class TestTransAddDelta(unittest.TestCase):
    """unit test TestTransAddDelta
    """

    def test_regress(self):
        """test regress
        """
        feature = np.zeros((14, 120), dtype="float32")
        feature[0:5, 0:40].fill(1)
        feature[0 + 5, 0:40].fill(1)
        feature[1 + 5, 0:40].fill(2)
        feature[2 + 5, 0:40].fill(3)
        feature[3 + 5, 0:40].fill(4)
        feature[8:14, 0:40].fill(4)
        trans = trans_add_delta.TransAddDelta()
        feature = feature.reshape((14 * 120))
        trans._regress(feature, 5 * 120, feature, 5 * 120 + 40, 40, 4, 120)
        trans._regress(feature, 5 * 120 + 40, feature, 5 * 120 + 80, 40, 4, 120)
        feature = feature.reshape((14, 120))
        tmp_feature = feature[5:5 + 4, :]
        self.assertAlmostEqual(1.0, tmp_feature[0][0])
        self.assertAlmostEqual(0.24, tmp_feature[0][119])
        self.assertAlmostEqual(2.0, tmp_feature[1][0])
        self.assertAlmostEqual(0.13, tmp_feature[1][119])
        self.assertAlmostEqual(3.0, tmp_feature[2][0])
        self.assertAlmostEqual(-0.13, tmp_feature[2][119])
        self.assertAlmostEqual(4.0, tmp_feature[3][0])
        self.assertAlmostEqual(-0.24, tmp_feature[3][119])

    def test_perform(self):
        """test perform
        """
        feature = np.zeros((4, 40), dtype="float32")
        feature[0, 0:40].fill(1)
        feature[1, 0:40].fill(2)
        feature[2, 0:40].fill(3)
        feature[3, 0:40].fill(4)
        trans = trans_add_delta.TransAddDelta()
        (feature, label, name) = trans.perform_trans((feature, None, None))
        self.assertAlmostEqual(feature.shape[0], 4)
        self.assertAlmostEqual(feature.shape[1], 120)
        self.assertAlmostEqual(1.0, feature[0][0])
        self.assertAlmostEqual(0.24, feature[0][119])
        self.assertAlmostEqual(2.0, feature[1][0])
        self.assertAlmostEqual(0.13, feature[1][119])
        self.assertAlmostEqual(3.0, feature[2][0])
        self.assertAlmostEqual(-0.13, feature[2][119])
        self.assertAlmostEqual(4.0, feature[3][0])
        self.assertAlmostEqual(-0.24, feature[3][119])


class TestTransSplict(unittest.TestCase):
    """unit test Test TransSplict
    """

    def test_perfrom(self):
        feature = np.zeros((8, 10), dtype="float32")
        for i in xrange(feature.shape[0]):
            feature[i, :].fill(i)

        trans = trans_splice.TransSplice()
        (feature, label, name) = trans.perform_trans((feature, None, None))
        self.assertEqual(feature.shape[1], 110)

        for i in xrange(8):
            nzero_num = 5 - i
            cur_val = 0.0
            if nzero_num < 0:
                cur_val = i - 5 - 1
            for j in xrange(11):
                if j <= nzero_num:
                    for k in xrange(10):
                        self.assertAlmostEqual(feature[i][j * 10 + k], cur_val)
                else:
                    if cur_val < 7:
                        cur_val += 1.0
                    for k in xrange(10):
                        self.assertAlmostEqual(feature[i][j * 10 + k], cur_val)


class TestTransDelay(unittest.TestCase):
    """unittest TransDelay
    """

    def test_perform(self):
        label = np.zeros((10, 1), dtype="int64")
        for i in xrange(10):
            label[i][0] = i

        trans = trans_delay.TransDelay(5)
        (_, label, _) = trans.perform_trans((None, label, None))

        for i in xrange(5):
            self.assertAlmostEqual(label[i + 5][0], i)

        for i in xrange(5):
            self.assertAlmostEqual(label[i][0], 0)


if __name__ == '__main__':
    unittest.main()
