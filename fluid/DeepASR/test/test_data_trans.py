#by zhxfl 2018.01.31
import sys
import unittest
import numpy
sys.path.append("../")
import data_utils.trans_mean_variance_norm as trans_mean_variance_norm
import data_utils.trans_add_delta as trans_add_delta
import data_utils.trans_splice as trans_splice

class TestTransMeanVarianceNorm(unittest.TestCase):
    """unit test for TransMeanVarianceNorm
    """
    def test(self):
        feature = numpy.zeros((2, 120), dtype="float32")
        feature.fill(1)
        trans = trans_mean_variance_norm.TransMeanVarianceNorm("../data/global_mean_var_search26kHr")
        (feature1, label1) = trans.perform_trans((feature, None))
        (mean, var) = trans.get_mean_var()
        feature_flat1 = feature1.flatten()
        feature_flat = feature.flatten()
        one = numpy.ones((1), dtype="float32")
        for idx, val in enumerate(feature_flat1):
            cur_idx = idx % 120
            self.assertAlmostEqual(val, (one[0] - mean[cur_idx]) * var[cur_idx])

class TestTransAddDelta(unittest.TestCase):
    """unit test TestTransAddDelta
    """
    def test_regress(self):
        """test regress
        """
        feature = numpy.zeros((14, 120), dtype="float32")
        feature[0 : 5, 0:40].fill(1)
        feature[0 + 5, 0:40].fill(1)
        feature[1 + 5, 0:40].fill(2)
        feature[2 + 5, 0:40].fill(3)
        feature[3 + 5, 0:40].fill(4)
        feature[8 :14, 0:40].fill(4)
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
        feature = numpy.zeros((4, 40), dtype="float32")
        feature[0, 0:40].fill(1)
        feature[1, 0:40].fill(2)
        feature[2, 0:40].fill(3)
        feature[3, 0:40].fill(4)
        trans = trans_add_delta.TransAddDelta()
        (feature, label) = trans.perform_trans((feature, None))
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
        feature = numpy.zeros((8, 10), dtype="float32")
        for i in xrange(feature.shape[0]):
            feature[i,:].fill(i)

        trans = trans_splice.TransSplice()
        (feature, label) = trans.perform_trans((feature, None))
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
                        print i, j, k
                        print feature[i].reshape(11, 10)
                        self.assertAlmostEqual(feature[i][j * 10 + k], cur_val)


if __name__ == '__main__':
    unittest.main()


