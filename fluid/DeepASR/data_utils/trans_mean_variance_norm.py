#by zhxfl 2018.01.29
import numpy
import math

class TransMeanVarianceNorm(object):
    """ normalization of mean variance for feature data 
    """
    def __init__(self, snorm_path):
        """init construction
            Args:
                snorm_path: the path of mean and variance
        """
        self._mean = None
        self._var = None
        self._load_norm(snorm_path)
    
    def _load_norm(self, snorm_path):
        """ load global mean var file
        """
        lLines = open(snorm_path).readlines()
        nLen = len(lLines)
        self._mean = numpy.zeros((nLen), dtype="float32")
        self._var = numpy.zeros((nLen), dtype="float32")
        self._nLen = nLen
        for nidx, l in enumerate(lLines):
            s = l.split()
            assert len(s) == 2
            self._mean[nidx] = float(s[0])
            self._var[nidx] = 1.0 / math.sqrt(float(s[1]))
            if self._var[nidx] > 100000.0: 
                self._var[nidx] = 100000.0

    def get_mean_var(self):
        """ get mean and var 
        """
        return (self._mean, self._var)

    def perform_trans(self, sample):
        """ feature = (feature - mean) * var
        """
        (feature, label) = sample
        shape = feature.shape
        assert len(shape) == 2
        nfeature_len = shape[0] * shape[1]
        assert nfeature_len % self._nLen == 0
        ncur_idx = 0
        feature = feature.reshape((nfeature_len))
        while ncur_idx < nfeature_len:
            block = feature[ncur_idx:ncur_idx + self._nLen]
            block = (block - self._mean) * self._var
            feature[ncur_idx:ncur_idx + self._nLen] = block
            ncur_idx += self._nLen 
        feature = feature.reshape(shape)
        return (feature, label)
