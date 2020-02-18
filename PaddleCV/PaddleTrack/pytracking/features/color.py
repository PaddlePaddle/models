from paddle.fluid import layers
from pytracking.features.featurebase import FeatureBase
from pytracking.libs.paddle_utils import PTensor
import numpy as np

class RGB(FeatureBase):
    """RGB feature normalized to [-0.5, 0.5]."""
    def dim(self):
        return 3

    def stride(self):
        return self.pool_stride

    def extract(self, im: np.ndarray):
        return im / 255 - 0.5


class Grayscale(FeatureBase):
    """Grayscale feature normalized to [-0.5, 0.5]."""
    def dim(self):
        return 1

    def stride(self):
        return self.pool_stride

    def extract(self, im: np.ndarray):
        return np.mean(im / 255 - 0.5, 1, keepdims=True)
