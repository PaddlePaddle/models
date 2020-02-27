import numpy as np
import pts_utils

a = np.random.random((16384, 3)).astype('float32')
b = np.random.random((64, 7)).astype('float32')
c = pts_utils.pts_in_boxes3d(a, b)
print(a, b, c, c.shape, np.sum(c))
