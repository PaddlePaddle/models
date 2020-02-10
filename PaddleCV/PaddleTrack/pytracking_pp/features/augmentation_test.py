import unittest
import numpy as np

import torch
import paddle.fluid as fluid
from paddle.fluid import layers

import os.path as osp
import sys

CURRENT_DIR = osp.dirname(__file__)
sys.path.append(osp.join(CURRENT_DIR, '..', '..'))

from pytracking_pp.libs.paddle_utils import n2p, n2t, t2n
import pytracking_pp.features.augmentation as paug
import pytracking.features.augmentation as taug

def p2n(x):
    return x

def batch_to_numpy(x):
    return np.transpose(np.squeeze(x, 0), [1, 2, 0])

def numpy_to_batch(x):
    return np.expand_dims(np.transpose(x, [2, 0, 1]), 0)

def test_unsqueeze():
    import numpy as np
    from paddle.fluid import dygraph, layers
    def n2p(x): return dygraph.to_variable(np.array(x))

    with dygraph.guard():
        x = n2p(np.random.rand(1, 3, 5, 5))
        layers.unsqueeze(x, [0])

def test_bilinear_resize():
    import torch
    import torch.nn.functional as F
    from paddle.fluid import layers, dygraph
    import numpy as np

    def n2p(x): return dygraph.to_variable(np.array(x))
    def n2t(x): return torch.from_numpy(np.array(x))
    def p2n(x): return x.numpy()
    def t2n(x): return x.detach().cpu().numpy()
    def try_fn(fn):
        success = True
        try:
            fn()
        except:
            success = False
        return success

    with dygraph.guard():
        rng = np.random.RandomState(0)
        im = rng.uniform(-1, 1, (1, 3, 127, 127)).astype('float32')
        align_corners_list = [True, False]
        for ac in align_corners_list:
            out_t = F.interpolate(n2t(im), [255, 255], mode='bilinear', align_corners=ac)
            out_p0 = layers.resize_bilinear(n2p(im), [255, 255], align_corners=ac, align_mode=0)
            out_p1 = layers.resize_bilinear(n2p(im), [255, 255], align_corners=ac, align_mode=1)

            s0 = try_fn(lambda: np.testing.assert_allclose(p2n(out_p0), t2n(out_t), atol=1e-5, rtol=1e-5))
            s1 = try_fn(lambda: np.testing.assert_allclose(p2n(out_p1), t2n(out_t), atol=1e-5, rtol=1e-5))
            print('align_corners: {}, align_mode_0 success: {}, align_mode_1 success: {}'.format(ac, s0, s1))
    # Outputs:
    # align_corners: True, align_mode_0 success: True, align_mode_1 success: True
    # align_corners: False, align_mode_0 success: False, align_mode_1 success: False

class TestAugmentation(unittest.TestCase):
    def test_Identity(self):
        repeat = range(50)
        rng = np.random.RandomState(0)
        for _ in repeat:
            im = rng.uniform(-1, 1, (1, 3, 255, 255))
            if rng.uniform() < 0.8:
                shift = rng.randint(-127, 127, (2,))
            else:
                shift = None
            output_sz = rng.randint(80, 300, (2,))

            aug_p = paug.Identity(output_sz, shift)
            aug_t = taug.Identity(output_sz, shift)

            print('shift: {}, output_sz: {}'.format(shift, output_sz))
            out_p = numpy_to_batch(aug_p(batch_to_numpy(im)))
            out_t = aug_t(n2t(im))

            out_p = p2n(out_p)
            out_t = t2n(out_t)
            np.testing.assert_allclose(out_p, out_t)

    def test_FlipHorizontal(self):
        repeat = range(50)
        rng = np.random.RandomState(0)
        for _ in repeat:
            im = rng.uniform(-1, 1, (1, 3, 255, 255))
            if rng.uniform() < 0.8:
                shift = rng.randint(-127, 127, (2,))
            else:
                shift = None
            output_sz = rng.randint(80, 300, (2,))

            aug_p = paug.FlipHorizontal(output_sz, shift)
            aug_t = taug.FlipHorizontal(output_sz, shift)

            print('shift: {}, output_sz: {}'.format(shift, output_sz))
            out_p = numpy_to_batch(aug_p(batch_to_numpy(im)))
            out_t = aug_t(n2t(im))
            out_p = p2n(out_p)
            out_t = t2n(out_t)
            np.testing.assert_allclose(out_p, out_t)

    def test_FlipVertical(self):
        use_gpu_list = [True, False]
        for use_gpu in use_gpu_list:
            place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
            with fluid.dygraph.guard(place):
                repeat = range(50)
                rng = np.random.RandomState(0)
                for _ in repeat:
                    im = rng.uniform(-1, 1, (1, 3, 255, 255))
                    if rng.uniform() < 0.8:
                        shift = rng.randint(-127, 127, (2,))
                    else:
                        shift = None
                    output_sz = rng.randint(80, 300, (2,))

            aug_p = paug.FlipVertical(output_sz, shift)
            aug_t = taug.FlipVertical(output_sz, shift)

            print('shift: {}, output_sz: {}'.format(shift, output_sz))
            out_p = numpy_to_batch(aug_p(batch_to_numpy(im)))
            out_t = aug_t(n2t(im))
            out_p = p2n(out_p)
            out_t = t2n(out_t)
            np.testing.assert_allclose(out_p, out_t)

    def test_Translation(self):
        repeat = range(50)
        rng = np.random.RandomState(0)
        for _ in repeat:
            im = rng.uniform(-1, 1, (1, 3, 255, 255))
            if rng.uniform() < 0.8:
                shift = rng.randint(-127, 127, (2,))
            else:
                shift = None
            translation = rng.randint(-127, 127, (2,))
            output_sz = rng.randint(80, 300, (2,))

            aug_p = paug.Translation(translation, output_sz, shift)
            aug_t = taug.Translation(translation, output_sz, shift)

            print('translation: {}, shift: {}, output_sz: {}'.format(translation, shift, output_sz))
            out_p = numpy_to_batch(aug_p(batch_to_numpy(im)))
            out_t = aug_t(n2t(im))
            out_p = p2n(out_p)
            out_t = t2n(out_t)
            np.testing.assert_allclose(out_p, out_t)

    def test_Rotate(self):
        repeat = range(50)
        rng = np.random.RandomState(0)
        for _ in repeat:
            im = rng.uniform(-1, 1, (1, 3, 255, 255)).astype('float32')
            if rng.uniform() < 0.8:
                shift = rng.randint(-127, 127, (2,))
            else:
                shift = None
            angle = rng.uniform(-180, 180)
            output_sz = rng.randint(80, 300, (2,))

            aug_p = paug.Rotate(angle, output_sz, shift)
            aug_t = taug.Rotate(angle, output_sz, shift)

            print('angle: {}, shift: {}, output_sz: {}'.format(angle, shift, output_sz))
            out_p = numpy_to_batch(aug_p(batch_to_numpy(im)))
            out_t = aug_t(n2t(im))
            out_p = p2n(out_p)
            out_t = t2n(out_t)
            np.testing.assert_allclose(out_p, out_t)

    def test_Blur(self):
        use_gpu_list = [True, False]
        for use_gpu in use_gpu_list:
            place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
            with fluid.dygraph.guard(place):
                repeat = range(50)
                rng = np.random.RandomState(0)
                for _ in repeat:
                    im = rng.uniform(-1, 1, (1, 3, 255, 255)).astype('float32')
                    if rng.uniform() < 0.8:
                        shift = rng.randint(-127, 127, (2,))
                    else:
                        shift = None
                    sigma = rng.uniform(0.1, 6)
                    output_sz = rng.randint(80, 300, (2,))

                    aug_p = paug.Blur(sigma, output_sz, shift)
                    aug_t = taug.Blur(sigma, output_sz, shift)

                    print('sigma: {}, shift: {}, output_sz: {}'.format(sigma, shift, output_sz))

                    out_p = numpy_to_batch(aug_p(batch_to_numpy(im)))
                    out_t = aug_t(n2t(im))

                    out_p = p2n(out_p)
                    out_t = t2n(out_t)
                    np.testing.assert_allclose(out_p, out_t, rtol=1e-07, atol=1e-06)

                    # for out_p, out_t in zip(aug_p.filter, aug_t.filter):
                    #     out_p = p2n(out_p)
                    #     out_t = t2n(out_t)
                    #     np.testing.assert_allclose(out_p, out_t)

if __name__ == '__main__':
    unittest.main()
