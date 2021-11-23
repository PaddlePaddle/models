import os
import sys
import shutil
import numpy as np
import cv2
from PIL import Image
import pdb

import paddle.v2 as paddle
import paddle.fluid as fluid


def process_dir(_dir):
    if os.path.exists(_dir):
        shutil.rmtree(_dir)
    os.makedirs(_dir)


def mse_loss(input, label):
    return fluid.layers.reduce_mean((input - label) * (input - label))


def cross_entropy_loss(input, label):
    eps = fluid.layers.fill_constant(shape=[1], value=1e-8, dtype='float32')
    input_l = fluid.layers.elementwise_max(input, eps)
    input_r = fluid.layers.elementwise_max(1 - input, eps)
    loss = -1.0 * label * fluid.layers.log(input_l) - 1.0 * (
        1 - label) * fluid.layers.log(input_r)
    return fluid.layers.reduce_mean(loss)


def scale_unit(arr, eps=1e-8):  #[0, 1]
    min = np.min(arr)
    arr -= min
    arr *= 1.0 / (np.max(arr) + eps)
    return arr


def get_da_weight(param_name):
    _tensor = fluid.global_scope().find_var(param_name).get_tensor()
    return np.array(_tensor)


def vis_weight(weight, noise_ratio, filter_size=(28, 28)):
    interval = 1  #interval between each filter
    filter_size = np.sqrt(weight.shape[0]).astype('int')
    nums = np.sqrt(weight.shape[1]).astype(
        'int')  # the number of filters each row or col

    vis_size = filter_size * nums + (nums - 1)
    vis_res = np.zeros((vis_size, vis_size))
    for row in range(nums):
        for col in range(nums):
            filter = scale_unit(weight[:, row * nums + col])
            filter = np.reshape(filter, (filter_size, filter_size)) * 255
            begin_row = row * filter_size + row * interval
            begin_col = col * filter_size + col * interval
            vis_res[begin_row:(begin_row + filter_size), begin_col:(
                begin_col + filter_size)] = filter
    cv2.imwrite('w_%.2f.png' % noise_ratio, vis_res)


def normal_noise(data, prop):
    img_data_noise = data + prop * np.random.randn(data.shape[0], data.shape[1])
    img_data_noise = img_data_noise.astype('float32')
    return img_data_noise


def masking_noise(data, prop):
    rand = np.random.uniform(size=data.shape)
    mask = np.where((rand - prop) < 0)
    for i in range(len(mask[0])):
        data[mask[0][i]][mask[1][i]] = 0
    return data


def salt_and_pepper_noise(data, prop):
    '''Apply salt and pepper noise to data in X.

    In other words a fraction v of elements in X (chosen at random) is set 
    to its maximum or minimum value according to a fair coin flip.
    If minimum or maximum are not given, the min (max) value in X is taken.
    
    Args:
        data: array_like, Input data
        prop: int, fraction of elements to distort
    Returns:
        transformed data
    '''

    X_noise = X.copy()
    n_features = X.shape[1]

    mn = X.min()
    mx = X.max()

    for i, sample in enumerate(X):
        mask = np.random.randint(0, n_features, v)

        for m in mask:

            if np.random.random() < 0.5:
                X_noise[i][m] = mn
            else:
                X_noise[i][m] = mx

    return X_noise


if __name__ == '__main__':
    # test filter visualization module
    encode_w = np.load('encode_w.npy')
    vis = vis_weight(encode_w, 0.5)
