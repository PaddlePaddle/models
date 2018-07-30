#!/bin/env python

#function:
#   a demo to show how to use the converted model genereated by caffe2fluid
#   

import os
import numpy as np
from PIL import Image
import time
import scipy.io
from utils import load_default_data
from utils import get_default_img_feat_path


def import_fluid():
    import paddle.fluid as fluid
    return fluid


def load_data(imgfile, shape):
    h, w = shape[1:]
    im = Image.open(imgfile)

    # The storage order of the loaded image is W(widht),
    # H(height), C(channel). PaddlePaddle requires
    # the CHW order, so transpose them.
    im = im.resize((w, h), Image.ANTIALIAS)
    im = np.array(im).astype(np.float32)
    im = im.transpose((2, 0, 1))  # CHW
    im = im[(2, 1, 0), :, :]  # BGR

    # The mean to be subtracted from each image.
    # By default, the per-channel ImageNet mean.
    mean = np.array([104., 117., 124.], dtype=np.float32)
    mean = mean.reshape([3, 1, 1])
    im = im - mean
    return im.reshape([1] + shape)


def get_shape(fluid, program, name):
    for var in program.list_vars():
        if var.name == 'data':
            return list(var.shape[1:])

    raise ValueError('not found shape for input layer[%s], '
                     'you can specify by yourself' % (name))


def load_inference_model(dirname, exe):
    """ load fluid's inference model
    """
    fluid = import_fluid()
    model_fn = 'model'
    params_fn = 'params'
    if os.path.exists(os.path.join(dirname, model_fn)) \
            and os.path.exists(os.path.join(dirname, params_fn)):
        program, feed_names, fetch_targets = fluid.io.load_inference_model(\
                dirname, exe, model_fn, params_fn)
    else:
        raise ValueError('not found model files in direcotry[%s]' % (dirname))

    input_shape = get_shape(fluid, program, feed_names[0])
    feed_shapes = [input_shape]

    return program, feed_names, fetch_targets, feed_shapes


def reduce_along_dim(img, dim, weights, indicies):
    '''
    Perform bilinear interpolation given along the image dimension dim
    -weights are the kernel weights
    -indicies are the crossponding indicies location
    return img resize along dimension dim
    '''
    other_dim = abs(dim - 1)
    if other_dim == 0:  # resizing image width
        weights = np.tile(weights[np.newaxis, :, :, np.newaxis],
                          (img.shape[other_dim], 1, 1, 3))
        out_img = img[:, indicies, :] * weights
        out_img = np.sum(out_img, axis=2)
    else:  # resize image height
        weights = np.tile(weights[:, :, np.newaxis, np.newaxis],
                          (1, 1, img.shape[other_dim], 3))
        out_img = img[indicies, :, :] * weights
        out_img = np.sum(out_img, axis=1)

    return out_img


def extract_feats(model_path, path_imgs):
    '''
    Function using the paddle python wrapper to extract 4096 from VGG_ILSVRC_16_layers.caffemodel model

    Inputs:
    ------
    path_imgs      : list of the full path of images to be processed
    model_path     : path to the model definition file
    Output:
    -------
    features           : return the features extracted
    '''

    fluid = import_fluid()

    place = fluid.CPUPlace()
    exe = fluid.Executor(place)

    ret = load_inference_model(model_path, exe)
    program, feed_names, fetch_targets, feed_shapes = ret
    input_name = feed_names[0]
    input_shape = feed_shapes[0]

    feats = np.zeros((4096, len(path_imgs)))

    start_time = time.time()
    for i in range(len(path_imgs)):
        np_images = np.array(load_data(path_imgs[i], input_shape))

        predictions = exe.run(program=program,
                              feed={input_name: np_images},
                              fetch_list=fetch_targets)

        feats[:, i] = predictions[0]

        print "%d out of %d done....." % (i, len(path_imgs))
        end_time = time.time()
        print("Duration time = %.4f" % (end_time - start_time))
        start_time = end_time

    return feats


if __name__ == "__main__":
    """ maybe more convenient to use 'run.sh' to call this tool
    """
    default_data_train_dir, default_data_test_dir, tar_token_filename = load_default_data(
    )

    weight_file = './caffe2fluid/fluid'
    feats_save_path = get_default_img_feat_path()
    for i, data_dir in enumerate(
        [default_data_train_dir, default_data_test_dir]):

        file_list = os.listdir(data_dir)
        file_list = [
            os.path.join(data_dir, img_path) for img_path in file_list
        ]
        feats = extract_feats(weight_file, file_list)
        vgg_feats = {}
        vgg_feats['feats'] = feats
        vgg_feats['img_paths'] = file_list
        scipy.io.savemat(feats_save_path[i], vgg_feats)
