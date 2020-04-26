import os
import fnmatch
import numpy as np
import cv2
from cv2 import imread
from scipy import linalg
import paddle.fluid as fluid
from inception import InceptionV3

def tqdm(x): return x

""" based on https://github.com/mit-han-lab/gan-compression/blob/master/metric/fid_score.py
"""

"""
inceptionV3 pretrain model is convert from pytorch, pretrain_model url is https://paddle-gan-models.bj.bcebos.com/params_inceptionV3.tar.gz
"""

def _calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    m1 = np.atleast_1d(mu1)
    m2 = np.atleast_1d(mu2)
 
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, 'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, 'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    t = sigma1.dot(sigma2)
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)


def _build_program(model):
    main_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.program_guard(main_program, startup_program):
        images = fluid.data(name='images', shape = [None, 3, None, None])
        output = model.network(images, class_dim=1008)
        pred = fluid.layers.pool2d(output[0], global_pooling=True)

    test_program = main_program.clone(for_test=True)
    return pred, test_program, startup_program

def _get_activations_from_ims(img, model, batch_size, dims, use_gpu, premodel_path):
    n_batches = (len(img) + batch_size - 1) // batch_size
    n_used_img = len(img)

    pred_arr = np.empty((n_used_img, dims))

    for i in tqdm(range(n_batches)):
        start = i * batch_size
        end = start + batch_size
        if end > len(img):
            end = len(img)
        images = img[start: end]
        if images.shape[1] != 3:
            images = images.transpose((0, 3, 1, 2))
        images /= 255

        output, main_program, startup_program = _build_program(model)
        place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(startup_program)

        fluid.load(main_program, os.path.join(premodel_path, 'paddle_inceptionv3'), exe)
        pred = exe.run(main_program, feed = {'images': images}, fetch_list = [output])[0]

        pred_arr[start: end] = pred.reshape(end - start, -1)

    return pred_arr

def _compute_statistic_of_img(img, model, batch_size, dims, use_gpu, premodel_path):
    act = _get_activations_from_ims(img, model, batch_size, dims, use_gpu, premodel_path) 
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu,sigma

def calculate_fid_given_img(img_fake, img_real, batch_size, use_gpu, dims, premodel_path, model=None):
    assert os.path.exists(premodel_path), 'pretrain_model path {} is not exists! Please download it first'.format(premodel_path)
    if model is None:
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        model = InceptionV3([block_idx])

    m1, s1 = _compute_statistic_of_img(img_fake, model, batch_size, dims, use_gpu, premodel_path)
    m2, s2 = _compute_statistic_of_img(img_real, model, batch_size, dims, use_gpu, premodel_path)

    fid_value = _calculate_frechet_distance(m1, s1, m2, s2)
    return fid_value


def _get_activations(files, model, batch_size, dims, use_gpu, premodel_path):
    if len(files) % batch_size != 0:
        print(('Warning: number of images is not a multiple of the '
            'batch size. Some samples are going to be ignored.'))
    if batch_size > len(files):
        print(('Warning: batch size is bigger than the datasets size. '
               'Setting batch size to datasets size'))
        batch_size = len(files)

    n_batches = len(files) // batch_size
    n_used_imgs = n_batches * batch_size

    pred_arr = np.empty((n_used_imgs, dims))
    for i in tqdm(range(n_batches)):
        start = i * batch_size
        end = start + batch_size
        images = np.array([imread(str(f)).astype(np.float32) for f in files[start:end]])

        if len(images.shape) != 4:
            images = imread(str(files[start]))
            images = cv2.cvtColor(images, cv2.COLOR_BGR2GRAY)
            images = np.array([images.astype(np.float32)])

        images = images.transpose((0, 3, 1, 2))
        images /= 255

        output, main_program, startup_program = _build_program(model)
        place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(startup_program)

        fluid.load(main_program, os.path.join(premodel_path, 'paddle_inceptionv3'), exe)
        pred = exe.run(main_program, feed = {'images': images}, fetch_list = [output])[0]

        pred_arr[start: end] = pred.reshape(end - start, -1)

    return pred_arr

def _calculate_activation_statistics(files, model, premodel_path, batch_size=50, dims=2048, use_gpu=False):
    act = _get_activations(files, model, batch_size, dims, use_gpu, premodel_path)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma

def _compute_statistics_of_path(path, model, batch_size, dims, use_gpu, premodel_path):
    if path.endswith('.npz'):
        f = np.load(path)
        m, s = f['mu'][:], f['sigma'][:]
        f.close()
    else:
        files = []
        for root, dirnames, filenames in os.walk(path):
            for filename in fnmatch.filter(filenames, '*.jpg') or fnmatch.filter(filenames, '*.png'):
                files.append(os.path.join(root, filename))
        m, s = _calculate_activation_statistics(files, model, premodel_path, batch_size, dims, use_gpu)
    return m, s

def calculate_fid_given_paths(paths, batch_size, use_gpu, dims, premodel_path, model=None):
    assert os.path.exists(premodel_path), 'pretrain_model path {} is not exists! Please download it first'.format(premodel_path)
    for p in paths:
        if not os.path.exists(p):
            raise RuntimeError('Invalid path: %s' % p)

    if model is None:
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        model = InceptionV3([block_idx])

    m1, s1 = _compute_statistics_of_path(paths[0], model, batch_size, dims, use_gpu, premodel_path)
    m2, s2 = _compute_statistics_of_path(paths[1], model, batch_size, dims, use_gpu, premodel_path)

    fid_value = _calculate_frechet_distance(m1, s1, m2, s2)
    return fid_value
    

