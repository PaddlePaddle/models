import os
import sys
import numpy as np
import shutil
import cv2
import pdb


def process_dir(_dir):
    if os.path.exists(_dir):
        shutil.rmtree(_dir)
    os.makedirs(_dir)


def pascal_palette():
    palette = {
        (0, 0, 0): 0,
        (128, 0, 0): 1,
        (0, 128, 0): 2,
        (128, 128, 0): 3,
        (0, 0, 128): 4,
        (128, 0, 128): 5,
        (0, 128, 128): 6,
        (128, 128, 128): 7,
        (64, 0, 0): 8,
        (192, 0, 0): 9,
        (64, 128, 0): 10,
        (192, 128, 0): 11,
        (64, 0, 128): 12,
        (192, 0, 128): 13,
        (64, 128, 128): 14,
        (192, 128, 128): 15,
        (0, 64, 0): 16,
        (128, 64, 0): 17,
        (0, 192, 0): 18,
        (128, 192, 0): 19,
        (0, 64, 128): 20,
        (224, 224, 192): 0
    }
    return palette


def convert_from_color_label(img):
    '''Convert the Pascal VOC label format to train.
    
    Args:
        img: The label result of Pascal VOC.
    '''
    palette = pascal_palette()
    for c, i in palette.items():
        _c = (c[2], c[1], c[0])  # the image channel read by opencv is (b, g, r)
        m = np.all(img == np.array(_c).reshape(1, 1, 3), axis=2)
        img[m] = i
    return img


def convert_to_color_label(img):
    '''Visualize the prediction result as Pascal VOC format.
    
    Args:
        img: The prediction result by FCN.
    '''
    palette = pascal_palette()
    palette = dict((k, v) for v, k in palette.items())

    r = np.zeros((img.shape[0], img.shape[1]))
    g = np.zeros((img.shape[0], img.shape[1]))
    b = np.zeros((img.shape[0], img.shape[1]))

    for l in range(21):
        r[img == l] = palette[l][0]
        g[img == l] = palette[l][1]
        b[img == l] = palette[l][2]

    result = np.zeros((img.shape[0], img.shape[1], 3))
    result[:, :, 0] = b
    result[:, :, 1] = g
    result[:, :, 2] = r

    return result


def resolve_caffe_model(pretrain_model):
    '''Resolve the pretrained model parameters for finetuning.
    
    Args:
        pretrain_model: The pretrained model path.
    
    Returns:
        weights_dict: The resolved model parameters as dictionary format.
    '''
    items = os.listdir(pretrain_model)

    weights_dict = {}
    for item in items:
        param_name = item.split('.')[0]
        param_value = np.load(os.path.join(pretrain_model, item))
        weights_dict[param_name] = param_value
    return weights_dict


def save_caffemodel_param(deploy_file, model_file, out_dir):
    '''Resolve VGG16 caffe model and save as numpy format.
    
    Args:
        deploy_file: The path of caffe deploy file.
        model_file: The path of caffe model file.
        out_dir: The directory to save the resolved model. 
    '''
    sys.path.insert(0, '/home/yczhao/work/caffe/python')
    import caffe

    net = caffe.Net(deploy_file, model_file, caffe.TEST)
    for param_name in net.params.keys():
        weight = net.params[param_name][0].data
        weight_name = param_name + '_weights'
        bias_name = param_name + '_biases'
        np.save(os.path.join(out_dir, weight_name + '.npy'), weight)

        if 'upscore' not in param_name:
            bias = net.params[param_name][1].data
            np.save(os.path.join(out_dir, bias_name + '.npy'), bias)


if __name__ == '__main__':
    deploy_file = './models/fcn_caffe/fcn32s_deploy.prototxt'
    model_file = './models/fcn_caffe/fcn32s.caffemodel'
    out_dir = './models/vgg16_weights'
    process_dir(out_dir)
    save_caffemodel_param(deploy_file, model_file, out_dir)
