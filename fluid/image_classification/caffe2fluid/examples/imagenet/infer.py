#!/bin/env python

#function:
#   a demo to show how to use the converted model genereated by caffe2fluid
#   
#notes:
#   only support imagenet data

import os
import sys
import inspect
import numpy as np
import paddle.v2 as paddle
import paddle.v2.fluid as fluid


def load_data(imgfile, shape):
    h, w = shape[1:]
    from PIL import Image
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


def build_model(net_file, net_name):
    print('build model with net_file[%s] and net_name[%s]' %
          (net_file, net_name))

    net_path = os.path.dirname(net_file)
    module_name = os.path.basename(net_file).rstrip('.py')
    if net_path not in sys.path:
        sys.path.insert(0, net_path)

    try:
        m = __import__(module_name, fromlist=[net_name])
        MyNet = getattr(m, net_name)
    except Exception as e:
        print('failed to load module[%s]' % (module_name))
        print(e)
        return None

    input_name = 'data'
    input_shape = MyNet.input_shapes()[input_name]
    images = fluid.layers.data(name='image', shape=input_shape, dtype='float32')
    #label = fluid.layers.data(name='label', shape=[1], dtype='int64')

    net = MyNet({input_name: images})
    input_shape = MyNet.input_shapes()[input_name]
    return net, input_shape


def dump_results(results, names, root):
    if os.path.exists(root) is False:
        os.path.mkdir(root)

    for i in range(len(names)):
        n = names[i]
        res = results[i]
        filename = os.path.join(root, n)
        np.save(filename + '.npy', res)


def infer(net_file, net_name, model_file, imgfile, debug=False):
    """ do inference using a model which consist 'xxx.py' and 'xxx.npy'
    """
    #1, build model
    net, input_shape = build_model(net_file, net_name)
    prediction = net.get_output()

    #2, load weights for this model
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    startup_program = fluid.default_startup_program()
    exe.run(startup_program)

    if model_file.find('.npy') > 0:
        net.load(data_path=model_file, exe=exe, place=place)
    else:
        net.load(data_path=model_file, exe=exe)

    #3, test this model
    test_program = fluid.default_main_program().clone()

    fetch_list_var = []
    fetch_list_name = []
    if debug is False:
        fetch_list_var.append(prediction)
    else:
        for k, v in net.layers.items():
            fetch_list_var.append(v)
            fetch_list_name.append(k)

    np_images = load_data(imgfile, input_shape)
    results = exe.run(program=test_program,
                      feed={'image': np_images},
                      fetch_list=fetch_list_var)

    if debug is True:
        dump_path = 'results.layers'
        dump_results(results, fetch_list_name, dump_path)
        print('all results dumped to [%s]' % (dump_path))
    else:
        result = results[0]
        print('predicted class:', np.argmax(result))


if __name__ == "__main__":
    """ maybe more convenient to use 'run.sh' to call this tool
    """
    net_file = 'models/resnet50/resnet50.py'
    weight_file = 'models/resnet50/resnet50.npy'
    imgfile = 'data/65.jpeg'
    net_name = 'ResNet50'

    argc = len(sys.argv)
    if argc == 5:
        net_file = sys.argv[1]
        weight_file = sys.argv[2]
        imgfile = sys.argv[3]
        net_name = sys.argv[4]
    elif argc > 1:
        print('usage:')
        print('\tpython %s [net_file] [weight_file] [imgfile] [net_name]' %
              (sys.argv[0]))
        print('\teg:python %s %s %s %s %s' % (sys.argv[0], net_file,
                                              weight_file, imgfile, net_name))
        sys.exit(1)

    infer(net_file, net_name, weight_file, imgfile)
