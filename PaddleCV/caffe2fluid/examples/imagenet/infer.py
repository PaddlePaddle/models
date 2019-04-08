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


def import_fluid():
    import paddle.fluid as fluid
    return fluid


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
    module_name = os.path.splitext(os.path.basename(net_file))[0]
    if net_path not in sys.path:
        sys.path.insert(0, net_path)

    try:
        m = __import__(module_name, fromlist=[net_name])
        MyNet = getattr(m, net_name)
    except Exception as e:
        print('failed to load module[%s.%s]' % (module_name, net_name))
        print(e)
        return None

    fluid = import_fluid()
    inputs_dict = MyNet.input_shapes()
    input_name = inputs_dict.keys()[0]
    input_shape = inputs_dict[input_name]
    images = fluid.layers.data(
        name=input_name, shape=input_shape, dtype='float32')
    #label = fluid.layers.data(name='label', shape=[1], dtype='int64')

    net = MyNet({input_name: images})
    return net, inputs_dict


def dump_results(results, names, root):
    if os.path.exists(root) is False:
        os.mkdir(root)

    for i in range(len(names)):
        n = names[i]
        res = results[i]
        filename = os.path.join(root, n)
        np.save(filename + '.npy', res)


def normalize_name(name_map):
    return {
        k.replace('/', '_'): v.replace('/', '_')
        for k, v in name_map.items()
    }


def rename_layer_name(names, net):
    """ because the names of output layers from caffe maybe changed for 'INPLACE' operation,
        and paddle's layers maybe fused, so we need to re-mapping their relationship for comparing
    """
    #build a mapping from paddle's name to caffe's name
    trace = getattr(net, 'name_trace', None)
    cf_trace = trace['caffe']
    real2cf = normalize_name(cf_trace['real2chg'])

    pd_trace = trace['paddle']
    pd2real = normalize_name(pd_trace['chg2real'])
    pd_deleted = normalize_name(pd_trace['deleted'])

    pd2cf_name = {}
    for pd_name, real_name in pd2real.items():
        if real_name in real2cf:
            pd2cf_name[pd_name] = '%s.%s.%s.both_changed' \
                    % (real2cf[real_name], real_name, pd_name)
        else:
            pd2cf_name[pd_name] = '%s.%s.pd_changed' % (real_name, pd_name)

    for pd_name, trace in pd_deleted.items():
        assert pd_name not in pd2cf_name, "this name[%s] has already exist" % (
            pd_name)
        pd2cf_name[pd_name] = '%s.pd_deleted' % (pd_name)

    for real_name, cf_name in real2cf.items():
        if cf_name not in pd2cf_name:
            pd2cf_name[cf_name] = '%s.cf_deleted' % (cf_name)

        if real_name not in pd2cf_name:
            pd2cf_name[real_name] = '%s.%s.cf_changed' % (cf_name, real_name)

    ret = []
    for name in names:
        new_name = pd2cf_name[name] if name in pd2cf_name else name
        print('remap paddle name[%s] to output name[%s]' % (name, new_name))
        ret.append(new_name)
    return ret


def load_model(exe, place, net_file, net_name, net_weight, debug):
    """ load model using xxxnet.py and xxxnet.npy
    """
    fluid = import_fluid()

    #1, build model
    net, input_map = build_model(net_file, net_name)
    feed_names = input_map.keys()
    feed_shapes = [v for k, v in input_map.items()]

    prediction = net.get_output()

    #2, load weights for this model
    startup_program = fluid.default_startup_program()
    exe.run(startup_program)

    #place = fluid.CPUPlace()
    if net_weight.find('.npy') > 0:
        net.load(data_path=net_weight, exe=exe, place=place)
    else:
        raise ValueError('not found weight file')

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

    return {
        'program': test_program,
        'feed_names': feed_names,
        'fetch_vars': fetch_list_var,
        'fetch_names': fetch_list_name,
        'feed_shapes': feed_shapes,
        'net': net
    }


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

    #print fluid.global_scope().find_var(feed_names[0])
    input_shape = get_shape(fluid, program, feed_names[0])
    feed_shapes = [input_shape]

    return program, feed_names, fetch_targets, feed_shapes


def infer(model_path, imgfile, net_file=None, net_name=None, debug=True):
    """ do inference using a model which consist 'xxx.py' and 'xxx.npy'
    """
    fluid = import_fluid()

    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    try:
        ret = load_inference_model(model_path, exe)
        program, feed_names, fetch_targets, feed_shapes = ret
        debug = False
        print('found a inference model for fluid')
    except ValueError as e:
        print('try to load model using net file and weight file')
        net_weight = model_path
        ret = load_model(exe, place, net_file, net_name, net_weight, debug)
        program = ret['program']
        feed_names = ret['feed_names']
        fetch_targets = ret['fetch_vars']
        fetch_list_name = ret['fetch_names']
        feed_shapes = ret['feed_shapes']
        net = ret['net']

    input_name = feed_names[0]
    input_shape = feed_shapes[0]

    np_images = load_data(imgfile, input_shape)
    results = exe.run(program=program,
                      feed={input_name: np_images},
                      fetch_list=fetch_targets)

    if debug is True:
        dump_path = 'results.paddle'
        dump_names = rename_layer_name(fetch_list_name, net)
        dump_results(results, dump_names, dump_path)
        print('all result of layers dumped to [%s]' % (dump_path))
    else:
        result = results[0]
        print('succeed infer with results[class:%d]' % (np.argmax(result)))

    return 0


def caffe_infer(prototxt, caffemodel, datafile):
    """ do inference using pycaffe for debug,
        all intermediate results will be dumpped to 'results.caffe'
    """
    import caffe

    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    input_layer = net.blobs.keys()[0]
    print('got name of input layer is:%s' % (input_layer))
    input_shape = list(net.blobs[input_layer].data.shape[1:])

    if '.npy' in datafile:
        np_images = np.load(datafile)
    else:
        np_images = load_data(datafile, input_shape)

    inputs = {input_layer: np_images}
    net.forward_all(**inputs)

    results = []
    names = []
    for k, v in net.blobs.items():
        k = k.replace('/', '_')
        names.append(k)
        results.append(v.data.copy())

    dump_path = 'results.caffe'
    dump_results(results, names, dump_path)
    print('all result of layers dumped to [%s]' % (dump_path))
    return 0


if __name__ == "__main__":
    """ maybe more convenient to use 'run.sh' to call this tool
    """
    net_file = 'models/resnet50/resnet50.py'
    weight_file = 'models/resnet50/resnet50.npy'
    datafile = 'data/65.jpeg'
    net_name = 'ResNet50'
    model_file = 'models/resnet50/fluid'

    ret = None
    if len(sys.argv) <= 2:
        pass
    elif sys.argv[1] == 'caffe':
        if len(sys.argv) != 5:
            print('usage:')
            print('\tpython %s caffe [prototxt] [caffemodel] [datafile]' %
                  (sys.argv[0]))
            sys.exit(1)
        prototxt = sys.argv[2]
        caffemodel = sys.argv[3]
        datafile = sys.argv[4]
        ret = caffe_infer(prototxt, caffemodel, datafile)
    elif sys.argv[1] == 'infer':
        if len(sys.argv) != 4:
            print('usage:')
            print('\tpython %s infer [fluid_model] [datafile]' % (sys.argv[0]))
            sys.exit(1)
        model_path = sys.argv[2]
        datafile = sys.argv[3]
        ret = infer(model_path, datafile)
    elif sys.argv[1] == 'dump':
        if len(sys.argv) != 6:
            print('usage:')
            print('\tpython %s dump [net_file] [weight_file] [datafile] [net_name]' \
                    % (sys.argv[0]))
            print('\teg:python %s dump %s %s %s %s' % (sys.argv[0],\
                net_file, weight_file, datafile, net_name))
            sys.exit(1)

        net_file = sys.argv[2]
        weight_file = sys.argv[3]
        datafile = sys.argv[4]
        net_name = sys.argv[5]
        ret = infer(weight_file, datafile, net_file, net_name)

    if ret is None:
        print('usage:')
        print(' python %s [infer] [fluid_model] [imgfile]' % (sys.argv[0]))
        print(' eg:python %s infer %s %s' % (sys.argv[0], model_file, datafile))
        sys.exit(1)

    sys.exit(ret)
