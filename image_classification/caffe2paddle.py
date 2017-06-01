# -*- coding: utf-8 -*-
import os
import functools
import inspect
import struct
import gzip
import tarfile
import cStringIO
import numpy as np
import caffe
from paddle.proto.ParameterConfig_pb2 import ParameterConfig


def __default_not_set_callback__(kwargs, name):
    return name not in kwargs or kwargs[name] is None


def wrap_param_default(param_names=None,
                       default_factory=None,
                       not_set_callback=__default_not_set_callback__):
    assert param_names is not None
    assert isinstance(param_names, list) or isinstance(param_names, tuple)
    for each_param_name in param_names:
        assert isinstance(each_param_name, basestring)

    def __impl__(func):
        @functools.wraps(func)
        def __wrapper__(*args, **kwargs):
            if len(args) != 0:
                argspec = inspect.getargspec(func)
                num_positional = len(argspec.args)
                if argspec.defaults:
                    num_positional -= len(argspec.defaults)
                assert argspec.varargs or len(
                    args
                ) <= num_positional, "Must use keyword arguments for non-positional args"
            for name in param_names:
                if not_set_callback(kwargs, name):  # Not set
                    kwargs[name] = default_factory(func)
            return func(*args, **kwargs)

        if hasattr(func, "argspec"):
            __wrapper__.argspec = func.argspec
        else:
            __wrapper__.argspec = inspect.getargspec(func)
        return __wrapper__

    return __impl__


class DefaultNameFactory(object):
    def __init__(self, name_prefix):
        self.__counter__ = 0
        self.__name_prefix__ = name_prefix

    def __call__(self, func):
        if self.__name_prefix__ is None:
            self.__name_prefix__ = func.__name__
        tmp = "__%s_%d__" % (self.__name_prefix__, self.__counter__)
        self.__check_name__(tmp)
        self.__counter__ += 1
        return tmp

    def __check_name__(self, nm):
        pass

    def reset(self):
        self.__counter__ = 0


def wrap_name_default(name_prefix=None, name_param="name"):
    """
    Decorator to set "name" arguments default to "{name_prefix}_{invoke_count}".

    ..  code:: python

        @wrap_name_default("some_name")
        def func(name=None):
            print name      # name will never be None. If name is not set,
                            # name will be "some_name_%d"

    :param name_prefix: name prefix. wrapped function"s __name__ if None.
    :type name_prefix: basestring
    :return: a decorator to set default name
    :rtype: callable
    """
    factory = DefaultNameFactory(name_prefix)
    return wrap_param_default([name_param], factory)


class ModelConverter(object):
    def __init__(self, caffe_model_file, caffe_pretrained_file,
                 paddle_tar_name):
        self.net = caffe.Net(caffe_model_file, caffe_pretrained_file,
                             caffe.TEST)
        self.tar_name = paddle_tar_name
        self.params = dict()
        self.pre_layer_name = ""
        self.pre_layer_type = ""

    def convert(self):
        layer_dict = self.net.layer_dict
        for layer_name in layer_dict.keys():
            layer = layer_dict[layer_name]
            layer_params = layer.blobs
            layer_type = layer.type
            if len(layer_params) > 0:
                self.pre_layer_name = getattr(
                    self, "convert_" + layer_type + "_layer")(layer_params)
            self.pre_layer_type = layer_type
        with gzip.open(self.tar_name, 'w') as f:
            self.to_tar(f)
        return

    def to_tar(self, f):
        tar = tarfile.TarFile(fileobj=f, mode='w')
        for param_name in self.params.keys():
            param_conf, param_data = self.params[param_name]

            confStr = param_conf.SerializeToString()
            tarinfo = tarfile.TarInfo(name="%s.protobuf" % param_name)
            tarinfo.size = len(confStr)
            buf = cStringIO.StringIO(confStr)
            buf.seek(0)
            tar.addfile(tarinfo, fileobj=buf)

            buf = cStringIO.StringIO()
            self.serialize(param_data, buf)
            tarinfo = tarfile.TarInfo(name=param_name)
            buf.seek(0)
            tarinfo.size = len(buf.getvalue())
            tar.addfile(tarinfo, buf)

    @staticmethod
    def serialize(data, f):
        f.write(struct.pack("IIQ", 0, 4, data.size))
        f.write(data.tobytes())

    @wrap_name_default("conv")
    def convert_Convolution_layer(self, params, name=None):
        for i in range(len(params)):
            data = np.array(params[i].data)
            if len(params) == 2:
                suffix = "0" if i == 0 else "bias"
                file_name = "_%s.w%s" % (name, suffix)
            else:
                file_name = "_%s.w%s" % (name, str(i))
            param_conf = ParameterConfig()
            param_conf.name = file_name
            param_conf.size = reduce(lambda a, b: a * b, data.shape)
            self.params[file_name] = (param_conf, data.flatten())
        return name

    @wrap_name_default("fc_layer")
    def convert_InnerProduct_layer(self, params, name=None):
        for i in range(len(params)):
            data = np.array(params[i].data)
            if len(params) == 2:
                suffix = "0" if i == 0 else "bias"
                file_name = "_%s.w%s" % (name, suffix)
            else:
                file_name = "_%s.w%s" % (name, str(i))
            data = np.transpose(data)
            param_conf = ParameterConfig()
            param_conf.name = file_name
            dims = list(data.shape)
            if len(dims) < 2:
                dims.insert(0, 1)
            param_conf.size = reduce(lambda a, b: a * b, dims)
            param_conf.dims.extend(dims)
            self.params[file_name] = (param_conf, data.flatten())
        return name

    @wrap_name_default("batch_norm")
    def convert_BatchNorm_layer(self, params, name=None):
        scale = np.array(params[-1].data)
        for i in range(2):
            data = np.array(params[i].data) * scale
            file_name = "_%s.w%s" % (name, str(i + 1))
            param_conf = ParameterConfig()
            param_conf.name = file_name
            dims = list(data.shape)
            assert len(dims) == 1
            dims.insert(0, 1)
            param_conf.size = reduce(lambda a, b: a * b, dims)
            param_conf.dims.extend(dims)
            self.params[file_name] = (param_conf, data.flatten())
        return name

    def convert_Scale_layer(self, params, name=None):
        assert self.pre_layer_type == "BatchNorm"
        name = self.pre_layer_name
        for i in range(len(params)):
            data = np.array(params[i].data)
            suffix = "0" if i == 0 else "bias"
            file_name = "_%s.w%s" % (name, suffix)
            param_conf = ParameterConfig()
            param_conf.name = file_name
            dims = list(data.shape)
            assert len(dims) == 1
            dims.insert(0, 1)
            param_conf.size = reduce(lambda a, b: a * b, dims)
            if i == 1:
                param_conf.dims.extend(dims)
            self.params[file_name] = (param_conf, data.flatten())
        return name


if __name__ == "__main__":
    converter = ModelConverter("./VGG_ILSVRC_16_layers_deploy.prototxt",
                               "./VGG_ILSVRC_16_layers.caffemodel",
                               "/Users/baidu/caffe/caffe/python/paddle_model",
                               "test_vgg16.tar.gz")
    converter.convert()
