import os
import functools
import inspect
import struct
import numpy as np
import caffe


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
                 paddle_output_path):
        self.net = caffe.Net(caffe_model_file, caffe_pretrained_file,
                             caffe.TEST)
        self.output_path = paddle_output_path
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
        return

    @staticmethod
    def write_parameter(outfile, feats):
        version = 0
        value_size = 4
        fo = open(outfile, "wb")
        header = ""
        header += struct.pack("i", version)
        header += struct.pack("I", value_size)
        header += struct.pack("Q", feats.size)
        fo.write(header + feats.tostring())

    @wrap_name_default("conv")
    def convert_Convolution_layer(self, params, name=None):
        for i in range(len(params)):
            data = np.array(params[i].data)
            if len(params) == 2:
                suffix = "0" if i == 0 else "bias"
                file = os.path.join(self.output_path,
                                    "_%s.w%s" % (name, suffix))
            else:
                file = os.path.join(self.output_path,
                                    "_%s.w%s" % (name, str(i)))
            ModelConverter.write_parameter(file, data.flatten())
        return name

    @wrap_name_default("fc_layer")
    def convert_InnerProduct_layer(self, params, name=None):
        for i in range(len(params)):
            data = np.array(params[i].data)
            if len(params) == 2:
                suffix = "0" if i == 0 else "bias"
                file = os.path.join(self.output_path,
                                    "_%s.w%s" % (name, suffix))
            else:
                file = os.path.join(self.output_path,
                                    "_%s.w%s" % (name, str(i)))
            data = np.transpose(data)
            ModelConverter.write_parameter(file, data.flatten())
        return name

    @wrap_name_default("batch_norm")
    def convert_BatchNorm_layer(self, params, name=None):
        scale = np.array(params[-1].data)
        for i in range(2):
            data = np.array(params[i].data) * scale
            file = os.path.join(self.output_path,
                                "_%s.w%s" % (name, str(i + 1)))
            ModelConverter.write_parameter(file, data.flatten())
        return name

    def convert_Scale_layer(self, params, name=None):
        assert self.pre_layer_type == "BatchNorm"
        name = self.pre_layer_name
        for i in range(len(params)):
            data = np.array(params[i].data)
            suffix = "0" if i == 0 else "bias"
            file = os.path.join(self.output_path, "_%s.w%s" % (name, suffix))
            ModelConverter.write_parameter(file, data.flatten())
        return name


if __name__ == "__main__":
    converter = ModelConverter("./ResNet-101-deploy.prototxt",
                               "./ResNet-101-model.caffemodel",
                               "./caffe2paddle_resnet/")
    converter.convert()
