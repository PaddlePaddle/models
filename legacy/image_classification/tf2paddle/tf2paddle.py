import os
import re
import collections
import struct
import gzip
import tarfile
import cStringIO
import numpy as np

import tensorflow as tf

from paddle.proto.ParameterConfig_pb2 import ParameterConfig
from paddle.trainer_config_helpers.default_decorators import wrap_name_default


class ModelConverter(object):
    def __init__(self,
                 paddle_tar_name,
                 param_name_map=None,
                 layer_name_map=None,
                 layer_type_map=None):
        self.tar_name = paddle_tar_name
        self.param_name_map = param_name_map
        self.layer_name_map = layer_name_map
        self.layer_type_map = layer_type_map
        self.params = dict()

    def convert(self):
        layers_params = self.arrange_layer_params()
        for layer_name in layers_params.keys():
            layer_params, layer_params_names, layer_type = layers_params[
                layer_name]
            if len(layer_params) > 0:
                if not layer_type:
                    assert layer_type_map and (
                        layer_type_map.get(layer_name) in ["conv", "bn", "fc"])
                    layer_type = layer_type_map[layer_name]
                self.pre_layer_name = getattr(
                    self, "convert_" + layer_type + "_layer")(
                        layer_params,
                        params_names=[
                            self.param_name_map.get(name)
                            if self.param_name_map else None
                            for name in layer_params_names
                        ],
                        name=None if self.layer_name_map == None else
                        self.layer_name_map.get(layer_name))
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


class TFModelConverter(ModelConverter):
    def __init__(self,
                 tf_net,
                 paddle_tar_name,
                 param_name_map=None,
                 layer_name_map=None,
                 layer_type_map=None):
        super(TFModelConverter, self).__init__(paddle_tar_name, param_name_map,
                                               layer_name_map, layer_type_map)
        self.sess = __import__(tf_net).build_model()

    def arrange_layer_params(self):
        all_vars = tf.global_variables()
        layers_params = collections.OrderedDict()
        for var in all_vars:
            var_name = var.name
            scope_pos = var_name.rfind('/')
            if scope_pos != -1:
                layer_scope = var_name[:scope_pos]
                if layers_params.has_key(layer_scope):
                    layer_params, layer_params_names, layer_type = layers_params[
                        layer_scope]
                    layer_params.append(var.eval(self.sess))
                    layer_params_names.append(var_name)
                else:
                    layer_type = re.search('conv|bn|fc', layer_scope)
                    layers_params[layer_scope] = ([var.eval(self.sess)],
                                                  [var_name], layer_type.group()
                                                  if layer_type else None)
        return layers_params

    @wrap_name_default("conv")
    def convert_conv_layer(self, params, params_names=None, name=None):
        for i in range(len(params)):
            data = np.transpose(params[i], (
                3, 2, 0, 1)) if len(params[i].shape) == 4 else params[i]
            if len(params) == 2:
                suffix = "0" if i == 0 else "bias"
                file_name = "_%s.w%s" % (name, suffix) if not (
                    params_names and params_names[i]) else params_names[i]
            else:
                file_name = "_%s.w%s" % (name, str(i)) if not (
                    params_names and params_names[i]) else params_names[i]
            param_conf = ParameterConfig()
            param_conf.name = file_name
            dims = list(data.shape)
            if len(dims) == 1:
                dims.insert(1, 1)
                param_conf.dims.extend(dims)
            param_conf.size = reduce(lambda a, b: a * b, data.shape)
            self.params[file_name] = (param_conf, data.flatten())

    @wrap_name_default("fc_layer")
    def convert_fc_layer(self, params, params_names=None, name=None):
        for i in range(len(params)):
            data = params[i]
            if len(params) == 2:
                suffix = "0" if i == 0 else "bias"
                file_name = "_%s.w%s" % (name, suffix) if not (
                    params_names and params_names[i]) else params_names[i]
            else:
                file_name = "_%s.w%s" % (name, str(i)) if not (
                    params_names and params_names[i]) else params_names[i]
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
    def convert_bn_layer(self, params, params_names=None, name=None):
        params = [params[i] for i in (0, 2, 3, 1)]
        params_names = [params_names[i]
                        for i in (0, 2, 3, 1)] if params_names else params_names
        for i in range(len(params)):
            data = params[i]
            file_name = "_%s.w%s" % (name, str(i)) if i < 3 else "_%s.w%s" % (
                name, "bias")
            file_name = file_name if not (params_names and
                                          params_names[i]) else params_names[i]
            param_conf = ParameterConfig()
            param_conf.name = file_name
            dims = list(data.shape)
            assert len(dims) == 1
            dims.insert(0, 1)
            param_conf.size = reduce(lambda a, b: a * b, dims)
            param_conf.dims.extend(dims)
            self.params[file_name] = (param_conf, data.flatten())
        return name


if __name__ == "__main__":
    tf_net = "TF_ResNet"
    paddle_tar_name = "Paddle_ResNet50.tar.gz"

    converter = TFModelConverter(tf_net=tf_net, paddle_tar_name=paddle_tar_name)
    converter.convert()
