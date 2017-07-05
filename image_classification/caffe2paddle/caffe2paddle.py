import os
import struct
import gzip
import tarfile
import cStringIO
import numpy as np
import cv2
import caffe
from paddle.proto.ParameterConfig_pb2 import ParameterConfig
from paddle.trainer_config_helpers.default_decorators import wrap_name_default


class ModelConverter(object):
    def __init__(self, caffe_model_file, caffe_pretrained_file,
                 paddle_tar_name):
        self.net = caffe.Net(caffe_model_file, caffe_pretrained_file,
                             caffe.TEST)
        self.tar_name = paddle_tar_name
        self.params = dict()
        self.pre_layer_name = ""
        self.pre_layer_type = ""

    def convert(self, name_map=None):
        layer_dict = self.net.layer_dict
        for layer_name in layer_dict.keys():
            layer = layer_dict[layer_name]
            layer_params = layer.blobs
            layer_type = layer.type
            if len(layer_params) > 0:
                self.pre_layer_name = getattr(
                    self, "convert_" + layer_type + "_layer")(
                        layer_params,
                        name=None
                        if name_map == None else name_map.get(layer_name))
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
            dims = list(data.shape)
            if len(dims) == 1:
                dims.insert(1, 1)
                param_conf.dims.extend(dims)
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
        scale = 1 / np.array(params[-1].data)[0] if np.array(
            params[-1].data)[0] != 0 else 0
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

    def caffe_predict(self,
                      img,
                      mean_file='./caffe/imagenet/ilsvrc_2012_mean.npy'):
        net = self.net

        net.blobs['data'].data[...] = load_image(img, mean_file=mean_file)
        out = net.forward()

        output_prob = net.blobs['prob'].data[0].flatten()
        print zip(np.argsort(output_prob)[::-1], np.sort(output_prob)[::-1])


def load_image(file, resize_size=256, crop_size=224, mean_file=None):
    # load image
    im = cv2.imread(file)
    # resize
    h, w = im.shape[:2]
    h_new, w_new = resize_size, resize_size
    if h > w:
        h_new = resize_size * h / w
    else:
        w_new = resize_size * w / h
    im = cv2.resize(im, (h_new, w_new), interpolation=cv2.INTER_CUBIC)
    # crop
    h, w = im.shape[:2]
    h_start = (h - crop_size) / 2
    w_start = (w - crop_size) / 2
    h_end, w_end = h_start + crop_size, w_start + crop_size
    im = im[h_start:h_end, w_start:w_end, :]
    # transpose to CHW order
    im = im.transpose((2, 0, 1))

    if mean_file:
        mu = np.load(mean_file)
        mu = mu.mean(1).mean(1)
        im = im - mu[:, None, None]
    im = im / 255.0
    return im


if __name__ == "__main__":
    caffe_model_file = "./ResNet-50-deploy.prototxt"
    caffe_pretrained_file = "./ResNet-50-model.caffemodel"
    paddle_tar_name = "Paddle_ResNet50.tar.gz"

    converter = ModelConverter(
        caffe_model_file=caffe_model_file,
        caffe_pretrained_file=caffe_pretrained_file,
        paddle_tar_name=paddle_tar_name)
    converter.convert()

    converter.caffe_predict("./cat.jpg",
                            "./caffe/imagenet/ilsvrc_2012_mean.npy")
