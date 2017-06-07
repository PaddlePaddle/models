from PIL import Image
import gzip
import numpy as np
import paddle.v2 as paddle
from image import load_and_transform

__all__ = ['resnet_imagenet', 'resnet_cifar10']


def conv_bn_layer(input,
                  ch_out,
                  filter_size,
                  stride,
                  padding,
                  active_type=paddle.activation.Relu(),
                  ch_in=None):
    tmp = paddle.layer.img_conv(
        input=input,
        filter_size=filter_size,
        num_channels=ch_in,
        num_filters=ch_out,
        stride=stride,
        padding=padding,
        act=paddle.activation.Linear(),
        bias_attr=False)
    return paddle.layer.batch_norm(input=tmp, act=active_type)


def shortcut(input, n_out, stride, b_projection):
    if b_projection:
        return conv_bn_layer(input, n_out, 1, stride, 0,
                             paddle.activation.Linear())
    else:
        return input


def basicblock(input, ch_out, stride, b_projection):
    # TODO: bug fix for ch_in = input.num_filters
    conv1 = conv_bn_layer(input, ch_out, 3, stride, 1)
    conv2 = conv_bn_layer(conv1, ch_out, 3, 1, 1, paddle.activation.Linear())
    short = shortcut(input, ch_out, stride, b_projection)
    return paddle.layer.addto(
        input=[conv2, short], act=paddle.activation.Relu())


def bottleneck(input, ch_out, stride, b_projection):
    # TODO: bug fix for ch_in = input.num_filters
    short = shortcut(input, ch_out * 4, stride, b_projection)
    conv1 = conv_bn_layer(input, ch_out, 1, stride, 0)
    conv2 = conv_bn_layer(conv1, ch_out, 3, 1, 1)
    conv3 = conv_bn_layer(conv2, ch_out * 4, 1, 1, 0,
                          paddle.activation.Linear())
    return paddle.layer.addto(
        input=[conv3, short], act=paddle.activation.Relu())


def layer_warp(block_func, input, features, count, stride):
    conv = block_func(input, features, stride, True)
    for i in range(1, count):
        conv = block_func(conv, features, 1, False)
    return conv


def resnet_imagenet(input, depth=50):
    cfg = {
        18: ([2, 2, 2, 1], basicblock),
        34: ([3, 4, 6, 3], basicblock),
        50: ([3, 4, 6, 3], bottleneck),
        101: ([3, 4, 23, 3], bottleneck),
        152: ([3, 8, 36, 3], bottleneck)
    }
    stages, block_func = cfg[depth]
    conv1 = conv_bn_layer(
        input, ch_in=3, ch_out=64, filter_size=7, stride=2, padding=3)
    pool1 = paddle.layer.img_pool(input=conv1, pool_size=3, stride=2)
    res1 = layer_warp(block_func, pool1, 64, stages[0], 1)
    res2 = layer_warp(block_func, res1, 128, stages[1], 2)
    res3 = layer_warp(block_func, res2, 256, stages[2], 2)
    res4 = layer_warp(block_func, res3, 512, stages[3], 2)
    pool2 = paddle.layer.img_pool(
        input=res4, pool_size=7, stride=1, pool_type=paddle.pooling.Avg())
    return pool2


def resnet_cifar10(input, depth=32):
    # depth should be one of 20, 32, 44, 56, 110, 1202
    assert (depth - 2) % 6 == 0
    n = (depth - 2) / 6
    nStages = {16, 64, 128}
    conv1 = conv_bn_layer(
        input, ch_in=3, ch_out=16, filter_size=3, stride=1, padding=1)
    res1 = layer_warp(basicblock, conv1, 16, n, 1)
    res2 = layer_warp(basicblock, res1, 32, n, 2)
    res3 = layer_warp(basicblock, res2, 64, n, 2)
    pool = paddle.layer.img_pool(
        input=res3, pool_size=8, stride=1, pool_type=paddle.pooling.Avg())
    return pool


def load_image(file, mean_file):
    im = load_and_transform(file, 256, 224, is_train=False)
    im = im[(2, 1, 0), :, :]
    mu = np.load(mean_file)
    mu = mu.mean(1).mean(1)
    im = im - mu[:, None, None]
    im = im.flatten()
    im = im / 255.0
    return im


DATA_DIM = 3 * 224 * 224
CLASS_DIM = 1000
BATCH_SIZE = 128

MODEL_FILE = 'paddle_resnet50.tar.gz'

if __name__ == "__main__":
    paddle.init(use_gpu=False, trainer_count=1)

    img = paddle.layer.data(
        "image", type=paddle.data_type.dense_vector(DATA_DIM))
    out = paddle.layer.fc(
        input=resnet_imagenet(img, 50),
        size=1000,
        act=paddle.activation.Softmax())

    parameters = paddle.parameters.Parameters.from_tar(gzip.open(MODEL_FILE))

    test_data = []
    test_data.append((load_image("./images/cat.jpg"), ))
    output_prob = paddle.infer(
        output_layer=out, parameters=parameters, input=test_data,
        field="value")[0]

    print np.sort(output_prob)[::-1]
    print np.argsort(output_prob)[::-1]
    print 'predicted class is:', output_prob.argmax()
