import os

import paddle.v2 as paddle
import paddle.fluid as fluid
from paddle.fluid.initializer import MSRA
from paddle.fluid.param_attr import ParamAttr

parameter_attr = ParamAttr(initializer=MSRA())


def conv_bn_layer(input,
                  filter_size,
                  num_filters,
                  stride,
                  padding,
                  channels=None,
                  num_groups=1,
                  act='relu',
                  use_cudnn=True):
    conv = fluid.layers.conv2d(
        input=input,
        num_filters=num_filters,
        filter_size=filter_size,
        stride=stride,
        padding=padding,
        groups=num_groups,
        act=None,
        use_cudnn=use_cudnn,
        param_attr=parameter_attr,
        bias_attr=False)
    return fluid.layers.batch_norm(input=conv, act=act)


def depthwise_separable(input, num_filters1, num_filters2, num_groups, stride,
                        scale):
    """
    """
    depthwise_conv = conv_bn_layer(
        input=input,
        filter_size=3,
        num_filters=int(num_filters1 * scale),
        stride=stride,
        padding=1,
        num_groups=int(num_groups * scale),
        use_cudnn=False)

    pointwise_conv = conv_bn_layer(
        input=depthwise_conv,
        filter_size=1,
        num_filters=int(num_filters2 * scale),
        stride=1,
        padding=0)
    return pointwise_conv


def mobile_net(img, class_dim, scale=1.0):

    # conv1: 112x112
    tmp = conv_bn_layer(
        img,
        filter_size=3,
        channels=3,
        num_filters=int(32 * scale),
        stride=2,
        padding=1)

    # 56x56
    tmp = depthwise_separable(
        tmp,
        num_filters1=32,
        num_filters2=64,
        num_groups=32,
        stride=1,
        scale=scale)

    tmp = depthwise_separable(
        tmp,
        num_filters1=64,
        num_filters2=128,
        num_groups=64,
        stride=2,
        scale=scale)

    # 28x28
    tmp = depthwise_separable(
        tmp,
        num_filters1=128,
        num_filters2=128,
        num_groups=128,
        stride=1,
        scale=scale)

    tmp = depthwise_separable(
        tmp,
        num_filters1=128,
        num_filters2=256,
        num_groups=128,
        stride=2,
        scale=scale)

    # 14x14
    tmp = depthwise_separable(
        tmp,
        num_filters1=256,
        num_filters2=256,
        num_groups=256,
        stride=1,
        scale=scale)

    tmp = depthwise_separable(
        tmp,
        num_filters1=256,
        num_filters2=512,
        num_groups=256,
        stride=2,
        scale=scale)

    # 14x14
    for i in range(5):
        tmp = depthwise_separable(
            tmp,
            num_filters1=512,
            num_filters2=512,
            num_groups=512,
            stride=1,
            scale=scale)
    # 7x7
    tmp = depthwise_separable(
        tmp,
        num_filters1=512,
        num_filters2=1024,
        num_groups=512,
        stride=2,
        scale=scale)

    tmp = depthwise_separable(
        tmp,
        num_filters1=1024,
        num_filters2=1024,
        num_groups=1024,
        stride=1,
        scale=scale)

    tmp = fluid.layers.pool2d(
        input=tmp,
        pool_size=0,
        pool_stride=1,
        pool_type='avg',
        global_pooling=True)

    tmp = fluid.layers.fc(input=tmp,
                          size=class_dim,
                          act='softmax',
                          param_attr=parameter_attr)
    return tmp


def train(learning_rate, batch_size, num_passes, model_save_dir='model'):
    class_dim = 102
    image_shape = [3, 224, 224]

    image = fluid.layers.data(name='image', shape=image_shape, dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')

    out = mobile_net(image, class_dim=class_dim)

    cost = fluid.layers.cross_entropy(input=out, label=label)
    avg_cost = fluid.layers.mean(x=cost)

    optimizer = fluid.optimizer.Momentum(
        learning_rate=learning_rate,
        momentum=0.9,
        regularization=fluid.regularizer.L2Decay(5 * 1e-5))
    opts = optimizer.minimize(avg_cost)
    accuracy = fluid.evaluator.Accuracy(input=out, label=label)

    inference_program = fluid.default_main_program().clone()
    with fluid.program_guard(inference_program):
        test_accuracy = fluid.evaluator.Accuracy(input=out, label=label)
        test_target = [avg_cost] + test_accuracy.metrics + test_accuracy.states
        inference_program = fluid.io.get_inference_program(test_target)

    place = fluid.CUDAPlace(0)
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    train_reader = paddle.batch(
        paddle.dataset.flowers.train(), batch_size=batch_size)
    test_reader = paddle.batch(
        paddle.dataset.flowers.test(), batch_size=batch_size)
    feeder = fluid.DataFeeder(place=place, feed_list=[image, label])

    for pass_id in range(num_passes):
        accuracy.reset(exe)
        for batch_id, data in enumerate(train_reader()):
            loss, acc = exe.run(fluid.default_main_program(),
                                feed=feeder.feed(data),
                                fetch_list=[avg_cost] + accuracy.metrics)
            print("Pass {0}, batch {1}, loss {2}, acc {3}".format(
                pass_id, batch_id, loss[0], acc[0]))
        pass_acc = accuracy.eval(exe)

        test_accuracy.reset(exe)
        for data in test_reader():
            loss, acc = exe.run(inference_program,
                                feed=feeder.feed(data),
                                fetch_list=[avg_cost] + test_accuracy.metrics)
        test_pass_acc = test_accuracy.eval(exe)
        print("End pass {0}, train_acc {1}, test_acc {2}".format(
            pass_id, pass_acc, test_pass_acc))
        if pass_id % 10 == 0:
            model_path = os.path.join(model_save_dir, str(pass_id))
            print 'save models to %s' % (model_path)
            fluid.io.save_inference_model(model_path, ['image'], [out], exe)


if __name__ == '__main__':
    train(learning_rate=0.005, batch_size=40, num_passes=300)
