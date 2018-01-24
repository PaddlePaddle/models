import os

import paddle.v2 as paddle
import paddle.v2.fluid as fluid
import reader


def conv_bn_layer(input, num_filters, filter_size, stride=1, groups=1,
                  act=None):
    conv = fluid.layers.conv2d(
        input=input,
        num_filters=num_filters,
        filter_size=filter_size,
        stride=stride,
        padding=(filter_size - 1) / 2,
        groups=groups,
        act=None,
        bias_attr=False)
    return fluid.layers.batch_norm(input=conv, act=act)


def squeeze_excitation(input, num_channels, reduction_ratio):
    pool = fluid.layers.pool2d(
        input=input, pool_size=0, pool_type='avg', global_pooling=True)
    squeeze = fluid.layers.fc(input=pool,
                              size=num_channels / reduction_ratio,
                              act='relu')
    excitation = fluid.layers.fc(input=squeeze,
                                 size=num_channels,
                                 act='sigmoid')
    scale = fluid.layers.elementwise_mul(x=input, y=excitation, axis=0)
    return scale


def shortcut(input, ch_out, stride):
    ch_in = input.shape[1]
    if ch_in != ch_out:
        if stride == 1:
            filter_size = 1
        else:
            filter_size = 3
        return conv_bn_layer(input, ch_out, filter_size, stride)
    else:
        return input


def bottleneck_block(input, num_filters, stride, cardinality, reduction_ratio):
    conv0 = conv_bn_layer(
        input=input, num_filters=num_filters, filter_size=1, act='relu')
    conv1 = conv_bn_layer(
        input=conv0,
        num_filters=num_filters,
        filter_size=3,
        stride=stride,
        groups=cardinality,
        act='relu')
    conv2 = conv_bn_layer(
        input=conv1, num_filters=num_filters * 2, filter_size=1, act=None)
    scale = squeeze_excitation(
        input=conv2,
        num_channels=num_filters * 2,
        reduction_ratio=reduction_ratio)

    short = shortcut(input, num_filters * 2, stride)

    return fluid.layers.elementwise_add(x=short, y=scale, act='relu')


def SE_ResNeXt(input, class_dim, infer=False):
    cardinality = 64
    reduction_ratio = 16
    depth = [3, 8, 36, 3]
    num_filters = [128, 256, 512, 1024]

    conv = conv_bn_layer(
        input=input, num_filters=64, filter_size=3, stride=2, act='relu')
    conv = conv_bn_layer(
        input=conv, num_filters=64, filter_size=3, stride=1, act='relu')
    conv = conv_bn_layer(
        input=conv, num_filters=128, filter_size=3, stride=1, act='relu')
    conv = fluid.layers.pool2d(
        input=conv, pool_size=3, pool_stride=2, pool_type='max')

    for block in range(len(depth)):
        for i in range(depth[block]):
            conv = bottleneck_block(
                input=conv,
                num_filters=num_filters[block],
                stride=2 if i == 0 and block != 0 else 1,
                cardinality=cardinality,
                reduction_ratio=reduction_ratio)

    pool = fluid.layers.pool2d(
        input=conv, pool_size=0, pool_type='avg', global_pooling=True)
    if not infer:
        drop = fluid.layers.dropout(x=pool, dropout_prob=0.2)
    else:
        drop = pool
    out = fluid.layers.fc(input=drop, size=class_dim, act='softmax')
    return out


def train(learning_rate, batch_size, num_passes, model_save_dir='model'):
    class_dim = 1000
    image_shape = [3, 224, 224]

    image = fluid.layers.data(name='image', shape=image_shape, dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')

    out = SE_ResNeXt(input=image, class_dim=class_dim)

    cost = fluid.layers.cross_entropy(input=out, label=label)
    avg_cost = fluid.layers.mean(x=cost)

    optimizer = fluid.optimizer.Momentum(
        learning_rate=learning_rate,
        momentum=0.9,
        regularization=fluid.regularizer.L2Decay(1e-4))
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

    train_reader = paddle.batch(reader.train(), batch_size=batch_size)
    test_reader = paddle.batch(reader.test(), batch_size=batch_size)
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

        model_path = os.path.join(model_save_dir, str(pass_id))
        fluid.io.save_inference_model(model_path, ['image'], [out], exe)


if __name__ == '__main__':
    train(learning_rate=0.1, batch_size=8, num_passes=100)
