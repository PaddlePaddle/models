import os
import numpy as np
import sys
import time
import paddle.v2 as paddle
import paddle.fluid as fluid
import reader

#fluid.default_startup_program().random_seed = 111


def load_persistables_if_exist(executor, dirname, main_program=None):
    filenames = next(os.walk(dirname))[2]
    filenames = set(filenames)

    def _is_presistable_and_exist_(var):
        if not fluid.io.is_persistable(var):
            return False
        else:
            return var.name in filenames

    fluid.io.load_vars(
        executor,
        dirname,
        main_program=main_program,
        vars=None,
        predicate=_is_presistable_and_exist_)


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
    return fluid.layers.batch_norm(input=conv, act=act, momentum=0.1)


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


def SE_ResNeXt50(input, class_dim):
    cardinality = 32
    reduction_ratio = 16
    depth = [3, 4, 6, 3]
    num_filters = [128, 256, 512, 1024]

    conv = conv_bn_layer(
        input=input, num_filters=64, filter_size=7, stride=2, act='relu')
    conv = fluid.layers.pool2d(
        input=conv, pool_size=3, pool_stride=2, pool_padding=1, pool_type='max')

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
    out = fluid.layers.fc(input=pool, size=class_dim, act='softmax')
    return out


def net_conf(image, label, class_dim):
    out = SE_ResNeXt50(input=image, class_dim=class_dim)
    cost = fluid.layers.cross_entropy(input=out, label=label)
    avg_cost = fluid.layers.mean(x=cost)
    #accuracy = fluid.evaluator.Accuracy(input=out, label=label)
    #accuracy5 = fluid.evaluator.Accuracy(input=out, label=label, k=5)
    accuracy = fluid.layers.accuracy(input=out, label=label)
    accuracy5 = fluid.layers.accuracy(input=out, label=label, k=5)
    return out, avg_cost, accuracy, accuracy5


def train(learning_rate,
          batch_size,
          num_passes,
          init_model=None,
          model_save_dir='model'):
    class_dim = 1000
    image_shape = [3, 224, 224]

    image = fluid.layers.data(name='image', shape=image_shape, dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')

    parallel = True
    use_nccl = True
    if parallel:
        places = fluid.layers.get_places()
        pd = fluid.layers.ParallelDo(places, use_nccl=use_nccl)
        with pd.do():
            img_ = pd.read_input(image)
            label_ = pd.read_input(label)
            prediction, avg_cost, accuracy, accuracy5 = net_conf(img_, label_,
                                                                 class_dim)

            for o in [avg_cost, accuracy, accuracy5]:
                pd.write_output(o)

        avg_cost, accuracy, accuracy5 = pd()
        # get mean loss and acc through every devices.
        avg_cost = fluid.layers.mean(x=avg_cost)
        accuracy = fluid.layers.mean(x=accuracy)
        accuracy5 = fluid.layers.mean(x=accuracy5)
    else:
        prediction, avg_cost, accuracy, accuracy5 = net_conf(image, label,
                                                             class_dim)

    #print("network:", fluid.default_main_program())
    #print("network:", fluid.default_startup_program())

    inference_program = fluid.default_main_program().clone()
    epoch = [30, 60, 90]
    total_images = 1281167
    pass_each_epoch = int(total_images / batch_size + 1)
    bd = [e * pass_each_epoch for e in epoch]
    lr = [0.1, 0.01, 0.001, 0.0001]

    print("Training with learning rates:", bd, lr)

    optimizer = fluid.optimizer.Momentum(
        learning_rate=fluid.layers.piecewise_decay(
            boundaries=bd, values=lr),
        momentum=0.9,
        regularization=fluid.regularizer.L2Decay(1e-4))
    opts = optimizer.minimize(avg_cost)
    fluid.memory_optimize(fluid.default_main_program())

    place = fluid.CUDAPlace(0)
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    if init_model is not None:
        load_persistables_if_exist(exe, init_model)
        #fluid.io.load_persistables(exe, init_model)

    train_reader = paddle.batch(reader.train(), batch_size=batch_size)
    test_reader = paddle.batch(reader.test(), batch_size=batch_size)
    feeder = fluid.DataFeeder(place=place, feed_list=[image, label])

    for pass_id in range(0, num_passes):
        train_info = [[], [], []]
        test_info = [[], [], []]
        for batch_id, data in enumerate(train_reader()):
            t1 = time.time()
            loss, acc, acc5 = exe.run(
                fluid.default_main_program(),
                feed=feeder.feed(data),
                fetch_list=[avg_cost, accuracy, accuracy5])
            t2 = time.time()
            period = t2 - t1
            train_info[0].append(loss[0])
            train_info[1].append(acc[0])
            train_info[2].append(acc5[0])
            if batch_id % 10 == 0:
                print(
                    "Pass {0}, trainbatch {1}, loss {2}, acc {3}, acc5 {4} time{5}".
                    format(pass_id, batch_id, loss[0], acc[0], acc5[0],
                           "%2.2f sec" % period))
                sys.stdout.flush()
            #if batch_id == 10:
            #    break
        train_loss = np.array(train_info[0]).mean()
        train_acc = np.array(train_info[1]).mean()
        train_acc5 = np.array(train_info[2]).mean()

        for batch_id, data in enumerate(test_reader()):
            t1 = time.time()
            loss, acc, acc5 = exe.run(
                inference_program,
                feed=feeder.feed(data),
                fetch_list=[avg_cost, accuracy, accuracy5])
            t2 = time.time()
            period = t2 - t1
            test_info[0].append(loss[0])
            test_info[1].append(acc[0])
            test_info[2].append(acc5[0])
            if batch_id % 10 == 0:
                print(
                    "Pass {0}, testbatch {1}, loss {2}, acc {3}, acc5 {4} time{5}".
                    format(pass_id, batch_id, loss[0], acc[0], acc5[0],
                           "%2.2f sec" % period))
                sys.stdout.flush()
            #if batch_id == 10:
            #    break

        test_loss = np.array(test_info[0]).mean()
        test_acc = np.array(test_info[1]).mean()
        test_acc5 = np.array(test_info[2]).mean()
        print(
            "End pass {0}, train_loss {1}, train_acc {2}, train_acc5 {3}, test_loss {4}, test_acc {5}, test_acc5 {6}".
            format(pass_id, train_loss, train_acc, train_acc5, test_loss,
                   test_acc, test_acc5))
        sys.stdout.flush()

        model_path = os.path.join(model_save_dir, str(pass_id))
        if not os.path.isdir(model_path):
            os.makedirs(model_path)
        fluid.io.save_persistables(exe, model_path)


if __name__ == '__main__':
    train(learning_rate=0.1, batch_size=256, num_passes=120, init_model=None)
