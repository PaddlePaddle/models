import os
import numpy as np
import time
import sys
#import paddle.v2 as paddle
import paddle
import paddle.fluid as fluid
import reader
import paddle.fluid.layers.control_flow as control_flow
import paddle.fluid.layers.nn as nn
import paddle.fluid.layers.ops as ops
import paddle.fluid.layers.tensor as tensor
from paddle.fluid.initializer import init_on_cpu
from paddle.fluid.layers.learning_rate_scheduler import _decay_step_counter
import math


def cosine_decay(learning_rate, step_each_epoch, epochs=120):
    """Applies cosine decay to the learning rate.
    lr = 0.05 * (math.cos(epoch * (math.pi / 120)) + 1)
    """
    global_step = _decay_step_counter()

    with init_on_cpu():
        # update 
        epoch = ops.floor(global_step / step_each_epoch)
        decayed_lr = learning_rate * \
                     (ops.cos(epoch * (math.pi / epochs)) + 1)/2
    #if global_step % step_each_epoch == 0:
    #    print("epoch={0}, global_step={1},decayed_lr={2} \
    #          (step_each_epoch={3})".format( \
    #          epoch,global_step,decayed_lr,step_each_epoch))
    return decayed_lr


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
    ### initializer parameter
    #print >> sys.stderr, "pool shape:", pool.shape
    stdv = 1.0 / math.sqrt(pool.shape[1] * 1.0)
    squeeze = fluid.layers.fc(input=pool,
                              size=num_channels / reduction_ratio,
                              act='relu',
                              param_attr=fluid.param_attr.ParamAttr(
                                  initializer=fluid.initializer.Uniform(-stdv,
                                                                        stdv)))
    #print >> sys.stderr, "squeeze shape:", squeeze.shape
    stdv = 1.0 / math.sqrt(squeeze.shape[1] * 1.0)
    excitation = fluid.layers.fc(input=squeeze,
                                 size=num_channels,
                                 act='sigmoid',
                                 param_attr=fluid.param_attr.ParamAttr(
                                     initializer=fluid.initializer.Uniform(
                                         -stdv, stdv)))
    scale = fluid.layers.elementwise_mul(x=input, y=excitation, axis=0)
    return scale


def shortcut_old(input, ch_out, stride):
    ch_in = input.shape[1]
    if ch_in != ch_out:
        if stride == 1:
            filter_size = 1
        else:
            filter_size = 3
        return conv_bn_layer(input, ch_out, filter_size, stride)
    else:
        return input


def shortcut(input, ch_out, stride):
    ch_in = input.shape[1]
    if ch_in != ch_out or stride != 1:
        filter_size = 1
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


def SE_ResNeXt(input, class_dim, infer=False, layers=50):
    supported_layers = [50, 152]
    if layers not in supported_layers:
        print("supported layers are", supported_layers, \
              "but input layer is ", layers)
        exit()
    if layers == 50:
        cardinality = 32
        reduction_ratio = 16
        depth = [3, 4, 6, 3]
        num_filters = [128, 256, 512, 1024]

        conv = conv_bn_layer(
            input=input, num_filters=64, filter_size=7, stride=2, act='relu')
        conv = fluid.layers.pool2d(
            input=conv,
            pool_size=3,
            pool_stride=2,
            pool_padding=1,
            pool_type='max')
    elif layers == 152:
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
            input=conv, pool_size=3, pool_stride=2, pool_padding=1, \
            pool_type='max')

    for block in range(len(depth)):
        for i in range(depth[block]):
            conv = bottleneck_block(
                input=conv,
                num_filters=num_filters[block],
                stride=2 if i == 0 and block != 0 else 1,
                cardinality=cardinality,
                reduction_ratio=reduction_ratio)

    pool = fluid.layers.pool2d(
        input=conv, pool_size=7, pool_type='avg', global_pooling=True)
    if not infer:
        drop = fluid.layers.dropout(x=pool, dropout_prob=0.5)
    else:
        drop = pool
    #print >> sys.stderr, "drop shape:", drop.shape
    stdv = 1.0 / math.sqrt(drop.shape[1] * 1.0)
    out = fluid.layers.fc(input=drop,
                          size=class_dim,
                          act='softmax',
                          param_attr=fluid.param_attr.ParamAttr(
                              initializer=fluid.initializer.Uniform(-stdv,
                                                                    stdv)))
    return out


def train(learning_rate,
          batch_size,
          num_epochs,
          init_model=None,
          model_save_dir='model',
          parallel=True,
          use_nccl=True,
          lr_strategy=None,
          class_dim=10000,
          layers=50):
    image_shape = [3, 224, 224]

    image = fluid.layers.data(name='image', shape=image_shape, dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')

    if parallel:
        places = fluid.layers.get_places()
        pd = fluid.layers.ParallelDo(places, use_nccl=use_nccl)

        with pd.do():
            image_ = pd.read_input(image)
            label_ = pd.read_input(label)
            out = SE_ResNeXt(input=image_, class_dim=class_dim, layers=layers)
            cost = fluid.layers.cross_entropy(input=out, label=label_)
            avg_cost = fluid.layers.mean(x=cost)
            acc_top1 = fluid.layers.accuracy(input=out, label=label_, k=1)
            acc_top5 = fluid.layers.accuracy(input=out, label=label_, k=5)
            pd.write_output(avg_cost)
            pd.write_output(acc_top1)
            pd.write_output(acc_top5)

        avg_cost, acc_top1, acc_top5 = pd()
        avg_cost = fluid.layers.mean(x=avg_cost)
        acc_top1 = fluid.layers.mean(x=acc_top1)
        acc_top5 = fluid.layers.mean(x=acc_top5)
    else:
        out = SE_ResNeXt(input=image, class_dim=class_dim, layers=layers)
        cost = fluid.layers.cross_entropy(input=out, label=label)
        avg_cost = fluid.layers.mean(x=cost)
        acc_top1 = fluid.layers.accuracy(input=out, label=label, k=1)
        acc_top5 = fluid.layers.accuracy(input=out, label=label, k=5)

    inference_program = fluid.default_main_program().clone(for_test=True)

    if "piecewise_decay" in lr_strategy:
        bd = lr_strategy["piecewise_decay"]["bd"]
        lr = lr_strategy["piecewise_decay"]["lr"]
        optimizer = fluid.optimizer.Momentum(
            learning_rate=fluid.layers.piecewise_decay(
                boundaries=bd, values=lr),
            momentum=0.9,
            regularization=fluid.regularizer.L2Decay(1e-4))
    elif "cosine_decay" in lr_strategy:
        step_each_epoch = lr_strategy["cosine_decay"]["step_each_epoch"]
        epochs = lr_strategy["cosine_decay"]["epochs"]
        optimizer = fluid.optimizer.Momentum(
            learning_rate=cosine_decay(
                learning_rate=learning_rate,
                step_each_epoch=step_each_epoch,
                epochs=epochs),
            momentum=0.9,
            regularization=fluid.regularizer.L2Decay(1e-4))
    else:
        optimizer = fluid.optimizer.Momentum(
            learning_rate=learning_rate,
            momentum=0.9,
            regularization=fluid.regularizer.L2Decay(1e-4))
    opts = optimizer.minimize(avg_cost)
    fluid.memory_optimize(fluid.default_main_program())

    #inference_program = fluid.default_main_program().clone()
    #with fluid.program_guard(inference_program):
    #    inference_program = fluid.io.get_inference_program([avg_cost, acc_top1, acc_top5])

    place = fluid.CUDAPlace(0)
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    #fluid.io.save_inference_model('./debug/',
    #                  ["image"],
    #                  [out],
    #                  exe,
    #                  main_program=None,
    #                  model_filename='model',
    #                  params_filename='params')
    #exit()

    start_epoch = 0
    if init_model is not None:
        load_model = model_save_dir + "/" + str(init_model)
        fluid.io.load_persistables(exe, load_model)
        start_epoch = int(init_model) + 1

    train_reader = paddle.batch(reader.train(), batch_size=batch_size)
    test_reader = paddle.batch(reader.test(), batch_size=batch_size)
    feeder = fluid.DataFeeder(place=place, feed_list=[image, label])

    for pass_id in range(start_epoch, num_epochs):
        train_info = [[], [], []]
        test_info = [[], [], []]
        for batch_id, data in enumerate(train_reader()):
            t1 = time.time()
            loss, acc1, acc5 = exe.run(
                fluid.default_main_program(),
                feed=feeder.feed(data),
                fetch_list=[avg_cost, acc_top1, acc_top5])
            t2 = time.time()
            period = t2 - t1
            train_info[0].append(loss[0])
            train_info[1].append(acc1[0])
            train_info[2].append(acc5[0])
            if batch_id % 10 == 0:
                print("Pass {0}, trainbatch {1}, loss {2}, acc1 {3},"
                      "acc5 {4} time {5}".format(pass_id,  \
                      batch_id, loss[0], acc1[0], acc5[0], \
                      "%2.2f sec" % period))
                sys.stdout.flush()

        train_loss = np.array(train_info[0]).mean()
        train_acc1 = np.array(train_info[1]).mean()
        train_acc5 = np.array(train_info[2]).mean()
        for batch_id, data in enumerate(test_reader()):
            t1 = time.time()
            loss, acc1, acc5 = exe.run(
                inference_program,
                feed=feeder.feed(data),
                fetch_list=[avg_cost, acc_top1, acc_top5])
            t2 = time.time()
            period = t2 - t1
            test_info[0].append(loss[0])
            test_info[1].append(acc1[0])
            test_info[2].append(acc5[0])
            if batch_id % 10 == 0:
                print("Pass {0}, testbatch {1}, loss {2}, acc1 {3},"
                      "acc5 {4} time {5}".format(pass_id,  \
                      batch_id, loss[0], acc1[0], acc5[0], \
                      "%2.2f sec" % period))
                sys.stdout.flush()

        test_loss = np.array(test_info[0]).mean()
        test_acc1 = np.array(test_info[1]).mean()
        test_acc5 = np.array(test_info[2]).mean()

        print("End pass {0}, train_loss {1}, train_acc1 {2}, train_acc5 {3},"
              "test_loss {4}, test_acc1 {5}, test_acc5 {6}".format(pass_id,  \
              train_loss, train_acc1, train_acc5, test_loss, \
              test_acc1, test_acc5))
        sys.stdout.flush()

        model_path = os.path.join(model_save_dir, str(pass_id))
        if not os.path.isdir(model_path):
            os.makedirs(model_path)
        fluid.io.save_persistables(exe, model_path)


if __name__ == '__main__':
    total_images = 1281167
    batch_size = 256
    step = int(total_images / batch_size + 1)
    num_epochs = 120

    # mode: piecewise_decay, cosine_decay
    learning_rate_mode = "cosine_decay"
    #learning_rate_mode = "piecewise_decay"
    lr_strategy = {}
    if learning_rate_mode == "piecewise_decay":
        epoch_points = [30, 60, 90]
        bd = [e * step for e in epoch_points]
        lr = [0.1, 0.01, 0.001, 0.0001]
        lr_strategy[learning_rate_mode] = {"bd": bd, "lr": lr}
    elif learning_rate_mode == "cosine_decay":
        lr_strategy[learning_rate_mode] = {
            "step_each_epoch": step,
            "epochs": num_epochs
        }
    else:
        lr_strategy = None
    use_nccl = True

    # class number
    class_dim = 1000
    # layers: 50, 152
    layers = 50

    train(
        learning_rate=0.1,
        batch_size=batch_size,
        num_epochs=num_epochs,
        init_model=None,  #index of saved model
        parallel=True,
        use_nccl=True,
        lr_strategy=lr_strategy,
        class_dim=class_dim,
        layers=layers)
