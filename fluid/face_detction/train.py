import os
import numpy as np
import time
import argparse
import functools

import reader
import paddle
import paddle.fluid as fluid
from pyramidbox import PyramidBox
from utility import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)

# yapf: disable
add_arg('parallel', bool, True, "parallel")
add_arg('learning_rate', float, 0.0001, "Learning rate.")
add_arg('batch_size', int, 16, "Minibatch size.")
add_arg('num_passes', int, 120, "Epoch number.")
add_arg('use_gpu', bool, True, "Whether use GPU.")
add_arg('use_pyramidbox', bool, False, "Whether use PyramidBox model.")
add_arg('dataset', str, 'WIDERFACE', "coco2014, coco2017, and pascalvoc.")
add_arg('model_save_dir', str, 'model', "The path to save model.")
add_arg('pretrained_model', str, './vgg_model/', "The init model path.")
add_arg('resize_h', int, 640, "The resized image height.")
add_arg('resize_w', int, 640, "The resized image height.")
#yapf: enable


def train(args, data_args, learning_rate, batch_size, pretrained_model,
          num_passes, optimizer_method):

    num_classes = 2

    devices = os.getenv("CUDA_VISIBLE_DEVICES") or ""
    devices_num = len(devices.split(","))

    image_shape = [3, data_args.resize_h, data_args.resize_w]

    if args.use_pyramidbox:
        network = PyramidBox(image_shape, sub_network=args.use_pyramidbox)
        face_loss, head_loss, loss = network.train()
    else:
        network = PyramidBox(image_shape, sub_network=args.use_pyramidbox)
        loss = network.vgg_ssd(num_classes, image_shape)

    epocs = 12880 / batch_size
    boundaries = [epocs * 100, epocs * 125, epocs * 150]
    values = [
        learning_rate, learning_rate * 0.1, learning_rate * 0.01,
        learning_rate * 0.001
    ]

    if optimizer_method == "momentum":
        optimizer = fluid.optimizer.Momentum(
            learning_rate=fluid.layers.piecewise_decay(
                boundaries=boundaries, values=values),
            momentum=0.9,
            regularization=fluid.regularizer.L2Decay(0.0005),
        )
    else:
        optimizer = fluid.optimizer.RMSProp(
            learning_rate=fluid.layers.piecewise_decay(boundaries, values),
            regularization=fluid.regularizer.L2Decay(0.0005),
        )

    optimizer.minimize(loss)

    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    # fluid.io.save_inference_model('./vgg_model/', ['image'], [loss], exe)
    if pretrained_model:
        def if_exist(var):
            return os.path.exists(os.path.join(pretrained_model, var.name))
        print('Load pre-trained model.')
        fluid.io.load_vars(exe, pretrained_model, predicate=if_exist)

    if args.parallel:
        train_exe = fluid.ParallelExecutor(
            use_cuda=args.use_gpu, loss_name=loss.name)

    train_reader = paddle.batch(
        reader.train(data_args, train_file_list), batch_size=batch_size)
    feeder = fluid.DataFeeder(
        place=place,
        feed_list=[
            network.image, network.gt_box, network.gt_label, network.difficult
        ])

    def save_model(postfix):
        model_path = os.path.join(model_save_dir, postfix)
        if os.path.isdir(model_path):
            shutil.rmtree(model_path)
        print 'save models to %s' % (model_path)
        fluid.io.save_persistables(exe, model_path)

    best_map = 0.

    for pass_id in range(num_passes):
        start_time = time.time()
        prev_start_time = start_time
        end_time = 0
        for batch_id, data in enumerate(train_reader()):
            prev_start_time = start_time
            start_time = time.time()
            if len(data) < devices_num: continue
            if args.parallel:
                loss_v, = train_exe.run(fetch_list=[loss.name],
                                        feed=feeder.feed(data))
            else:
                loss_v, = exe.run(fluid.default_main_program(),
                                  feed=feeder.feed(data),
                                  fetch_list=[loss])
            end_time = time.time()
            loss_v = np.mean(np.array(loss_v))
            if batch_id % 1 == 0:
                print("Pass {0}, batch {1}, loss {2}, time {3}".format(
                    pass_id, batch_id, loss_v, start_time - prev_start_time))
        if pass_id % 10 == 0 or pass_id == num_passes - 1:
            save_model(str(pass_id))
    print("Best test map {0}".format(best_map))


if __name__ == '__main__':
    args = parser.parse_args()
    print_arguments(args)

    data_dir = 'data/WIDERFACE/WIDER_train/images/'
    train_file_list = 'label/train_gt_widerface.res'
    val_file_list = 'label/val_gt_widerface.res'
    model_save_dir = args.model_save_dir

    data_args = reader.Settings(
        dataset=args.dataset,
        data_dir=data_dir,
        resize_h=args.resize_h,
        resize_w=args.resize_w,
        apply_expand=False,
        mean_value=[104., 117., 123],
        ap_version='11point')
    train(
        args,
        data_args=data_args,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        pretrained_model=args.pretrained_model,
        num_passes=args.num_passes,
        optimizer_method="momentum")
