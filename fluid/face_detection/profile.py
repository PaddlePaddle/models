import os
import shutil
import numpy as np
import time
import argparse
import functools

import reader
import paddle
import paddle.fluid as fluid
import paddle.fluid.profiler as profiler
from pyramidbox import PyramidBox
from utility import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)

# yapf: disable
add_arg('parallel',         bool,  True,            "parallel")
add_arg('learning_rate',    float, 0.001,           "Learning rate.")
add_arg('batch_size',       int,   20,              "Minibatch size.")
add_arg('num_iteration',    int,   10,              "Epoch number.")
add_arg('skip_reader',      bool,  False,            "Whether to skip data reader.")
add_arg('use_gpu',          bool,  True,            "Whether use GPU.")
add_arg('use_pyramidbox',   bool,  True,            "Whether use PyramidBox model.")
add_arg('model_save_dir',   str,   'output',        "The path to save model.")
add_arg('pretrained_model', str,   './pretrained/', "The init model path.")
add_arg('resize_h',         int,   640,             "The resized image height.")
add_arg('resize_w',         int,   640,             "The resized image height.")
#yapf: enable


def train(args, config, train_file_list, optimizer_method):
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    height = args.resize_h
    width = args.resize_w
    use_gpu = args.use_gpu
    use_pyramidbox = args.use_pyramidbox
    model_save_dir = args.model_save_dir
    pretrained_model = args.pretrained_model
    skip_reader = args.skip_reader
    num_iterations = args.num_iteration
    parallel = args.parallel

    num_classes = 2
    image_shape = [3, height, width]

    devices = os.getenv("CUDA_VISIBLE_DEVICES") or ""
    devices_num = len(devices.split(","))

    fetches = []
    network = PyramidBox(image_shape, num_classes,
                         sub_network=use_pyramidbox)
    if use_pyramidbox:
        face_loss, head_loss, loss = network.train()
        fetches = [face_loss, head_loss]
    else:
        loss = network.vgg_ssd_loss()
        fetches = [loss]

    epocs = 12880 / batch_size
    boundaries = [epocs * 40, epocs * 60, epocs * 80, epocs * 100]
    values = [
        learning_rate, learning_rate * 0.5, learning_rate * 0.25,
        learning_rate * 0.1, learning_rate * 0.01
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
    fluid.memory_optimize(fluid.default_main_program())

    place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    start_pass = 0
    if pretrained_model:
        if pretrained_model.isdigit():
            start_pass = int(pretrained_model) + 1
            pretrained_model = os.path.join(model_save_dir, pretrained_model)
            print("Resume from %s " %(pretrained_model))

        if not os.path.exists(pretrained_model):
            raise ValueError("The pre-trained model path [%s] does not exist." %
                             (pretrained_model))
        def if_exist(var):
            return os.path.exists(os.path.join(pretrained_model, var.name))
        fluid.io.load_vars(exe, pretrained_model, predicate=if_exist)

    if parallel:
        train_exe = fluid.ParallelExecutor(
            use_cuda=use_gpu, loss_name=loss.name)

    train_reader = reader.train_batch_reader(config, train_file_list, batch_size=batch_size)

    def tensor(data, place, lod=None):
        t = fluid.core.LoDTensor()
        t.set(data, place)
        if lod:
            t.set_lod(lod)
        return t

    im, face_box, head_box, labels, lod = next(train_reader)
    im_t = tensor(im, place)
    box1 = tensor(face_box, place, [lod])
    box2 = tensor(head_box, place, [lod])
    lbl_t = tensor(labels, place, [lod])
    feed_data = {'image': im_t, 'face_box': box1,
                 'head_box': box2, 'gt_label': lbl_t}

    def run(iterations, feed_data):
        # global feed_data
        reader_time = []
        run_time = []
        for batch_id in range(iterations):
            start_time = time.time()
            if not skip_reader:
                im, face_box, head_box, labels, lod = next(train_reader)
                im_t = tensor(im, place)
                box1 = tensor(face_box, place, [lod])
                box2 = tensor(head_box, place, [lod])
                lbl_t = tensor(labels, place, [lod])
                feed_data = {'image': im_t, 'face_box': box1,
                             'head_box': box2, 'gt_label': lbl_t}
            end_time = time.time()
            reader_time.append(end_time - start_time)

            start_time = time.time()
            if parallel:
                fetch_vars = train_exe.run(fetch_list=[v.name for v in fetches],
                                           feed=feed_data)
            else:
                fetch_vars = exe.run(fluid.default_main_program(),
                                     feed=feed_data,
                                     fetch_list=fetches)
            end_time = time.time()
            run_time.append(end_time - start_time)
            fetch_vars = [np.mean(np.array(v)) for v in fetch_vars]
            if not args.use_pyramidbox:
                print("Batch {0}, loss {1}".format(batch_id, fetch_vars[0]))
            else:
                print("Batch {0}, face loss {1}, head loss {2}".format(
                       batch_id, fetch_vars[0], fetch_vars[1]))

        return reader_time, run_time

    # start-up
    run(2, feed_data)

    # profiling
    start = time.time()
    if not parallel:
        with profiler.profiler('All', 'total', '/tmp/profile_file'):
            reader_time, run_time = run(num_iterations, feed_data)
    else:
        reader_time, run_time = run(num_iterations, feed_data)
    end = time.time()
    total_time = end - start
    print("Total time: {0}, reader time: {1} s, run time: {2} s".format(
        total_time, np.sum(reader_time), np.sum(run_time)))


if __name__ == '__main__':
    args = parser.parse_args()
    print_arguments(args)

    data_dir = 'data/WIDERFACE/WIDER_train/images/'
    train_file_list = 'label/train_gt_widerface.res'

    config = reader.Settings(
        data_dir=data_dir,
        resize_h=args.resize_h,
        resize_w=args.resize_w,
        apply_expand=False,
        mean_value=[104., 117., 123.],
        ap_version='11point')
    train(args, config, train_file_list, optimizer_method="momentum")
