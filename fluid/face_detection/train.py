import os
import shutil
import numpy as np
import time
import argparse
import functools

import paddle.fluid as fluid
from pyramidbox import PyramidBox
import reader
from utility import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)

# yapf: disable
add_arg('parallel',         bool,  True,            "Whether use multi-GPU/threads or not.")
add_arg('learning_rate',    float, 0.001,           "The start learning rate.")
add_arg('batch_size',       int,   16,              "Minibatch size.")
add_arg('num_passes',       int,   160,             "Epoch number.")
add_arg('use_gpu',          bool,  True,            "Whether use GPU.")
add_arg('use_pyramidbox',   bool,  True,            "Whether use PyramidBox model.")
add_arg('model_save_dir',   str,   'output',        "The path to save model.")
add_arg('resize_h',         int,   640,             "The resized image height.")
add_arg('resize_w',         int,   640,             "The resized image width.")
add_arg('with_mem_opt',     bool,  True,            "Whether to use memory optimization or not.")
add_arg('pretrained_model', str,   './vgg_ilsvrc_16_fc_reduced/', "The init model path.")
#yapf: enable


def train(args, config, train_file_list, optimizer_method):
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    num_passes = args.num_passes
    height = args.resize_h
    width = args.resize_w
    use_gpu = args.use_gpu
    use_pyramidbox = args.use_pyramidbox
    model_save_dir = args.model_save_dir
    pretrained_model = args.pretrained_model
    with_memory_optimization = args.with_mem_opt

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

    steps_per_pass = 12880 / batch_size
    boundaries = [steps_per_pass * 50, steps_per_pass * 80,
                  steps_per_pass * 120, steps_per_pass * 140]
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
    if with_memory_optimization:
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

    if args.parallel:
        train_exe = fluid.ParallelExecutor(
            use_cuda=use_gpu, loss_name=loss.name)

    train_reader = reader.train_batch_reader(config, train_file_list, batch_size=batch_size)

    def save_model(postfix):
        model_path = os.path.join(model_save_dir, postfix)
        if os.path.isdir(model_path):
            shutil.rmtree(model_path)
        print 'save models to %s' % (model_path)
        fluid.io.save_persistables(exe, model_path)

    def tensor(data, place, lod=None):
        t = fluid.core.LoDTensor()
        t.set(data, place)
        if lod:
            t.set_lod(lod)
        return t

    for pass_id in range(start_pass, num_passes):
        start_time = time.time()
        prev_start_time = start_time
        end_time = 0
        for batch_id in range(steps_per_pass):
            im, face_box, head_box, labels, lod = next(train_reader)
            im_t = tensor(im, place)
            box1 = tensor(face_box, place, [lod])
            box2 = tensor(head_box, place, [lod])
            lbl_t = tensor(labels, place, [lod])
            feeding = {'image': im_t, 'face_box': box1,
                       'head_box': box2, 'gt_label': lbl_t}

            prev_start_time = start_time
            start_time = time.time()
            if args.parallel:
                fetch_vars = train_exe.run(fetch_list=[v.name for v in fetches],
                                           feed=feeding)
            else:
                fetch_vars = exe.run(fluid.default_main_program(),
                                     feed=feeding,
                                     fetch_list=fetches)
            end_time = time.time()
            fetch_vars = [np.mean(np.array(v)) for v in fetch_vars]
            if batch_id % 10 == 0:
                if not args.use_pyramidbox:
                    print("Pass {0}, batch {1}, loss {2}, time {3}".format(
                        pass_id, batch_id, fetch_vars[0],
                        start_time - prev_start_time))
                else:
                    print("Pass {0}, batch {1}, face loss {2}, head loss {3}, " \
                          "time {4}".format(pass_id,
                           batch_id, fetch_vars[0], fetch_vars[1],
                           start_time - prev_start_time))

        if pass_id % 1 == 0 or pass_id == num_passes - 1:
            save_model(str(pass_id))


if __name__ == '__main__':
    args = parser.parse_args()
    print_arguments(args)

    data_dir = 'data/WIDER_train/images/'
    train_file_list = 'data/wider_face_split/wider_face_train_bbx_gt.txt'

    config = reader.Settings(
        data_dir=data_dir,
        resize_h=args.resize_h,
        resize_w=args.resize_w,
        apply_distort=True,
        apply_expand=False,
        mean_value=[104., 117., 123.],
        ap_version='11point')
    train(args, config, train_file_list, optimizer_method="momentum")
