import os
import shutil
import numpy as np
import time
import argparse
import functools

import paddle
import paddle.fluid as fluid
from pyramidbox import PyramidBox
import reader
from utility import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)

# yapf: disable
add_arg('parallel',         bool,  True,            "Whether use multi-GPU/threads or not.")
add_arg('learning_rate',    float, 0.001,           "The start learning rate.")
add_arg('batch_size',       int,   5,              "Minibatch size.")
add_arg('num_passes',       int,   160,             "Epoch number.")
add_arg('use_gpu',          bool,  True,            "Whether use GPU.")
add_arg('use_pyramidbox',   bool,  True,            "Whether use PyramidBox model.")
add_arg('model_save_dir',   str,   'output',        "The path to save model.")
add_arg('resize_h',         int,   640,             "The resized image height.")
add_arg('resize_w',         int,   640,             "The resized image width.")
add_arg('with_mem_opt',     bool,  True,            "Whether to use memory optimization or not.")
add_arg('pretrained_model', str,   './vgg_ilsvrc_16_fc_reduced/', "The init model path.")
#yapf: enable

def build_program(optimizer_method, steps_per_pass, main_prog, startup_prog, args):
    height = args.resize_h
    width = args.resize_w
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    use_pyramidbox = args.use_pyramidbox
    image_shape = [3, height, width]
    num_classes = 2
    def get_optimizer():
        boundaries = [steps_per_pass * 50, steps_per_pass * 80,
                      steps_per_pass * 120, steps_per_pass * 140]
        values = [
            learning_rate, learning_rate * 0.5, learning_rate * 0.25,
            learning_rate * 0.1, learning_rate * 0.01]
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
        return optimizer
    with fluid.program_guard(main_prog, startup_prog):
        py_reader = fluid.layers.py_reader(
            capacity=8,
            shapes=[[-1] + image_shape, [-1, 4], [-1, 4], [-1, 1]],
            lod_levels=[0, 1, 1, 1],
            dtypes=["float32", "float32", "float32", "int32"],
            use_double_buffer=True)
        with fluid.unique_name.guard():
            image, face_box, head_box, gt_label = fluid.layers.read_file(py_reader)
            fetches = []
            network = PyramidBox(image, face_box, head_box, gt_label,
                                 num_classes, sub_network=use_pyramidbox)
            if use_pyramidbox:
                face_loss, head_loss, loss = network.train()
                fetches = [face_loss, head_loss]
            else:
                loss = network.vgg_ssd_loss()
                fetches = [loss]
            optimizer = get_optimizer()
            optimizer.minimize(loss)
    return py_reader, fetches, loss

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
    devices = os.getenv("CUDA_VISIBLE_DEVICES") or ""
    devices_num = len(devices.split(","))
    steps_per_pass = 12880 // batch_size // devices_num

    startup_prog = fluid.Program()
    train_prog = fluid.Program()

    train_py_reader, fetches, loss = build_program(
        optimizer_method = optimizer_method,
        steps_per_pass = steps_per_pass,
        main_prog = train_prog,
        startup_prog = startup_prog,
        args=args)

    if with_memory_optimization:
        fluid.memory_optimize(train_prog)

    place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(startup_prog)

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
        fluid.io.load_vars(
            exe, pretrained_model, main_program=train_prog, predicate=if_exist)
    train_reader = reader.train_batch_reader(config, train_file_list, batch_size=batch_size)
    train_py_reader.decorate_paddle_reader(train_reader)

    if args.parallel:
        train_exe = fluid.ParallelExecutor(
            main_program = train_prog,
            use_cuda=use_gpu,
            loss_name=loss.name)

    def save_model(postfix, program):
        model_path = os.path.join(model_save_dir, postfix)
        if os.path.isdir(model_path):
            shutil.rmtree(model_path)
        print 'save models to %s' % (model_path)
        fluid.io.save_persistables(exe, model_path, main_program=program)

    for pass_id in range(start_pass, num_passes):
        start_time = time.time()
        train_py_reader.start()
        prev_start_time = start_time
        end_time = 0
        batch_id = 0
        try:
            while True:
                prev_start_time = start_time
                start_time = time.time()
                if args.parallel:
                    fetch_vars = train_exe.run(fetch_list=[v.name for v in fetches])
                else:
                    fetch_vars = exe.run(train_prog,
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
                batch_id += 1
                if batch_id > steps_per_pass:
                    break
        except fluid.core.EOFException:
            train_py_reader.reset()

        if pass_id % 1 == 0 or pass_id == num_passes - 1:
            save_model(str(pass_id), train_prog)


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
