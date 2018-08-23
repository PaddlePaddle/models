import os
import time
import numpy as np
import argparse
import functools
import shutil

import paddle
import paddle.fluid as fluid
import reader
from mobilenet_ssd import mobile_net
from utility import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('learning_rate',    float, 0.001,     "Learning rate.")
add_arg('batch_size',       int,   16,        "Minibatch size.")
add_arg('num_passes',       int,   120,       "Epoch number.")
add_arg('use_gpu',          bool,  True,      "Whether use GPU.")
add_arg('parallel',         bool,  True,      "Parallel.")
add_arg('dataset',          str,   'pascalvoc', "coco2014, coco2017, and pascalvoc.")
add_arg('model_save_dir',   str,   'model',     "The path to save model.")
add_arg('pretrained_model', str,   'pretrained/ssd_mobilenet_v1_coco/', "The init model path.")
add_arg('apply_distort',    bool,  True,   "Whether apply distort.")
add_arg('apply_expand',     bool,  True,   "Whether apply expand.")
add_arg('nms_threshold',    float, 0.45,   "NMS threshold.")
add_arg('ap_version',       str,   '11point',   "integral, 11point.")
add_arg('resize_h',         int,   300,    "The resized image height.")
add_arg('resize_w',         int,   300,    "The resized image height.")
add_arg('mean_value_B',     float, 127.5,  "Mean value for B channel which will be subtracted.")  #123.68
add_arg('mean_value_G',     float, 127.5,  "Mean value for G channel which will be subtracted.")  #116.78
add_arg('mean_value_R',     float, 127.5,  "Mean value for R channel which will be subtracted.")  #103.94
add_arg('is_toy',           int,   0, "Toy for quick debug, 0 means using all data, while n means using only n sample.")
add_arg('data_dir',         str,   'data/pascalvoc', "data directory")
add_arg('enable_ce',     bool,  False, "Whether use CE to evaluate the model")
#yapf: enable

def build_program(is_train, main_prog, startup_prog, args, data_args,
                  values=None, train_file_list=None):
    image_shape = [3, data_args.resize_h, data_args.resize_w]
    if 'coco' in data_args.dataset:
        num_classes = 91
    elif 'pascalvoc' in data_args.dataset:
        num_classes = 21

    def get_optimizer():
        optimizer = fluid.optimizer.RMSProp(
            learning_rate=fluid.layers.piecewise_decay(boundaries, values),
            regularization=fluid.regularizer.L2Decay(0.00005), )
        return optimizer

    with fluid.program_guard(main_prog, startup_prog):
        py_reader = fluid.layers.py_reader(
            capacity=64,
            shapes=[[-1] + image_shape, [-1, 4], [-1, 1], [-1, 1]],
            lod_levels=[0, 1, 1, 1],
            dtypes=["float32", "float32", "int32", "int32"],
            use_double_buffer=True)
        with fluid.unique_name.guard():
            image, gt_box, gt_label, difficult = fluid.layers.read_file(py_reader)
            locs, confs, box, box_var = mobile_net(num_classes, image, image_shape)
            if is_train:
                loss = fluid.layers.ssd_loss(locs, confs, gt_box, gt_label, box,
                    box_var)
                loss = fluid.layers.reduce_sum(loss)
                optimizer = get_optimizer()
                optimizer.minimize(loss)
            else:
                nmsed_out = fluid.layers.detection_output(
                   locs, confs, box, box_var, nms_threshold=args.nms_threshold)
                with fluid.program_guard(main_prog):
                    loss = fluid.evaluator.DetectionMAP(
                        nmsed_out,
                        gt_label,
                        gt_box,
                        difficult,
                        num_classes,
                        overlap_threshold=0.5,
                        evaluate_difficult=False,
                        ap_version=args.ap_version)
    if not is_train:
        main_prog = main_prog.clone(for_test=True)
    return py_reader, loss

def train(args,
          train_file_list,
          val_file_list,
          data_args,
          learning_rate,
          batch_size,
          num_passes,
          model_save_dir,
          pretrained_model=None):

    startup_prog = fluid.Program()
    train_prog = fluid.Program()
    test_prog = fluid.Program()

    devices = os.getenv("CUDA_VISIBLE_DEVICES") or ""
    devices_num = len(devices.split(","))
    if 'coco' in data_args.dataset:
        # learning rate decay in 12, 19 pass, respectively
        if '2014' in train_file_list:
            epocs = 82783 // batch_size // devices_num
            boundaries = [epocs * 12, epocs * 19]
        elif '2017' in train_file_list:
            epocs = 118287 // batch_size // devices_num
            boundaries = [epocs * 12, epocs * 19]
        values = [learning_rate, learning_rate * 0.5,
            learning_rate * 0.25]
    elif 'pascalvoc' in data_args.dataset:
        epocs = 19200 // batch_size // devices_num
        boundaries = [epocs * 40, epocs * 60, epocs * 80, epocs * 100]
        values = [
            learning_rate, learning_rate * 0.5, learning_rate * 0.25,
            learning_rate * 0.1, learning_rate * 0.01]

    if args.enable_ce:
        startup_prog.random_seed = 111
        train_prog.random_seed = 111
        test_prog.random_seed = 111

    train_py_reader, loss = build_program(
        is_train=True,
        main_prog=train_prog,
        startup_prog=startup_prog,
        args=args,
        data_args=data_args,
        values = values,
        train_file_list=train_file_list)
    test_py_reader, map_eval = build_program(
        is_train=False,
        main_prog=test_prog,
        startup_prog=startup_prog,
        args=args,
        data_args=data_args)
    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(startup_prog)

    if pretrained_model:
        def if_exist(var):
            return os.path.exists(os.path.join(pretrained_model, var.name))
        fluid.io.load_vars(exe, pretrained_model, main_program=train_prog, predicate=if_exist)

    if args.parallel:
        train_exe = fluid.ParallelExecutor(main_program=train_prog,
            use_cuda=args.use_gpu, loss_name=loss.name)
        test_exe = fluid.ParallelExecutor(main_program=test_prog,
            use_cuda=args.use_gpu, share_vars_from=train_exe)
    if not args.enable_ce:
        train_reader = reader.train_batch_reader(data_args, train_file_list, batch_size)
    else:
        import random
        random.seed(0)
        np.random.seed(0)
        train_reader = reader.train_batch_reader(data_args, train_file_list, batch_size, shuffle=False)
    test_reader = reader.test(data_args, val_file_list, batch_size)
    train_py_reader.decorate_paddle_reader(train_reader)
    test_py_reader.decorate_paddle_reader(test_reader)

    def save_model(postfix, main_prog):
        model_path = os.path.join(model_save_dir, postfix)
        if os.path.isdir(model_path):
            shutil.rmtree(model_path)
        print('save models to %s' % (model_path))
        fluid.io.save_persistables(exe, model_path, main_program=main_prog)

    best_map = 0.
    def test(pass_id, best_map):
        _, accum_map = map_eval.get_map_var()
        map_eval.reset(test_exe)
        every_pass_map=[]
        test_py_reader.start()
        batch_id = 0
        try:
            while True:
                test_map, = test_exe.run(test_prog,
                                   fetch_list=[accum_map])
                if batch_id % 20 == 0:
                    every_pass_map.append(test_map)
                    print("Batch {0}, map {1}".format(batch_id, test_map))
                batch_id += 1
        except fluid.core.EOFException:
            test_py_reader.reset()
        mean_map = np.mean(every_pass_map)
        if test_map[0] > best_map:
            best_map = test_map[0]
            save_model('best_model', test_prog)
        print("Pass {0}, test map {1}".format(pass_id, test_map))
        return best_map, mean_map

    total_time = 0.0
    for pass_id in range(num_passes):
        epoch_idx = pass_id + 1
        start_time = time.time()
        train_py_reader.start()
        prev_start_time = start_time
        every_pass_loss = []
        batch_id = 0
        try:
            while True:
                prev_start_time = start_time
                start_time = time.time()

                if args.parallel:
                    loss_v, = train_exe.run(fetch_list=[loss.name])
                else:
                    loss_v, = exe.run(train_prog, fetch_list=[loss])
                loss_v = np.mean(np.array(loss_v))
                every_pass_loss.append(loss_v)
                if batch_id % 20 == 0:
                    print("Pass {0}, batch {1}, loss {2}, time {3}".format(
                        pass_id, batch_id, loss_v, start_time - prev_start_time))
                batch_id += 1
                if batch_id > epocs:
                    break
        except fluid.core.EOFException:
            train_py_reader.reset()

        end_time = time.time()
        best_map, mean_map = test(pass_id, best_map)
        if args.enable_ce and pass_id == num_passes - 1:
            total_time += end_time - start_time
            train_avg_loss = np.mean(every_pass_loss)
            if devices_num == 1:
                print("kpis	train_cost	%s" % train_avg_loss)
                print("kpis	test_acc	%s" % mean_map)
                print("kpis	train_speed	%s" % (total_time / epoch_idx))
            else:
                print("kpis	train_cost_card%s	%s" %
                       (devices_num, train_avg_loss))
                print("kpis	test_acc_card%s	%s" %
                       (devices_num, mean_map))
                print("kpis	train_speed_card%s	%f" %
                       (devices_num, total_time / epoch_idx))

        if pass_id % 10 == 0 or pass_id == num_passes - 1:
            save_model(str(pass_id), train_prog)
    print("Best test map {0}".format(best_map))

if __name__ == '__main__':
    args = parser.parse_args()
    print_arguments(args)

    data_dir = args.data_dir
    label_file = 'label_list'
    model_save_dir = args.model_save_dir
    train_file_list = 'trainval.txt'
    val_file_list = 'test.txt'
    if 'coco' in args.dataset:
        data_dir = 'data/coco'
        if '2014' in args.dataset:
            train_file_list = 'annotations/instances_train2014.json'
            val_file_list = 'annotations/instances_val2014.json'
        elif '2017' in args.dataset:
            train_file_list = 'annotations/instances_train2017.json'
            val_file_list = 'annotations/instances_val2017.json'

    data_args = reader.Settings(
        dataset=args.dataset,
        data_dir=data_dir,
        label_file=label_file,
        resize_h=args.resize_h,
        resize_w=args.resize_w,
        mean_value=[args.mean_value_B, args.mean_value_G, args.mean_value_R],
        apply_distort=args.apply_distort,
        apply_expand=args.apply_expand,
        ap_version = args.ap_version,
        toy=args.is_toy)
    train(
        args,
        train_file_list=train_file_list,
        val_file_list=val_file_list,
        data_args=data_args,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_passes=args.num_passes,
        model_save_dir=model_save_dir,
        pretrained_model=args.pretrained_model)
