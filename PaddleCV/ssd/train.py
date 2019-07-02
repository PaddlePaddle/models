import os
import time
import numpy as np
import argparse
import functools
import shutil
import math
import multiprocessing

import paddle
import paddle.fluid as fluid
import reader
from mobilenet_ssd import build_mobilenet_ssd
from utility import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('learning_rate',    float, 0.001,     "Learning rate.")
add_arg('batch_size',       int,   64,        "Minibatch size of all devices.")
add_arg('epoc_num',         int,   120,       "Epoch number.")
add_arg('use_gpu',          bool,  True,      "Whether use GPU.")
add_arg('parallel',         bool,  True,      "Whether train in parallel on multi-devices.")
add_arg('dataset',          str,   'pascalvoc', "dataset can be coco2014, coco2017, and pascalvoc.")
add_arg('model_save_dir',   str,   'model',     "The path to save model.")
add_arg('pretrained_model', str,   'pretrained/ssd_mobilenet_v1_coco/', "The init model path.")
add_arg('ap_version',       str,   '11point',           "mAP version can be integral or 11point.")
add_arg('image_shape',      str,   '3,300,300',         "Input image shape.")
add_arg('mean_BGR',         str,   '127.5,127.5,127.5', "Mean value for B,G,R channel which will be subtracted.")
add_arg('data_dir',         str,   'data/pascalvoc', "Data directory.")
add_arg('enable_ce',        bool,  False, "Whether use CE to evaluate the model.")
#yapf: enable

train_parameters = {
    "pascalvoc": {
        "train_images": 16551,
        "image_shape": [3, 300, 300],
        "class_num": 21,
        "batch_size": 64,
        "lr": 0.001,
        "lr_epochs": [40, 60, 80, 100],
        "lr_decay": [1, 0.5, 0.25, 0.1, 0.01],
        "ap_version": '11point',
    },
    "coco2014": {
        "train_images": 82783,
        "image_shape": [3, 300, 300],
        "class_num": 91,
        "batch_size": 64,
        "lr": 0.001,
        "lr_epochs": [12, 19],
        "lr_decay": [1, 0.5, 0.25],
        "ap_version": 'integral', # should use eval_coco_map.py to test model
    },
    "coco2017": {
        "train_images": 118287,
        "image_shape": [3, 300, 300],
        "class_num": 91,
        "batch_size": 64,
        "lr": 0.001,
        "lr_epochs": [12, 19],
        "lr_decay": [1, 0.5, 0.25],
        "ap_version": 'integral', # should use eval_coco_map.py to test model
    }
}

def optimizer_setting(train_params):
    batch_size = train_params["batch_size"]
    iters = train_params["train_images"] // batch_size
    lr = train_params["lr"]
    boundaries = [i * iters  for i in train_params["lr_epochs"]]
    values = [ i * lr for i in train_params["lr_decay"]]

    optimizer = fluid.optimizer.RMSProp(
        learning_rate=fluid.layers.piecewise_decay(boundaries, values),
        regularization=fluid.regularizer.L2Decay(0.00005), )

    return optimizer


def build_program(main_prog, startup_prog, train_params, is_train):
    image_shape = train_params['image_shape']
    class_num = train_params['class_num']
    ap_version = train_params['ap_version']
    outs = []
    with fluid.program_guard(main_prog, startup_prog):
        py_reader = fluid.layers.py_reader(
            capacity=64,
            shapes=[[-1] + image_shape, [-1, 4], [-1, 1], [-1, 1]],
            lod_levels=[0, 1, 1, 1],
            dtypes=["float32", "float32", "int32", "int32"],
            use_double_buffer=True)
        with fluid.unique_name.guard():
            image, gt_box, gt_label, difficult = fluid.layers.read_file(py_reader)
            locs, confs, box, box_var = build_mobilenet_ssd(image, class_num, image_shape)
            if is_train:
                with fluid.unique_name.guard("train"):
                    loss = fluid.layers.ssd_loss(locs, confs, gt_box, gt_label, box,
                        box_var)
                    loss = fluid.layers.reduce_sum(loss)
                    optimizer = optimizer_setting(train_params)
                    optimizer.minimize(loss)
                outs = [py_reader, loss]
            else:
                with fluid.unique_name.guard("inference"):
                    nmsed_out = fluid.layers.detection_output(
                        locs, confs, box, box_var, nms_threshold=0.45)
                    map_eval = fluid.metrics.DetectionMAP(
                        nmsed_out,
                        gt_label,
                        gt_box,
                        difficult,
                        class_num,
                        overlap_threshold=0.5,
                        evaluate_difficult=False,
                        ap_version=ap_version)
                # nmsed_out and image is used to save mode for inference
                outs = [py_reader, map_eval, nmsed_out, image]
    return outs


def train(args,
          data_args,
          train_params,
          train_file_list,
          val_file_list):

    model_save_dir = args.model_save_dir
    pretrained_model = args.pretrained_model
    use_gpu = args.use_gpu
    parallel = args.parallel
    enable_ce = args.enable_ce
    is_shuffle = True

    if not use_gpu:
        devices_num = int(os.environ.get('CPU_NUM',
                          multiprocessing.cpu_count()))
    else:
        devices_num = fluid.core.get_cuda_device_count()

    batch_size = train_params['batch_size']
    epoc_num = train_params['epoc_num']
    batch_size_per_device = batch_size // devices_num
    num_workers = 8

    startup_prog = fluid.Program()
    train_prog = fluid.Program()
    test_prog = fluid.Program()

    if enable_ce:
        import random
        random.seed(0)
        np.random.seed(0)
        is_shuffle = False
        startup_prog.random_seed = 111
        train_prog.random_seed = 111
        test_prog.random_seed = 111

    train_py_reader, loss = build_program(
        main_prog=train_prog,
        startup_prog=startup_prog,
        train_params=train_params,
        is_train=True)
    test_py_reader, map_eval, _, _ = build_program(
        main_prog=test_prog,
        startup_prog=startup_prog,
        train_params=train_params,
        is_train=False)

    test_prog = test_prog.clone(for_test=True)
    place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(startup_prog)

    if pretrained_model:
        def if_exist(var):
            return os.path.exists(os.path.join(pretrained_model, var.name))
        fluid.io.load_vars(exe, pretrained_model, main_program=train_prog,
                           predicate=if_exist)

    if parallel:
        loss.persistable = True
        build_strategy = fluid.BuildStrategy()
        build_strategy.enable_inplace = True
        build_strategy.memory_optimize = True
        train_exe = fluid.ParallelExecutor(main_program=train_prog,
            use_cuda=use_gpu, loss_name=loss.name, build_strategy=build_strategy)
    train_reader = reader.train(data_args,
                                train_file_list,
                                batch_size_per_device,
                                shuffle=is_shuffle,
                                num_workers=num_workers,
                                enable_ce=enable_ce)
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
    def test(epoc_id, best_map):
        _, accum_map = map_eval.get_map_var()
        map_eval.reset(exe)
        every_epoc_map=[] # for CE
        test_py_reader.start()
        try:
            batch_id = 0
            while True:
                test_map, = exe.run(test_prog, fetch_list=[accum_map])
                if batch_id % 10 == 0:
                    every_epoc_map.append(test_map)
                    print("Batch {0}, map {1}".format(batch_id, test_map))
                batch_id += 1
        except fluid.core.EOFException:
            test_py_reader.reset()
        mean_map = np.mean(every_epoc_map)
        print("Epoc {0}, test map {1}".format(epoc_id, test_map[0]))
        if test_map[0] > best_map:
            best_map = test_map[0]
            save_model('best_model', test_prog)
        return best_map, mean_map


    total_time = 0.0
    for epoc_id in range(epoc_num):
        train_reader = reader.train(data_args,
                                train_file_list,
                                batch_size_per_device,
                                shuffle=is_shuffle,
                                num_workers=num_workers,
                                enable_ce=enable_ce)
        train_py_reader.decorate_paddle_reader(train_reader)
        epoch_idx = epoc_id + 1
        start_time = time.time()
        prev_start_time = start_time
        every_epoc_loss = []
        batch_id = 0
        train_py_reader.start()
        while True:
            try:
                prev_start_time = start_time
                start_time = time.time()
                if parallel:
                    loss_v, = train_exe.run(fetch_list=[loss.name])
                else:
                    loss_v, = exe.run(train_prog, fetch_list=[loss])
                loss_v = np.mean(np.array(loss_v))
                every_epoc_loss.append(loss_v)
                if batch_id % 10 == 0:
                    print("Epoc {:d}, batch {:d}, loss {:.6f}, time {:.5f}".format(
                        epoc_id, batch_id, loss_v, start_time - prev_start_time))
                batch_id += 1
            except (fluid.core.EOFException, StopIteration):
                train_reader().close()
                train_py_reader.reset()
                break

        end_time = time.time()
        total_time += end_time - start_time
        if epoc_id % 10 == 0 or epoc_id == epoc_num - 1:
            best_map, mean_map = test(epoc_id, best_map)
            print("Best test map {0}".format(best_map))
            # save model
            save_model(str(epoc_id), train_prog)

    if enable_ce:
        train_avg_loss = np.mean(every_epoc_loss)
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


def main():
    args = parser.parse_args()
    print_arguments(args)

    data_dir = args.data_dir
    dataset = args.dataset
    assert dataset in ['pascalvoc', 'coco2014', 'coco2017']

    # for pascalvoc
    label_file = 'label_list'
    train_file_list = 'trainval.txt'
    val_file_list = 'test.txt'

    if dataset == 'coco2014':
        train_file_list = 'annotations/instances_train2014.json'
        val_file_list = 'annotations/instances_val2014.json'
    elif dataset == 'coco2017':
        train_file_list = 'annotations/instances_train2017.json'
        val_file_list = 'annotations/instances_val2017.json'

    mean_BGR = [float(m) for m in args.mean_BGR.split(",")]
    image_shape = [int(m) for m in args.image_shape.split(",")]
    train_parameters[dataset]['image_shape'] = image_shape
    train_parameters[dataset]['batch_size'] = args.batch_size
    train_parameters[dataset]['lr'] = args.learning_rate
    train_parameters[dataset]['epoc_num'] = args.epoc_num
    train_parameters[dataset]['ap_version'] = args.ap_version

    data_args = reader.Settings(
        dataset=args.dataset,
        data_dir=data_dir,
        label_file=label_file,
        resize_h=image_shape[1],
        resize_w=image_shape[2],
        mean_value=mean_BGR,
        apply_distort=True,
        apply_expand=True,
        ap_version = args.ap_version)
    train(args,
          data_args,
          train_parameters[dataset],
          train_file_list=train_file_list,
          val_file_list=val_file_list)


if __name__ == '__main__':
    main()