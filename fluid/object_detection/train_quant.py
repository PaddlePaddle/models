import os
import time
import numpy as np
import argparse
import functools
import shutil
import math

import paddle
import paddle.fluid as fluid
import reader
from mobilenet_ssd import mobile_net
from utility import add_arguments, print_arguments
from train import build_program
from train import train_parameters

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('learning_rate',    float, 0.0001,              "Learning rate.")
add_arg('batch_size',       int,   64,                  "Minibatch size.")
add_arg('epoc_num',         int,   10,                  "Epoch number.")
add_arg('use_gpu',          bool,  True,                "Whether use GPU.")
add_arg('parallel',         bool,  True,                "Whether train in parallel training.")
add_arg('model_save_dir',   str,   'model',             "The path to save model.")
add_arg('pretrained_model', str,   '',                  "The init model path.")
add_arg('ap_version',       str,   '11point',           "Integral, 11point.")
add_arg('image_shape',      str,   '3,300,300',         "Input image shape.")
add_arg('mean_BGR',         str,   '127.5,127.5,127.5', "Mean value for B,G,R channel which will be subtracted.")
add_arg('data_dir',         str,   'data/pascalvoc',    "Data directory")
add_arg('act_quant_type',   str,   'abs_max',           "Quantize type of activation.")
#yapf: enable

def train(args,
          data_args,
          train_params,
          train_file_list,
          val_file_list):

    model_save_dir = args.model_save_dir
    pretrained_model = args.pretrained_model
    epoc_num = args.epoc_num
    use_gpu = args.use_gpu
    parallel = args.parallel
    is_shuffle = True
    act_quant_type = args.act_quant_type

    devices = os.getenv("CUDA_VISIBLE_DEVICES") or ""
    devices_num = len(devices.split(","))
    batch_size = train_params['batch_size']
    batch_size_per_device = batch_size // devices_num
    iters_per_epoc = train_params["train_images"] // batch_size
    num_workers = 4

    startup_prog = fluid.Program()
    train_prog = fluid.Program()
    test_prog = fluid.Program()

    train_py_reader, loss = build_program(
        main_prog=train_prog,
        startup_prog=startup_prog,
        train_params=train_params,
        is_train=True)
    test_py_reader, map_eval = build_program(
        main_prog=test_prog,
        startup_prog=startup_prog,
        train_params=train_params,
        is_train=False)

    test_prog = test_prog.clone(for_test=True)

    transpiler = fluid.contrib.QuantizeTranspiler(weight_bits=8,
        activation_bits=8,
        activation_quantize_type=act_quant_type,
        weight_quantize_type='abs_max')

    transpiler.training_transpile(train_prog, startup_prog)
    transpiler.training_transpile(test_prog, startup_prog)

    place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(startup_prog)

    if pretrained_model:
        print('Load init model %s.' % pretrained_model)
        def if_exist(var):
            return os.path.exists(os.path.join(pretrained_model, var.name))
        fluid.io.load_vars(exe, pretrained_model, main_program=train_prog,
                           predicate=if_exist)
    else:
        print('There is no init model.')

    if parallel:
        train_exe = fluid.ParallelExecutor(main_program=train_prog,
            use_cuda=use_gpu, loss_name=loss.name)

    train_reader = reader.train(data_args,
                                train_file_list,
                                batch_size_per_device,
                                shuffle=is_shuffle,
                                use_multiprocessing=True,
                                num_workers=num_workers,
                                max_queue=24)
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
        every_epoc_map=[]
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
        print("Epoc {0}, test map {1}".format(epoc_id, test_map))
        if test_map[0] > best_map:
            best_map = test_map[0]
            save_model('best_model', test_prog)
        return best_map, mean_map

    train_py_reader.start()
    total_time = 0.0
    try:
        for epoc_id in range(epoc_num):
            # test
            best_map, mean_map = test(epoc_id, best_map)
            print("Best test map {0}".format(best_map))

            # train
            start_time = time.time()
            prev_start_time = start_time
            every_epoc_loss = []
            for batch_id in range(iters_per_epoc):
                prev_start_time = start_time
                start_time = time.time()
                if parallel:
                    loss_v, = train_exe.run(fetch_list=[loss.name])
                else:
                    loss_v, = exe.run(train_prog, fetch_list=[loss])
                loss_v = np.mean(np.array(loss_v))
                every_epoc_loss.append(loss_v)
                if batch_id % 20 == 0:
                    print("Epoc {0}, batch {1}, loss {2}, time {3}".format(
                        epoc_id, batch_id, loss_v, start_time - prev_start_time))
            end_time = time.time()
            total_time += end_time - start_time

            save_model(str(epoc_id), train_prog)

    except fluid.core.EOFException:
        train_py_reader.reset()
    except StopIteration:
        train_py_reader.reset()
    train_py_reader.reset()


if __name__ == '__main__':
    args = parser.parse_args()
    print_arguments(args)

    # for pascalvoc
    label_file = 'label_list'
    train_file_list = 'trainval.txt'
    val_file_list = 'test.txt'

    mean_BGR = [float(m) for m in args.mean_BGR.split(",")]
    image_shape = [int(m) for m in args.image_shape.split(",")]
    train_parameters[dataset]['image_shape'] = image_shape
    train_parameters[dataset]['batch_size'] = args.batch_size
    train_parameters[dataset]['lr'] = args.learning_rate
    train_parameters[dataset]['epoc_num'] = args.epoc_num
    train_parameters[dataset]['ap_version'] = args.ap_version

    data_args = reader.Settings(
        dataset=args.dataset,
        data_dir=args.data_dir,
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
