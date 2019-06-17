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
from mobilenet_ssd import mobile_net
from utility import add_arguments, print_arguments
from train import build_program
from train import train_parameters
from infer import draw_bounding_box_on_image

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('learning_rate',    float, 0.0001,              "Learning rate.")
add_arg('batch_size',       int,   64,                  "Minibatch size.")
add_arg('epoc_num',         int,   20,                  "Epoch number.")
add_arg('use_gpu',          bool,  True,                "Whether use GPU.")
add_arg('parallel',         bool,  True,                "Whether train in parallel on multi-devices.")
add_arg('model_save_dir',   str,   'quant_model',       "The path to save model.")
add_arg('init_model',       str,   'ssd_mobilenet_v1_pascalvoc', "The init model path.")
add_arg('ap_version',       str,   '11point',           "mAP version can be integral or 11point.")
add_arg('image_shape',      str,   '3,300,300',         "Input image shape.")
add_arg('mean_BGR',         str,   '127.5,127.5,127.5', "Mean value for B,G,R channel which will be subtracted.")
add_arg('lr_epochs',        str,   '30,60',             "The learning decay steps.")
add_arg('lr_decay_rates',   str,   '1,0.1,0.01',        "The learning decay rates for each step.")
add_arg('data_dir',         str,   'data/pascalvoc',    "Data directory")
add_arg('act_quant_type',   str,   'abs_max',           "Quantize type of activation, whicn can be abs_max or range_abs_max")
add_arg('image_path',       str,   '',                  "The image used to inference and visualize.")
add_arg('confs_threshold',  float, 0.5,                 "Confidence threshold to draw bbox.")
add_arg('mode',             str,   'train',             "Job mode can be one of ['train', 'test', 'infer'].")
#yapf: enable

def test(exe, test_prog, map_eval, test_py_reader):
    _, accum_map = map_eval.get_map_var()
    map_eval.reset(exe)
    test_py_reader.start()
    try:
        batch = 0
        while True:
            test_map, = exe.run(test_prog, fetch_list=[accum_map])
            if batch % 10 == 0:
                print("Batch {0}, map {1}".format(batch, test_map))
            batch += 1
    except fluid.core.EOFException:
        test_py_reader.reset()
    finally:
        test_py_reader.reset()
    print("Test map {0}".format(test_map))
    return test_map


def save_model(exe, main_prog, model_save_dir, postfix):
    model_path = os.path.join(model_save_dir, postfix)
    if os.path.isdir(model_path):
        shutil.rmtree(model_path)
    fluid.io.save_persistables(exe, model_path, main_program=main_prog)


def train(args,
          data_args,
          train_params,
          train_file_list,
          val_file_list):

    model_save_dir = args.model_save_dir
    init_model = args.init_model
    epoc_num = args.epoc_num
    use_gpu = args.use_gpu
    parallel = args.parallel
    is_shuffle = True
    act_quant_type = args.act_quant_type

    if use_gpu:
        devices_num = fluid.core.get_cuda_device_count()
    else:
        devices_num = int(os.environ.get('CPU_NUM', multiprocessing.cpu_count()))

    batch_size = train_params['batch_size']
    batch_size_per_device = batch_size // devices_num
    num_workers = 4

    startup_prog = fluid.Program()
    train_prog = fluid.Program()
    test_prog = fluid.Program()

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

    transpiler = fluid.contrib.QuantizeTranspiler(weight_bits=8,
        activation_bits=8,
        activation_quantize_type=act_quant_type,
        weight_quantize_type='abs_max')

    transpiler.training_transpile(train_prog, startup_prog)
    transpiler.training_transpile(test_prog, startup_prog)

    place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(startup_prog)

    if init_model:
        print('Load init model %s.' % init_model)
        def if_exist(var):
            return os.path.exists(os.path.join(init_model, var.name))
        fluid.io.load_vars(exe, init_model, main_program=train_prog,
                           predicate=if_exist)
    else:
        print('There is no init model.')

    if parallel:
        train_exe = fluid.ParallelExecutor(main_program=train_prog,
            use_cuda=True if use_gpu else False, loss_name=loss.name)

    train_reader = reader.train(data_args,
                                train_file_list,
                                batch_size_per_device,
                                shuffle=is_shuffle,
                                num_workers=num_workers)
    test_reader = reader.test(data_args, val_file_list, batch_size)
    train_py_reader.decorate_paddle_reader(train_reader)
    test_py_reader.decorate_paddle_reader(test_reader)

    train_py_reader.start()
    best_map = 0.
    for epoc in range(epoc_num):
        if epoc == 0:
            # test quantized model without quantization-aware training.
            test_map = test(exe, test_prog, map_eval, test_py_reader)
        batch = 0
        train_py_reader.start()
        while True:
            try:
                # train
                start_time = time.time()
                if parallel:
                    outs = train_exe.run(fetch_list=[loss.name])
                else:
                    outs = exe.run(train_prog, fetch_list=[loss])
                end_time = time.time()
                avg_loss = np.mean(np.array(outs[0]))
                if batch % 10 == 0:
                    print("Epoc {:d}, batch {:d}, loss {:.6f}, time {:.5f}".format(
                        epoc , batch, avg_loss, end_time - start_time))
            except (fluid.core.EOFException, StopIteration):
                train_reader().close()
                train_py_reader.reset()
                break
        test_map = test(exe, test_prog, map_eval, test_py_reader)
        save_model(exe, train_prog, model_save_dir, str(epoc))
        if test_map > best_map:
            best_map = test_map
            save_model(exe, train_prog, model_save_dir, 'best_map')
        print("Best test map {0}".format(best_map))


def eval(args, data_args, configs, val_file_list):
    init_model = args.init_model
    use_gpu = args.use_gpu
    act_quant_type = args.act_quant_type
    model_save_dir = args.model_save_dir

    batch_size = configs['batch_size']
    batch_size_per_device = batch_size

    startup_prog = fluid.Program()
    test_prog = fluid.Program()
    test_py_reader, map_eval, nmsed_out, image = build_program(
        main_prog=test_prog,
        startup_prog=startup_prog,
        train_params=configs,
        is_train=False)
    test_prog = test_prog.clone(for_test=True)

    transpiler = fluid.contrib.QuantizeTranspiler(weight_bits=8,
        activation_bits=8,
        activation_quantize_type=act_quant_type,
        weight_quantize_type='abs_max')
    transpiler.training_transpile(test_prog, startup_prog)

    place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(startup_prog)

    def if_exist(var):
        return os.path.exists(os.path.join(init_model, var.name))
    fluid.io.load_vars(exe, init_model, main_program=test_prog,
                       predicate=if_exist)

    # freeze after load parameters
    transpiler.freeze_program(test_prog, place)

    test_reader = reader.test(data_args, val_file_list, batch_size)
    test_py_reader.decorate_paddle_reader(test_reader)

    test_map = test(exe, test_prog, map_eval, test_py_reader)
    print("Test model {0}, map {1}".format(init_model, test_map))
    # convert model to 8-bit before saving, but now Paddle can't load
    # the 8-bit model to do inference.
    # transpiler.convert_to_int8(test_prog, place)
    fluid.io.save_inference_model(model_save_dir, [image.name],
                                  [nmsed_out], exe, test_prog)


def infer(args, data_args):
    model_dir = args.init_model
    image_path = args.image_path
    confs_threshold = args.confs_threshold

    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    [inference_program, feed , fetch] = fluid.io.load_inference_model(
        dirname=model_dir,
        executor=exe,
        model_filename='__model__')

    #print(np.array(fluid.global_scope().find_var('conv2d_20.w_0').get_tensor()))
    #print(np.max(np.array(fluid.global_scope().find_var('conv2d_20.w_0').get_tensor())))
    infer_reader = reader.infer(data_args, image_path)
    data = infer_reader()
    data = data.reshape((1,) + data.shape)
    outs = exe.run(inference_program,
                   feed={feed[0]: data},
                   fetch_list=fetch,
                   return_numpy=False)
    out = np.array(outs[0])
    draw_bounding_box_on_image(image_path, out, confs_threshold,
                               data_args.label_list)


if __name__ == '__main__':
    args = parser.parse_args()
    print_arguments(args)

    # for pascalvoc
    label_file = 'label_list'
    train_list = 'trainval.txt'
    val_list = 'test.txt'
    dataset = 'pascalvoc'

    mean_BGR = [float(m) for m in args.mean_BGR.split(",")]
    image_shape = [int(m) for m in args.image_shape.split(",")]
    lr_epochs = [int(m) for m in args.lr_epochs.split(",")]
    lr_rates = [float(m) for m in args.lr_decay_rates.split(",")]
    train_parameters[dataset]['image_shape'] = image_shape
    train_parameters[dataset]['batch_size'] = args.batch_size
    train_parameters[dataset]['lr'] = args.learning_rate
    train_parameters[dataset]['epoc_num'] = args.epoc_num
    train_parameters[dataset]['ap_version'] = args.ap_version
    train_parameters[dataset]['lr_epochs'] = lr_epochs
    train_parameters[dataset]['lr_decay'] = lr_rates

    data_args = reader.Settings(
        dataset=dataset,
        data_dir=args.data_dir,
        label_file=label_file,
        resize_h=image_shape[1],
        resize_w=image_shape[2],
        mean_value=mean_BGR,
        apply_distort=True,
        apply_expand=True,
        ap_version = args.ap_version)
    if args.mode == 'train':
        train(args, data_args, train_parameters[dataset], train_list, val_list)
    elif args.mode == 'test':
        eval(args, data_args, train_parameters[dataset], val_list)
    else:
        infer(args, data_args)
