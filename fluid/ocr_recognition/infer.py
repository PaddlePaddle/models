import paddle.v2 as paddle
import paddle.fluid as fluid
import paddle.fluid.profiler as profiler
from utility import add_arguments, print_arguments, to_lodtensor, get_feeder_data
from crnn_ctc_model import ctc_infer
import numpy as np
import ctc_reader
import argparse
import functools
import os
import time

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('model_path',         str,  None,   "The model path to be used for inference.")
add_arg('input_images_dir',   str,  None,   "The directory of images.")
add_arg('input_images_list',  str,  None,   "The list file of images.")
add_arg('dict',               str,  None,   "The dictionary. The result of inference will be index sequence if the dictionary was None.")
add_arg('use_gpu',            bool,  True,      "Whether use GPU to infer.")
add_arg('iterations',         int,  0,      "The number of iterations. Zero or less means whole test set. More than 0 means the test set might be looped until # of iterations is reached.")
add_arg('profile',            bool, False,  "Whether to use profiling.")
add_arg('skip_batch_num',     int,  0,      "The number of first minibatches to skip as warm-up for better performance test.")
add_arg('batch_size',         int,  1,      "The minibatch size.")
# yapf: enable


def inference(args, infer=ctc_infer, data_reader=ctc_reader):
    """OCR inference"""
    num_classes = data_reader.num_classes()
    data_shape = data_reader.data_shape()
    # define network
    images = fluid.layers.data(name='pixel', shape=data_shape, dtype='float32')
    sequence = infer(
        images, num_classes, use_cudnn=True if args.use_gpu else False)
    # data reader
    infer_reader = data_reader.inference(
        batch_size=args.batch_size,
        infer_images_dir=args.input_images_dir,
        infer_list_file=args.input_images_list,
        cycle=True if args.iterations > 0 else False)
    # prepare environment
    place = fluid.CPUPlace()
    if args.use_gpu:
        place = fluid.CUDAPlace(0)

    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    # load dictionary
    dict_map = None
    if args.dict is not None and os.path.isfile(args.dict):
        dict_map = {}
        with open(args.dict) as dict_file:
            for i, word in enumerate(dict_file):
                dict_map[i] = word.strip()
        print "Loaded dict from %s" % args.dict

    # load init model
    model_dir = args.model_path
    model_file_name = None
    if not os.path.isdir(args.model_path):
        model_dir = os.path.dirname(args.model_path)
        model_file_name = os.path.basename(args.model_path)
    fluid.io.load_params(exe, dirname=model_dir, filename=model_file_name)
    print "Init model from: %s." % args.model_path

    batch_times = []
    iters = 0
    for data in infer_reader():
        if args.iterations > 0 and iters == args.iterations + args.skip_batch_num:
            break
        if iters < args.skip_batch_num:
            print("Warm-up itaration")
        if iters == args.skip_batch_num:
            profiler.reset_profiler()

        start = time.time()
        result = exe.run(fluid.default_main_program(),
                         feed=get_feeder_data(
                             data, place, need_label=False),
                         fetch_list=[sequence],
                         return_numpy=False)
        batch_time = time.time() - start
        fps = args.batch_size / batch_time
        batch_times.append(batch_time)
        indexes = np.array(result[0]).flatten()
        if dict_map is not None:
            print "Iteration %d, latency: %.5f s, fps: %f, result: %s" % (
                iters,
                batch_time,
                fps,
                [dict_map[index] for index in indexes], )
        else:
            print "Iteration %d, latency: %.5f s, fps: %f, result: %s" % (
                iters,
                batch_time,
                fps,
                indexes, )

        iters += 1

    latencies = batch_times[args.skip_batch_num:]
    latency_avg = np.average(latencies)
    latency_pc99 = np.percentile(latencies, 99)
    fpses = np.divide(args.batch_size, latencies)
    fps_avg = np.average(fpses)
    fps_pc99 = np.percentile(fpses, 1)

    # Benchmark output
    print('\nTotal examples (incl. warm-up): %d' % (iters * args.batch_size))
    print('average latency: %.5f s, 99pc latency: %.5f s' % (latency_avg,
                                                             latency_pc99))
    print('average fps: %.5f, fps for 99pc latency: %.5f' % (fps_avg, fps_pc99))


def main():
    args = parser.parse_args()
    print_arguments(args)
    if args.profile:
        if args.use_gpu:
            with profiler.cuda_profiler("cuda_profiler.txt", 'csv') as nvprof:
                inference(args, data_reader=ctc_reader)
        else:
            with profiler.profiler("CPU", sorted_key='total') as cpuprof:
                inference(args, data_reader=ctc_reader)
    else:
        inference(args, data_reader=ctc_reader)


if __name__ == "__main__":
    main()
