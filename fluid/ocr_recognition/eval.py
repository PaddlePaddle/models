import paddle.v2 as paddle
import paddle.fluid as fluid
from utility import add_arguments, print_arguments, to_lodtensor, get_feeder_data
from crnn_ctc_model import ctc_infer
from crnn_ctc_model import ctc_eval
import ctc_reader
import argparse
import functools
import os

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('model_path',         str,  None,   "The model path to be used for inference.")
add_arg('input_images_dir',   str,  None,   "The directory of images.")
add_arg('input_images_list',  str,  None,   "The list file of images.")
add_arg('use_gpu',            bool,  True,      "Whether use GPU to eval.")
# yapf: enable


def evaluate(args, eval=ctc_eval, data_reader=ctc_reader):
    """OCR inference"""
    num_classes = data_reader.num_classes()
    data_shape = data_reader.data_shape()
    # define network
    images = fluid.layers.data(name='pixel', shape=data_shape, dtype='float32')
    label = fluid.layers.data(
        name='label', shape=[1], dtype='int32', lod_level=1)
    evaluator, cost = eval(images, label, num_classes)

    # data reader
    test_reader = data_reader.test(
        test_images_dir=args.input_images_dir,
        test_list_file=args.input_images_list)

    # prepare environment
    place = fluid.CPUPlace()
    if use_gpu:
        place = fluid.CUDAPlace(0)

    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    # load init model
    model_dir = args.model_path
    model_file_name = None
    if not os.path.isdir(args.model_path):
        model_dir = os.path.dirname(args.model_path)
        model_file_name = os.path.basename(args.model_path)
    fluid.io.load_params(exe, dirname=model_dir, filename=model_file_name)
    print "Init model from: %s." % args.model_path

    evaluator.reset(exe)
    count = 0
    for data in test_reader():
        count += 1
        exe.run(fluid.default_main_program(), feed=get_feeder_data(data, place))
    avg_distance, avg_seq_error = evaluator.eval(exe)
    print "Read %d samples; avg_distance: %s; avg_seq_error: %s" % (
        count, avg_distance, avg_seq_error)


def main():
    args = parser.parse_args()
    print_arguments(args)
    evaluate(args, data_reader=ctc_reader)


if __name__ == "__main__":
    main()
