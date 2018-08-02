import paddle.v2 as paddle
import paddle.fluid as fluid
from utility import add_arguments, print_arguments, to_lodtensor, get_ctc_feeder_data, get_attention_feeder_for_infer
from crnn_ctc_model import ctc_infer
from attention_model import attention_infer
import numpy as np
import data_reader
import argparse
import functools
import os

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('model',    str,   "crnn_ctc",           "Which type of network to be used. 'crnn_ctc' or 'attention'")
add_arg('model_path',         str,  None,   "The model path to be used for inference.")
add_arg('input_images_dir',   str,  None,   "The directory of images.")
add_arg('input_images_list',  str,  None,   "The list file of images.")
add_arg('dict',               str,  None,   "The dictionary. The result of inference will be index sequence if the dictionary was None.")
add_arg('use_gpu',            bool,  True,      "Whether use GPU to infer.")
# yapf: enable


def inference(args):
    """OCR inference"""
    if args.model == "crnn_ctc":
        infer = ctc_infer
        get_feeder_data = get_ctc_feeder_data
    else:
        infer = attention_infer
        get_feeder_data = get_attention_feeder_for_infer

    num_classes = data_reader.num_classes()
    data_shape = data_reader.data_shape()
    # define network
    images = fluid.layers.data(name='pixel', shape=data_shape, dtype='float32')
    ids = infer(images, num_classes)

    # data reader
    infer_reader = data_reader.inference(
        infer_images_dir=args.input_images_dir,
        infer_list_file=args.input_images_list,
        model=args.model)
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

    for data in infer_reader():
        feed_dict = get_feeder_data(data, place)
        result = exe.run(fluid.default_main_program(),
                         feed=feed_dict,
                         fetch_list=[ids],
                         return_numpy=False)
        indexes = np.array(result[0]).flatten()
        if dict_map is not None:
            print "result: %s" % ([dict_map[index] for index in indexes], )
        else:
            print "result: %s" % (indexes, )


def main():
    args = parser.parse_args()
    print_arguments(args)
    inference(args)


if __name__ == "__main__":
    main()
