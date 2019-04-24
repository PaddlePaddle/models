"""Infer for ICNet model."""
from __future__ import print_function
import cityscape
import argparse
import functools
import sys
import os
import cv2

import paddle.fluid as fluid
import paddle
from icnet import icnet
from utils import add_arguments, print_arguments, get_feeder_data
from paddle.fluid.layers.learning_rate_scheduler import _decay_step_counter
from paddle.fluid.initializer import init_on_cpu
import numpy as np

IMG_MEAN = np.array((103.939, 116.779, 123.68), dtype=np.float32)
parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('model_path',        str,   None,         "Model path.")
add_arg('images_list',       str,   None,         "List file with images to be infered.")
add_arg('images_path',       str,   None,         "The images path.")
add_arg('out_path',          str,   "./output",         "Output path.")
add_arg('use_gpu',           bool,  True,       "Whether use GPU to test.")
# yapf: enable

data_shape = [3, 1024, 2048]
num_classes = 19

label_colours = [
    [128, 64, 128],
    [244, 35, 231],
    [69, 69, 69]
    # 0 = road, 1 = sidewalk, 2 = building
    ,
    [102, 102, 156],
    [190, 153, 153],
    [153, 153, 153]
    # 3 = wall, 4 = fence, 5 = pole
    ,
    [250, 170, 29],
    [219, 219, 0],
    [106, 142, 35]
    # 6 = traffic light, 7 = traffic sign, 8 = vegetation
    ,
    [152, 250, 152],
    [69, 129, 180],
    [219, 19, 60]
    # 9 = terrain, 10 = sky, 11 = person
    ,
    [255, 0, 0],
    [0, 0, 142],
    [0, 0, 69]
    # 12 = rider, 13 = car, 14 = truck
    ,
    [0, 60, 100],
    [0, 79, 100],
    [0, 0, 230]
    # 15 = bus, 16 = train, 17 = motocycle
    ,
    [119, 10, 32]
]

# 18 = bicycle


def color(input):
    """
    Convert infered result to color image.
    """
    result = []
    for i in input.flatten():
        result.append(
            [label_colours[i][2], label_colours[i][1], label_colours[i][0]])
    result = np.array(result).reshape([input.shape[0], input.shape[1], 3])
    return result


def infer(args):
    data_shape = cityscape.test_data_shape()
    num_classes = cityscape.num_classes()
    # define network
    images = fluid.layers.data(name='image', shape=data_shape, dtype='float32')
    _, _, sub124_out = icnet(images, num_classes,
                             np.array(data_shape[1:]).astype("float32"))
    predict = fluid.layers.resize_bilinear(
        sub124_out, out_shape=data_shape[1:3])
    predict = fluid.layers.transpose(predict, perm=[0, 2, 3, 1])
    predict = fluid.layers.reshape(predict, shape=[-1, num_classes])
    _, predict = fluid.layers.topk(predict, k=1)
    predict = fluid.layers.reshape(
        predict,
        shape=[data_shape[1], data_shape[2], -1])  # batch_size should be 1
    inference_program = fluid.default_main_program().clone(for_test=True)
    # prepare environment
    place = fluid.CPUPlace()
    if args.use_gpu:
        place = fluid.CUDAPlace(0)
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())
    assert os.path.exists(args.model_path)
    fluid.io.load_params(exe, args.model_path)
    print("loaded model from: %s" % args.model_path)
    sys.stdout.flush()

    if not os.path.isdir(args.out_path):
        os.makedirs(args.out_path)

    for line in open(args.images_list):
        image_file = args.images_path + "/" + line.strip()
        filename = os.path.basename(image_file)
        image = paddle.dataset.image.load_image(
            image_file, is_color=True).astype("float32")
        image -= IMG_MEAN
        img = paddle.dataset.image.to_chw(image)[np.newaxis, :]
        image_t = fluid.LoDTensor()
        image_t.set(img, place)
        result = exe.run(inference_program,
                         feed={"image": image_t},
                         fetch_list=[predict])
        cv2.imwrite(args.out_path + "/" + filename + "_result.png",
                    color(result[0]))
    print("Saved images into: %s" % args.out_path)


def main():
    args = parser.parse_args()
    print_arguments(args)
    infer(args)


if __name__ == "__main__":
    main()
