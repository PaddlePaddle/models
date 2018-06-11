import os
import time
import numpy as np
import argparse
import functools
from PIL import Image
from PIL import ImageDraw

import paddle
import paddle.fluid as fluid
import reader
from pyramidbox import PyramidBox
from utility import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('use_gpu',          bool,  True,      "Whether use GPU.")
add_arg('use_pyramidbox',   bool,  False, "Whether use PyramidBox model.")
add_arg('confs_threshold',  float, 0.15,    "Confidence threshold to draw bbox.")
add_arg('image_path',       str,   '',        "The data root path.")
add_arg('model_dir',        str,   '',     "The model path.")
add_arg('resize_h',         int,   0,    "The resized image height.")
add_arg('resize_w',         int,   0,    "The resized image height.")
# yapf: enable


def draw_bounding_box_on_image(image_path, nms_out, confs_threshold):
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size

    for dt in nms_out:
        category_id, score, xmin, ymin, xmax, ymax = dt.tolist()
        if score < confs_threshold:
            continue
        bbox = dt[2:]
        xmin, ymin, xmax, ymax = bbox
        (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                      ymin * im_height, ymax * im_height)
        draw.line(
            [(left, top), (left, bottom), (right, bottom), (right, top),
             (left, top)],
            width=4,
            fill='red')
    image_name = image_path.split('/')[-1]
    print("image with bbox drawed saved as {}".format(image_name))
    image.save(image_name)


def infer(args, data_args):
    num_classes = 2
    infer_reader = reader.infer(data_args, args.image_path)
    data = infer_reader()

    if args.resize_h and args.resize_w:
        image_shape = [3, args.resize_h, args.resize_w]
    else:
        image_shape = data.shape[1:]

    fetches = []

    network = PyramidBox(
        image_shape,
        num_classes,
        sub_network=args.use_pyramidbox,
        is_infer=True)
    infer_program, nmsed_out = network.infer()
    fetches = [nmsed_out]

    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)

    model_dir = args.model_dir
    if not os.path.exists(model_dir):
        raise ValueError("The model path [%s] does not exist." % (model_dir))

    def if_exist(var):
        return os.path.exists(os.path.join(model_dir, var.name))

    fluid.io.load_vars(exe, model_dir, predicate=if_exist)

    feed = {'image': fluid.create_lod_tensor(data, [], place)}
    predict, = exe.run(infer_program,
                       feed=feed,
                       fetch_list=fetches,
                       return_numpy=False)
    predict = np.array(predict)
    draw_bounding_box_on_image(args.image_path, predict, args.confs_threshold)


if __name__ == '__main__':
    args = parser.parse_args()
    print_arguments(args)

    data_dir = 'data/WIDERFACE/WIDER_val/images/'
    file_list = 'label/val_gt_widerface.res'

    data_args = reader.Settings(
        data_dir=data_dir,
        resize_h=args.resize_h,
        resize_w=args.resize_w,
        mean_value=[104., 117., 123],
        apply_distort=False,
        apply_expand=False,
        ap_version='11point')
    infer(args, data_args=data_args)
