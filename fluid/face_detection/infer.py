import os
import time
import numpy as np
import argparse
import functools
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

import paddle.fluid as fluid
import reader
from pyramidbox import PyramidBox
from widerface_eval import *
from visualize import draw_bboxes
from utility import add_arguments, print_arguments
parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)

# yapf: disable
add_arg('use_gpu',          bool,  True,   "Whether use GPU or not.")
add_arg('use_pyramidbox',   bool,  True,   "Whether use PyramidBox model.")
add_arg('confs_threshold',  float, 0.15,   "Confidence threshold to draw bbox.")
add_arg('image_path',       str,   '',     "The image used to inference and visualize.")
add_arg('model_dir',        str,   '',     "The model path.")
# yapf: enable


def infer(args, config):
    choose = (args.use_gpu, args.use_pyramidbox, args.model_dir)
    threshold = args.confs_threshold
    model_dir = args.model_dir
    image_path = args.image_path
    if not os.path.exists(model_dir):
        raise ValueError("The model path [%s] does not exist." % (model_dir))

    image = Image.open(image_path)
    if image.mode == 'L':
        image = img.convert('RGB')

    shrink, max_shrink = get_shrink(image.size[1], image.size[0])

    det0 = detect_face(image, shrink, choose)
    det1 = flip_test(image, shrink, choose)
    [det2, det3] = multi_scale_test(image, max_shrink, choose)
    det4 = multi_scale_test_pyramid(image, max_shrink, choose)
    det = np.row_stack((det0, det1, det2, det3, det4))
    dets = bbox_vote(det)
    keep_index = np.where(dets[:, 4] >= threshold)[0]
    dets = dets[keep_index, :]
    draw_bboxes(image_path, dets[:, 0:4])


if __name__ == '__main__':
    args = parser.parse_args()
    print_arguments(args)
    config = reader.Settings()
    infer(args, config)
