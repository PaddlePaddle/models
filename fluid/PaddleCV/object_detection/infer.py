import os
import time
import numpy as np
import argparse
import functools
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

import paddle
import paddle.fluid as fluid
import reader
from mobilenet_ssd import mobile_net
from utility import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('dataset',          str,   'pascalvoc',    "coco and pascalvoc.")
add_arg('use_gpu',          bool,  True,      "Whether use GPU.")
add_arg('image_path',       str,   '',        "The image used to inference and visualize.")
add_arg('model_dir',        str,   '',     "The model path.")
add_arg('nms_threshold',    float, 0.45,   "NMS threshold.")
add_arg('confs_threshold',  float, 0.5,    "Confidence threshold to draw bbox.")
add_arg('resize_h',         int,   300,    "The resized image height.")
add_arg('resize_w',         int,   300,    "The resized image height.")
add_arg('mean_value_B',     float, 127.5,  "Mean value for B channel which will be subtracted.")  #123.68
add_arg('mean_value_G',     float, 127.5,  "Mean value for G channel which will be subtracted.")  #116.78
add_arg('mean_value_R',     float, 127.5,  "Mean value for R channel which will be subtracted.")  #103.94
# yapf: enable


def infer(args, data_args, image_path, model_dir):
    image_shape = [3, data_args.resize_h, data_args.resize_w]
    if 'coco' in data_args.dataset:
        num_classes = 91
        # cocoapi
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
        label_fpath = os.path.join(data_dir, label_file)
        coco = COCO(label_fpath)
        category_ids = coco.getCatIds()
        label_list = {
            item['id']: item['name']
            for item in coco.loadCats(category_ids)
        }
        label_list[0] = ['background']
    elif 'pascalvoc' in data_args.dataset:
        num_classes = 21
        label_list = data_args.label_list

    image = fluid.layers.data(name='image', shape=image_shape, dtype='float32')
    locs, confs, box, box_var = mobile_net(num_classes, image, image_shape)
    nmsed_out = fluid.layers.detection_output(
        locs, confs, box, box_var, nms_threshold=args.nms_threshold)

    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    # yapf: disable
    if model_dir:
        def if_exist(var):
            return os.path.exists(os.path.join(model_dir, var.name))
        fluid.io.load_vars(exe, model_dir, predicate=if_exist)
    # yapf: enable
    infer_reader = reader.infer(data_args, image_path)
    feeder = fluid.DataFeeder(place=place, feed_list=[image])

    data = infer_reader()

    # switch network to test mode (i.e. batch norm test mode)
    test_program = fluid.default_main_program().clone(for_test=True)
    nmsed_out_v, = exe.run(test_program,
                           feed=feeder.feed([[data]]),
                           fetch_list=[nmsed_out],
                           return_numpy=False)
    nmsed_out_v = np.array(nmsed_out_v)
    draw_bounding_box_on_image(image_path, nmsed_out_v, args.confs_threshold,
                               label_list)


def draw_bounding_box_on_image(image_path, nms_out, confs_threshold,
                               label_list):
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size

    for dt in nms_out:
        if dt[1] < confs_threshold:
            continue
        category_id = dt[0]
        bbox = dt[2:]
        xmin, ymin, xmax, ymax = clip_bbox(dt[2:])
        (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                      ymin * im_height, ymax * im_height)
        draw.line(
            [(left, top), (left, bottom), (right, bottom), (right, top),
             (left, top)],
            width=4,
            fill='red')
        if image.mode == 'RGB':
            draw.text((left, top), label_list[int(category_id)], (255, 255, 0))
    image_name = image_path.split('/')[-1]
    print("image with bbox drawed saved as {}".format(image_name))
    image.save(image_name)


def clip_bbox(bbox):
    xmin = max(min(bbox[0], 1.), 0.)
    ymin = max(min(bbox[1], 1.), 0.)
    xmax = max(min(bbox[2], 1.), 0.)
    ymax = max(min(bbox[3], 1.), 0.)
    return xmin, ymin, xmax, ymax


if __name__ == '__main__':
    args = parser.parse_args()
    print_arguments(args)

    data_dir = 'data/pascalvoc'
    label_file = 'label_list'

    if not os.path.exists(args.model_dir):
        raise ValueError("The model path [%s] does not exist." %
                         (args.model_dir))
    if 'coco' in args.dataset:
        data_dir = 'data/coco'
        label_file = 'annotations/instances_val2014.json'

    data_args = reader.Settings(
        dataset=args.dataset,
        data_dir=data_dir,
        label_file=label_file,
        resize_h=args.resize_h,
        resize_w=args.resize_w,
        mean_value=[args.mean_value_B, args.mean_value_G, args.mean_value_R],
        apply_distort=False,
        apply_expand=False,
        ap_version='')
    infer(
        args,
        data_args=data_args,
        image_path=args.image_path,
        model_dir=args.model_dir)
