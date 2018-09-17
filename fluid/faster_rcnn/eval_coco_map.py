import os
import time
import numpy as np
import argparse
import functools
from eval_helper import get_nmsed_box
import paddle
import paddle.fluid as fluid
import reader
from fasterrcnn_model_test import FasterRcnn_test
from utility import add_arguments, print_arguments
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
# A special mAP metric for COCO dataset, which averages AP in different IoUs.
# To use this eval_cocoMAP.py, [cocoapi](https://github.com/cocodataset/cocoapi) is needed.
import models.model_builder as model_builder
import models.resnet as resnet
import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval, Params

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('dataset',          str,   'coco2017',  "coco2014, coco2017.")
add_arg('batch_size',       int,   1,        "Minibatch size.")
add_arg('use_gpu',          bool,  True,      "Whether use GPU.")
add_arg('data_dir',         str,   'data/COCO17',        "The data root path.")
add_arg('model_dir',        str,   '',     "The model path.")
add_arg('nms_threshold',    float, 0.3,    "NMS threshold.")
add_arg('score_threshold',    float, 0.05,    "score threshold for NMS.")
add_arg('confs_threshold',  float, 9.,    "Confidence threshold to draw bbox.")
add_arg('image_path',       str,   '',        "The image used to inference and visualize.")
add_arg('anchor_sizes',     int,    [32,64,128,256,512],  "The size of anchors.")
add_arg('aspect_ratios',    float,  [0.5,1.0,2.0],    "The ratio of anchors.")
add_arg('ap_version',       str,   'cocoMAP',   "cocoMAP.")
add_arg('max_size',         int,   1333,    "The resized image height.")
add_arg('scales', int,  [800],    "The resized image height.")
add_arg('mean_value',     float,   [102.9801, 115.9465, 122.7717], "pixel mean")
add_arg('class_num',        int,   81,          "Class number.")
add_arg('variance',         float,  [1.,1.,1.,1.],    "The variance of anchors.")

# yapf: enable


def eval(args):

    if '2014' in args.dataset:
        test_list = 'annotations/instances_val2014.json'
    elif '2017' in args.dataset:
        test_list = 'annotations/instances_val2017.json'

    image_shape = [3, args.max_size, args.max_size]
    class_nums = args.class_num
    batch_size = args.batch_size

    cocoGt = COCO(os.path.join(data_args.data_dir, test_list))
    numId_to_catId_map = {i + 1: v for i, v in enumerate(cocoGt.getCatIds())}
    category_ids = cocoGt.getCatIds()
    label_list = {
        item['id']: item['name']
        for item in cocoGt.loadCats(category_ids)
    }
    label_list[0] = ['background']
    print(label_list)

    model = model_builder.FasterRCNN(
        cfg=args,
        add_conv_body_func=resnet.add_ResNet50_conv4_body,
        add_roi_box_head_func=resnet.add_ResNet_roi_conv5_head,
        use_pyreader=False,
        is_train=False,
        use_random=False)
    model.build_model(image_shape)
    rpn_rois, confs, locs = model.eval_out()
    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    # yapf: disable
    if args.model_dir:
        def if_exist(var):
            return os.path.exists(os.path.join(args.model_dir, var.name))
        fluid.io.load_vars(exe, args.model_dir, predicate=if_exist)
    # yapf: enable
    test_reader = reader.test(args, batch_size)
    feeder = fluid.DataFeeder(place=place, feed_list=model.feeds())

    dts_res = []
    fetch_list = [rpn_rois, confs, locs]
    for batch_id, data in enumerate(test_reader()):
        start = time.time()
        #image, gt_box, gt_label, is_crowd, im_info, im_id = data[0]
        rpn_rois_v, confs_v, locs_v = exe.run(
            fetch_list=[v.name for v in fetch_list],
            feed=feeder.feed(data),
            return_numpy=False)

        im_info = []
        for i in range(len(data)):
            im_info.append(data[i][4])
        new_lod, nmsed_out = get_nmsed_box(args, rpn_rois_v, confs_v, locs_v,
                                           class_nums, im_info,
                                           numId_to_catId_map)
        for i in range(len(data)):
            if str(data[i][5]) in args.image_path:
                draw_bounding_box_on_image(args.image_path, nmsed_out,
                                           args.confs_threshold, label_list)
        dts_res += get_dt_res(new_lod, nmsed_out, data)
        end = time.time()
        print('batch id: {}, time: {}'.format(batch_id, end - start))
    with open("detection_result.json", 'w') as outfile:
        json.dump(dts_res, outfile)
    print("start evaluate using coco api")
    cocoDt = cocoGt.loadRes("detection_result.json")
    cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
    #cocoEval.params.imgIds = im_id
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()


def get_dt_res(lod, nmsed_out, data):
    dts_res = []
    nmsed_out_v = np.array(nmsed_out)
    assert (len(lod) == args.batch_size + 1), \
      "Error Lod Tensor offset dimension. Lod({}) vs. batch_size({})"\
                    .format(len(lod), batch_size)
    k = 0
    for i in range(args.batch_size):
        dt_num_this_img = lod[i + 1] - lod[i]
        image_id = int(data[i][-1])
        image_width = int(data[i][4][1])
        image_height = int(data[i][4][2])
        for j in range(dt_num_this_img):
            dt = nmsed_out_v[k]
            k = k + 1
            xmin, ymin, xmax, ymax, score, category_id = dt.tolist()
            w = xmax - xmin + 1
            h = ymax - ymin + 1
            bbox = [xmin, ymin, w, h]
            dt_res = {
                'image_id': image_id,
                'category_id': category_id,
                'bbox': bbox,
                'score': score
            }
            dts_res.append(dt_res)
    return dts_res


def draw_bounding_box_on_image(image_path, nms_out, confs_threshold,
                               label_list):
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size

    for dt in nms_out:
        xmin, ymin, xmax, ymax, score, category_id = dt.tolist()
        if score < confs_threshold:
            continue
        bbox = dt[:4]
        xmin, ymin, xmax, ymax = bbox
        draw.line(
            [(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin),
             (xmin, ymin)],
            width=4,
            fill='red')
        if image.mode == 'RGB':
            draw.text((xmin, ymin), label_list[int(category_id)], (255, 255, 0))
    image_name = image_path.split('/')[-1]
    print("image with bbox drawed saved as {}".format(image_name))
    image.save(image_name)


if __name__ == '__main__':
    args = parser.parse_args()
    print_arguments(args)

    data_args = reader.Settings(args)
    eval(data_args)
