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

# A special mAP metric for COCO dataset, which averages AP in different IoUs.
# To use this eval_cocoMAP.py, [cocoapi](https://github.com/cocodataset/cocoapi) is needed.
import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('dataset',          str,   'coco2017',  "coco2014, coco2017.")
add_arg('batch_size',       int,   2,        "Minibatch size.")
add_arg('use_gpu',          bool,  True,      "Whether use GPU.")
add_arg('data_dir',         str,   'data/COCO17',        "The data root path.")
add_arg('model_dir',        str,   '',     "The model path.")
add_arg('nms_threshold',    float, 0.5,    "NMS threshold.")
add_arg('confs_threshold',  float, 0.5,    "Confidence threshold to draw bbox.")
add_arg('image_path',       str,   '',        "The image used to inference and visualize.")
add_arg('anchor_sizes',     int,    [32,64,128,256,512],  "The size of anchors.")
add_arg('aspect_ratios',    float,  [0.5,1.0,2.0],    "The ratio of anchors.")
add_arg('ap_version',       str,   'cocoMAP',   "cocoMAP.")
add_arg('max_size',         int,   1333,    "The resized image height.")
add_arg('scales', int,  [800],    "The resized image height.")
add_arg('mean_value',     float,   [102.9801, 115.9465, 122.7717], "pixel mean")

# yapf: enable


def eval(args):

    if '2014' in args.dataset:
        test_list = 'annotations/instances_val2014.json'
    elif '2017' in args.dataset:
        test_list = 'annotations/instances_val2017.json'

    image_shape = [3, args.max_size, args.max_size]
    class_nums = 81
    label_fpath = os.path.join(args.data_dir, test_list)
    coco = COCO(label_fpath)
    category_ids = coco.getCatIds()
    label_list = {
        item['id']: item['name']
        for item in coco.loadCats(category_ids)
    }
    label_list[0] = ['background']

    image = fluid.layers.data(name='image', shape=image_shape, dtype='float32')
    im_info = fluid.layers.data(name='im_info', shape=[3], dtype='float32')
    im_id = fluid.layers.data(name='im_id', shape=[1], dtype='int32')
    # model
    rpn_rois, confs, locs = FasterRcnn_test(
        input=image,
        depth=50,
        anchor_sizes=[32, 64, 128, 256, 512],
        variance=[1., 1., 1., 1.],
        aspect_ratios=[0.5, 1.0, 2.0],
        im_info=im_info,
        class_nums=class_nums)

    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    # yapf: disable
    if args.model_dir:
        def if_exist(var):
            return os.path.exists(os.path.join(args.model_dir, var.name))
        fluid.io.load_vars(exe, args.model_dir, predicate=if_exist)
    # yapf: enable
    test_reader = reader.test(args)

    dts_res = []
    fetch_list = [rpn_rois, confs, locs]
    #if True:
    for batch_id, data in enumerate(test_reader()):
        #data = test_reader()
        #print(batch_id)
        image, gt_box, gt_label, is_crowd, im_info, lod, im_id = data
        image_t = fluid.core.LoDTensor()
        image_t.set(image, place)

        im_info_t = fluid.core.LoDTensor()
        im_info_t.set(im_info, place)

        im_id_t = fluid.core.LoDTensor()
        im_id_t.set(im_id, place)
        feeding = {}
        feeding['image'] = image_t
        feeding['im_info'] = im_info_t
        feeding['im_id'] = im_id_t

        rpn_rois_v, confs_v, locs_v = exe.run(fluid.default_main_program(),
                                              feed=feeding,
                                              fetch_list=fetch_list,
                                              return_numpy=False)
        if batch_id % 20 == 0:
            print("Batch {0}".format(batch_id))
        print('im_info: {}'.format(im_info))
        new_lod, nmsed_out = get_nmsed_box(rpn_rois_v, confs_v, locs_v,
                                           class_nums, im_info)
        #draw_bounding_box_on_image(image_path, nmsed_out, 
        #			args.confs_threshold, label_list)
        dts_res += get_dt_res(new_lod, nmsed_out, data)
        break
    with open("detection_result.json", 'w') as outfile:
        json.dump(dts_res, outfile)
    print("start evaluate using coco api")
    cocoGt = COCO(os.path.join(data_args.data_dir, test_list))
    cocoDt = cocoGt.loadRes("detection_result.json")
    cocoEval = COCOeval(cocoGt, cocoDt, "bbox")
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()


def get_dt_res(lod, nmsed_out, data):
    dts_res = []
    nmsed_out_v = np.array(nmsed_out)
    real_batch_size = min(args.batch_size, len(data))
    assert (len(lod) == real_batch_size + 1), \
      "Error Lod Tensor offset dimension. Lod({}) vs. batch_size({})"\
                    .format(len(lod), batch_size)
    k = 0
    for i in range(real_batch_size):
        dt_num_this_img = lod[i + 1] - lod[i]
        image_id = int(data[-1][i])
        image_width = int(data[4][i][1])
        image_height = int(data[4][i][2])
        for j in range(dt_num_this_img):
            dt = nmsed_out_v[k]
            #print('k: {}'.format(k))
            k = k + 1
            xmin, ymin, xmax, ymax, score, category_id = dt.tolist()
            w = xmax - xmin + 1
            h = ymax - ymin + 1
            bbox = [xmin, ymin, w, h]
            dt_res = {
                'image_id': image_id,
                'category_id': int(category_id),
                'bbox': bbox,
                'score': score
            }
            print('de_res: {}'.format(dt_res))
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


if __name__ == '__main__':
    args = parser.parse_args()
    print_arguments(args)

    data_args = reader.Settings(args)
    eval(data_args)
