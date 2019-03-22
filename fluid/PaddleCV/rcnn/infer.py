import os
import time
import numpy as np
from eval_helper import *
import paddle
import paddle.fluid as fluid
import reader
from utility import print_arguments, parse_args
import models.model_builder as model_builder
import models.resnet as resnet
from config import cfg
from data_utils import DatasetPath


def infer():

    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval, Params

        data_path = DatasetPath('val')
        test_list = data_path.get_file_list()
        coco_api = COCO(test_list)
        cid = coco_api.getCatIds()
        cat_id_to_num_id_map = {
            v: i + 1
            for i, v in enumerate(coco_api.getCatIds())
        }
        category_ids = coco_api.getCatIds()
        labels_map = {
            cat_id_to_num_id_map[item['id']]: item['name']
            for item in coco_api.loadCats(category_ids)
        }
        labels_map[0] = 'background'
    except:
        print("The COCO dataset or COCO API is not exist, use the default "
              "mapping of class index and real category name on COCO17.")
        assert cfg.dataset == 'coco2017'
        labels_map = coco17_labels()

    image_shape = [3, cfg.TEST.max_size, cfg.TEST.max_size]
    class_nums = cfg.class_num

    model = model_builder.RCNN(
        add_conv_body_func=resnet.add_ResNet50_conv4_body,
        add_roi_box_head_func=resnet.add_ResNet_roi_conv5_head,
        use_pyreader=False,
        mode='infer')
    model.build_model(image_shape)
    pred_boxes = model.eval_bbox_out()
    if cfg.MASK_ON:
        masks = model.eval_mask_out()
    place = fluid.CUDAPlace(0) if cfg.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    # yapf: disable
    if not os.path.exists(cfg.pretrained_model):
        raise ValueError("Model path [%s] does not exist." % (cfg.pretrained_model))

    def if_exist(var):
        return os.path.exists(os.path.join(cfg.pretrained_model, var.name))
    fluid.io.load_vars(exe, cfg.pretrained_model, predicate=if_exist)
    # yapf: enable
    infer_reader = reader.infer(cfg.image_path)
    feeder = fluid.DataFeeder(place=place, feed_list=model.feeds())

    dts_res = []
    segms_res = []
    if cfg.MASK_ON:
        fetch_list = [pred_boxes, masks]
    else:
        fetch_list = [pred_boxes]
    data = next(infer_reader())
    im_info = [data[0][1]]
    result = exe.run(fetch_list=[v.name for v in fetch_list],
                     feed=feeder.feed(data),
                     return_numpy=False)
    pred_boxes_v = result[0]
    if cfg.MASK_ON:
        masks_v = result[1]
    new_lod = pred_boxes_v.lod()
    nmsed_out = pred_boxes_v
    image = None
    if cfg.MASK_ON:
        segms_out = segm_results(nmsed_out, masks_v, im_info)
        image = draw_mask_on_image(cfg.image_path, segms_out,
                                   cfg.draw_threshold)

    draw_bounding_box_on_image(cfg.image_path, nmsed_out, cfg.draw_threshold,
                               labels_map, image)


if __name__ == '__main__':
    args = parse_args()
    print_arguments(args)
    infer()
