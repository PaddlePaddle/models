import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["FLAGS_fraction_of_gpu_memory_to_use"] = "0.3"
import sys
sys.path.insert(0, ".")
import argparse
import functools

import paddle.fluid as fluid
import reader
from utils import *
import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import tempfile

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('batch_size',       int,   32,        "Minibatch size.")
add_arg('data_dir',         str,   '',        "The data root path.")
add_arg('test_list',        str,   '',        "The testing data lists.")
add_arg('model_dir',        str,   '',     "The model path.")
add_arg('nms_threshold',    float, 0.45,    "NMS threshold.")
add_arg('ap_version',       str,   'cocoMAP',   "cocoMAP.")
add_arg('mean_value_B',     float, 127.5,  "Mean value for B channel which will be subtracted.")  #123.68
add_arg('mean_value_G',     float, 127.5,  "Mean value for G channel which will be subtracted.")  #116.78
add_arg('mean_value_R',     float, 127.5,  "Mean value for R channel which will be subtracted.")  #103.94

def use_coco_api_compute_mAP(data_args, test_list, num_classes, test_reader, exe, infer_program,
                             feeded_var_names, feeder, target_var, batch_size):
    cocoGt = COCO(os.path.join(data_args.data_dir, test_list))
    json_category_id_to_contiguous_id = {
        v: i + 1
        for i, v in enumerate(cocoGt.getCatIds())
    }
    contiguous_category_id_to_json_id = {
        v: k
        for k, v in json_category_id_to_contiguous_id.items()
    }

    dts_res = []

    executor = fluid.Executor(fluid.CUDAPlace(0))
    test_program = fluid.Program()
    with fluid.program_guard(test_program):
        boxes = fluid.layers.data(
            name='boxes', shape=[-1, -1, 4], dtype='float32')
        scores = fluid.layers.data(
            name='scores', shape=[-1, num_classes, -1], dtype='float32')
        pred_result = fluid.layers.multiclass_nms(
            bboxes=boxes,
            scores=scores,
            score_threshold=0.01,
            nms_top_k=-1,
            nms_threshold=0.45,
            keep_top_k=-1,
            normalized=False)

    executor.run(fluid.default_startup_program())

    for batch_id, data in enumerate(test_reader()):
        boxes_np, scores_np = exe.run(program=infer_program,
                                      feed={feeded_var_names[0]: feeder.feed(data)['image']},
                                      fetch_list=target_var)

        nms_out = executor.run(
            program=test_program,
            feed={
                'boxes': boxes_np,
                'scores': scores_np
            },
            fetch_list=[pred_result], return_numpy=False)
        if batch_id % 20 == 0:
            print("Batch {0}".format(batch_id))
        dts_res += get_batch_dt_res(nms_out, data, contiguous_category_id_to_json_id, batch_size)

    _, tmp_file = tempfile.mkstemp()
    with open(tmp_file, 'w') as outfile:
        json.dump(dts_res, outfile)
    print("start evaluate using coco api")
    cocoDt = cocoGt.loadRes(tmp_file)
    cocoEval = COCOeval(cocoGt, cocoDt, "bbox")
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    mAP = cocoEval.stats[0]
    return mAP

def compute_score(model_dir, data_dir, test_list='annotations/instances_val2017.json', batch_size=32, height=300, width=300, num_classes=81,
                          mean_value=[127.5, 127.5, 127.5]):
    """
        compute score, mAP, flops of a model

        Args:
            model_dir (string): directory of model
            data_dir (string): directory of coco dataset, like '/your/path/to/coco', '/work/datasets/coco'

        Returns:
            tuple: score, mAP, flops.

        """

    place = fluid.CUDAPlace(0)
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    [infer_program, feeded_var_names, target_var] = fluid.io.load_inference_model(dirname=model_dir, executor=exe)

    image_shape = [3, height, width]

    data_args = reader.Settings(
            dataset='coco2017',
            data_dir=data_dir,
            resize_h=height,
            resize_w=width,
            mean_value=mean_value,
            apply_distort=False,
            apply_expand=False,
            ap_version='cocoMAP')

    image = fluid.layers.data(name='image', shape=image_shape, dtype='float32')
    gt_box = fluid.layers.data(
        name='gt_box', shape=[4], dtype='float32', lod_level=1)
    gt_label = fluid.layers.data(
        name='gt_label', shape=[1], dtype='int32', lod_level=1)
    gt_iscrowd = fluid.layers.data(
        name='gt_iscrowd', shape=[1], dtype='int32', lod_level=1)
    gt_image_info = fluid.layers.data(
        name='gt_image_id', shape=[3], dtype='int32')

    test_reader = reader.test(data_args, test_list, batch_size)
    feeder = fluid.DataFeeder(
        place=place,
        feed_list=[image, gt_box, gt_label, gt_iscrowd, gt_image_info])

    mAP = use_coco_api_compute_mAP(data_args, test_list, num_classes, test_reader, exe, infer_program,
                             feeded_var_names, feeder, target_var, batch_size)
    total_flops_params, is_quantize = summary(infer_program)
    MAdds = np.sum(total_flops_params['flops']) / 2000000.0

    if is_quantize:
        MAdds /= 2.0

    print('mAP:', mAP)
    print('MAdds:', MAdds)

    if MAdds < 160.0:
        MAdds = 160.0

    if MAdds > 1300.0:
        score = 0.0
    else:
        score = mAP * 100 - (5.1249 * np.log(MAdds) - 14.499)

    print('score:', score)

    return score, mAP, MAdds


if __name__ == '__main__':
    args = parser.parse_args()
    print_arguments(args)
    score, mAP, flops = compute_score(args.model_dir, args.data_dir, batch_size=args.batch_size)
