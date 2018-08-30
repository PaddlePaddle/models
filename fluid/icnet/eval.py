"""Evaluator for ICNet model."""
import paddle.fluid as fluid
import numpy as np
from utils import add_arguments, print_arguments, get_feeder_data
from paddle.fluid.layers.learning_rate_scheduler import _decay_step_counter
from paddle.fluid.initializer import init_on_cpu
from icnet import icnet
import cityscape
import argparse
import functools
import sys
import os

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('model_path',        str,   None,         "Model path.")
add_arg('use_gpu',           bool,  True,       "Whether use GPU to test.")
# yapf: enable


def cal_mean_iou(wrong, correct):
    sum = wrong + correct
    true_num = (sum != 0).sum()
    for i in range(len(sum)):
        if sum[i] == 0:
            sum[i] = 1
    return (correct.astype("float64") / sum).sum() / true_num


def create_iou(predict, label, mask, num_classes, image_shape):
    predict = fluid.layers.resize_bilinear(predict, out_shape=image_shape[1:3])
    predict = fluid.layers.transpose(predict, perm=[0, 2, 3, 1])
    predict = fluid.layers.reshape(predict, shape=[-1, num_classes])
    label = fluid.layers.reshape(label, shape=[-1, 1])
    _, predict = fluid.layers.topk(predict, k=1)
    predict = fluid.layers.cast(predict, dtype="float32")
    predict = fluid.layers.gather(predict, mask)
    label = fluid.layers.gather(label, mask)
    label = fluid.layers.cast(label, dtype="int32")
    predict = fluid.layers.cast(predict, dtype="int32")
    iou, out_w, out_r = fluid.layers.mean_iou(predict, label, num_classes)
    return iou, out_w, out_r


def eval(args):
    data_shape = cityscape.test_data_shape()
    num_classes = cityscape.num_classes()
    # define network
    images = fluid.layers.data(name='image', shape=data_shape, dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int32')
    mask = fluid.layers.data(name='mask', shape=[-1], dtype='int32')

    _, _, sub124_out = icnet(images, num_classes,
                             np.array(data_shape[1:]).astype("float32"))
    iou, out_w, out_r = create_iou(sub124_out, label, mask, num_classes,
                                   data_shape)
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

    fetch_vars = [iou, out_w, out_r]
    out_wrong = np.zeros([num_classes]).astype("int64")
    out_right = np.zeros([num_classes]).astype("int64")
    count = 0
    test_reader = cityscape.test()
    for data in test_reader():
        count += 1
        result = exe.run(inference_program,
                         feed=get_feeder_data(
                             data, place, for_test=True),
                         fetch_list=fetch_vars)
        out_wrong += result[1]
        out_right += result[2]
        sys.stdout.flush()
    iou = cal_mean_iou(out_wrong, out_right)
    print("\nmean iou: %.3f" % iou)
    print("kpis	test_acc	%f" % iou)


def main():
    args = parser.parse_args()
    print_arguments(args)
    eval(args)


if __name__ == "__main__":
    main()
