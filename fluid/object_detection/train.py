import paddle.v2 as paddle
import paddle.fluid as fluid
import reader
import load_model as load_model
from mobilenet_ssd import mobile_net
from utility import add_arguments, print_arguments
import os
import time
import numpy as np
import argparse
import functools

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('learning_rate', float, 0.001, "Learning rate.")
add_arg('batch_size', int, 64, "Minibatch size.")
add_arg('num_passes', int, 20, "Epoch number.")
add_arg('parallel', bool, True, "Whether use parallel training.")
add_arg('use_gpu', bool, True, "Whether use GPU.")
add_arg('train_file_list', str,
        './data/COCO17/annotations/instances_train2017.json', "train file list")
add_arg('val_file_list', str,
        './data/COCO17/annotations/instances_val2017.json', "vaild file list")
add_arg('model_save_dir', str, 'model_coco_pretrain', "where to save model")

add_arg('dataset', str, 'coco', "coco or pascalvoc")
add_arg(
    'is_toy', int, 0,
    "Is Toy for quick debug, 0 means using all data, while n means using only n sample"
)
add_arg('data_dir', str, './data/COCO17', "Root path of data")
add_arg('label_file', str, 'label_list',
        "Lable file which lists all label name")
add_arg('apply_distort', bool, True, "Whether apply distort")
add_arg('apply_expand', bool, True, "Whether appley expand")
add_arg('resize_h', int, 300, "resize image size")
add_arg('resize_w', int, 300, "resize image size")
add_arg('mean_value_B', float, 127.5,
        "mean value which will be subtracted")  #123.68
add_arg('mean_value_G', float, 127.5,
        "mean value which will be subtracted")  #116.78
add_arg('mean_value_R', float, 127.5,
        "mean value which will be subtracted")  #103.94


def train(args,
          train_file_list,
          val_file_list,
          data_args,
          learning_rate,
          batch_size,
          num_passes,
          model_save_dir='model',
          init_model_path=None):
    image_shape = [3, data_args.resize_h, data_args.resize_w]

    image = fluid.layers.data(name='image', shape=image_shape, dtype='float32')
    gt_box = fluid.layers.data(
        name='gt_box', shape=[4], dtype='float32', lod_level=1)
    gt_label = fluid.layers.data(
        name='gt_label', shape=[1], dtype='int32', lod_level=1)
    difficult = fluid.layers.data(
        name='gt_difficult', shape=[1], dtype='int32', lod_level=1)

    if args.parallel:
        places = fluid.layers.get_places()
        pd = fluid.layers.ParallelDo(places)
        with pd.do():
            image_ = pd.read_input(image)
            gt_box_ = pd.read_input(gt_box)
            gt_label_ = pd.read_input(gt_label)
            difficult_ = pd.read_input(difficult)
            locs, confs, box, box_var = mobile_net(data_args, image_,
                                                   image_shape)
            loss = fluid.layers.ssd_loss(locs, confs, gt_box_, gt_label_, box,
                                         box_var)
            nmsed_out = fluid.layers.detection_output(
                locs, confs, box, box_var, nms_threshold=0.45)
            loss = fluid.layers.reduce_sum(loss)
            pd.write_output(loss)
            pd.write_output(nmsed_out)

        loss, nmsed_out = pd()
        loss = fluid.layers.mean(loss)
    else:
        locs, confs, box, box_var = mobile_net(data_args, image, image_shape)
        nmsed_out = fluid.layers.detection_output(
            locs, confs, box, box_var, nms_threshold=0.45)
        loss = fluid.layers.ssd_loss(locs, confs, gt_box, gt_label, box,
                                     box_var)
        loss = fluid.layers.reduce_sum(loss)

    test_program = fluid.default_main_program().clone(for_test=True)
    if data_args.dataset == 'coco':
        num_classes = 81
    elif data_args.dataset == 'pascalvoc':
        num_classes = 21
    with fluid.program_guard(test_program):
        map_eval = fluid.evaluator.DetectionMAP(
            nmsed_out,
            gt_label,
            gt_box,
            difficult,
            num_classes,
            overlap_threshold=0.5,
            evaluate_difficult=False,
            ap_version='11point')

    boundaries = [160000, 240000]
    values = [0.001, 0.0005, 0.00025]
    optimizer = fluid.optimizer.RMSProp(
        learning_rate=fluid.layers.piecewise_decay(boundaries, values),
        regularization=fluid.regularizer.L2Decay(0.00005), )

    optimizer.minimize(loss)

    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    load_model.load_and_set_vars(place)
    #load_model.load_paddlev1_vars(place)
    train_reader = paddle.batch(
        reader.train(data_args, train_file_list), batch_size=batch_size)
    test_reader = paddle.batch(
        reader.test(data_args, val_file_list), batch_size=batch_size)
    feeder = fluid.DataFeeder(
        place=place, feed_list=[image, gt_box, gt_label, difficult])

    def test(pass_id):
        _, accum_map = map_eval.get_map_var()
        map_eval.reset(exe)
        test_map = None
        for _, data in enumerate(test_reader()):
            test_map = exe.run(test_program,
                               feed=feeder.feed(data),
                               fetch_list=[accum_map])
        print("Test {0}, map {1}".format(pass_id, test_map[0]))

    for pass_id in range(num_passes):
        start_time = time.time()
        prev_start_time = start_time
        end_time = 0
        for batch_id, data in enumerate(train_reader()):
            prev_start_time = start_time
            start_time = time.time()
            #print("Batch {} start at {:.2f}".format(batch_id, start_time))
            loss_v = exe.run(fluid.default_main_program(),
                             feed=feeder.feed(data),
                             fetch_list=[loss])
            end_time = time.time()
            if batch_id % 20 == 0:
                print("Pass {0}, batch {1}, loss {2}, time {3}".format(
                    pass_id, batch_id, loss_v[0], start_time - prev_start_time))
        test(pass_id)

        if pass_id % 10 == 0:
            model_path = os.path.join(model_save_dir, str(pass_id))
            print 'save models to %s' % (model_path)
            fluid.io.save_inference_model(model_path, ['image'], [nmsed_out],
                                          exe)


if __name__ == '__main__':
    args = parser.parse_args()
    print_arguments(args)
    data_args = reader.Settings(
        dataset=args.dataset,  # coco or pascalvoc
        toy=args.is_toy,
        data_dir=args.data_dir,
        label_file=args.label_file,
        apply_distort=args.apply_distort,
        apply_expand=args.apply_expand,
        resize_h=args.resize_h,
        resize_w=args.resize_w,
        mean_value=[args.mean_value_B, args.mean_value_G, args.mean_value_R])
    train(
        args,
        train_file_list=args.train_file_list,
        val_file_list=args.val_file_list,
        data_args=data_args,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_passes=args.num_passes,
        model_save_dir=args.model_save_dir)
